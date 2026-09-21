import copy
import torch
import pytest
from cine.model import JointModel, rasterize, select_detections
from cine.utils import read_config


def test_nms_retains_original_candidate_ids():
    boxes = torch.tensor([[20, 20, 40, 40], [21, 21, 40, 40], [100, 100, 110, 120]], dtype=torch.float)
    scores = torch.tensor([0.5, 0.9, 0.8])
    ids, _ = select_detections(boxes, scores, 0.1)
    assert ids.tolist() == [1, 2]
    attributes = torch.tensor([0.1, 0.2, 0.3])
    torch.testing.assert_close(attributes[ids], torch.tensor([0.2, 0.3]))


def test_masks_max_overlap_and_detachment():
    boxes = torch.tensor([[[0, 8, 100, 100], [50, 50, 150, 150]]], dtype=torch.float, requires_grad=True)
    scores = torch.tensor([[0.8, 0.6]], requires_grad=True)
    rel = torch.tensor([[0.25, 1.0]], requires_grad=True)
    states = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], requires_grad=True)
    features = [torch.zeros(1, 64, 92, 160)]
    masks = rasterize(boxes, scores, rel, states, features)
    assert not masks[0].requires_grad
    torch.testing.assert_close(masks[0][0, :, 8, 8], torch.tensor([0.8, 0.6, 0.8, 0.6, 0.]))
    empty = rasterize(boxes, scores * 0, rel, states, features)
    assert empty[0].count_nonzero() == 0


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(4)
    return JointModel(read_config('configs/dtld.yaml'), pretrained=False).cuda().eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU acceptance test")
def test_global_gradient_does_not_reach_local_heads(model):
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda', enabled=False):
        p = model(torch.rand(1, 3, 736, 1280, device='cuda'))
        p['global_logits'].square().sum().backward()
    assert model.detector.model[0].conv.weight.grad.abs().sum() > 0
    assert model.global_classifier.weight.grad.abs().sum() > 0
    for head in [model.detector.model[-1], model.attr_project, model.relevance, model.state]:
        assert all(x.grad is None or not x.grad.any() for x in head.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU acceptance test")
def test_empty_targets_and_masked_global_are_finite(model):
    model.zero_grad(set_to_none=True)
    batch = dict(batch_idx=torch.empty(0, device='cuda', dtype=torch.long), cls=torch.empty(0,1,device='cuda'), bboxes=torch.empty(0,4,device='cuda'), relevance=torch.empty(0,device='cuda'), states=torch.empty(0,device='cuda',dtype=torch.long), global_label=torch.tensor([-1],device='cuda'))
    with torch.autocast('cuda'):
        p = model(torch.rand(1,3,736,1280,device='cuda'))
        loss, components = model.loss(p,batch)
    assert torch.isfinite(loss)
    assert all(components[k] == 0 for k in ['relevance', 'state', 'global_loss'])
    loss.backward()


def test_foreground_targets_follow_assigner_indices(model):
    # Deliberately reverse matching order and include an empty middle image.
    class Criterion:
        def get_assigned_targets_and_loss(self, p, batch):
            fg = torch.tensor([[True, True], [False, False], [True, False]])
            ids = torch.tensor([[1, 0], [0, 0], [0, 0]])
            return (fg, ids), torch.zeros(3), {}
    old = model.criterion
    model.criterion = Criterion()
    pred = dict(raw={}, relevance_logits=torch.tensor([[10., -10.], [0., 0.], [10., 0.]]), state_logits=torch.tensor([[[0., 10., 0.], [10., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]], [[0., 0., 10.], [0., 0., 0.]]]), global_logits=torch.zeros(3,3))
    batch = dict(batch_idx=torch.tensor([0,0,2]), relevance=torch.tensor([0.,1.,1.]), states=torch.tensor([0,1,2]), global_label=torch.tensor([-1,-1,-1]))
    try:
        loss, parts = model.loss(pred,batch)
        assert parts['relevance'] < 0.001 and parts['state'] < 0.001
    finally:
        model.criterion = old
