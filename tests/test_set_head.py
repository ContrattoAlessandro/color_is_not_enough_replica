import pytest
import torch
from cine.model import JointModel, SetEvidencePool, box_evidence
from cine.utils import read_config


def test_evidence_features_topk_and_invalid_padding():
    boxes = torch.tensor([[[0., 8., 1280., 728.], [0., 8., 640., 368.],
                           [20., 20., 10., 30.], [0., 8., 10., 20.]]], requires_grad=True)
    q = torch.tensor([[.8, .9, .99, float('nan')]], requires_grad=True)
    rel = torch.full((1, 4), .5, requires_grad=True)
    states = torch.tensor([[[.2, .3, .5]] * 4], requires_grad=True)
    features = [torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2)]
    e, valid = box_evidence(boxes, q, rel, states, features, topk=3)
    assert valid.tolist() == [[True, True, False]]
    assert not e.requires_grad
    torch.testing.assert_close(e[0, 0], torch.tensor([0., 0., .5, .5, torch.tensor(.25).log(), 0., .9, .45, .18, .27, .45]))
    torch.testing.assert_close(e[0, 1, :6], torch.tensor([0., 0., 1., 1., 0., 0.]))
    one, _ = box_evidence(boxes, q, rel, states, features, topk=1)
    torch.testing.assert_close(one[:, 0], e[:, 0])


def test_permutation_invariance_including_cutoff_ties():
    torch.manual_seed(3)
    n = 80
    xy = torch.rand(1, n, 2) * 100 + 20
    boxes = torch.cat([xy, xy + 10], -1)
    q = torch.ones(1, n) * .7
    rel = torch.rand(1, n)
    states = torch.randn(1, n, 3).softmax(-1)
    features = [torch.zeros(1, 1, 1, n)]
    pool = SetEvidencePool()
    e, v = box_evidence(boxes, q, rel, states, features)
    permutation = torch.randperm(n)
    e2, v2 = box_evidence(boxes[:, permutation], q[:, permutation], rel[:, permutation], states[:, permutation], features)
    torch.testing.assert_close(pool(e, v)[0], pool(e2, v2)[0])
    permutation = torch.randperm(64)
    torch.testing.assert_close(pool(e, v)[0], pool(e[:, permutation], v[:, permutation])[0])


def test_null_only_is_finite_and_trainable_and_padding_is_ignored():
    pool = SetEvidencePool()
    e = torch.randn(2, 64, 11, requires_grad=True)
    valid = torch.zeros(2, 64, dtype=torch.bool)
    pooled, attention = pool(e, valid)
    torch.testing.assert_close(pooled, pool.null_token[0].expand(2, -1))
    assert attention[:, :-1].count_nonzero() == 0
    assert (attention[:, -1] == 1).all()
    pooled.square().sum().backward()
    assert e.grad is None
    assert pool.null_token.grad.abs().sum() > 0
    valid[0, :3] = True
    expected, _ = pool(e, valid)
    changed = e.detach().clone()
    changed[~valid] = 1000
    torch.testing.assert_close(pool(changed, valid)[0], expected)


@pytest.mark.parametrize('amp', [False, True])
def test_global_detach_boundary_and_checkpoint_roundtrip(amp):
    if amp and not torch.cuda.is_available():
        pytest.skip('CUDA AMP')
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = read_config('configs/dtld_p2_set_null.yaml')
    config['model']['prior_conf'] = 0.0  # Exercise real tokens, not just NULL.
    model = JointModel(config, pretrained=False).to(device).eval()
    assert not hasattr(model, 'global_conv')
    x = torch.rand(1, 3, 128, 128, device=device)
    with torch.autocast(device, enabled=amp):
        p = model(x)
        loss = p['global_logits'].float().square().sum()
    assert p['evidence_valid'].any()
    assert not p['box_evidence'].requires_grad
    assert torch.isfinite(loss)
    scaler = torch.amp.GradScaler(device, enabled=amp)
    scaler.scale(loss).backward()
    for module in [model.detector.model[0], model.global_project, model.global_set_pool, model.global_classifier]:
        assert any(w.grad is not None and w.grad.abs().sum() > 0 for w in module.parameters())
    assert model.global_set_pool.null_token.grad.abs().sum() > 0
    for module in [model.detector.model[-1], model.attr_project, model.relevance, model.state]:
        assert all(w.grad is None or not w.grad.any() for w in module.parameters())
    restored = JointModel(config, pretrained=False).to(device).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad(), torch.autocast(device, enabled=amp):
        torch.testing.assert_close(restored(x)['global_logits'], p['global_logits'])


def test_nonfinite_boxes_are_excluded_before_clipping():
    boxes = torch.tensor([[[0., 8., float('inf'), 100.], [0., 8., 10., 100.]]])
    e, valid = box_evidence(boxes, torch.tensor([[.9, .01]]), torch.ones(1, 2),
                            torch.ones(1, 2, 3) / 3, [torch.zeros(1, 1, 1, 2)])
    assert not valid.any()
    assert torch.isfinite(e).all()
    assert e.count_nonzero() == 0
