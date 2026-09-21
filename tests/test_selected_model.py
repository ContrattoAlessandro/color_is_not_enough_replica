from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from cine.model import JointModel
from cine.utils import read_config
from cine.generalization import audit, bootstrap_global, partition_counts


def selected_model():
    return JointModel(read_config('configs/dtld.yaml'), pretrained=False)


def test_training_weights_exclude_masked_labels_and_have_unit_expectation():
    model = selected_model()
    rows = [dict(global_label=i) for i, n in [(0, 9), (1, 16), (2, 1), (-1, 100)] for _ in range(n)]
    model.configure_global_weights(rows)
    assert model.global_class_counts == [9, 16, 1]
    weights = model.global_class_weights
    assert weights[2] > weights[0] > weights[1]
    torch.testing.assert_close((weights * torch.tensor([9, 16, 1]) / 26).sum(), torch.tensor(1.))
    with pytest.raises(ValueError, match='all three'):
        model.configure_global_weights([dict(global_label=0)])


def test_global_loss_uses_valid_image_mean_and_masks_gradients():
    model = selected_model()
    model.configure_global_weights([dict(global_label=i) for i in [0, 0, 0, 1, 1, 2]])
    class Criterion:
        def get_assigned_targets_and_loss(self, raw, batch):
            return (torch.zeros(3, 1, dtype=torch.bool), torch.zeros(3, 1, dtype=torch.long)), torch.zeros(3), {}
    model.criterion = Criterion()
    logits = torch.tensor([[2., 0., 0.], [0., 0., 0.], [10., 0., 0.]], requires_grad=True)
    pred = dict(raw={}, relevance_logits=torch.zeros(3, 1, requires_grad=True),
                state_logits=torch.zeros(3, 1, 3, requires_grad=True), global_logits=logits)
    batch = dict(batch_idx=torch.empty(0, dtype=torch.long), global_label=torch.tensor([0, 2, -1]))
    loss, parts = model.loss(pred, batch)
    expected = (F.cross_entropy(logits[:2], torch.tensor([0, 2]), reduction='none') *
                model.global_class_weights[[0, 2]]).mean()
    torch.testing.assert_close(parts['global_loss'], expected.detach())
    loss.backward()
    assert logits.grad[2].count_nonzero() == 0
    batch['global_label'][:] = -1
    assert model.loss(pred, batch)[1]['global_loss'] == 0


def test_selected_architecture_state_roundtrip():
    torch.set_num_threads(4)
    model = selected_model().eval()
    assert model.feature_channels == [128, 256, 512]
    assert all(m[0].in_channels == 69 for m in model.global_conv)
    restored = selected_model().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        expected = model(torch.zeros(1, 3, 64, 64))
        actual = restored(torch.zeros(1, 3, 64, 64))
    torch.testing.assert_close(expected['global_logits'], actual['global_logits'])
    assert actual['state_logits'].shape[1] == actual['boxes'].shape[1]


def test_checkpoint_restores_class_weights(tmp_path):
    from cine.engine import save_checkpoint, load_checkpoint
    model = selected_model()
    model.configure_global_weights([dict(global_label=i) for i in [0, 0, 1, 2]])
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, .95)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, model, optimizer, scheduler, scaler, 1, model.config, 4, {})
    loaded, saved = load_checkpoint(path, device='cpu')
    torch.testing.assert_close(loaded.global_class_weights, model.global_class_weights)
    assert loaded.global_class_counts == [2, 1, 1]
    saved.pop('global_class_weights')
    torch.save(saved, path)
    with pytest.raises(ValueError, match='missing saved'):
        load_checkpoint(path, device='cpu')


def test_validation_audit_and_bootstrap(tmp_path):
    rows = [dict(id=i, path=str(Path(f'image{i}.jpg').resolve()), sequence=f'seq{i//3}',
                 global_label=i % 3, boxes=[], relevance=[], states=[]) for i in range(12)]
    predictions = [dict(id=r['id'], path=r['path'], sequence=r['sequence'], label=r['global_label'],
                        probabilities=np.eye(3)[r['global_label']].tolist()) for r in rows]
    assert partition_counts(rows)['classes']['NoR'] == dict(images=4, sequences=4)
    report = audit(rows, predictions, tmp_path)
    assert report['overall']['global_mAP'] == 1.
    intervals = bootstrap_global(predictions, samples=40)
    assert intervals['sequences'] == 4
    assert intervals['intervals']['global_mAP'] == [1., 1.]
    predictions[0]['id'] = 999
    with pytest.raises(ValueError, match='identity'):
        audit(rows, predictions, tmp_path)


def test_global_metrics_for_entirely_masked_groups():
    from cine.monitoring import global_metrics
    for labels, probabilities in [([], []), ([-1], [[.2, .3, .5]])]:
        metrics = global_metrics(labels, probabilities)
        assert metrics['global_mAP'] is None
        assert metrics['balanced_accuracy'] is None
        assert metrics['confusion_matrix'] == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
