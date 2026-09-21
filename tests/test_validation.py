import json
import numpy as np
import pytest
import torch
from cine.split import grouped_split, assert_disjoint
from cine.loading import EpochSampler, loader
from cine.monitoring import LossMeter, global_metrics


def rows():
    return [dict(id=i, sequence=f'city/seq{i // 3}', path=f'image{i}', image_sha256=f'hash{i}', global_label=i % 3) for i in range(60)]


def test_split_is_complete_disjoint_and_deterministic():
    original = rows()
    train, val = grouped_split(original, 0.1, 42)
    assert len(val) == 6
    assert len(train) + len(val) == len(original)
    assert {r['id'] for r in train} | {r['id'] for r in val} == set(range(60))
    assert_disjoint(dict(train=train, val=val))
    assert grouped_split(original, 0.1, 42) == (train, val)
    assert {r['sequence'] for r in grouped_split(original[::-1], 0.1, 42)[1]} == {r['sequence'] for r in val}


@pytest.mark.parametrize('field', ['sequence', 'path', 'image_sha256'])
def test_leakage_rejected_even_if_only_hash_matches(field):
    a, b = dict(sequence='a', path='a', image_sha256='a'), dict(sequence='b', path='b', image_sha256='b')
    b[field] = a[field]
    with pytest.raises(ValueError, match='leakage'):
        assert_disjoint(dict(train=[a], val=[b]))


def test_epoch_order_and_resume_do_not_depend_on_worker_state():
    sampler = EpochSampler(list(range(32)), seed=42)
    first = list(sampler)
    sampler.set_epoch(3)
    third = list(sampler)
    restored = EpochSampler(list(range(32)), seed=42)
    restored.set_epoch(3)
    assert list(restored) == third
    assert [i for i, e in first] != [i for i, e in third]
    assert sorted(i for i, e in third) == list(range(32))
    assert all(e == 3 for i, e in third)


def test_worker_zero_omits_multiprocessing_only_options():
    batches = loader(list(range(2)), 1, workers=0, prefetch_factor=4, persistent_workers=True)
    assert batches.num_workers == 0
    assert batches.prefetch_factor is None
    assert not batches.persistent_workers


def test_loss_meter_weights_attributes_and_masks_correctly():
    meter = LossMeter('cpu')
    names = meter.names[:-1]
    meter.update(torch.tensor(6.), {k: torch.tensor(1.) for k in names}, {'_foreground_count': torch.tensor(2)}, {'img': torch.zeros(2, 1), 'global_label': torch.tensor([0, -1])})
    meter.update(torch.tensor(12.), {k: torch.tensor(2.) for k in names}, {'_foreground_count': torch.tensor(6)}, {'img': torch.zeros(2, 1), 'global_label': torch.tensor([-1, -1])})
    result = meter.result()
    assert result['box'] == 1.5
    assert result['relevance'] == 1.75
    assert result['global_loss'] == 1.0
    assert result['total'] == 9.0


def test_global_metrics_exclude_ambiguous_frames():
    metrics = global_metrics([0, 1, 2, -1], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [1., 0., 0.]])
    assert metrics['global_mAP'] == 1
    assert metrics['balanced_accuracy'] == 1
    assert metrics['global_coverage'] == 0.75
    assert metrics['confusion_matrix'] == [[1,0,0], [0,1,0], [0,0,1]]

