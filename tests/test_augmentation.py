"""Regression checks for legacy blur and the retained PS augmentation."""
import json
import random

import cv2
import numpy as np
import pytest
import torch

from cine.augmentation import policy, validate_policy, schedule
from cine.data import DTLDDataset, process_image
from cine.loading import loader, shutdown_loader


@pytest.fixture
def manifest(tmp_path):
    rng = np.random.default_rng(8)
    image = rng.integers(0, 256, (1024, 2048, 3), dtype=np.uint8)
    path = tmp_path / 'frame.png'
    cv2.imwrite(str(path), image)
    rows = [
        dict(id=i, path=str(path), sequence='seq', boxes=boxes,
             relevance=rel, states=states, global_label=label)
        for i, (boxes, rel, states, label) in enumerate([
            ([], [], [], 2),
            ([[0, 8, 2, 12]], [1], [2], -1),
            ([[10, 20, 14, 30], [30, 20, 34, 30]], [1, 1], [0, 1], -1),
            ([[100, 100, 120, 140]], [1], [0], 0),
        ])
    ]
    file = tmp_path / 'rows.json'
    file.write_text(json.dumps(rows))
    return file, image


def test_legacy_blur_exact(manifest):
    file, image = manifest
    dataset = DTLDDataset(file, 1., augment_seed=42)
    expected = process_image(image)
    rng = random.Random(42 + 2 * 1000003 + 3 * 9176)
    rng.random()
    expected = cv2.GaussianBlur(expected, (3, 3), rng.uniform(.1, 1.))
    np.testing.assert_array_equal(
        dataset[(3, 2)]['img'].numpy(), expected[:, :, ::-1].transpose(2, 0, 1))


def test_ps_preserves_labels_geometry_padding_and_rng(manifest):
    file, _ = manifest
    clean = DTLDDataset(file)
    p = policy()
    p['clean_fraction'] = 0
    augmented = DTLDDataset(file, augmentation=p)
    numpy_before, random_before = np.random.get_state(), random.getstate()
    for i in range(4):
        a, b = clean[i], augmented[(i, 5)]
        for key in ('bboxes', 'states', 'relevance'):
            assert torch.equal(a[key], b[key])
        assert a['global_label'] == b['global_label']
        assert b['img'].shape == (3, 736, 1280)
        assert b['img'].dtype == torch.uint8
        assert (b['img'][:, :8] == 114).all()
        assert (b['img'][:, -8:] == 114).all()
        assert torch.equal(b['img'], augmented[(i, 5)]['img'])
    assert random.getstate() == random_before
    np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
    assert 'augmentation_operations' not in clean[0]


def test_ps_schedule_constraints():
    p = policy()
    p['clean_fraction'] = 0
    p['probabilities'] = {name: 1 for name in p['probabilities']}
    seen = set()
    for i in range(500):
        ops = schedule(p, 42, 0, i)
        assert len(ops) <= 2
        assert not ('blur' in ops and any(op in ops for op in ('motion', 'downsample')))
        seen.update(ops)
    assert {'exposure', 'gamma', 'contrast', 'noise', 'motion', 'downsample'} <= seen
    p['clean_fraction'] = 1
    assert schedule(p, 42, 0, 0) == []


def test_archived_ps_policy_compatible_but_other_groups_rejected():
    expected = policy()
    archived = dict(expected, version=1, groups=['P', 'S'],
                    disabled=['jpeg'])
    archived['probabilities'] = dict(expected['probabilities'], C=.2, A=.15)
    archived['ranges'] = dict(expected['ranges'], saturation=[.85, 1.],
                              gains=[.95, 1.05], jpeg=[75, 95],
                              haze=[.03, .10], haze_gray=[220, 255])
    assert validate_policy(archived) == expected
    archived['groups'] = ['P', 'C']
    with pytest.raises(ValueError, match='Only the frozen PS'):
        validate_policy(archived)


def test_worker_and_epoch_resume(manifest):
    file, _ = manifest
    p = policy()
    p['clean_fraction'] = 0
    p['probabilities'] = {name: 1 for name in p['probabilities']}
    data = DTLDDataset(file, augmentation=p)
    expected = {i: data[(i, 3)]['img'] for i in range(len(data))}
    batches = loader(data, 2, 2, shuffle=True, persistent_workers=True)
    try:
        for epoch in (0, 3, 3):
            batches.sampler.set_epoch(epoch)
            for batch in batches:
                if epoch == 3:
                    for idx, img in zip(batch['ids'], batch['img']):
                        assert torch.equal(img, expected[idx])
    finally:
        shutdown_loader(batches)


def test_invalid_ps_configuration():
    p = policy()
    p['groups'] = ['P', 'C']
    with pytest.raises(ValueError):
        validate_policy(p)
    p = policy()
    p['ranges']['gamma'] = [0, 1]
    with pytest.raises(ValueError):
        validate_policy(p)
