import numpy as np
import pytest
from cine.data import transform_boxes, inverse_boxes, global_label, STATE_MAP, collate
import torch


def test_geometry_round_trip_and_crop():
    original = np.array([[300, 100, 400, 200], [0, 100, 10, 200], [100, 10, 150, 50]], dtype=float)
    result, keep = transform_boxes(original)
    assert keep.tolist() == [True, False, True]
    np.testing.assert_allclose(inverse_boxes(result[:1]), original[:1], atol=1e-4)
    assert result[1, 0] == 0
    assert result[:, 1].min() >= 8


@pytest.mark.parametrize("rel,states,expected", [([], [], 2), ([0], [0], 2), ([1], [0], 0), ([1], [1], 1), ([1], [2], -1), ([1, 1], [0, 1], -1), ([1, 1], [1, 2], 1), ([1, 0], [0, 1], 0)])
def test_global_policy(rel, states, expected):
    assert global_label(rel, states) == expected


def test_state_mapping():
    assert [STATE_MAP[k] for k in ['red', 'yellow', 'red_yellow', 'green', 'off', 'unknown']] == [0, 0, 0, 1, 2, 2]


def test_collate_preserves_image_major_targets():
    def item(n, idx):
        return dict(img=torch.zeros(3, 32, 32, dtype=torch.uint8), bboxes=torch.zeros(n, 4), relevance=torch.arange(n).float(), states=torch.arange(n), global_label=-1, id=idx)
    b = collate([item(2, 0), item(0, 1), item(1, 2)])
    assert b['batch_idx'].tolist() == [0, 0, 2]
    assert b['states'].tolist() == [0, 1, 0]
