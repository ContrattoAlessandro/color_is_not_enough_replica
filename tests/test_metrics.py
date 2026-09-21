import pytest
from cine.evaluate import coco_metrics


@pytest.mark.parametrize('mode', ['object', 'relevance', 'state'])
def test_perfect_predictions_have_unit_ap(mode):
    rows = [dict(id=0, boxes=[[10,20,30,60]], relevance=[1], states=[0])]
    preds = [dict(boxes=[[10,20,30,60]], scores=[0.9], relevance=[0.9], states=[[0.9,0.05,0.05]])]
    assert coco_metrics(rows, preds, mode)['AP50_95'] == pytest.approx(1.0)


def test_empty_predictions_have_zero_ap():
    rows = [dict(id=0, boxes=[[10,20,30,60]], relevance=[1], states=[0])]
    preds = [dict(boxes=[], scores=[], relevance=[], states=[])]
    assert coco_metrics(rows, preds, 'object')['AP50_95'] == 0
