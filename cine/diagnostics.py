"""Validation diagnostics; never substitutes for the COCO evaluator."""
import numpy as np

BINS = ('<4', '4-8', '8-16', '>=16')

def matched_boxes(boxes, prediction, threshold):
    """Score-ordered one-to-one matching, class agnostic, at confidence 0.05."""
    gt = np.asarray(boxes, dtype=float).reshape(-1, 4)
    used, matches = set(), []
    order = sorted(range(len(prediction['scores'])), key=lambda i: -prediction['scores'][i])
    for j in order:
        if prediction['scores'][j] < .05 or not len(gt):
            continue
        box = np.asarray(prediction['boxes'][j])
        inter = np.maximum(0, np.minimum(gt[:,2:],box[2:])-np.maximum(gt[:,:2],box[:2])).prod(1)
        union = (gt[:,2:]-gt[:,:2]).prod(1)+(box[2:]-box[:2]).prod()-inter
        ious = inter / np.maximum(union, 1e-12)
        for i in used:
            ious[i] = -1
        i = int(ious.argmax())
        if ious[i] >= threshold:
            used.add(i)
            matches.append((i,j))
    return matches

def local_diagnostics(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError('Rows and predictions differ')
    output = {}
    for threshold in (.5, .75):
        totals, matched = np.zeros(4,dtype=int), np.zeros(4,dtype=int)
        confusion = np.zeros((4,3,3),dtype=int)
        for row, pred in zip(rows, predictions):
            boxes = np.asarray(row['boxes']).reshape(-1,4)
            bins = np.searchsorted([4,8,16], boxes[:,2]-boxes[:,0], side='right')
            totals += np.bincount(bins,minlength=4)
            for i,j in matched_boxes(boxes,pred,threshold):
                b = bins[i]
                matched[b] += 1
                confusion[b,row['states'][i],int(np.argmax(pred['states'][j]))] += 1
        output[str(threshold)] = {name: dict(ground_truth=int(totals[i]),matched=int(matched[i]),
            recall=float(matched[i]/totals[i]) if totals[i] else None,
            state_accuracy=float(np.trace(confusion[i])/matched[i]) if matched[i] else None,
            state_confusion=confusion[i].tolist()) for i,name in enumerate(BINS)}
    return dict(confidence=.05, matching='score-ordered one-to-one, class-agnostic', width_bins=BINS, by_iou=output)
