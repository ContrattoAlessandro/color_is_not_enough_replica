"""Predeclared multi-task campaign selection rules (all values are fractions)."""
import math

KEYS = ('global_mAP', 'balanced_accuracy', 'NoR_AP', 'object', 'state', 'relevance')

def vector(m):
    return dict(global_mAP=m['global_mAP'], balanced_accuracy=m['balanced_accuracy'],
                NoR_AP=m['global_AP']['NoR'], **{k:m[k]['AP50_95'] for k in ('object','state','relevance')})

def score(m):
    v = vector(m)
    return sum(v[k] for k in ('global_mAP','object','state')) / 3

def eligible(m, reference):
    a, b = vector(m), vector(reference)
    return all(a[k] is not None and math.isfinite(a[k]) and a[k] >= b[k] - .005 for k in KEYS)

def selection_key(m, validation):
    mode = validation.get('selection_metric', 'global_mAP')
    if mode == 'global_mAP':
        return m['global_mAP']
    if mode != 'multitask':
        raise ValueError('Unknown selection metric')
    if not eligible(m, validation['reference_metrics']):
        return None
    return (score(m), m['global_AP']['NoR'])

def advances(m, reference):
    return eligible(m, reference) and score(m) - score(reference) >= .005

def confirms(candidates, baselines):
    if len(candidates) != 3 or len(baselines) != 3:
        raise ValueError('Confirmation requires three matched splits')
    deltas = [score(a)-score(b) for a,b in zip(candidates, baselines)]
    means = {k:sum(vector(a)[k]-vector(b)[k] for a,b in zip(candidates, baselines))/3 for k in KEYS}
    return sum(d > 0 for d in deltas) >= 2 and sum(deltas)/3 >= .005 and means['NoR_AP'] >= 0 and all(v >= -.005 for v in means.values())
