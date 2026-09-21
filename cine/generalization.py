"""Validation-only sequence audits and cluster bootstrap uncertainty."""
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path

import numpy as np

from .data import GLOBAL_NAMES
from .monitoring import global_metrics
from .utils import write_json


def partition_counts(rows):
    valid = [r for r in rows if r['global_label'] >= 0]
    return dict(images=len(rows), sequences=len({r['sequence'] for r in rows}),
                classes={name: dict(images=sum(r['global_label'] == i for r in rows),
                                    sequences=len({r['sequence'] for r in rows if r['global_label'] == i}))
                         for i, name in enumerate(GLOBAL_NAMES)},
                masked_images=len(rows) - len(valid),
                masked_sequences=len({r['sequence'] for r in rows if r['global_label'] < 0}))


def scene_group(row):
    if row['global_label'] < 0:
        relevant_states = {s for r, s in zip(row['relevance'], row['states']) if r}
        return 'ambiguous_conflicting_states' if len(relevant_states - {2}) > 1 else 'ambiguous_unknown_only'
    if row['global_label'] == 2:
        return 'empty_scene' if not row['boxes'] else 'visible_irrelevant_lights'
    return 'relevant_lights'


def audit(rows, predictions, output):
    """Scene labels are annotation-derived; intersection exit requires review."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if len(rows) != len(predictions):
        raise ValueError('Prediction count differs from manifest')
    groups, sequences = defaultdict(list), defaultdict(list)
    reviewed = {}
    if (output / 'scene_review.csv').exists():
        with (output / 'scene_review.csv').open(newline='', encoding='utf-8') as f:
            reviewed = {int(r['id']): r for r in csv.DictReader(f)}
    records = []
    for row, pred in zip(rows, predictions):
        if (row['id'], row['sequence'], row['path']) != (pred['id'], pred['sequence'], pred['path']):
            raise ValueError('Predictions do not match manifest order/identity')
        probabilities = pred['probabilities']
        record = dict(id=row['id'], path=row['path'], sequence=row['sequence'],
                      label=row['global_label'], predicted=int(np.argmax(probabilities)),
                      probabilities=probabilities, scene_group=scene_group(row))
        records.append(record)
        groups[record['scene_group']].append(record)
        sequences[row['sequence']].append(record)

    def metrics(items):
        m = global_metrics([r['label'] for r in items], [r['probabilities'] for r in items])
        m.update(images=len(items), sequences=len({r['sequence'] for r in items}),
                 NoR_images=sum(r['label'] == 2 for r in items))
        return m

    summary = dict(counts=partition_counts(rows), overall=metrics(records),
                   scenes={name: metrics(items) for name, items in groups.items()},
                   sequences={name: metrics(items) for name, items in sequences.items()},
                   notes=['Scene groups use cropped annotation content, not visual semantic labels.',
                          'Intersection exits are unreviewed; final frames are review candidates only.',
                          'AP is undefined for absent classes; inspect per-class recall for homogeneous groups.'])
    summary['reviewed_scene_tags'] = {}
    for tag in ('intersection_exit', 'annotation_issue'):
        tagged = [r for r in records if reviewed.get(r['id'], {}).get(tag, '').lower() == 'yes'
                  and reviewed[r['id']].get('path') == r['path']]
        summary['reviewed_scene_tags'][tag] = dict(
            reviewed_images=sum(reviewed.get(r['id'], {}).get(tag, '').lower() in ('yes', 'no')
                                and reviewed[r['id']].get('path') == r['path'] for r in records),
            positive_images=len(tagged), metrics=metrics(tagged) if tagged else None)
    write_json(output / 'audit.json', summary)
    write_json(output / 'global_predictions.json', predictions)
    last_frames = {r['id'] for items in sequences.values()
                   for r in sorted(items, key=lambda x: x['path'])[-3:]}
    review = [r for r in records if r['label'] in (-1, 2)]
    review.sort(key=lambda r: (r['label'] != 2, r['predicted'] == r['label'], r['probabilities'][2]))
    with (output / 'scene_review.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'path', 'sequence', 'label', 'predicted',
                                               'NoR_probability', 'scene_group', 'sequence_tail_candidate',
                                               'intersection_exit', 'annotation_issue', 'review_notes'])
        writer.writeheader()
        for r in review:
            prior = reviewed.get(r['id'], {})
            if prior.get('path') != r['path']:
                prior = {}
            writer.writerow({k: r[k] for k in ('id', 'path', 'sequence', 'label', 'predicted', 'scene_group')} |
                            dict(NoR_probability=r['probabilities'][2], sequence_tail_candidate=r['id'] in last_frames,
                                 intersection_exit=prior.get('intersection_exit', 'unreviewed'),
                                 annotation_issue=prior.get('annotation_issue', 'unreviewed'),
                                 review_notes=prior.get('review_notes', '')))
    cards = []
    # One image per sequence before repeating sequences.
    chosen, seen = [], set()
    for r in review:
        if r['sequence'] not in seen:
            chosen.append(r)
            seen.add(r['sequence'])
        if len(chosen) == 60:
            break
    for r in chosen:
        truth = GLOBAL_NAMES[r['label']] if r['label'] >= 0 else 'masked'
        cards.append('<article><img loading="lazy" src="' + html.escape(Path(r['path']).as_uri(), quote=True) +
                     '"><p>' + html.escape(f"{truth} -> {GLOBAL_NAMES[r['predicted']]} | NoR={r['probabilities'][2]:.3f} | {r['sequence']}") +
                     '</p></article>')
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>NoR validation audit</title>'
                                      '<style>body{font:16px sans-serif;background:#eee}main{display:grid;grid-template-columns:1fr 1fr;gap:16px}'
                                      'img{width:100%}article{background:white;padding:10px}</style>'
                                      '<h1>NoR validation audit</h1><p>One example per sequence; errors first. '
                                      'Annotate intersection exits and label issues in scene_review.csv.</p><main>' +
                                      ''.join(cards) + '</main>', encoding='utf-8')
    return summary


def bootstrap_global(records, samples=1000, seed=42):
    """Resample whole sequences; weights preserve every frame's multiplicity."""
    from sklearn.metrics import average_precision_score
    valid = [r for r in records if r['label'] >= 0]
    if not valid:
        raise ValueError('No valid global labels for bootstrap')
    labels = np.array([r['label'] for r in valid])
    probabilities = np.array([r['probabilities'] for r in valid])
    _, inverse = np.unique([r['sequence'] for r in valid], return_inverse=True)
    n = inverse.max() + 1
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    for _ in range(samples):
        weights = np.bincount(rng.integers(0, n, n), minlength=n)[inverse]
        totals = np.bincount(labels, weights=weights, minlength=3)
        if np.any(totals == 0):
            continue
        aps = [average_precision_score(labels == i, probabilities[:, i], sample_weight=weights) for i in range(3)]
        recalls = [weights[(labels == i) & (probabilities.argmax(1) == i)].sum() / totals[i] for i in range(3)]
        for name, score in zip(GLOBAL_NAMES, aps):
            values[name + '_AP'].append(float(score))
        values['global_mAP'].append(float(np.mean(aps)))
        values['balanced_accuracy'].append(float(np.mean(recalls)))
    return dict(method='percentile 95% CI, whole-sequence resampling, valid global labels',
                requested_samples=samples, accepted_samples=len(values['global_mAP']), seed=seed,
                sequences=int(n), intervals={key: np.quantile(v, [.025, .975]).tolist() for key, v in values.items() if v})


def audit_checkpoint(checkpoint, output):
    """Recompute validation-only diagnostics for a retained checkpoint."""
    from .engine import load_checkpoint
    from .data import DTLDDataset
    from .loading import loader, shutdown_loader
    from .monitoring import validate_epoch
    model, saved = load_checkpoint(checkpoint)
    config = saved['config']
    dataset = DTLDDataset(Path(config['data']['prepared']) / 'val.json')
    ec = config['evaluate']
    batches = loader(dataset, ec['batch_size'], ec['workers'],
                     prefetch_factor=ec.get('prefetch_factor', 2),
                     persistent_workers=ec.get('persistent_workers', True))
    output = Path(output)
    try:
        validate_epoch(model, config, batches, output / 'metrics.json')
        predictions = json.loads((output / 'metrics.global.json').read_text(encoding='utf-8'))
        return audit(dataset.rows, predictions, output)
    finally:
        shutdown_loader(batches)


def summarize_seeds(metrics):
    values = {
        'global_mAP': [m['global_mAP'] for m in metrics],
        'balanced_accuracy': [m['balanced_accuracy'] for m in metrics],
        **{name + '_recall': [m.get('class_recall', {}).get(name,
             m['confusion_matrix'][i][i] / sum(m['confusion_matrix'][i])) for m in metrics]
           for i, name in enumerate(GLOBAL_NAMES)},
        **{name + '_AP': [m['global_AP'][name] for m in metrics] for name in GLOBAL_NAMES},
        **{name + '_AP50_95': [m[name]['AP50_95'] for m in metrics] for name in ('object', 'relevance', 'state')}}
    return {name: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                       runs=len(v)) for name, v in values.items()}
