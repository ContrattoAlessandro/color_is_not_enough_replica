"""GPU-side loss aggregation and per-epoch validation, without test-set access."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix

from .data import GLOBAL_NAMES
from .utils import move_batch, write_json


class LossMeter:
    names = ('box', 'cls', 'dfl', 'relevance', 'state', 'global_loss', 'total')

    def __init__(self, device='cuda'):
        self.sums = torch.zeros(7, device=device, dtype=torch.float64)
        self.counts = torch.zeros_like(self.sums)

    def update(self, total, components, prediction, batch):
        n = self.sums.new_tensor(len(batch['img']))
        fg = prediction['_foreground_count'].to(self.sums)
        valid = (batch['global_label'] >= 0).sum().to(self.sums)
        weights = torch.stack([n, n, n, fg, fg, valid, n])
        values = torch.stack([components[k].detach() for k in self.names[:-1]] + [total.detach()]).to(self.sums)
        self.sums += values * weights
        self.counts += weights

    def result(self):
        return dict(zip(self.names, (self.sums / self.counts.clamp_min(1)).cpu().tolist()))


def global_metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities).reshape(-1, 3)
    valid = labels >= 0
    aps = {name: float(average_precision_score(labels[valid] == i, probabilities[valid, i])) if np.any(labels[valid] == i) else None for i, name in enumerate(GLOBAL_NAMES)}
    # Macro AP is undefined if any of the required classes is absent.
    macro = float(np.mean(list(aps.values()))) if all(v is not None for v in aps.values()) else None
    matrix = confusion_matrix(labels[valid], probabilities[valid].argmax(1), labels=[0, 1, 2]) if valid.any() else np.zeros((3, 3), dtype=int)
    recall = {name: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else None for i, name in enumerate(GLOBAL_NAMES)}
    return dict(class_recall=recall, global_valid=int(valid.sum()), global_masked=int((~valid).sum()), global_coverage=float(valid.mean()) if len(labels) else 0.0, global_AP=aps, global_mAP=macro, balanced_accuracy=float(np.mean([v for v in recall.values() if v is not None])) if valid.any() else None, confusion_matrix=matrix.tolist(), class_order=GLOBAL_NAMES)


@torch.no_grad()
def validate_epoch(model, config, batches, output=None):
    from .evaluate import coco_metrics, serialize
    was_training = model.training
    model.eval()
    meter, predictions = LossMeter(), []
    begin = time.perf_counter()
    try:
        for step, batch in enumerate(batches):
            batch = move_batch(batch, 'cuda')
            with torch.autocast('cuda', enabled=config['train']['amp']):
                prediction = model(batch['img'].float() / 255)
                loss, components = model.loss(prediction, batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite validation loss')
            meter.update(loss, components, prediction, batch)
            predictions.extend(serialize(d) for d in model.detections(prediction, config['evaluate']['conf']))
            if step % 100 == 0:
                print(f'Validation: {len(predictions)}/{len(batches.dataset)}', flush=True)
        rows = batches.dataset.rows
        metrics = dict(images=len(rows), losses=meter.result(), **global_metrics([r['global_label'] for r in rows], [p['global_probabilities'] for p in predictions]))
        for mode in ['object', 'relevance', 'state']:
            metrics[mode] = coco_metrics(rows, predictions, mode)
        from .diagnostics import local_diagnostics
        metrics['local_diagnostics'] = local_diagnostics(rows, predictions)
        metrics['seconds'] = time.perf_counter() - begin
        if output:
            write_json(output, metrics)
            write_json(Path(output).with_suffix('.global.json'), [
                dict(id=r['id'], path=r['path'], sequence=r['sequence'],
                     label=r['global_label'], probabilities=p['global_probabilities'])
                for r, p in zip(rows, predictions)])
        return metrics
    finally:
        model.train(was_training)


def plot_history(history, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.flat[:3], ['total', 'state', 'global_loss']):
        ax.plot(epochs, [h['train_losses'][name] for h in history], label='train')
        ax.plot(epochs, [h['validation']['losses'][name] for h in history], label='validation')
        ax.set_title(name + ' loss')
        ax.legend()
    ax = axes[1, 0]
    for mode in ['object', 'relevance', 'state']:
        ax.plot(epochs, [h['validation'][mode]['AP50_95'] for h in history], label=mode)
    ax.set_title('Validation local AP50:95')
    ax.legend()
    ax = axes[1, 1]
    for metric in ['global_mAP', 'balanced_accuracy']:
        ax.plot(epochs, [h['validation'][metric] for h in history], label=metric)
    ax.set_title('Validation global metrics')
    ax.legend()
    ax = axes[1, 2]
    for name in GLOBAL_NAMES:
        ax.plot(epochs, [h['validation']['global_AP'][name] for h in history], label=name)
    ax.set_title('Validation per-class AP')
    ax.legend()
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.2)
    fig.suptitle('Train vs clean sequence-held-out validation (no test metrics)')
    fig.tight_layout()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Windows can reject reopening an existing PNG while a viewer or scanner holds it.
    temporary = output.with_name(f'{output.stem}.{os.getpid()}.tmp.png')
    fig.savefig(temporary, dpi=140)
    try:
        os.replace(temporary, output)
    except OSError:
        # A locked previous plot must not invalidate a completed training epoch.
        fallback = output.with_name(f'{output.stem}.epoch_{epochs[-1]:03d}.png')
        os.replace(temporary, fallback)
    plt.close(fig)
