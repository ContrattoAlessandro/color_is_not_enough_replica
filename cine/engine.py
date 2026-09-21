"""Training with persistent workers, clean per-epoch validation and resumable state."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import DTLDDataset
from .loading import loader, shutdown_loader
from .model import JointModel
from .utils import move_batch, seed_everything, sha256, versions, write_json


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, config, batch_size, fingerprints, best_score=None, history=None):
    obj = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch, config=config, batch_size=batch_size, fingerprints=fingerprints, versions=versions(), transfer_report=model.transfer_report, best_score=best_score, history=history or [], rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    obj['global_class_weights'] = model.global_class_weights.detach().cpu().tolist()
    obj['global_class_counts'] = model.global_class_counts
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def load_checkpoint(path, device='cuda'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = JointModel(checkpoint['config'], pretrained=False).to(device)
    model.load_state_dict(checkpoint['model'])
    model.transfer_report = checkpoint.get('transfer_report', {})
    if checkpoint['config']['train'].get('global_weighting', 'none') != 'none' and 'global_class_weights' not in checkpoint:
        raise ValueError('Weighted checkpoint is missing saved training class weights')
    model.global_class_weights.copy_(torch.tensor(checkpoint.get('global_class_weights', [1., 1., 1.]), device=device))
    model.global_class_counts = checkpoint.get('global_class_counts')
    return model, checkpoint


def train(config, resume=None, epochs=None, limit=None, output=None, batch_size=None):
    from .monitoring import LossMeter, validate_epoch, plot_history
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    tc = config['train']
    out = Path(output or tc['output'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and not resume:
        raise FileExistsError(f'Use --resume or a fresh output directory: {out}')
    prepared = Path(config['data']['prepared'])
    if not (prepared / 'audit.json').is_file():
        raise RuntimeError('Audited manifests are required')
    validation_enabled = config.get('validation', {}).get('enabled', False)
    names = ['train', 'val', 'test'] if validation_enabled else ['train', 'test']
    fingerprints = {name: sha256(prepared / f'{name}.json') for name in names}
    if validation_enabled:
        from .split import assert_disjoint
        assert_disjoint({name: json.loads((prepared / f'{name}.json').read_text(encoding='utf-8')) for name in names})
    dataset = DTLDDataset(prepared / 'train.json', tc['blur_probability'], limit, tc['seed'])
    max_epochs = epochs or tc['epochs']
    if max_epochs < 1:
        raise ValueError('epochs must be positive')
    seed_everything(tc['seed'])
    torch.backends.cudnn.benchmark = tc.get('cudnn_benchmark', False)
    if resume:
        model, checkpoint = load_checkpoint(resume)
        if checkpoint['fingerprints'] != fingerprints or checkpoint['config'] != config:
            raise ValueError('Configuration or dataset changed since checkpoint; clean restart required')
        bs = checkpoint['batch_size']
    else:
        bs = batch_size or tc['batch_size']
        model = JointModel(config).cuda()
    previous_weights = model.global_class_weights.clone()
    model.configure_global_weights(dataset.rows)
    if resume and not torch.equal(previous_weights, model.global_class_weights):
        raise ValueError('Checkpoint global weights differ from training partition')
    if bs < 1 or tc['effective_batch'] % bs:
        raise ValueError('Microbatch must divide effective batch')
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc['lr'], weight_decay=tc['weight_decay'])
    update_counter = {'steps': 0}
    def count_update(optimizer, args, kwargs):
        update_counter['steps'] += 1
    update_hook = optimizer.register_step_post_hook(count_update)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=tc['gamma'])
    scaler = torch.amp.GradScaler('cuda', enabled=tc['amp'])
    start_epoch, best_score, history = 0, None, []
    if resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        if max_epochs < start_epoch:
            raise ValueError('Checkpoint already reached requested epoch count')
        best_score, history = checkpoint.get('best_score'), checkpoint.get('history', [])
        rng = checkpoint['rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state_all(rng['cuda'])
    write_json(out / 'experiment.json', dict(config=config, versions=versions(), fingerprints=fingerprints, batch_size=bs, samples=len(dataset), requested_epochs=max_epochs, transfer=model.transfer_report, feature_channels=model.feature_channels, global_class_weights=model.global_class_weights.cpu().tolist(), global_class_counts=model.global_class_counts, parameters=sum(p.numel() for p in model.parameters()), initialization='resumed clean-split checkpoint' if resume else 'fresh COCO compatible tensors; no weights from previous run or benchmark', loss_averaging='detector/total by images, attributes by assigned foreground candidates, global weighted sum divided by valid image count'))
    batches = loader(dataset, bs, tc['workers'], True, tc['seed'], tc.get('prefetch_factor', 2), tc.get('persistent_workers', False))
    val_batches = None
    if validation_enabled:
        ec = config['evaluate']
        validation = DTLDDataset(prepared / 'val.json')
        val_batches = loader(validation, ec['batch_size'], ec['workers'], False, tc['seed'], ec.get('prefetch_factor', 2), ec.get('persistent_workers', True))
    accumulation = tc['effective_batch'] // bs
    begin = time.perf_counter()
    try:
        for epoch in range(start_epoch, max_epochs):
            updates_before = update_counter['steps']
            model.train()
            batches.sampler.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            meter, seen, data_wait = LossMeter(), 0, 0.0
            epoch_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            iterator = iter(batches)
            for step in range(len(batches)):
                waiting = time.perf_counter()
                cpu_batch = next(iterator)
                data_wait += time.perf_counter() - waiting
                batch = move_batch(cpu_batch, 'cuda')
                n = len(batch['img'])
                if step % accumulation == 0:
                    group_samples = min(tc['effective_batch'], len(dataset) - seen)
                with torch.autocast('cuda', enabled=tc['amp']):
                    prediction = model(batch['img'].float() / 255)
                    loss, components = model.loss(prediction, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch {epoch + 1}, step {step}')
                scaler.scale(loss * n / group_samples).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(batches):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                meter.update(loss, components, prediction, batch)
                seen += n
                if step % 100 == 0 or step + 1 == len(batches):
                    status = dict(phase='training', epoch=epoch + 1, epochs=max_epochs, step=step + 1, steps=len(batches), seen=seen, images_per_second=seen / (time.perf_counter() - epoch_start), loss=meter.result(), elapsed_seconds=time.perf_counter() - begin)
                    write_json(out / 'status.json', status)
                    print(json.dumps(status), flush=True)
                del prediction, loss, batch, components
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - epoch_start
            row = dict(epoch=epoch + 1, train_seconds=elapsed, images_per_second=seen / elapsed, data_wait_seconds=data_wait, peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30, train_losses=meter.result())
            row['optimizer_updates'] = update_counter['steps'] - updates_before
            row['amp_skipped_updates'] = (len(batches) + accumulation - 1) // accumulation - row['optimizer_updates']
            if row['optimizer_updates'] == 0:
                raise RuntimeError('All optimizer steps were skipped; refusing to advance the learning-rate schedule or save a trained epoch')
            improved = False
            if val_batches is not None:
                write_json(out / 'status.json', dict(phase='validation', epoch=epoch + 1, epochs=max_epochs, images=len(val_batches.dataset)))
                metrics = validate_epoch(model, config, val_batches, out / 'validation' / f'epoch_{epoch + 1:03d}.json')
                row['validation'] = metrics
                score = metrics['global_mAP']
                improved = score is not None and (best_score is None or score > best_score)
                if improved:
                    best_score = score
                row['best_validation_global_mAP'] = best_score
                print('Validation metrics: ' + json.dumps(metrics), flush=True)
            scheduler.step()
            row['next_lr'] = scheduler.get_last_lr()[0]
            history.append(row)
            # Rewrite atomically, so resume cannot duplicate a partially logged epoch.
            write_json(out / 'history.json', history)
            if val_batches is not None:
                plot_history(history, out / 'learning_curves.png')
            save_checkpoint(out / 'last.pt', model, optimizer, scheduler, scaler, epoch + 1, config, bs, fingerprints, best_score, history)
            if improved:
                save_checkpoint(out / 'best.pt', model, optimizer, scheduler, scaler, epoch + 1, config, bs, fingerprints, best_score, history)
            print(f'Saved epoch {epoch + 1}; best validation global mAP: {best_score}', flush=True)
        save_checkpoint(out / 'final.pt', model, optimizer, scheduler, scaler, max_epochs, config, bs, fingerprints, best_score, history)
        write_json(out / 'status.json', dict(phase='training_complete', epochs=max_epochs, elapsed_seconds=time.perf_counter() - begin))
        return out / 'final.pt'
    finally:
        update_hook.remove()
        shutdown_loader(batches)
        if val_batches is not None:
            shutdown_loader(val_batches)
