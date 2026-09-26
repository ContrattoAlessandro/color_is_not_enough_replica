"""Full-resolution training/validation memory probe; run in a fresh process."""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cine.data import DTLDDataset, collate
from cine.model import JointModel
from cine.utils import read_config, move_batch, write_json, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = read_config(args.config)
    seed_everything(config['train']['seed'])
    torch.backends.cudnn.benchmark = config['train']['cudnn_benchmark']
    dataset = DTLDDataset(Path(config['data']['prepared']) / 'train.json')
    ids = sorted(range(len(dataset)), key=lambda i: -len(dataset.rows[i]['boxes']))[:args.batch]
    report = dict(config=args.config, batch=args.batch, images=[dataset.rows[i]['id'] for i in ids], accepted=False)
    try:
        model = JointModel(config).cuda().train()
        model.configure_global_weights(dataset.rows)
        batch = move_batch(collate([dataset[i] for i in ids]), 'cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
        scaler = torch.amp.GradScaler('cuda')
        updates = {'count': 0}
        def updated(*args):
            updates['count'] += 1
        optimizer.register_step_post_hook(updated)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(12):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.autocast('cuda'):
                prediction = model(batch['img'].float() / 255)
                loss, _ = model.loss(prediction, batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - begin)
            print(f'step={step} updates={updates["count"]} scale={scaler.get_scale()} seconds={times[-1]:.2f}', flush=True)
            del prediction, loss
            if updates['count'] >= 3 and step >= 4:
                break
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad(), torch.autocast('cuda'):
            prediction = model(batch['img'].float() / 255)
            validation_loss, _ = model.loss(prediction, batch)
            assert torch.isfinite(validation_loss)
        free, total = torch.cuda.mem_get_info()
        report.update(free_gb=free/2**30, total_gb=total/2**30,
                      peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30,
                      images_per_second=args.batch * min(3, len(times)) / sum(times[-3:]),
                      optimizer_updates=updates['count'], parameters=sum(p.numel() for p in model.parameters()),
                      feature_channels=model.feature_channels, transfer=model.transfer_report,
                      accepted=free >= 2**30 and updates['count'] >= 3)
    except torch.cuda.OutOfMemoryError:
        report.update(reason='CUDA out of memory')
    except Exception as error:
        report.update(reason=f'{type(error).__name__}: {error}')
        write_json(args.output, report)
        raise
    write_json(args.output, report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
