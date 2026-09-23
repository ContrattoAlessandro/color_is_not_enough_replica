"""Isolated GPU memory probe using dense real training annotations."""
from pathlib import Path
import argparse
import time
import torch
from .data import DTLDDataset, collate
from .model import JointModel
from .utils import read_config, move_batch, write_json, seed_everything


def probe(batch_size, architecture, output):
    torch.set_num_threads(4)
    seed_everything(42)
    torch.backends.cudnn.benchmark = True
    config = read_config('configs/dtld.yaml')
    config['model']['architecture'] = architecture
    dataset = DTLDDataset(Path(config['data']['prepared'])/'train.json')
    ids = sorted(range(len(dataset)), key=lambda i: -len(dataset.rows[i]['boxes']))[:batch_size]
    report = dict(batch_size=batch_size, architecture=architecture, images=[dataset.rows[i]['id'] for i in ids])
    try:
        model = JointModel(config).cuda().train()
        model.configure_global_weights(dataset.rows)
        batch = move_batch(collate([dataset[i] for i in ids]),'cuda')
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        updates = {'count':0}
        def updated(*args): updates['count'] += 1
        optimizer.register_step_post_hook(updated)
        torch.cuda.reset_peak_memory_stats()
        times=[]
        for step in range(5):
            torch.cuda.synchronize(); start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                prediction=model(batch['img'].float()/255)
                loss,_=model.loss(prediction,batch)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite probe loss')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            torch.cuda.synchronize(); times.append(time.perf_counter()-start)
            del prediction,loss
        free,total=torch.cuda.mem_get_info()
        peak=torch.cuda.max_memory_reserved()
        report.update(free_gb=free/2**30, total_gb=total/2**30, peak_reserved_gb=peak/2**30,
                      images_per_second=batch_size*3/sum(times[2:]), optimizer_updates=updates['count'], accepted=free>=2**30 and updates['count']>0)
    except torch.cuda.OutOfMemoryError:
        report.update(accepted=False,reason='CUDA out of memory')
    write_json(output,report)
    print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--batch',type=int,required=True)
    p.add_argument('--architecture',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); probe(a.batch,a.architecture,a.output)
