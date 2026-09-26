"""Live view of queue progress and train/validation divergence.

From the repository root: python scripts/watch_yolov8m_comparison.py
Use --once for one snapshot; Ctrl+C exits the live view.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'artifacts/yolov8m_comparison'
RUNS = [('raster_seed42','Medium raster'),('set_null_seed42','Medium Top-64/NULL')]


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def pct(value):
    return '--' if value is None else f'{100*value:.2f}%'


def loss(value):
    return '--' if value is None else f'{value:.3f}'


def snapshot():
    print('\n' + datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))
    queue=read(ROOT/'status.json') or {}
    print(f"Queue: {queue.get('phase','not started')}" + (f" | current run: {queue['run']}" if queue.get('run') else '') + (f" | error: {queue['error']}" if queue.get('error') else ''))
    print('Run                  Stage                 Progress     Train loss  Val loss  Train/val global loss  Val mAP latest/best  RR recall  NoR AP')
    for directory,label in RUNS:
        folder=ROOT/directory
        state=read(folder/'status.json') or {}
        history=read(folder/'history.json') or []
        validated=[row for row in history if row.get('validation')]
        last=validated[-1] if validated else {}
        val=last.get('validation',{})
        val_loss=val.get('losses',{})
        train_loss=last.get('train_losses',{})
        current=state.get('loss',{})
        step='--'
        if state.get('epoch'):
            step=f"{state['step']}/{state['steps']}"
            progress=f"{state['epoch']}/{state.get('epochs','?')} ({step})"
        elif state.get('phase')=='training_complete':
            progress=f"{state.get('epochs',len(history))}/{state.get('epochs',len(history))}"
        else:
            progress=state.get('phase','queued')
        best=max((row['validation'].get('global_mAP') for row in validated if row['validation'].get('global_mAP') is not None),default=None)
        latest=val.get('global_mAP')
        tl=train_loss.get('global_loss')
        vl=val_loss.get('global_loss')
        divergence='--' if tl is None or vl is None else f'{tl:.3f}/{vl:.3f}'
        print(f"{label:<21} {state.get('phase','queued'):<21} {progress:<12} {loss(current.get('total')):<11} {loss(val_loss.get('total')):<9} {divergence:<22} {pct(latest)}/{pct(best):<9} {pct(val.get('class_recall',{}).get('RR')):<10} {pct(val.get('global_AP',{}).get('NoR'))}")
        if len(validated)>=2:
            previous=validated[-2]['validation'].get('global_mAP');now=latest
            if previous is not None and now is not None:
                print(f'  Latest validation change: {100*(now-previous):+.2f} percentage points (epoch {last["epoch"]}). Watch the trend across several epochs alongside validation loss and NoR/RR recall.')
    if not any((ROOT/name/'history.json').exists() for name,_ in RUNS):
        print(f'Logs: {ROOT}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval',type=int,default=30,help='seconds between live snapshots (default: 30)')
    parser.add_argument('--once',action='store_true',help='show one snapshot and exit')
    args=parser.parse_args()
    if args.interval<1:parser.error('--interval must be positive')
    try:
        while True:
            snapshot()
            if args.once:break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('Live view closed.')


if __name__=='__main__':
    main()
