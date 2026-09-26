"""Serial, resumable YOLOv8m head comparison using the frozen run manifest.

Run from the repository root. Check artifacts/yolov8m_comparison/status.json.
All source/config hashes are checked before each subprocess. Both checkpoints
are chosen on validation and frozen before either official-test evaluation.
"""
import json
import msvcrt
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cine.utils import read_config, sha256, write_json

ROOT = REPO / 'artifacts/yolov8m_comparison'


def now():
    return datetime.now(timezone.utc).isoformat()


def status(phase, **kwargs):
    record = dict(phase=phase, controller_pid=os.getpid(), updated_at=now(), **kwargs)
    write_json(ROOT / 'status.json', record)
    print(json.dumps(record), flush=True)


def verify(manifest):
    for path, expected in manifest['inputs'].items():
        if sha256(REPO / path) != expected:
            raise RuntimeError(f'Run provenance changed: {path}')


def child(manifest, stage, name, args, directory):
    verify(manifest)
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-m', 'cine', *args]
    with (directory / f'{stage}.log').open('a', encoding='utf-8') as stdout, (directory / f'{stage}.stderr.log').open('a', encoding='utf-8') as stderr:
        process = subprocess.Popen(command, cwd=REPO, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        status(stage, run=name, child_pid=process.pid, command=command,
               log=str(directory / f'{stage}.log'), run_status=str(directory / 'status.json'))
        code = process.wait()
    if code:
        raise RuntimeError(f'{name} {stage} failed with exit code {code}; inspect {directory}')


def metric_vector(m):
    return dict(global_mAP=m['global_mAP'], balanced_accuracy=m['balanced_accuracy'],
                **{f'{c}_AP':m['global_AP'][c] for c in ('RR','RG','NoR')},
                **{f'{c}_recall':m['class_recall'][c] for c in ('RR','RG','NoR')},
                **{f'{c}_AP50_95':m[c]['AP50_95'] for c in ('object','relevance','state')})


def report(manifest):
    load = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    paths = dict(s_raster=REPO/'artifacts/results/seed42/metrics.json',
                 s_set_null=REPO/'artifacts/p2_set_null/seed42/test_best_epoch005/metrics.json')
    paths.update({f'm_{r["name"]}':ROOT/r['directory']/'evaluation/metrics.json' for r in manifest['runs']})
    metrics = {name:load(path) for name,path in paths.items()}
    for m in metrics.values():
        assert (m['images'],m['global_valid'],m['class_order']) == (12453,11901,['RR','RG','NoR'])
    vectors = {name:metric_vector(m) for name,m in metrics.items()}
    deltas = {head:{k:100*(vectors[f'm_{head}'][k]-vectors[f's_{head}'][k]) for k in vectors[f'm_{head}']}
              for head in ('raster','set_null')}
    write_json(ROOT/'comparison.json',dict(test_images=12453,global_valid=11901,metrics=vectors,
               medium_minus_small_pp=deltas,source_metrics={n:str(p) for n,p in paths.items()}))
    lines = ['# YOLOv8m comparison: seed 42', '',
             'All results use the same official-test partition and metric definitions. Each checkpoint was selected by validation global mAP. Both medium checkpoints were frozen before either was tested.', '',
             '| Metric | Small raster | Small Top-64/NULL | Medium raster | Medium Top-64/NULL |',
             '|---|---:|---:|---:|---:|']
    for key in vectors['s_raster']:
        lines.append('| '+key+' | '+' | '.join(f'{100*vectors[name][key]:.2f}%' for name in paths)+' |')
    lines += ['', '## Medium minus small, percentage points', '',
              '| Metric | Raster | Top-64/NULL |','|---|---:|---:|']
    for key in vectors['s_raster']:
        lines.append(f'| {key} | {deltas["raster"][key]:+.2f} | {deltas["set_null"][key]:+.2f} |')
    lines += ['', '## Interpretation', '',
              'These are single-seed comparisons. Medium runs share physical batch 8 and effective batch 16, while the historical small runs used physical batch 16; accumulation does not reproduce identical batch-normalization statistics. Medium runs complete 30 epochs, and the historical small NULL run was stopped after 13 completed epochs (its epoch-5 checkpoint was tested). Treat size effects against those historical runs as preliminary.', '',
              'Global mAP is the primary comparison metric. Also inspect RR, RG and NoR recall and the three local AP metrics before preferring a model; an increase in NoR recall can come with more missed relevant-red frames. The table does not establish superiority across training seeds. The sampler has no official-test result.', '',
              'Training recipe, checkpoint selection and input hashes are recorded in manifest.json, per-run experiment.json and selection.json. Test scores are reported once per validation-selected model and are not used to change the runs.', '']
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines), encoding='utf-8')


def run(manifest):
    for spec in manifest['runs']:
        verify(manifest)
        directory = ROOT/spec['directory']
        config_path = REPO/spec['config']
        config = read_config(config_path)
        history_path = directory/'history.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        completed = len(history) == config['train']['epochs'] and (directory/'final.pt').exists()
        if not completed:
            args = ['--config',str(config_path),'train']
            if (directory/'last.pt').exists():
                args += ['--resume',str(directory/'last.pt')]
            child(manifest,'training',spec['name'],args,directory)
        history = json.loads(history_path.read_text())
        if len(history) != config['train']['epochs']:
            raise RuntimeError('Training did not complete the planned epochs')
    # Freeze BOTH validation selections before exposing ANY new test result.
    import torch
    for spec in manifest['runs']:
        directory = ROOT/spec['directory']
        frozen = directory/'selected_for_test.pt'
        selection_path = directory/'selection.json'
        if selection_path.exists():
            saved = json.loads(selection_path.read_text())
            if sha256(frozen) != saved['checkpoint_sha256']:
                raise RuntimeError('Frozen checkpoint changed')
            continue
        shutil.copy2(directory/'best.pt', frozen)
        ck = torch.load(frozen,map_location='cpu',weights_only=False)
        history = json.loads((directory/'history.json').read_text())
        best = max(history,key=lambda r:r['validation']['global_mAP'])
        if ck['epoch'] != best['epoch']:
            raise RuntimeError('Checkpoint is not the validation-selected best')
        write_json(selection_path,dict(epoch=ck['epoch'],validation_global_mAP=best['validation']['global_mAP'],
                   checkpoint_sha256=sha256(frozen),fingerprints=ck['fingerprints'],frozen_at=now()))
        del ck
    for spec in manifest['runs']:
        directory = ROOT/spec['directory']
        output = directory/'evaluation'
        success = output/'completed.json'
        if not success.exists():
            if (output/'metrics.json').exists():
                raise RuntimeError('Metrics exist without completion marker; inspect before repeating test evaluation')
            child(manifest,'evaluating',spec['name'],['evaluate','--checkpoint',str(directory/'selected_for_test.pt'),
                                                   '--output',str(output)],output)
            write_json(success,dict(completed_at=now(),checkpoint_sha256=sha256(directory/'selected_for_test.pt')))
    report(manifest)
    status('complete', comparison=str(ROOT/'COMPARISON.md'))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'controller.lock').open('a+b') as lock:
        if lock.seek(0,2) == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            raise SystemExit('Comparison controller is already running')
        try:
            manifest=json.loads((ROOT/'manifest.json').read_text())
            run(manifest)
        except Exception as error:
            status('failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__ == '__main__':
    main()
