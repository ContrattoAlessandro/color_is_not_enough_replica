"""Bounded, resumable P2-first campaign. Run: python -m cine.campaign run."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml
from .utils import read_config, write_json, sha256, versions
from .selection import advances, confirms, score, vector, eligible
from .generalization import audit, summarize_seeds, bootstrap_global
from .split import create_split, assert_disjoint

ROOT = Path('artifacts/improvement_v2')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def status(phase, **details):
    write_json(ROOT/'status.json',dict(phase=phase,controller_pid=os.getpid(),updated_at=time.time(),**details))


def code_hashes():
    return {str(p):sha256(p) for p in sorted(Path('cine').glob('*.py'))}


def command(args, log):
    with open(log,'a',encoding='utf-8') as stream:
        stream.write('\n'+repr(args)+'\n'); stream.flush()
        subprocess.run([sys.executable,'-X','utf8','-u',*args],stdout=stream,stderr=subprocess.STDOUT,check=True)


def establish_provenance():
    path=ROOT/'provenance.json'
    manifest=Path('artifacts/data_validation')
    now=dict(code=code_hashes(),config=sha256('configs/dtld.yaml'),pretrained=sha256('artifacts/pretrained/yolov8s.pt'),
             manifests={k:sha256(manifest/f'{k}.json') for k in ('train','val','test')},versions=versions())
    if path.exists():
        previous=read(path)
        if previous!=now:
            raise ValueError('Campaign code/config/data/environment changed; refusing to mix experiments')
    else:
        write_json(path,now)
    parts={k:read(manifest/f'{k}.json') for k in ('train','val','test')}
    assert_disjoint(parts)
    base=read_config('configs/dtld.yaml')
    source=read(Path(base['data']['annotations'])/'DTLD_test.json')['images']
    names={Path(r['image_path']).stem+'.jpg' for r in source}
    prepared_names={Path(r['path']).name for r in parts['test']}
    if names!=prepared_names or len(source)!=len(parts['test']):
        raise ValueError('Prepared test identities do not match DTLD_test.json')
    write_json(ROOT/'protocol.json',dict(test_source='DTLD_test.json',test_images=len(source),
        identical_source_image_set=True,test_byte_identical=sha256(manifest/'test.json')==sha256('artifacts/data/test.json'),
        state_policy='red/yellow/red_yellow -> red; green -> green; off/unknown -> unknown',
        limitation='Paper calls evaluation partition official validation; exact author mapping remains unverified. Follow-up campaign informed by previously reported test performance.'))


def choose_batch():
    path=ROOT/'preflight'/'selected.json'; path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): return read(path)['batch_size']
    for batch in (16,8,4,2):
        reports=[]
        for arch,short in [('yolov8s-p2.yaml','p2'),('yolov8s.yaml','standard')]:
            output=path.parent/f'{short}_b{batch}.json'
            if not output.exists() or (read(output).get('accepted') and 'optimizer_updates' not in read(output)):
                status('preflight',architecture=arch,batch_size=batch)
                command(['-m','cine.preflight','--batch',str(batch),'--architecture',arch,'--output',str(output)],path.parent/f'{short}_b{batch}.log')
            reports.append(read(output))
            if not reports[-1]['accepted']: break
        if len(reports)==2 and all(r['accepted'] for r in reports):
            write_json(path,dict(batch_size=batch,effective_batch=16,reports=reports))
            return batch
    raise RuntimeError('No common batch fits with 1 GiB headroom')


def config_for(flags, split, seed, batch, reference=None):
    c=copy.deepcopy(read_config('configs/dtld.yaml'))
    c['train'].update(seed=seed,batch_size=batch,effective_batch=16,epochs=30,
                      sampler='sequence_class' if 'S' in flags else 'image',
                      recipe='staged' if 'T' in flags else 'joint',
                      task_weights=dict(detection=1.,relevance=1.,state=1.,global_loss=1.))
    c['evaluate']['batch_size']=batch
    c['model']['architecture']='yolov8s-p2.yaml' if 'P' in flags else 'yolov8s.yaml'
    c['data']['prepared']='artifacts/data_validation' if split==42 else str(ROOT/'splits'/str(split))
    if reference is not None:
        reference = {k:copy.deepcopy(reference[k]) for k in ('global_mAP','balanced_accuracy','global_AP','object','state','relevance')}
        c['validation'].update(selection_metric='multitask',reference_metrics=reference)
    return c


def run_trial(name, flags, split, seed, batch, reference=None):
    if code_hashes()!=read(ROOT/'provenance.json')['code']:
        raise ValueError('Code changed during campaign')
    directory=ROOT/'runs'/name; directory.mkdir(parents=True,exist_ok=True)
    config=config_for(flags,split,seed,batch,reference); config['train']['output']=str(directory)
    cfg=directory/'config.yaml'
    if cfg.exists() and read_config(cfg)!=config: raise ValueError('Run configuration mismatch')
    if not cfg.exists(): cfg.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    status('training',run=name,flags=flags,split=split,seed=seed,run_status=str(directory/'status.json'))
    complete=(directory/'status.json').exists() and read(directory/'status.json')['phase']=='training_complete'
    if not complete:
        args=['-m','cine','--config',str(cfg),'train','--output',str(directory)]
        if (directory/'last.pt').exists(): args.extend(['--resume',str(directory/'last.pt')])
        if '--evaluate-after' in args: raise AssertionError('Test evaluation forbidden during training')
        command(args,directory/'train.log')
    best=directory/'best.pt'
    if not best.exists():
        write_json(directory/'selection.json',dict(selected=False,reason='No eligible joint checkpoint'))
        return None
    ck=torch.load(best,map_location='cpu',weights_only=False)
    selected=dict(name=name,flags=flags,split=split,seed=seed,checkpoint=str(best),
                  checkpoint_sha256=sha256(best),epoch=ck['epoch'],metrics=ck['history'][-1]['validation'],config=config)
    del ck
    write_json(directory/'selection.json',selected)
    predictions=read(directory/'validation'/f"epoch_{selected['epoch']:03d}.global.json")
    audit(read(Path(config['data']['prepared'])/'val.json'),predictions,directory/'audit')
    return selected


def screening(batch):
    base=run_trial('baseline_split42_seed42','',42,42,batch)
    candidates=[]; decisions=[]
    for flags in ('S','P','T'):
        candidate=run_trial(f'{flags}_split42_seed42',flags,42,42,batch,base['metrics'])
        accepted=candidate is not None and advances(candidate['metrics'],base['metrics'])
        decisions.append(dict(flags=flags,advance=accepted))
        if accepted: candidates.append(candidate)
        write_json(ROOT/'screening_progress.json',dict(decisions=decisions))
    if len(candidates)>=2:
        flags=''.join(sorted(c['flags'] for c in candidates))
        candidate=run_trial(f'{flags}_split42_seed42',flags,42,42,batch,base['metrics'])
        accepted=candidate is not None and advances(candidate['metrics'],base['metrics'])
        decisions.append(dict(flags=flags,advance=accepted))
        if accepted: candidates.append(candidate)
    winner=max(candidates,key=lambda c:(score(c['metrics']),c['metrics']['global_AP']['NoR'],-c['epoch'])) if candidates else None
    result=dict(baseline=base,winner=winner,decisions=decisions)
    write_json(ROOT/'screening.json',result)
    return result


def confirmation(screen,batch):
    winner=screen['winner']; candidates=[winner]; baselines=[screen['baseline']]
    for split in (43,44):
        directory=ROOT/'splits'/str(split)
        if not directory.exists(): create_split(output=directory,seed=split)
        base=run_trial(f'baseline_split{split}_seed42','',split,42,batch)
        candidate=run_trial(f"{winner['flags']}_split{split}_seed42",winner['flags'],split,42,batch,base['metrics'])
        baselines.append(base); candidates.append(candidate)
    accepted=all(c is not None for c in candidates) and confirms([c['metrics'] for c in candidates],[b['metrics'] for b in baselines])
    result=dict(accepted=accepted,baselines=baselines,candidates=candidates)
    write_json(ROOT/'confirmation.json',result)
    return result


def require_frozen():
    if not (ROOT/'frozen.json').exists(): raise RuntimeError('Test evaluation requires a frozen campaign')
    frozen=read(ROOT/'frozen.json')
    if frozen['code']!=code_hashes(): raise ValueError('Code differs from frozen campaign')
    return frozen


def final_runs(screen,batch):
    winner=screen['winner']; path=ROOT/'frozen.json'
    frozen=dict(flags=winner['flags'],config=winner['config'],code=code_hashes(),
                seeds=[42,43,44,45,46],reference_metrics=screen['baseline']['metrics'],
                selection='mean(global,object,state), six baseline nonregression guards 0.005, NoR tie-break, earlier epoch',
                test_manifest=sha256('artifacts/data_validation/test.json'))
    if path.exists() and read(path)!=frozen: raise ValueError('Frozen campaign differs')
    write_json(path,frozen)
    selected=[winner]
    for seed in range(43,47):
        selected.append(run_trial(f"{winner['flags']}_split42_seed{seed}",winner['flags'],42,seed,batch,screen['baseline']['metrics']))
    if any(s is None for s in selected):
        finish('final_selection_failed',reason='At least one seed has no eligible checkpoint; official test not evaluated')
        return
    require_frozen()
    tests=[]; bootstraps=[]
    for chosen in selected:
        require_frozen()
        if sha256(chosen['checkpoint'])!=chosen['checkpoint_sha256']: raise ValueError('Selected checkpoint changed')
        directory=ROOT/'test'/f"seed{chosen['seed']}"; directory.mkdir(parents=True,exist_ok=True)
        status('final_evaluation',seed=chosen['seed'])
        if not (directory/'metrics.json').exists():
            command(['-m','cine','evaluate','--checkpoint',chosen['checkpoint'],'--output',str(directory)],directory/'evaluate.log')
        metrics=read(directory/'metrics.json'); tests.append(metrics)
        bootpath=directory/'bootstrap.json'
        if not bootpath.exists():
            rows=read('artifacts/data_validation/test.json'); predictions=read(directory/'predictions_tensor_coordinates.json')
            records=[dict(label=r['global_label'],sequence=r['sequence'],probabilities=p['global_probabilities']) for r,p in zip(rows,predictions)]
            write_json(bootpath,bootstrap_global(records))
        bootstraps.append(read(bootpath))
    summary=summarize_seeds(tests)
    targets=dict(global_mAP=.831,balanced_accuracy=.808,NoR_AP=.546,object_AP50_95=.444,state_AP50_95=.386,relevance_AP50_95=.5274-.005)
    achieved={k:summary[k]['mean']>=v for k,v in targets.items()}
    baseline=read('artifacts/results/final_report.json')['seed_summary']
    monitored=('global_mAP','balanced_accuracy','NoR_AP','object_AP50_95','state_AP50_95','relevance_AP50_95')
    promotion=all(summary[k]['mean']>=baseline[k]['mean']-.005 for k in monitored) and sum(summary[k]['mean']-baseline[k]['mean'] for k in ('global_mAP','object_AP50_95','state_AP50_95'))>0
    report=dict(outcome='evaluated',selected=selected,seed_summary=summary,bootstrap=bootstraps,targets=targets,targets_met=achieved,
                promotion_eligible=promotion,limitation='Follow-up adaptation; historical test results informed campaign motivation; no exact reproduction claim.')
    write_json(ROOT/'final_report.json',report)
    if promotion: write_json(ROOT/'recommended_model.json',dict(checkpoint=winner['checkpoint'],reason='Five-seed improvement with nonregression guards'))
    lines=['# Improvement campaign results','', '| Metric | Mean | Sample SD | Target | Met |','|---|---:|---:|---:|---|']
    for k,target in targets.items():
        lines.append(f"| {k} | {summary[k]['mean']*100:.2f}% | {summary[k]['std']*100:.2f} pp | {target*100:.2f}% | {achieved[k]} |")
    lines.extend(['',f'Promotion eligible: {promotion}. Existing default retained pending report review.', '',report['limitation']])
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    finish('complete',targets_met=achieved,promotion_eligible=promotion)


def finish(outcome,**details):
    status(outcome,**details)
    if outcome!='complete':
        (ROOT/'RESULTS.md').write_text(f'# Improvement campaign\n\nOutcome: {outcome}.\n\n{json.dumps(details,indent=2)}\n\nNo new official-test results were used to expand the search. See screening.json and confirmation.json when present.\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','screen','confirm','final','run','status'])
    args=parser.parse_args(); ROOT.mkdir(parents=True,exist_ok=True)
    if args.stage=='status':
        print(json.dumps(read(ROOT/'status.json') if (ROOT/'status.json').exists() else {},indent=2)); return
    # An OS-held lock is released even on process failure; prevents duplicate GPU campaigns.
    import msvcrt
    with open(ROOT/'controller.lock','a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            establish_provenance(); batch=choose_batch()
            if args.stage=='preflight': status('preflight_complete',batch_size=batch); return
            if args.stage in ('screen','run'):
                screen=screening(batch)
            else:
                screen=read(ROOT/'screening.json')
            if screen['winner'] is None: finish('screening_stopped',reason='No isolated change passed'); return
            if args.stage=='screen': status('screening_complete'); return
            confirm=confirmation(screen,batch) if args.stage in ('confirm','run') else read(ROOT/'confirmation.json')
            if not confirm['accepted']: finish('confirmation_stopped',reason='Winner did not meet cross-split gates'); return
            if args.stage=='confirm': status('confirmation_complete'); return
            final_runs(screen,batch)
        except Exception as error:
            previous=read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
            write_json(ROOT/'failure.json',dict(error=type(error).__name__,message=str(error),previous=previous))
            status('failed',error=str(error)); raise

if __name__=='__main__': main()
