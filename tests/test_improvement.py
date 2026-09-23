import copy
from collections import Counter
from types import SimpleNamespace
import pytest
import torch
from cine.loading import SequenceClassSampler
from cine.selection import selection_key, advances, confirms
from cine.phases import phase_at, set_phase, optimizer_for
from cine.diagnostics import local_diagnostics, matched_boxes
from cine.model import JointModel
from cine.utils import read_config


def config():
    return read_config('configs/dtld.yaml')

def metrics(value=.5):
    return dict(global_mAP=value, balanced_accuracy=value, global_AP={'NoR':value},
                **{k:{'AP50_95':value} for k in ('object','state','relevance')})

def test_sequence_sampler():
    class Dataset:
        rows = [dict(global_label=c,sequence=s) for c in (-1,0,1,2) for s,n in [('a',90),('b',10)] for _ in range(n)]
        def __len__(self): return len(self.rows)
    data = Dataset()
    sampler = SequenceClassSampler(data,42)
    first = list(sampler)
    assert len(first)==len(data) and first==list(sampler)
    assert Counter(data.rows[i]['global_label'] for i,e in first)==Counter(r['global_label'] for r in data.rows)
    b = sum(data.rows[i]['sequence']=='b' for i,e in first)
    assert 150 < b < 250
    sampler.set_epoch(1)
    assert list(sampler)!=first
    restored=SequenceClassSampler(data,42); restored.set_epoch(1)
    assert list(restored)==list(sampler)

def test_selection_guards_and_ties():
    base=metrics(); vc=dict(selection_metric='multitask',reference_metrics=base)
    assert selection_key(metrics(.51),vc)>selection_key(metrics(.50),vc)
    bad=metrics(.6); bad['relevance']['AP50_95']=.49
    assert selection_key(bad,vc) is None
    assert advances(metrics(.51),base)
    assert not advances(metrics(.501),base)
    tie=metrics(.51); tie['global_AP']['NoR']=.52
    assert selection_key(tie,vc)>selection_key(metrics(.51),vc)
    assert confirms([metrics(.51)]*3,[base]*3)
    assert not confirms([bad]*3,[base]*3)
    with pytest.raises(ValueError): selection_key(base,dict(selection_metric='invalid'))

def test_diagnostic_matching():
    row=dict(boxes=[[0,8,4,20],[10,8,18,20],[30,8,46,20]],states=[0,1,2])
    pred=dict(boxes=[[0,8,4,20],[0,8,4,20],[10,8,18,20]],scores=[.9,.8,.7],states=[[1,0,0],[1,0,0],[1,0,0]])
    assert matched_boxes(row['boxes'],pred,.5)==[(0,0),(1,2)]
    d=local_diagnostics([row],[pred])['by_iou']['0.5']
    assert d['4-8']['recall']==1 and d['8-16']['state_accuracy']==0
    assert d['>=16']['recall']==0 and d['<4']['recall'] is None

def test_p2_transfer_and_outputs():
    torch.set_num_threads(4)
    c=config(); c['model']['architecture']='yolov8s-p2.yaml'
    model=JointModel(c,pretrained=True).eval()
    assert model.detector.model[-1].stride.tolist()==[4,8,16,32]
    assert len(model.feature_channels)==4
    assert model.transfer_report['initialized']
    with torch.no_grad():
        p=model(torch.zeros(1,3,64,64))
    assert p['boxes'].shape[1]==340
    assert p['state_logits'].shape==(1,340,3)
    assert p['relevance_logits'].shape==(1,340)
    assert all(not m.requires_grad for m in p['priors'])
    restored=JointModel(c,pretrained=False).eval(); restored.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad(): torch.testing.assert_close(p['global_logits'],restored(torch.zeros(1,3,64,64))['global_logits'])

def test_phase_groups_and_batch_norm():
    c=config(); c['train']['recipe']='staged'
    assert [phase_at(c,e) for e in [0,9,10,11,12,29]]==['local','local','global','global','joint','joint']
    m=JointModel(c,pretrained=False); m.train(); set_phase(m,'global')
    assert not m.detector.training and m.global_conv.training
    assert all(p.requires_grad==n.startswith('global_') for n,p in m.named_parameters())
    opt=optimizer_for(m,c,'joint')
    assert [g['lr'] for g in opt.param_groups]==[1e-5,1e-4]
    state=opt.state_dict(); other=optimizer_for(m,c,'joint'); other.load_state_dict(state)
    assert [g['lr'] for g in other.param_groups]==[1e-5,1e-4]


def test_campaign_freeze_guard(tmp_path, monkeypatch):
    from cine import campaign
    from cine.utils import write_json
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    with pytest.raises(RuntimeError,match='frozen'):
        campaign.require_frozen()
    write_json(tmp_path/'frozen.json',dict(code={'wrong':'hash'}))
    with pytest.raises(ValueError,match='Code differs'):
        campaign.require_frozen()
    write_json(tmp_path/'frozen.json',dict(code=campaign.code_hashes()))
    assert campaign.require_frozen()['code']==campaign.code_hashes()


def test_campaign_configs_are_yaml_roundtrippable():
    import yaml
    from cine.campaign import config_for
    ref=metrics(); ref['local_diagnostics']={'width_bins':('a','b')}
    c=config_for('PST',42,43,4,ref)
    assert yaml.safe_load(yaml.safe_dump(c))==c
    assert c['train']['sampler']=='sequence_class'
    assert c['train']['recipe']=='staged'
    assert c['evaluate']['batch_size']==4
    assert 'local_diagnostics' not in c['validation']['reference_metrics']


def test_global_phase_freezes_local_statistics_and_weights():
    from cine.data import collate
    c=config(); c['train']['recipe']='staged'
    m=JointModel(c,pretrained=False); m.train()
    optimizer=optimizer_for(m,c,'global')
    before={k:v.clone() for k,v in m.detector.state_dict().items()}
    x=torch.rand(2,3,64,64)
    out=m(x)
    out['global_logits'].square().mean().backward(); optimizer.step()
    for k,v in m.detector.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    assert m.global_classifier.weight.grad is not None
    assert all(p.grad is None for p in m.detector.parameters())


def test_screening_combines_only_advancing_changes(tmp_path,monkeypatch):
    from cine import campaign
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    calls=[]
    def trial(name,flags,split,seed,batch,reference=None):
        calls.append(flags)
        if flags=='T': return None
        return dict(name=name,flags=flags,epoch=3,metrics=metrics({'':.5,'S':.51,'P':.52,'PS':.53}[flags]))
    monkeypatch.setattr(campaign,'run_trial',trial)
    result=campaign.screening(8)
    assert calls==['','S','P','T','PS']
    assert result['winner']['flags']=='PS'


def test_no_eligible_final_seed_prevents_test_evaluation(tmp_path,monkeypatch):
    from cine import campaign
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    monkeypatch.setattr(campaign,'run_trial',lambda *a,**k:None)
    def forbidden(*a,**k): raise AssertionError('Official test must not be evaluated')
    monkeypatch.setattr(campaign,'command',forbidden)
    screen=dict(winner=dict(flags='P',config={},metrics=metrics(.51)),baseline=dict(metrics=metrics()))
    campaign.final_runs(screen,8)
    assert campaign.read(tmp_path/'status.json')['phase']=='final_selection_failed'


def test_phase_changes_preserve_permanently_frozen_parameters():
    c=config(); m=JointModel(c,pretrained=False)
    frozen={n for n,p in m.named_parameters() if not p.requires_grad}
    assert any('dfl' in n for n in frozen)
    for phase in ('joint','local','global','joint'):
        set_phase(m,phase)
        assert all(not p.requires_grad for n,p in m.named_parameters() if n in frozen)
