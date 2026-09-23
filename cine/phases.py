"""Phase boundaries and optimizer groups for the fixed 10/2/18 recipe."""
import torch

def phase_at(config, epoch_index):
    recipe = config['train'].get('recipe', 'joint')
    if recipe == 'joint':
        return 'joint'
    if recipe != 'staged':
        raise ValueError('Unknown training recipe')
    return 'local' if epoch_index < 10 else 'global' if epoch_index < 12 else 'joint'

def set_phase(model, phase):
    model.training_phase = phase
    if not hasattr(model, '_phase_trainable'):
        model._phase_trainable = {name: p.requires_grad for name,p in model.named_parameters()}
    for name, parameter in model.named_parameters():
        global_parameter = name.startswith('global_')
        parameter.requires_grad_(model._phase_trainable[name] and (phase == 'joint' or (global_parameter if phase == 'global' else not global_parameter)))
    if phase == 'global':
        for name, module in model.named_children():
            if not name.startswith('global_'):
                module.eval()

def optimizer_for(model, config, phase):
    set_phase(model, phase)
    tc = config['train']
    if tc.get('recipe', 'joint') == 'staged' and phase == 'joint':
        groups = [dict(params=[p for n,p in model.named_parameters() if not n.startswith('global_')], lr=1e-5),
                  dict(params=[p for n,p in model.named_parameters() if n.startswith('global_')], lr=1e-4)]
    else:
        groups = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(groups, lr=tc['lr'], weight_decay=tc['weight_decay'])
