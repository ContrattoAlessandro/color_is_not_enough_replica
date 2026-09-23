"""Explicit semantic transfer from standard YOLOv8s to its P2 variant."""
import torch


def transfer_p2(source, target):
    src, dst = source.model, target.model
    if len(src) != 23 or len(dst) != 29 or list(src[-1].stride.tolist()) != [8, 16, 32]:
        raise ValueError('P2 transfer requires standard YOLOv8s COCO source')
    # Shared backbone/top-down neck, then the equivalent bottom-up P4/P5 path.
    mapping = {**{i: i for i in range(16)}, 22: 16, 23: 17, 24: 18, 25: 19, 26: 20, 27: 21}
    copied = []
    for new_i, old_i in mapping.items():
        old, new = src[old_i], dst[new_i]
        if type(old) is not type(new):
            raise ValueError('Unexpected module correspondence')
        a, b = old.state_dict(), new.state_dict()
        if a.keys() != b.keys() or any(a[k].shape != b[k].shape for k in b):
            raise ValueError(f'Incompatible semantic module {old_i} -> {new_i}')
        new.load_state_dict(a)
        copied.extend(f'model.{new_i}.{k}' for k in b)
    # Detection branches are matched by stride, never by list position.
    for j, stride in enumerate(dst[-1].stride.tolist()):
        if stride == 4:
            continue
        i = src[-1].stride.tolist().index(stride)
        for branch in ('cv2', 'cv3'):
            old, new = getattr(src[-1], branch)[i], getattr(dst[-1], branch)[j]
            a, b = old.state_dict(), new.state_dict()
            compatible = {k:v for k,v in a.items() if k in b and v.shape == b[k].shape}
            new.load_state_dict(compatible, strict=False)
            copied.extend(f'model.28.{branch}.{j}.{k}' for k in compatible)
            if branch == 'cv3' and old[-1].weight.shape[1:] == new[-1].weight.shape[1:]:
                with torch.no_grad():
                    new[-1].weight.copy_(old[-1].weight[9:10])
                    new[-1].bias.copy_(old[-1].bias[9:10])
                copied.extend([f'model.28.cv3.{j}.2.weight', f'model.28.cv3.{j}.2.bias'])
    copied = sorted(set(copied))
    return dict(compatible_tensors=len(copied), total_tensors=len(target.state_dict()),
                module_mapping=mapping, copied=copied,
                initialized=sorted(set(target.state_dict())-set(copied)),
                note='P2 and new P3 fusion initialized; equivalent P4/P5 modules and stride-matched branches transferred.')
