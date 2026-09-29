"""The frozen photometric + sensor (PS) training augmentation.

Inputs and outputs are BGR uint8 images. Version-1 PS checkpoints remain
resumable: their inactive C/A fields and disabled JPEG choice are normalized
to the same effective version-2 policy.
"""
import copy
import hashlib
import random

import cv2
import numpy as np


DEFAULT = dict(
    version=2,
    clean_fraction=.25,
    max_operations=2,
    probabilities=dict(P=.5, S=.25, blur=.1),
    ranges=dict(
        exposure=[-.35, .35],
        gamma=[.85, 1.15],
        contrast=[.85, 1.15],
        noise=[1., 4.],
        downsample=[.8, 1.],
        blur=[.1, 1.],
    ),
)
OPERATIONS = dict(P=('exposure', 'gamma', 'contrast'),
                  S=('noise', 'motion', 'downsample'), blur=('blur',))
ORDER = {op: rank for rank, group in enumerate(('P', 'S', 'blur'))
         for op in OPERATIONS[group]}


def policy():
    return copy.deepcopy(DEFAULT)


def validate_policy(value):
    """Accept only PS; normalize the archived version-1 PS configuration."""
    if not isinstance(value, dict):
        raise ValueError('Augmentation policy must be a dictionary')
    if value.get('version') == 1:
        if value.get('groups') != ['P', 'S'] or value.get('disabled') != ['jpeg']:
            raise ValueError('Only the frozen PS recipe is supported')
        try:
            value = dict(
                version=2,
                clean_fraction=value['clean_fraction'],
                max_operations=value['max_operations'],
                probabilities={k: value['probabilities'][k] for k in DEFAULT['probabilities']},
                ranges={k: value['ranges'][k] for k in DEFAULT['ranges']},
            )
        except KeyError as exc:
            raise ValueError('Incomplete archived PS policy') from exc
    if set(value) != set(DEFAULT) or value['version'] != 2:
        raise ValueError('Only the PS version-2 policy is supported')
    if (not isinstance(value['clean_fraction'], (int, float))
            or not np.isfinite(value['clean_fraction'])
            or not 0 <= value['clean_fraction'] <= 1
            or type(value['max_operations']) is not int
            or not 1 <= value['max_operations'] <= 2):
        raise ValueError('Invalid clean fraction or operation cap')
    if (set(value['probabilities']) != set(DEFAULT['probabilities'])
            or any(not isinstance(p, (int, float)) or not np.isfinite(p) or not 0 <= p <= 1
                   for p in value['probabilities'].values())):
        raise ValueError('Invalid PS probabilities')
    if set(value['ranges']) != set(DEFAULT['ranges']):
        raise ValueError('Incomplete PS ranges')
    for op, bounds in value['ranges'].items():
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or not all(isinstance(v, (int, float)) for v in bounds)
                or not np.isfinite(bounds).all() or bounds[0] > bounds[1]
                or (op != 'exposure' and bounds[0] <= 0)):
            raise ValueError('Invalid PS range: ' + op)
    if value['ranges']['downsample'][1] > 1:
        raise ValueError('Downsample factor must be at most one')
    return copy.deepcopy(value)


def seed_for(seed, epoch, identity, operation):
    key = repr((seed, epoch, identity, operation)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')


def schedule(value, seed, epoch, identity):
    rng = random.Random(seed_for(seed, epoch, identity, 'schedule'))
    if rng.random() < value['clean_fraction']:
        return []
    selected = []
    for group in ('P', 'S', 'blur'):
        grng = random.Random(seed_for(seed, epoch, identity, group))
        if grng.random() < value['probabilities'][group]:
            selected.append(grng.choice(OPERATIONS[group]))
    if 'blur' in selected and any(op in selected for op in ('motion', 'downsample')):
        if rng.random() < .5:
            selected.remove('blur')
        else:
            selected = [op for op in selected if op not in ('motion', 'downsample')]
    if len(selected) > value['max_operations']:
        selected = rng.sample(selected, value['max_operations'])
    return sorted(selected, key=ORDER.get)


def transform(content, op, magnitude, rng):
    """Clip at each operation boundary; quantize only when returning the image."""
    x = content.astype(np.float32)
    if op == 'exposure':
        x *= 2. ** magnitude
    elif op == 'gamma':
        x = 255. * np.power(x / 255., magnitude)
    elif op == 'contrast':
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = mean + magnitude * (x - mean)
    elif op == 'noise':
        x += rng.normal(0., magnitude, size=x.shape).astype(np.float32)
    elif op == 'motion':
        kernel = np.zeros((3, 3), np.float32)
        kernel[1, :] = 1
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((1, 1), magnitude, 1), (3, 3))
        kernel /= kernel.sum()
        x = cv2.filter2D(x, -1, kernel)
    elif op == 'downsample':
        h, w = x.shape[:2]
        small = cv2.resize(x, (max(1, round(w * magnitude)),
                               max(1, round(h * magnitude))), interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif op == 'blur':
        x = cv2.GaussianBlur(x, (3, 3), magnitude)
    else:
        raise ValueError('Unsupported PS operation: ' + op)
    return np.clip(x, 0, 255)


def apply(image, value, seed, epoch, identity):
    from .data import PAD, CONTENT_HEIGHT

    ops = schedule(value, seed, epoch, identity)
    if not ops:
        return image, []
    x = image[PAD:PAD + CONTENT_HEIGHT].astype(np.float32)
    for op in ops:
        rng = np.random.default_rng(seed_for(seed, epoch, identity, op))
        magnitude = (rng.uniform(0, 180) if op == 'motion'
                     else rng.uniform(*value['ranges'][op]))
        x = transform(x, op, magnitude, rng)
    out = np.full_like(image, 114)
    out[PAD:PAD + CONTENT_HEIGHT] = np.rint(x).astype(np.uint8)
    return out, ops
