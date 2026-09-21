"""Create a fixed sequence-disjoint validation holdout from audited training data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil

from .utils import sha256, write_json


def grouped_split(rows, fraction=0.1, seed=42):
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    sequences = sorted({row['sequence'] for row in rows})
    if len(sequences) < 2:
        raise ValueError('At least two sequences required')
    n = max(1, min(len(sequences) - 1, round(len(sequences) * fraction)))
    validation_sequences = set(random.Random(seed).sample(sequences, n))
    train = [r for r in rows if r['sequence'] not in validation_sequences]
    val = [r for r in rows if r['sequence'] in validation_sequences]
    return train, val


def assert_disjoint(parts):
    keys = list(parts)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            for field in ('sequence', 'path', 'image_sha256'):
                left = {r[field] for r in parts[a] if field in r}
                right = {r[field] for r in parts[b] if field in r}
                overlap = left & right
                if overlap:
                    raise ValueError(f'{a}/{b} {field} leakage: {len(overlap)}')


def create_split(source='artifacts/data', output='artifacts/data_validation', fraction=0.1, seed=42):
    source, output = Path(source), Path(output)
    if not (source / 'audit.json').is_file():
        raise ValueError('Source data must be fully audited first')
    rows = json.loads((source / 'train.json').read_text(encoding='utf-8'))
    test = json.loads((source / 'test.json').read_text(encoding='utf-8'))
    train, val = grouped_split(rows, fraction, seed)
    parts = dict(train=train, val=val, test=test)
    assert_disjoint(parts)
    for name, part in parts.items():
        if {r['global_label'] for r in part if r['global_label'] >= 0} != {0, 1, 2}:
            raise ValueError(f'{name} does not contain every global class')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing split: {output}')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'train.json', train)
    write_json(output / 'val.json', val)
    shutil.copyfile(source / 'test.json', output / 'test.json')
    report = dict(seed=seed, validation_fraction_of_sequences=fraction, method='Uniform seeded sample of sorted intersection-sequence IDs; complete sequences retained; no test-driven selection', source_train_sha256=sha256(source / 'train.json'), source_audit_sha256=sha256(source / 'audit.json'), splits={})
    for name, part in parts.items():
        report['splits'][name] = dict(images=len(part), sequences=len({r['sequence'] for r in part}), global_counts=dict(Counter(str(r['global_label']) for r in part)), cities=dict(Counter(r['sequence'].split('/')[0] for r in part)), manifest_sha256=sha256(output / f'{name}.json'))
    report['test_manifest_byte_identical'] = sha256(output / 'test.json') == sha256(source / 'test.json')
    report['overlap_sequences_images_hashes'] = 0
    write_json(output / 'audit.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='artifacts/data')
    parser.add_argument('--output', default='artifacts/data_validation')
    parser.add_argument('--fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    create_split(args.source, args.output, args.fraction, args.seed)
