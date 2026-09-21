"""DTLD JSON -> audited manifests; one fixed image/box transform."""
import collections
import concurrent.futures
import json
import random
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import sha256, write_json

STATE_NAMES = ["red", "green", "unknown"]
GLOBAL_NAMES = ["RR", "RG", "NoR"]
STATE_MAP = {"red": 0, "yellow": 0, "red_yellow": 0, "green": 1, "off": 2, "unknown": 2}
WIDTH, CONTENT_HEIGHT, HEIGHT, PAD = 1280, 720, 736, 8
SCALE = 720 / 1024
AFFINE = np.array([[SCALE, 0, -80], [0, SCALE, PAD]], dtype=np.float32)


def transform_boxes(boxes):
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]] * SCALE - 80
    b[:, [1, 3]] = b[:, [1, 3]] * SCALE + PAD
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, WIDTH)
    b[:, [1, 3]] = b[:, [1, 3]].clip(PAD, PAD + CONTENT_HEIGHT)
    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return b[keep], keep


def inverse_boxes(boxes):
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    b[:, [0, 2]] = ((b[:, [0, 2]] + 80) / SCALE).clip(0, 2048)
    b[:, [1, 3]] = ((b[:, [1, 3]] - PAD) / SCALE).clip(0, 1024)
    return b


def global_label(relevance, states):
    relevant = [s for r, s in zip(relevance, states) if r]
    if not relevant:
        return 2
    known = set(relevant) - {2}
    return next(iter(known)) if len(known) == 1 else -1


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.shape != (1024, 2048, 3):
        raise ValueError(f"Unreadable image or unexpected dimensions: {path}; {None if img is None else img.shape}")
    return img


def process_image(img):
    out = cv2.warpAffine(img, AFFINE, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    out[:PAD] = 114
    out[PAD + CONTENT_HEIGHT:] = 114
    return out


def audit_image(path):
    read_image(path)
    return sha256(path)


def prepare(config):
    cv2.setNumThreads(1)
    cfg = config["data"]
    out = Path(cfg["prepared"])
    out.mkdir(parents=True, exist_ok=True)
    report = {"transform": AFFINE.tolist(), "input_wh": [WIDTH, HEIGHT], "splits": {}, "versions": "DTLD v2 JSON + existing JPEGs"}
    sequences = {}
    for split in ("train", "test"):
        source = Path(cfg["annotations"]) / f"DTLD_{split}.json"
        images = json.loads(source.read_text(encoding="utf-8"))["images"]
        rows, counts, seqs = [], collections.Counter(), set()
        seen = set()
        for i, im in enumerate(images):
            name = PurePosixPath(im["image_path"]).stem + ".jpg"
            if name in seen:
                raise ValueError(f"Duplicate basename: {name}")
            seen.add(name)
            path = (Path(cfg["images"]) / split / name).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            seq = str(PurePosixPath(im["image_path"]).parent)
            seqs.add(seq)
            boxes, rel, states = [], [], []
            for o in im["labels"]:
                x, y, w, h = [o[k] for k in ("x", "y", "w", "h")]
                if not np.isfinite([x, y, w, h]).all() or w < 0 or h < 0:
                    raise ValueError(f"Invalid box: {name}: {o}")
                if w == 0 or h == 0:
                    counts["source_zero_area_boxes_removed"] += 1
                    continue
                if x < 0 or y < 0 or x + w > 2048 or y + h > 1024:
                    counts["source_boxes_outside_image"] += 1
                attr = o["attributes"]
                if attr["relevance"] not in ("relevant", "not_relevant"):
                    raise ValueError(f"Unknown relevance: {attr}")
                boxes.append([x, y, x + w, y + h])
                rel.append(int(attr["relevance"] == "relevant"))
                states.append(STATE_MAP[attr["state"]])
            b, keep = transform_boxes(boxes)
            r, s = np.asarray(rel, dtype=int)[keep].tolist(), np.asarray(states, dtype=int)[keep].tolist()
            label = global_label(r, s)
            counts["boxes_before_crop"] += len(boxes)
            counts["boxes_after_crop"] += len(b)
            counts[GLOBAL_NAMES[label] if label >= 0 else "global_masked"] += 1
            rows.append(dict(id=i, path=str(path), sequence=seq, boxes=b.tolist(), relevance=r, states=s, global_label=label))
        print(f"Auditing every {split} JPEG ({len(rows)} files)...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for i, digest in enumerate(pool.map(audit_image, [r["path"] for r in rows])):
                rows[i]["image_sha256"] = digest
                if (i + 1) % 2000 == 0:
                    print(f"{split}: {i + 1}/{len(rows)} decoded and hashed", flush=True)
        manifest = out / f"{split}.json"
        write_json(manifest, rows)
        report["splits"][split] = dict(images=len(rows), sequences=len(seqs), counts=dict(counts), source_sha256=sha256(source), manifest_sha256=sha256(manifest))
        sequences[split] = seqs
        # Representative, deterministic source images with transformed annotations.
        for n, row in enumerate(rows[::max(1, len(rows) // 4)][:4]):
            img = process_image(read_image(row["path"]))
            for box, r, s in zip(row["boxes"], row["relevance"], row["states"]):
                a, b, c, d = map(round, box)
                cv2.rectangle(img, (a, b), (c, d), [(0, 0, 255), (0, 255, 0), (255, 160, 0)][s], 2 if r else 1)
            cv2.imwrite(str(out / f"audit_{split}_{n}.jpg"), img)
    overlap = sequences["train"] & sequences["test"]
    if overlap:
        raise ValueError(f"Sequence leakage: {sorted(overlap)[:5]}")
    report["sequence_overlap"] = 0
    write_json(out / "audit.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


class DTLDDataset(Dataset):
    def __init__(self, manifest, blur_probability=0.0, limit=None, augment_seed=42):
        self.rows = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if limit:
            self.rows = self.rows[:limit]
        self.blur_probability = blur_probability
        self.augment_seed = augment_seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        cv2.setNumThreads(1)
        epoch = 0
        if isinstance(index, tuple):
            index, epoch = index
        # Independent of worker assignment/prefetch; exact epoch-boundary resume.
        rng = random.Random(self.augment_seed + epoch * 1000003 + index * 9176)
        row = self.rows[index]
        img = process_image(read_image(row["path"]))
        if rng.random() < self.blur_probability:
            img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.1, 1.0))
        # BGR -> RGB CHW. Keep uint8 until transfer to GPU.
        tensor = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))
        boxes = torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4)
        xywh = boxes.clone()
        xywh[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        xywh[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        xywh /= torch.tensor([WIDTH, HEIGHT, WIDTH, HEIGHT])
        return dict(img=tensor, bboxes=xywh, relevance=torch.tensor(row["relevance"], dtype=torch.float32), states=torch.tensor(row["states"], dtype=torch.long), global_label=row["global_label"], id=row["id"])


def collate(items):
    counts = [len(x["bboxes"]) for x in items]
    return dict(img=torch.stack([x["img"] for x in items]), bboxes=torch.cat([x["bboxes"] for x in items]), cls=torch.zeros(sum(counts), 1), batch_idx=torch.repeat_interleave(torch.arange(len(items)), torch.tensor(counts)), relevance=torch.cat([x["relevance"] for x in items]), states=torch.cat([x["states"] for x in items]), global_label=torch.tensor([x["global_label"] for x in items]), ids=[x["id"] for x in items])
