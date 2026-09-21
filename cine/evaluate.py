import contextlib
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix

# A project-local wheel avoids changing the user's shared Python installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps"))
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .data import DTLDDataset, GLOBAL_NAMES, STATE_NAMES, inverse_boxes, process_image, read_image
from .engine import loader, load_checkpoint
from .utils import move_batch, sha256, write_json


def coco_metrics(rows, predictions, mode):
    categories = [dict(id=i, name=n) for i, n in enumerate(STATE_NAMES if mode == "state" else [mode])]
    gt = dict(images=[dict(id=r["id"], width=1280, height=736) for r in rows], annotations=[], categories=categories, info={})
    results = []
    for row, pred in zip(rows, predictions):
        for box, relevant, state in zip(row["boxes"], row["relevance"], row["states"]):
            if mode == "relevance" and not relevant:
                continue
            x, y, x2, y2 = box
            gt["annotations"].append(dict(id=len(gt["annotations"]) + 1, image_id=row["id"], category_id=state if mode == "state" else 0, bbox=[x, y, x2-x, y2-y], area=(x2-x)*(y2-y), iscrowd=0))
        for j, box in enumerate(pred["boxes"]):
            x, y, x2, y2 = box
            for cls in range(len(categories)):
                score = pred["scores"][j]
                if mode == "relevance":
                    score *= pred["relevance"][j]
                elif mode == "state":
                    score *= pred["states"][j][cls]
                results.append(dict(image_id=row["id"], category_id=cls, bbox=[x, y, x2-x, y2-y], score=float(score)))
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = gt
        truth.createIndex()
        if results:
            detected = truth.loadRes(results)
        else:
            detected = COCO()
            detected.dataset = dict(images=gt["images"], categories=categories, annotations=[])
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return dict(AP50_95=float(evaluator.stats[0]), AP50=float(evaluator.stats[1]), AP75=float(evaluator.stats[2]))


def serialize(pred):
    return {k: v.detach().float().cpu().tolist() for k, v in pred.items() if k != "candidate_indices"}


def draw_prediction(image, prediction, threshold=0.25):
    for box, score, rel, state in zip(prediction["boxes"], prediction["scores"], prediction["relevance"], prediction["states"]):
        if score < threshold:
            continue
        x, y, x2, y2 = map(round, box)
        s = int(np.argmax(state))
        color = [(0, 0, 255), (0, 220, 0), (255, 150, 0)][s]
        cv2.rectangle(image, (x, y), (x2, y2), color, 2)
        cv2.putText(image, f"{STATE_NAMES[s]} {score:.2f} rel={rel:.2f}", (x, max(15, y-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    probs = prediction["global_probabilities"]
    cv2.putText(image, " ".join(f"{n}: {p:.3f}" for n, p in zip(GLOBAL_NAMES, probs)), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return image


@torch.no_grad()
def evaluate(config, checkpoint, output=None, limit=None):
    out = Path(output or Path(checkpoint).parent / "evaluation")
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load_checkpoint(checkpoint)
    if ck["config"] != config:
        raise ValueError("Use the same config as the training checkpoint")
    manifest = Path(config["data"]["prepared"]) / "test.json"
    if sha256(manifest) != ck["fingerprints"]["test"]:
        raise ValueError("Test manifest differs from checkpoint")
    model.eval()
    dataset = DTLDDataset(manifest, limit=limit)
    ec = config["evaluate"]
    batches = loader(dataset, ec["batch_size"], ec["workers"], prefetch_factor=ec.get('prefetch_factor', 2), persistent_workers=ec.get('persistent_workers', True))
    predictions, latencies = [], []
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    for step, batch in enumerate(batches):
        batch = move_batch(batch, "cuda")
        images = batch["img"].float() / 255
        if step == 0:
            for _ in range(3):
                with torch.autocast("cuda", enabled=config["train"]["amp"]):
                    model(images)
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", enabled=config["train"]["amp"]):
            pred = model(images)
        det = model.detections(pred, ec["conf"])
        torch.cuda.synchronize()
        latencies.append((time.perf_counter()-t, len(images)))
        predictions.extend(serialize(d) for d in det)
        if step % 100 == 0:
            print(f"Evaluating {len(predictions)}/{len(dataset)}", flush=True)
    write_json(out / "predictions_tensor_coordinates.json", predictions)
    labels = np.array([r["global_label"] for r in dataset.rows])
    valid = labels >= 0
    probabilities = np.array([p["global_probabilities"] for p in predictions])
    aps = {name: float(average_precision_score(labels[valid] == i, probabilities[valid, i])) if np.any(labels[valid] == i) else None for i, name in enumerate(GLOBAL_NAMES)}
    metrics = dict(images=len(dataset), global_valid=int(valid.sum()), global_masked=int((~valid).sum()), global_coverage=float(valid.mean()), global_AP=aps, global_mAP=float(np.mean([v for v in aps.values() if v is not None])) if any(v is not None for v in aps.values()) else None, balanced_accuracy=float(balanced_accuracy_score(labels[valid], probabilities[valid].argmax(1))) if valid.any() else None, confusion_matrix=confusion_matrix(labels[valid], probabilities[valid].argmax(1), labels=[0, 1, 2]).tolist(), class_order=GLOBAL_NAMES, inference_ms_per_image=1000 * sum(t for t, n in latencies) / sum(n for t, n in latencies), inference_batch_size=ec["batch_size"], inference_timing="GPU model + both NMS paths, synchronized, excludes loading and CPU serialization", peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30, elapsed_seconds=time.perf_counter()-begin, checkpoint_epoch=ck["epoch"])
    matrix = np.asarray(metrics['confusion_matrix'])
    metrics['class_recall'] = {name: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else None
                               for i, name in enumerate(GLOBAL_NAMES)}
    for mode in ["object", "relevance", "state"]:
        metrics[mode] = coco_metrics(dataset.rows, predictions, mode)
    write_json(out / "metrics.json", metrics)
    for i in np.linspace(0, len(dataset)-1, min(8, len(dataset)), dtype=int):
        row, p = dataset.rows[i], predictions[i]
        orig = dict(p, boxes=inverse_boxes(p["boxes"]).tolist())
        cv2.imwrite(str(out / f"example_{i}.jpg"), draw_prediction(read_image(row["path"]), orig))
        write_json(out / f"example_{i}.json", dict(image=row["path"], **orig))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


@torch.no_grad()
def predict(checkpoint, image_path, output):
    model, ck = load_checkpoint(checkpoint)
    model.eval()
    img = read_image(image_path)
    transformed = process_image(img)
    tensor = torch.from_numpy(np.ascontiguousarray(transformed[:, :, ::-1].transpose(2, 0, 1))).unsqueeze(0).cuda().float() / 255
    with torch.autocast("cuda", enabled=ck["config"]["train"]["amp"]):
        result = model(tensor)
    p = serialize(model.detections(result, ck["config"]["evaluate"]["conf"])[0])
    p["boxes"] = inverse_boxes(p["boxes"]).tolist()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "prediction.json", dict(image=str(image_path), coordinate_space="original 2048x1024 image", state_order=STATE_NAMES, global_order=GLOBAL_NAMES, **p))
    cv2.imwrite(str(output / "prediction.jpg"), draw_prediction(img, p))
