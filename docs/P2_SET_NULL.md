# P2: set evidence with a learned NULL token

This P2 names the global-head proposal, not the earlier YOLOv8s-P2 stride-4 detector experiment. The new config uses standard YOLOv8s. The retained raster model remains the default, including strict loading of its original checkpoints.

## Implementation

`configs/dtld_p2_set_null.yaml` changes the global head and output directory relative to `configs/dtld.yaml`. Training remains COCO initialization, seed 42, batch/effective batch 16, inverse-square-root global weighting, image sampling, 30 epochs and the same prepared split. No sampler change is included in this isolated architecture ablation.

The set path takes at most 64 dense boxes by confidence q, preserving the existing q >= 0.05 threshold and content clipping. It removes NMS from global evidence selection; local detection evaluation still uses the existing NMS. Invalid/nonfinite or degenerate boxes are excluded. Score ties at the cutoff are resolved lexicographically by feature values to preserve permutation invariance. Short sets are padded with an explicit validity mask.

Each token has 11 detached values: four content-normalized xyxy coordinates, log normalized area (floored at 1e-12), normalized pyramid-level index, q, q*P(relevant), q*P(red), q*P(green), q*P(unknown). The optional ROI vector is omitted in this first variant. A shared Linear(11,128)-ReLU-Linear(128,128)-ReLU embeds each box. A learned 128-dimensional NULL token is appended before learned scalar attention and softmax pooling. NULL remains valid for every image; an empty set pools to NULL exactly. NULL is a learned evidence embedding, not a fixed NoR label.

GAP of each existing global feature projection yields the 192-dimensional scene vector. Concatenating it with the 128-dimensional pooled evidence gives a 320-dimensional input to the three-class RR/RG/NoR classifier. There are no prior raster maps or per-level global convolutions on this path. Evidence and geometry are detached; the scene path continues to train shared features, while global loss cannot reach detector output or local attribute heads. Diagnostics expose detached `box_evidence`, `evidence_valid`, and 65 attention weights with NULL last; `priors` is empty for this variant.

## Which experiment was best?

The sequence/class sampler (S) used `yolov8s.yaml`, not the stride-4 detector. It won the newer campaign screening against its matched batch-8 baseline, but did not pass confirmation. It is not the retained release model. All three columns below use internal split 42 and training seed 42; the historical retained recipe used physical batch 16, while the campaign used physical batch 8 with accumulation to 16, and candidate checkpoint selection used additional multitask guards.

| Validation metric | Retained model (epoch 3) | Campaign baseline (epoch 20) | Sampler S (epoch 15) |
|---|---:|---:|---:|
| Global mAP | 91.83% | 90.29% | 91.22% |
| Balanced accuracy | 88.24% | 89.86% | 90.54% |
| NoR AP | 79.21% | 75.24% | 77.52% |
| NoR recall | 70.63% | 75.40% | 80.16% |
| Detection AP50:95 | 37.03% | 36.27% | 36.79% |
| Relevance AP50:95 | 52.62% | 52.01% | 51.78% |
| State AP50:95 | 32.40% | 31.46% | 32.18% |

On split 43, no sampler checkpoint met all six non-regression guards. Its best-global checkpoint (epoch 10) improved global mAP from 81.96% to 83.29%, but detection fell from 37.15% to 35.84%, relevance from 52.80% to 50.49%, and state from 33.47% to 32.33%. Each exceeds the permitted 0.5 pp loss. Split 44 had an eligible sampler checkpoint (77.64% global mAP versus 77.01% baseline), but that does not resolve split 43's failure. `confirmation.json` records accepted=false and the campaign stopped without new official-test evaluation.

Sources: `artifacts/selected_model/seed42/history.json`, `artifacts/improvement_v2/screening.json`, `artifacts/improvement_v2/confirmation.json`, `artifacts/improvement_v2/runs/S_split43_seed42/history.json` and its `selection.json`.

## Retained official-test reference

These are five independent seeds (42-46), mean plus sample standard deviation, from `artifacts/results/RESULTS.md`. They must not be compared numerically with validation scores as if they were the same partition.

| Metric | Mean | Sample SD |
|---|---:|---:|
| Global mAP | 82.36% | 1.27 pp |
| Balanced accuracy | 80.59% | 2.52 pp |
| RR / RG / NoR AP | 97.56% / 98.28% / 51.25% | 0.35 / 0.60 / 4.33 pp |
| RR / RG / NoR recall | 92.50% / 98.20% / 51.06% | 1.18 / 0.51 / 7.96 pp |
| Detection AP50:95 | 33.83% | 0.36 pp |
| Relevance AP50:95 | 52.74% | 0.52 pp |
| State mAP50:95 | 31.27% | 0.39 pp |

## Run and compare

From the repository root:

    python -m pytest -q tests/test_set_head.py
    python -u -m cine --config configs/dtld_p2_set_null.yaml train

Resume with the saved configuration:

    python -u -m cine train --resume artifacts/p2_set_null/seed42/last.pt

Training validates every epoch and saves best.pt by validation global mAP. Do not pass --evaluate-after during exploration: that invokes official-test evaluation. The seed-42 run was launched on 2026-09-25 and stopped during epoch 14 after 13 completed epochs at the user's request to evaluate the best validation checkpoint. Epoch 5 was frozen before the official-test evaluation. Logs, status, launch provenance, and source/config snapshots are in artifacts/p2_set_null/seed42. The completed comparison is in [COMPARISON.md](../artifacts/p2_set_null/seed42/test_best_epoch005/COMPARISON.md). First compare on internal validation using the retained seed-42 metrics above, then confirm promising results across seeds/splits before freezing and testing. A sampler-plus-set-head run would be a separate interaction experiment and needs a matched sampler baseline; it should not be silently folded into this head-only ablation.

## Verification

The full test suite passes (51 tests), including NULL-only inputs, cutoff ties, padding, probability features, nonfinite candidates, CUDA AMP, checkpoint roundtrips, and global-only gradient isolation. A batch of two real training images at 1280x736 completed two AMP optimizer updates after automatic initial loss-scale backoff; this was an in-memory smoke check with no saved trained checkpoint. The retained seed-42 checkpoint also loaded strictly. The set-head model has 11,271,155 parameters versus 11,482,674 for the retained raster model. The full batch-16 run completed 13 epochs before being stopped during epoch 14. The frozen epoch-5 checkpoint has now been evaluated on all 12,453 official-test images: global mAP 82.73%, balanced accuracy 81.59%, NoR AP 51.87%. See the comparison report for class tradeoffs and uncertainty.

## Official-test outcome

The epoch-5 P2 checkpoint scored 82.73% global mAP versus 82.12% for retained seed 42 and 82.36% for the retained five-seed mean. NoR recall improved, while relevant-red recall decreased. Local AP metrics remained close to the retained averages. This single-seed result does not establish a new overall architecture winner; the retained model remains the default. Training is stopped, and last.pt preserves epoch 13 for explicit resume. The original no-test command above documents the training protocol; the later official-test evaluation was explicitly requested by the user.
