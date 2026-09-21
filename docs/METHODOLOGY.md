# Retained method and evaluation

COCO-initialized YOLOv8s has a one-class detector, relevance/state heads, and a global RR/RG/NoR classifier. Detection uses native task-aligned assignment and box/classification/DFL gains 7.5/0.5/1.5. Local attribute heads receive 64 projected channels plus four normalized predicted box coordinates.

The global head combines pyramid features with five detached spatial priors: confidence q, q times relevance, and q times each state probability. Rectangle overlaps use maximum; candidates use q >= 0.05, NMS IoU 0.7, max 300. Global gradients update shared features but do not traverse the priors to local heads.

## Labels and geometry

2048x1024 images are centrally cropped to 16:9, resized to 1280x720, then padded to 1280x736. The affine transform is x' = 0.703125*x - 80 and y' = 0.703125*y + 8. Predictions can be mapped to original coordinates.

States: red/yellow/red-yellow -> red; green -> green; off/unknown -> unknown. Relevant objects determine RR or RG. No relevant objects means NoR. Conflicting known relevant states or only unknown relevant states are masked globally and retain local supervision.

The clean split has 25,618 training images (1,330 sequences), 2,907 validation images (148 sequences), and 12,453 official-test images (632 sequences), disjoint by sequence, path, and image hash. Global test coverage is 11,901 images (95.57%).

## Training

AdamW, learning rate 1e-4, weight decay 1e-4, exponential factor 0.95, 30 epochs, physical/effective batch 16, AMP, four persistent workers. Gaussian blur probability 0.1 is the only augmentation. Sampling and blur are seeded per image/epoch.

Global weights use valid training labels only: w_c = f_c^(-1/2) / sum_j sqrt(f_j), giving expected weight one. Global loss is mean(w_y * cross_entropy) over valid images. Relevance and state use foreground means. Empty/masked batches yield finite connected zero losses. Current weights are approximately RR 1.1150, RG 0.7982, NoR 2.9433.

Every epoch validates all tasks; best.pt maximizes validation global mAP. Reports average independent seeds 42-46. Joint-prior and photometric trials did not improve selection performance and were removed. Historical comparisons remain in the compact final report.

## Evaluation and evidence

COCO AP50:95 uses maxDets [1,10,100]. Relevant-object scores are q*p(relevant); state scores are q*p(state). Evaluation confidence is 0.001 and NMS IoU 0.7. Global AP uses sklearn average precision; balanced accuracy averages per-class recall.

Configuration selection preceded the five official-test evaluations. Reports include sample standard deviation and 1,000 whole-sequence bootstrap resamples per model. Seed 42 is the primary checkpoint; no seed was selected using test results.

Retained checkpoints and final reports are unchanged originals. Historical frozen-selection metadata retains original paths; cleanup_manifest.json maps checkpoints to their new paths and verifies SHA-256 hashes. Optimizer state remains inside best.pt for explicit resume. Extra checkpoints and raw prediction dumps were removed.

Validation audit groups use annotations to distinguish empty scenes, visible irrelevant lights, relevant lights, and masked cases. Intersection-exit and annotation-issue fields require visual review; sequence tails are only review candidates. Audits preserve existing yes/no tags.

The architecture, internal holdout, and label handling differ from the paper. Do not claim exact reproduction or a controlled Faster R-CNN speed advantage. Recorded latency is amortized batch latency, not batch-one deployment latency.
