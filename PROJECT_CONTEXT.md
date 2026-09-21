# DTLD "Color Is Not Enough" project context

This file is an LLM-ready description of the repository, the retained method, the evaluation protocol, and the measured results. It should be read together with README.md, docs/METHODOLOGY.md, configs/dtld.yaml, and artifacts/results/RESULTS.md.

## Project objective and current status

The project adapts the paper "Color Is Not Enough: Dataset and Method for Identifying Relevant Traffic Lights in Driving Scenes" to the German Traffic Light Dataset (DTLD). The aim is to predict both local traffic-light attributes and the global driving-relevant state while retaining a YOLO detector:

- local object detection;
- per-object relevance (relevant vs irrelevant);
- per-object light state;
- image-level global classification: RR (relevant red), RG (relevant green), or NoR (no relevant light).

The completed campaign selected a YOLOv8s model with inverse-square-root weighting for the global classification loss. Five independent seeds (42, 43, 44, 45, 46) were trained and evaluated after the configuration was frozen. Seed 42 is the primary checkpoint; the five checkpoints are independent models, not an ensemble. No training process is currently running.

The repository was cleaned after selection. Discarded experiments, duplicate checkpoints, raw prediction dumps, smoke outputs, and campaign-only orchestration tools were removed. The retained code and artifacts are sufficient to prepare data, train, evaluate, audit generalization, and run inference.

## Dataset, splits, and label policy

The external DTLD source data are not copied or modified by this repository. The default paths in configs/dtld.yaml point to:

- annotations: C:/Users/alexa/Desktop/Tesi_Autonomous_Driving/DTLD/v2.0
- images: C:/Users/alexa/Desktop/Tesi_Autonomous_Driving/DTLD_jpg_plain

The retained clean split is sequence-disjoint and was checked for path and image-hash leakage:

| Partition | Images | Sequences |
|---|---:|---:|
| train | 25,618 | 1,330 |
| internal validation | 2,907 | 148 |
| official test | 12,453 | 632 |

The official test manifest is unchanged. Global labels are valid for 11,901 of the 12,453 test images (95.57% coverage); ambiguous global cases are masked rather than forced into a class. The manifests and audit are in artifacts/data_validation/. The original audited manifests used to rebuild the split are in artifacts/data/.

Input geometry:

- source images are 2048 x 1024;
- a central crop makes a 16:9 image;
- the crop is resized to 1280 x 720;
- 8 pixels of vertical padding produce a 1280 x 736 model canvas;
- source-to-model coordinates are x' = 0.703125*x - 80 and y' = 0.703125*y + 8.

Local state handling is deliberately simplified relative to the paper:

- red, yellow, and red_yellow map to red;
- green maps to green;
- off and unknown map to unknown.

Global labels are derived from relevant objects:

- at least one relevant known red object and no relevant known green object -> RR;
- at least one relevant known green object and no relevant known red object -> RG;
- no relevant objects -> NoR;
- conflicting known relevant states, or only unknown relevant objects, -> masked global label (-1), while local detection/relevance/state supervision remains active.

The development partition is strongly imbalanced. The motivating audit counted roughly 1,146 NoR images versus 15,584 RG images, so the selected recipe weights the global loss without oversampling.

## Retained model

The selected architecture is configurable YOLOv8s (configs/yolov8s.yaml) initialized from artifacts/pretrained/yolov8s.pt. It has one detector class and custom attribute/global heads:

1. YOLO objectness/classification and box regression use the native task-aligned assignment and YOLO box, classification, and DFL losses.
2. A 64-channel projection plus four normalized predicted box coordinates feeds the per-object relevance and state heads.
3. A global head combines pyramid features with five detached spatial priors:
   - detection confidence q;
   - q * relevance_probability;
   - q * red_probability;
   - q * green_probability;
   - q * unknown_probability.
4. Priors are rasterized by maximum rectangle overlap. Candidate filtering uses q >= 0.05, NMS IoU 0.70, and at most 300 candidates.
5. The priors are detached: global gradients update shared features but do not back-propagate through local relevance/state predictions. This preserves the intended multi-task separation.

The final retained configuration is called small_weighted: YOLOv8s, inverse-square-root global class weighting, the five original priors, and Gaussian blur only. A trial adding three joint priors (q * relevance * state) and a trial adding mild brightness/contrast augmentation did not improve validation selection performance and were not retained.

## Training recipe

The default recipe is configs/dtld.yaml:

- COCO initialization;
- AdamW, learning rate 1e-4, weight decay 1e-4;
- exponential learning-rate factor 0.95;
- 30 epochs;
- physical batch 16 and effective batch 16;
- AMP mixed precision;
- four persistent data-loader workers;
- Gaussian blur probability 0.1;
- no hue or geometry augmentation in the selected run;
- seeded sampling and blur;
- validation after every epoch;
- checkpoint selection by validation global mAP.

The global class weights are computed from valid training labels only:

  w_c = f_c^(-1/2) / sum_j sqrt(f_j)

so their frequency-weighted expected value is one. The selected approximate weights are RR=1.1150, RG=0.7982, and NoR=2.9433. Local relevance and state losses are unchanged.

The optimizer state is preserved inside each retained best.pt, so a selected run can be explicitly resumed. Evaluation and prediction load the saved checkpoint settings automatically.

## Evaluation protocol

Detection, relevance, and state are evaluated with COCO-style AP50:95, with max detections [1, 10, 100]. The evaluator uses confidence 0.001 and NMS IoU 0.70.

- Detection score: detector confidence.
- Relevance score: q * P(relevant).
- State score: q * P(state).
- Global AP: scikit-learn average precision for each of RR, RG, and NoR, then their mean (global mAP).
- Balanced accuracy: mean recall over the three global classes.

The test set was not used for architecture, weighting, augmentation, threshold, or checkpoint selection. The final report contains sample standard deviations over five seeds and 1,000 whole-sequence bootstrap resamples. Bootstrap resampling uses 623 valid test sequences; for the selected aggregate, the 95% intervals are approximately:

- global mAP: 78.42% to 86.08%;
- balanced accuracy: 73.76% to 81.26%;
- NoR AP: 39.18% to 61.60%;
- RR AP: 96.94% to 98.53%;
- RG AP: 98.04% to 99.30%.

## Final official-test results

The following are means and sample standard deviations over the five selected seeds:

| Metric | Mean | Std. dev. |
|---|---:|---:|
| global mAP | 82.36% | 1.27 pp |
| balanced accuracy | 80.59% | 2.52 pp |
| RR recall | 92.50% | 1.18 pp |
| RG recall | 98.20% | 0.51 pp |
| NoR recall | 51.06% | 7.96 pp |
| RR AP | 97.56% | 0.35 pp |
| RG AP | 98.28% | 0.60 pp |
| NoR AP | 51.25% | 4.33 pp |
| detection AP50:95 | 33.83% | 0.36 pp |
| relevance AP50:95 | 52.74% | 0.52 pp |
| state mAP50:95 | 31.27% | 0.39 pp |

The original YOLOv8n one-seed baseline on the earlier protocol was:

| Metric | YOLOv8n baseline |
|---|---:|
| global mAP | 77.34% |
| balanced accuracy | 75.10% |
| RR AP / RG AP / NoR AP | 97.15% / 97.11% / 37.76% |
| detection AP50:95 | 31.19% |
| relevance AP50:95 | 50.04% |
| state mAP50:95 | 28.81% |

The selected five-seed mean therefore improves the directional comparison by approximately +5.02 pp global mAP, +5.49 pp balanced accuracy, +13.49 pp NoR AP, +2.64 pp detection AP, +2.70 pp relevance AP, and +2.46 pp state mAP. The baseline was not rerun as a five-seed matched control, so these gains should not be treated as a fully controlled statistical comparison.

## Comparison with the paper

The paper values used as targets are global mAP 83.1%, balanced accuracy 80.8%, NoR AP 54.6%, detection AP50:95 44.4%, relevance AP50:95 35.0%, and state mAP50:95 38.6%. On the selected five-seed mean:

| Metric | This project | Paper target | Difference |
|---|---:|---:|---:|
| global mAP | 82.36% | 83.10% | -0.74 pp |
| balanced accuracy | 80.59% | 80.80% | -0.21 pp |
| NoR AP | 51.25% | 54.60% | -3.35 pp |
| detection AP50:95 | 33.83% | 44.40% | -10.57 pp |
| relevance AP50:95 | 52.74% | 35.00% | +17.74 pp |
| state mAP50:95 | 31.27% | 38.60% | -7.33 pp |

For the paper's global per-class AP values (RR=96.3%, RG=98.4%, NoR=54.6%), this project obtains RR=97.56% (+1.26 pp), RG=98.28% (-0.12 pp), and NoR=51.25% (-3.35 pp). The strongest remaining global limitation is NoR ranking/generalization. Detection and state quality are also materially below the paper's reported values; relevance AP is substantially higher.

These are not exact reproduction numbers. The paper uses a different detector/backbone and protocol (reported Faster R-CNN/ResNet50 results), while this project uses YOLOv8s, a sequence-disjoint internal holdout, a reduced state taxonomy, and masked ambiguous global labels. The comparison is therefore a useful target comparison, not a claim of a controlled reproduction.

## Generalization evidence and interpretation

NoR is rare and scene-dependent. In the initial development run, NoR AP was 74.6% on validation at epoch 5 but 37.8% on test, showing that adjacent-frame image counts do not represent independent scene diversity. The selected weighted YOLOv8s recipe raises the five-seed test mean to 51.25%, but NoR recall still has high seed variation (7.96 pp standard deviation) and the sequence bootstrap interval is wide.

Two additional sequence-disjoint validation confirmations were used before the final test freeze:

- split 43: baseline global mAP 80.73%, NoR AP 47.86%, balanced accuracy 79.64%; winner 82.40%, 50.18%, and 80.13%;
- split 44: baseline global mAP 75.84%, NoR AP 35.78%, balanced accuracy 74.41%; winner 76.67%, 37.64%, and 75.75%.

The weighted model is consistently better, but split-to-split variation remains material. Threshold movement may change balanced accuracy at one operating point, but it cannot repair poor NoR AP/ranking.

## Recommended next work

1. Harmonize the protocol before claiming paper-level reproduction. Train/evaluate on the paper's official split and full official training pool where possible, reproduce its state taxonomy and relevance/global-label definitions, and report the current masked-label results separately.
2. Attack NoR generalization directly. Audit errors by sequence and scene type (empty scenes, visible irrelevant lights, intersection exits, and ambiguous annotations); use sequence-balanced sampling or hard-negative mining; and keep global mAP as the checkpoint-selection metric.
3. Improve local evidence if state/detection remains limiting. Test YOLOv8m after the s-model is established, then consider a P2/high-resolution feature level or a crop/RoI attribute head for tiny traffic lights. Keep architecture changes isolated.
4. Restore the paper's state detail in a separate experiment. The current red/yellow/red-yellow collapse makes the comparison easier but loses information; a four-state local head can be evaluated while retaining the three global classes.
5. Use calibration only as a separate validation analysis. It can improve balanced accuracy reporting, but it does not substitute for better ranking or detection.
6. Defer temporal modeling, external data, and large resolution changes until protocol and local/global failure modes are established.

## Training time and hardware

For the primary seed 42, artifacts/selected_model/seed42/history.json records:

- 30 epochs;
- training time: 11,459.3 s = about 3 h 11 min;
- validation time: 1,387.3 s = about 23 min;
- total train plus validation: 12,846.5 s = about 3 h 34 min;
- mean throughput: 67.25 images/s;
- peak reserved GPU memory: 7.97 GB;
- parameters: 11,482,674;
- physical/effective batch: 16.

The five retained seeds together used approximately 17.66 hours when run serially (roughly 3.5 to 3.6 hours per seed). The full exploratory campaign took longer; discarded exploratory outputs were removed.

The recorded environment is Python 3.12, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, Ultralytics 8.4.116, and an RTX 5070 with 12 GB VRAM.

## Repository map and commands

- cine/: data preparation, model, training engine, inference, evaluation, split audit, and generalization audit.
- configs/dtld.yaml: selected training/evaluation recipe.
- artifacts/data/: original audited manifests.
- artifacts/data_validation/: clean sequence-disjoint manifests and audits.
- artifacts/pretrained/yolov8s.pt: retained COCO initialization.
- artifacts/selected_model/seed42 through seed46/: one byte-preserved best.pt per seed, history, curves, and metadata.
- artifacts/results/: per-seed metrics, bootstrap intervals, final report, selection/provenance records, and cleanup manifest.
- tests/: retained pipeline tests.
- .deps/: local COCO evaluator dependency.

Typical commands from the repository root:

    python -m cine predict --checkpoint artifacts/selected_model/seed42/best.pt --image "C:/path/frame.jpg" --output artifacts/prediction
    python -m cine evaluate --checkpoint artifacts/selected_model/seed42/best.pt --output artifacts/evaluation
    python -m cine audit --checkpoint artifacts/selected_model/seed42/best.pt --output artifacts/validation_audit
    python -u -m cine train --evaluate-after
    python -m pytest -q

README.md describes the supported workflow. artifacts/results/cleanup_manifest.json maps retained checkpoint paths to their original paths and verifies SHA-256 preservation. No source DTLD files or the supplied paper PDF were altered.
