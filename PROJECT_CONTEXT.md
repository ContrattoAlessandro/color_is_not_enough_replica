# DTLD “Color Is Not Enough” — coding-agent context

Updated 2026-09-24 from the retained source, experiment records, and reports in this repository. Read this with README.md and docs/METHODOLOGY.md before changing the pipeline.

## Project and status

This repository adapts the paper “Color Is Not Enough: Dataset and Method for Identifying Relevant Traffic Lights in Driving Scenes” to the German Traffic Light Dataset (DTLD). It predicts traffic-light boxes, per-object relevance and state, and a frame-level driving-relevant class: RR (relevant red), RG (relevant green), or NoR (no relevant light).

The strongest completed, protocol-approved experiment is the retained small_weighted configuration: COCO-initialized YOLOv8s with inverse-square-root weighting of the global RR/RG/NoR loss. It was frozen before final evaluation and trained with five independent seeds (42–46). The five checkpoints are separate models, not an ensemble. Seed 42 is the primary checkpoint used in examples.

A newer bounded campaign in artifacts/improvement_v2 tried sequence/class sampling, a YOLOv8s-P2 detector, and staged training. Its terminal status is confirmation_stopped: the screened sequence/class sampler did not pass the predeclared cross-split confirmation gates; P2 and staged training did not advance from screening. No official-test results were used for that campaign. Do not describe its best single-split validation checkpoint as the project winner or substitute it for the retained five-seed result.

This is a methodological adaptation, not an exact reproduction of the paper. The detector/backbone, split handling, masked global labels, and state mapping differ. The DTLD_test.json source partition is used as the project’s official test partition, but equivalence to the authors’ exact evaluation sample/mapping is unverified.

## Data, labels, and geometry

DTLD source data are external to the repository. The defaults in configs/dtld.yaml refer to:

- Annotations: C:/Users/alexa/Desktop/Tesi_Autonomous_Driving/DTLD/v2.0
- JPEGs: C:/Users/alexa/Desktop/Tesi_Autonomous_Driving/DTLD_jpg_plain

The retained prepared split is sequence-disjoint and audited for path and image-hash leakage:

| Partition | Images | Sequences |
|---|---:|---:|
| Train | 25,618 | 1,330 |
| Internal validation | 2,907 | 148 |
| Official test source partition | 12,453 | 632 |

Of the official test source images, 11,901 (95.57%) have a valid global label. Ambiguous global labels are masked; their local object labels remain usable. Prepared manifests and audits live in artifacts/data_validation; source manifests and split provenance are retained in artifacts/data and artifacts/results.

Each source image is 2048×1024. The transform scales by 0.703125, center-crops horizontally to 1280 pixels, resizes the content to 1280×720, and pads 8 pixels at the top and bottom to make a 1280×736 model canvas. In source coordinates, x' = 0.703125*x - 80 and y' = 0.703125*y + 8.

Local states use three classes: red, green, unknown. Source red, yellow, and red_yellow map to red; green maps to green; off and unknown map to unknown. Global labels come from relevant objects:

- At least one known relevant red and no known relevant green → RR.
- At least one known relevant green and no known relevant red → RG.
- No relevant objects → NoR.
- Both known red and green among relevant objects, or relevant objects with no known state → masked (-1).

The label equivalence to the paper remains unverified. The training set is imbalanced (7,985 RR, 15,584 RG, 1,146 NoR), which motivated global loss weighting rather than oversampling in the winning recipe.

## Best-performing retained architecture

The selected model is implemented in cine/model.py. It uses Ultralytics YOLOv8s initialized from the retained COCO checkpoint at artifacts/pretrained/yolov8s.pt. The YOLO architecture definition comes from the installed Ultralytics package; configs/yolov8s.yaml is not a repository file. The detector predicts one traffic-light class and uses the native YOLO task-aligned assignment and box, classification, and DFL losses (gains 7.5, 0.5, 1.5). Seed 42 has 11,482,674 parameters and feature channels [128, 256, 512].

The model adds two tasks to the detector:

1. Per-object heads: each detector pyramid feature is projected to 64 channels. At each detector location, the projected feature and four normalized decoded-box coordinates feed a binary relevance head and a three-class state head.
2. Global head: at each pyramid level, a separate projected feature map is concatenated with five rasterized spatial priors: detector confidence q, q×P(relevant), q×P(red), q×P(green), and q×P(unknown). A small convolutional branch pools each level; the pooled level vectors are concatenated and classified as RR/RG/NoR.

Prior boxes are selected at q ≥ 0.05, NMS IoU 0.70, up to 300 candidates per image. Each prior is rasterized over its box using maximum overlap. The prior values are detached: global loss can update shared detector features through the global branch but cannot backpropagate through local relevance/state probabilities. This is the retained multi-task design; the rejected joint-prior trial is not enabled.

## Winning experiment and measured results

The configuration is configs/dtld.yaml; the retained per-seed provenance is in artifacts/selected_model/seed42–seed46/experiment.json.

- YOLOv8s, COCO initialization; AdamW, learning rate 1e-4, weight decay 1e-4, exponential LR factor 0.95.
- 30 epochs; batch size/effective batch 16; AMP; four persistent data-loader workers.
- Gaussian blur probability 0.1 is the only selected augmentation. No hue, geometric, brightness, or contrast augmentation.
- Image-level sampling; seeded sampling and blur. No oversampling.
- Global loss uses valid training labels only. Weights are proportional to inverse square-root class frequency and normalized to have frequency-weighted mean 1: RR 1.1150, RG 0.7982, NoR 2.9433. Local losses are not reweighted.
- Validate every epoch and select each seed’s checkpoint by validation global mAP. Freeze the recipe before evaluating the official test source partition.

Official-test results are the mean and sample standard deviation across five seeds:

| Metric | Mean | Sample SD |
|---|---:|---:|
| Global mAP | 82.36% | 1.27 pp |
| Balanced accuracy | 80.59% | 2.52 pp |
| RR / RG / NoR recall | 92.50% / 98.20% / 51.06% | 1.18 / 0.51 / 7.96 pp |
| RR / RG / NoR AP | 97.56% / 98.28% / 51.25% | 0.35 / 0.60 / 4.33 pp |
| Detection AP50:95 | 33.83% | 0.36 pp |
| Relevance AP50:95 | 52.74% | 0.52 pp |
| State mAP50:95 | 31.27% | 0.39 pp |

Global AP is per-class scikit-learn average precision averaged over RR, RG, and NoR. Balanced accuracy is mean class recall. Detection, relevance, and state use COCO-style AP50:95 with maxDets [1, 10, 100]; relevance scores are q×P(relevant), state scores q×P(state). Evaluation confidence is 0.001 and NMS IoU 0.70. The report includes whole-sequence bootstrap intervals; see artifacts/results/final_report.json and artifacts/results/RESULTS.md. The 95% whole-sequence bootstrap intervals for global mAP, balanced accuracy, NoR AP, RR AP, and RG AP are approximately 78.42–86.08%, 73.76–81.26%, 39.18–61.60%, 96.94–98.53%, and 98.04–99.30%, respectively.

NoR is the principal generalization weakness: it is rare, has high seed/split variation, and its five-seed AP is 51.25% (SD 4.33 pp). The five-seed confidence intervals and sequence bootstrap intervals are material. An earlier single-seed YOLOv8n baseline was evaluated under an earlier protocol, so its difference from this result is directional, not a matched statistical comparison. The paper comparison is also not controlled: its detector and protocol differ.

## Follow-up campaign outcome

docs/IMPROVEMENT_CAMPAIGN.md defines the bounded P2-first experiment. Outputs, provenance, and terminal records are under artifacts/improvement_v2.

- S = sequence/class-aware sampling; P = YOLOv8s-P2 with stride 4; T = staged local/global/joint training.
- S advanced from screening, with split-42 validation global mAP 91.22%, but it did not pass cross-split confirmation. Its higher single-split score is not final evidence.
- P and T did not advance from screening.
- Campaign outcome is confirmation_stopped (“Winner did not meet cross-split gates”). No new test-set results were produced.

Keep this follow-up separate from the selected small_weighted method. If continuing model research, first state which split and metric are being used; preserve the frozen official-test protocol and do not tune on test results.

## Repository map and working commands

- cine/data.py: source-label parsing, auditing, fixed image/box transform, dataset and collation.
- cine/split.py: sequence-grouped train/validation split and disjointness checks.
- cine/model.py: selected joint model, heads, priors, and losses.
- cine/engine.py, cine/phases.py, cine/monitoring.py: training, checkpoint/resume, validation and checkpoint selection.
- cine/evaluate.py: COCO-style local metrics, global metrics, prediction and rendering.
- cine/generalization.py: sequence/scene audit.
- cine/campaign.py, cine/selection.py, cine/transfer.py: bounded improvement campaign and its selection/transfer logic.
- configs/dtld.yaml: default selected recipe.
- artifacts/selected_model/seed42–seed46/: retained best checkpoints, histories, and experiment metadata.
- artifacts/results/: frozen selection, final metrics, bootstrap reports, and provenance.
- tests/: pipeline tests. .deps/: local COCO evaluator dependency.

From the repository root, the usual commands are:

    python -m cine predict --checkpoint artifacts/selected_model/seed42/best.pt --image "C:/path/frame.jpg" --output artifacts/prediction
    python -m cine evaluate --checkpoint artifacts/selected_model/seed42/best.pt --output artifacts/evaluation
    python -m cine audit --checkpoint artifacts/selected_model/seed42/best.pt --output artifacts/validation_audit
    python -u -m cine train --evaluate-after
    python -m cine.campaign status

The CLI loads the saved configuration from a checkpoint for evaluate, predict, and resume. Only pass an explicit config when it intentionally matches the checkpoint. Training requires CUDA and audited prepared manifests; defaults expect source DTLD at the external paths above. The recorded successful-run environment is Python 3.12, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, Ultralytics 8.4.116, and an RTX 5070 with 12 GB VRAM. Seed 42 took about 3 h 34 min for training plus validation, averaging 67.25 images/s with 7.97 GB peak reserved memory; these are run-specific measurements. New training outputs go under artifacts/training. Use a new output directory or explicit resume; do not overwrite retained checkpoints/results. The clean split is already prepared. Do not run the original preparation command against it: with validation enabled, prepare intentionally refuses to overwrite; split regeneration is handled by cine.split.

## Guidance for the next agent

Treat artifacts/results/frozen_selection.json and the five retained seed checkpoints as the final selected evidence. Treat artifacts/improvement_v2/RESULTS.md and confirmation.json as evidence that the follow-up was stopped, not as a new release result. Keep the label taxonomy, split manifests, evaluation definitions, and frozen-test boundary explicit in any future comparison. The safest next research priorities are to audit NoR errors by sequence/scene and to improve tiny-light detection/state quality while isolating architecture changes. Resolve the paper’s exact state and split protocol before making a reproduction claim.
