# Color Is Not Enough: YOLOv8s on DTLD

Traffic-light detection, per-object relevance and state, and global RR/RG/NoR classification. The retained model is YOLOv8s with inverse-square-root global class weighting.

## Use the trained model

Run from this directory with the existing CUDA-enabled Python installation:

    python -m cine predict --checkpoint artifacts/selected_model/seed42/best.pt --image "C:/path/frame.jpg" --output artifacts/prediction
    python -m cine evaluate --checkpoint artifacts/selected_model/seed42/best.pt --output artifacts/evaluation
    python -m cine audit --checkpoint artifacts/selected_model/seed42/best.pt --output artifacts/validation_audit

Seed 42 is the primary checkpoint from configuration selection. Seeds 43-46 support the five-seed results; these are independent models, not an ensemble. All checkpoints were selected by validation global mAP, not test scores.

Evaluation, prediction, and resume load the original checkpoint settings automatically. Historical paths inside checkpoint metadata describe the original run; checkpoint files remain byte-identical after cleanup. Supply --config only for an intentionally matching saved configuration.

## Train

The default recipe is configs/dtld.yaml, using retained COCO weights and audited manifests.

    python -u -m cine train --evaluate-after
    python -u -m cine train --resume artifacts/training/last.pt --evaluate-after
    python -m pytest -q

New runs write to artifacts/training. Existing outputs require explicit resume or a different --output directory. After training, --evaluate-after evaluates the validation-selected best checkpoint. The optional run_training.ps1 launcher supports -Resume CHECKPOINT and -OutputDirectory DIRECTORY.

## Retained files

- cine/: preparation, model, training, inference, evaluation, validation audits, and utilities.
- configs/dtld.yaml: the selected training recipe.
- tests/: tests for the retained pipeline.
- artifacts/data/: original audited manifests for rebuilding the clean split.
- artifacts/data_validation/: audited train/validation/test manifests.
- artifacts/pretrained/yolov8s.pt: COCO initialization.
- artifacts/selected_model/seed42 through seed46/: best checkpoints, histories, curves, and original experiment metadata.
- artifacts/results/: final report, per-seed test metrics/bootstrap intervals, primary-model examples, and provenance/cleanup records.
- .deps/: required local COCO evaluator dependency.

Rejected runs, search/benchmark scripts, smoke artifacts, duplicate final/last checkpoints, and regenerable prediction dumps were removed. External source DTLD files are untouched.

## Results

Official-test mean and sample standard deviation over five seeds:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Global mAP | 82.36% | 1.27 pp |
| Balanced accuracy | 80.59% | 2.52 pp |
| NoR AP | 51.25% | 4.33 pp |
| Detection AP50:95 | 33.83% | 0.36 pp |
| Relevance AP50:95 | 52.74% | 0.52 pp |
| State mAP50:95 | 31.27% | 0.39 pp |

See artifacts/results/RESULTS.md and final_report.json. This remains a methodological adaptation: the internal holdout, state mapping, and masked global labels differ from the paper. Global targets were approached but not reached on the five-seed means.

Environment: Python 3.12, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, Ultralytics 8.4.116, RTX 5070 12 GB. See requirements.txt and docs/METHODOLOGY.md.

## New improvement campaign

The P2-first campaign adds controlled sampling, P2 detection and staged-training trials. Run `python -m cine.campaign run`; inspect progress with `python -m cine.campaign status`. All new outputs are isolated under `artifacts/improvement_v2` and existing selected models remain available. See [docs/IMPROVEMENT_CAMPAIGN.md](docs/IMPROVEMENT_CAMPAIGN.md) for selection gates, resume behavior and reporting.
