# YOLOv8m comparison, seed 42

Two fresh runs replace YOLOv8s with COCO-initialized YOLOv8m while preserving the two existing global-head designs:

- `configs/dtld_m_raster.yaml`: retained five-channel raster priors and convolutional pooling.
- `configs/dtld_m_set_null.yaml`: Top-64 detached box evidence, shared 128-dimensional MLP, attention with learned NULL, and scene GAP. The optional ROI vector remains omitted.

Both use 30 epochs, seed 42, AdamW lr 1e-4, weight decay 1e-4, exponential gamma .95, image sampling, inverse-square-root global class weighting, blur probability .1, AMP, and unchanged audited data and geometry. Best checkpoints maximize validation global mAP. Local labels, detach boundary, evaluation definitions and thresholds are unchanged.

## Batch size and preflight

The physical batch is 8 for both runs, with two-step gradient accumulation to effective batch 16. Evaluation also uses batch 8. On the RTX 5070 12 GB, dense full-resolution real-image probes completed optimizer updates and validation loss calculation. Peak reserved memory: 6.90 GB raster, 6.95 GB NULL. The original physical batch 16 was not probed for these larger models. The selected batch leaves GPU headroom.

Raster has 26,236,354 parameters; NULL has 26,024,835. Both have pyramid channels [192,384,576]. Compatible COCO transfer copied 469/475 detector tensors and all three traffic-light output layers. Both start from the same medium pretrained file, not from the earlier small checkpoints.

## Execution and outputs

The controller runs serially: train raster, train NULL, freeze both best-validation checkpoints, evaluate each once on the official test partition, then write the comparison. A failure stops the queue and records the reason. A process lock prevents duplicate controllers. Code, weights, saved configs and data hashes are checked before each subprocess; modifications stop the queue rather than mix sources.

    python -u scripts/run_yolov8m_comparison.py

Re-running that command after an interruption resumes incomplete training from last.pt and skips completed stages. An incomplete evaluation with metrics already present requires inspection before repetition.

Outputs: `artifacts/yolov8m_comparison/`. Read `status.json` for the current queue stage. Per-run directories `raster_seed42` and `set_null_seed42` contain training.log, training.stderr.log, status.json, experiment.json, history.json and checkpoints. Evaluation outputs go into each run's evaluation directory. The completed tables are `COMPARISON.md` and `comparison.json` at the comparison root. Both runs and official-test evaluations have finished. `manifest.json` records frozen inputs; source and config copies preserve provenance.

## Comparison limits

The medium runs are matched to each other. Historical small runs used physical batch 16: gradient accumulation retains effective batch size but does not reproduce the same batch-normalization statistics. The earlier small NULL run was stopped after 13 completed epochs and selected epoch 5; both medium runs completed 30 epochs. Their best checkpoints were selected at epochs 3 (raster) and 9 (Top-64/NULL). Size comparisons against historical results are therefore preliminary, not a perfectly controlled size-only ablation. These are single-seed results. Inspect global mAP, RR/RG/NoR recall and local AP jointly before preferring a model.

The completed test comparison is [COMPARISON.md](../artifacts/yolov8m_comparison/COMPARISON.md). The medium Top-64/NULL model reached 85.15% test global mAP, compared with 81.15% for medium raster and 82.36% (sample SD 1.27 pp) for the retained YOLOv8s five-seed mean. Both medium runs show validation overfitting after their early selected epochs; the test uses each validation-selected checkpoint. Each medium result is one seed and should be confirmed across seeds before it replaces the retained model.

## Live monitoring

Open another PowerShell window at the repository root and run:

    python scripts/watch_yolov8m_comparison.py

It refreshes every 30 seconds. Ctrl+C exits the monitor without stopping training. For a single snapshot, use `python scripts/watch_yolov8m_comparison.py --once`. It reads each run's `status.json` and epoch-level `history.json`; it does not load a model or use GPU. Follow `controller.log` for queue changes and each run's `training.log` for detailed batch progress. The queue is complete; the script can still inspect its saved run histories. Compare train and validation global loss, validation global mAP, NoR AP and RR recall across several completed epochs. A sustained training-loss decrease while validation loss rises and validation metrics fall is a stronger overfitting signal than a one-epoch wobble. During an epoch it shows the most recent completed validation until the next epoch finishes.
