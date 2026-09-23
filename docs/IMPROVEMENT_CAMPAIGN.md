# P2-first improvement campaign

This is a follow-up campaign informed by previously reported test results. It is an adaptation, not an exact paper reproduction. Existing data, checkpoints and result reports are preserved.

## Commands

Run from the repository root:

    python -m cine.campaign preflight
    python -m cine.campaign run
    python -m cine.campaign status

The controller runs serially, resumes incomplete runs from last.pt, and holds an OS lock to prevent duplicate campaigns. Code, pretrained weights, configuration, software versions and primary manifests are fingerprinted. Changed provenance causes a stop rather than mixing experiments. Stages screen, confirm and final are also available; final requires successful screening and confirmation. The controller never passes --evaluate-after to exploratory training.

Outputs live under artifacts/improvement_v2. status.json identifies the current run; each run has train.log, status.json, config.yaml, history.json and selection.json. failure.json records exceptions. RESULTS.md records the terminal outcome. Do not overwrite the retained artifacts/results or artifacts/selected_model directories.

## Fixed experiment design

Use a common physical batch, chosen from 16, 8, 4, 2 by full-resolution probes with at least 1 GiB GPU headroom; effective batch remains 16. Baseline and all candidates use the same batch and original preprocessing and evaluation conventions.

Screen a fresh baseline, S (sequence/class sampler), P (YOLOv8s-P2), and T (staged training), each with split 42 and seed 42. If two or more isolated changes pass, test their combination once. At most five screening runs.

S preserves exact image counts per global class, including masked images, per epoch. It samples sequences uniformly within class and frames uniformly within sequence with replacement. The original class weights remain valid because class frequencies are preserved. Seed plus epoch determines the complete order.

P adds stride 4 to strides 8/16/32. P2 transfer explicitly maps the shared backbone/top-down neck and equivalent P4/P5 bottom-up modules. Detection branches transfer by stride and compatible shape. The new P2/P3 fusion and incompatible tensors are initialized normally. Transfer reports enumerate copied and newly initialized tensors.

T uses 10 local epochs, selects by mean detection/state AP, restores that checkpoint, trains only global modules for 2 epochs with local batch-normalization frozen, then jointly fine-tunes for 18 epochs. Local/head-only phases start at 1e-4. Joint fine-tuning uses local parameters at 1e-5 and global parameters at 1e-4. AdamW, weight decay 1e-4 and exponential gamma .95 are retained. Optimizer and scheduler restart at transitions. Training resumes from the checkpoint phase and requires local_best.pt for the transition after epoch 10.

New config fields: train.sampler (image or sequence_class), train.recipe (joint or staged), train.task_weights (detection, relevance, state, global_loss; defaults one), validation.selection_metric (global_mAP or multitask), validation.reference_metrics for multitask. Existing configs retain their original behavior. Predictions and checkpoint model-state interfaces remain compatible with retained models.

## Selection

For each split, the baseline reference is its best-global-mAP checkpoint. Candidate joint checkpoints must keep all six monitored metrics (global mAP, balanced accuracy, NoR AP, detection AP, state AP, relevance AP) within .005 of that reference or improve. Rank eligible checkpoints by mean(global mAP, detection AP, state AP), then NoR AP; exact ties retain the earlier epoch. Advance changes only when that mean improves at least .005.

Confirm the strongest candidate against freshly trained baselines on sequence splits 43 and 44, training seed 42. Require a positive ranking-score difference on at least two of three splits, a mean improvement of at least .005, no decrease in mean NoR AP, and no other mean metric loss greater than .005. A failure ends the bounded search.

Freeze the winning recipe before final evaluation. Complete training seeds 42-46 on the primary split, reusing the identical selected screening seed 42. If any seed has no eligible checkpoint, report that outcome without test evaluation. Otherwise evaluate each checkpoint once, report mean/sample SD and whole-sequence bootstrap intervals. Final test targets: global .831, balanced accuracy .808, NoR .546, detection .444, state .386, relevance at least .5224. A recommendation pointer is produced only if five-seed results improve the three-metric mean and satisfy all non-regression guards against the historical five-seed baseline. Existing default checkpoints remain preserved.

## Diagnostics and limitations

Validation reports include score-ordered class-agnostic one-to-one matching at confidence .05, recall at IoU .50/.75 for ground-truth widths <4, 4-8, 8-16 and >=16 pixels, state confusion conditional on matching, and match coverage. These are diagnostic metrics, separate from unchanged COCO AP. Selected-run audits group NoR failures by sequence and annotation-derived scene type; intersection-exit labels require visual review.

The local state taxonomy remains red/green/unknown. The paper mentions both four annotation states and a three-class classifier; exact author mappings remain unresolved. Prepared test image identities are checked against the source DTLD_test.json. Calling that partition test versus validation does not establish equivalence to the authors' exact sample set.

RoI pooling, YOLOv8m, resolution changes, external data, temporal modeling, calibration and full-training-pool retraining are deferred. Training time is estimated from probes, not promised; actual epoch throughput is recorded.
