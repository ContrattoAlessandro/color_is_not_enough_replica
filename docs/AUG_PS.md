# Retained PS augmentation

The retained augmentation is **P + S**: moderate photometric exposure, gamma,
and contrast changes; Gaussian noise, three-pixel motion blur, and mild
downsample/upsample. The gallery review disabled JPEG. No color-response or
atmosphere augmentation is active. All operations affect only the 1280×720 image
content; the eight-pixel top and bottom padding, boxes, states, relevance, and
global labels remain unchanged.

The policy reserves 25% of samples as clean. On the remaining samples, one P
operation triggers with probability 0.50, one S operation with 0.25, and the
original Gaussian blur with 0.10. At most two operations are used. Blur never
co-occurs with motion blur or downsampling. Randomness is derived from the
training seed, epoch, sample identity, and operation, so worker count and
epoch-boundary resume do not change an image's transform.

`configs/dtld_m_set_null_aug_ps.yaml` runs YOLOv8m Top-64/NULL without ROI for
15 epochs with batch 8, effective batch 16, and the original optimizer. It
writes to `artifacts/augmentation_ps/seed42`, leaving the original screening
checkpoint and results untouched. From the repository root:

```powershell
python -m cine --config configs/dtld_m_set_null_aug_ps.yaml train
python -m cine evaluate --checkpoint artifacts/augmentation_ps/seed42/best.pt --split val --output artifacts/augmentation_ps/seed42/evaluation_val
```

The original PS checkpoint remains at
`artifacts/augmentation_null_v1/runs/screening/PS_split42_seed42/best.pt`.
Its archived version-1 policy is accepted by the retained PS code and produces
the same augmentation schedule and transformed images. The original selected
checkpoint can still be evaluated with `python -m cine evaluate --checkpoint
artifacts/augmentation_null_v1/runs/screening/PS_split42_seed42/best.pt
--split val`.

The original validation and official-test comparisons remain under
`artifacts/augmentation_null_v1`. They are historical results, not evidence
that PS generalized to another dataset. The original PS screening result did
not meet its validation safeguards, so no default checkpoint was changed.
