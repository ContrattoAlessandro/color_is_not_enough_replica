"""Dense YOLO local attributes and detached detection-prior global fusion."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from .data import WIDTH, CONTENT_HEIGHT, HEIGHT, PAD


def select_detections(boxes, scores, conf, iou=0.7, maximum=300):
    """Return original dense candidate indices, never a reindexed attribute array."""
    boxes = boxes.float().clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, WIDTH)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(PAD, PAD + CONTENT_HEIGHT)
    valid = (scores >= conf) & torch.isfinite(scores) & torch.isfinite(boxes).all(-1)
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    ids = valid.nonzero(as_tuple=True)[0]
    selected = nms(boxes[ids], scores[ids].float(), iou)[:maximum]
    return ids[selected], boxes[ids[selected]]


@torch.no_grad()
def rasterize(boxes, scores, relevance, states, features, conf=0.05, iou=0.7, maximum=300):
    """Exact max-over-rectangles masks; chunking bounds temporary GPU memory."""
    masks = [f.new_zeros((len(boxes), 5, *f.shape[-2:])) for f in features]
    for b in range(len(boxes)):
        ids, selected = select_detections(boxes[b], scores[b], conf, iou, maximum)
        values = torch.cat([scores[b, ids, None], scores[b, ids, None] * relevance[b, ids, None], scores[b, ids, None] * states[b, ids]], 1).float()
        for mask in masks:
            h, w = mask.shape[-2:]
            lo = torch.floor(selected[:, :2] * selected.new_tensor([w / WIDTH, h / HEIGHT]))
            hi = torch.ceil(selected[:, 2:] * selected.new_tensor([w / WIDTH, h / HEIGHT]))
            yy = torch.arange(h, device=mask.device)[None, :, None]
            xx = torch.arange(w, device=mask.device)[None, None, :]
            for start in range(0, len(ids), 32):
                l, r = lo[start:start + 32], hi[start:start + 32]
                inside = (xx >= l[:, 0, None, None]) & (xx < r[:, 0, None, None]) & (yy >= l[:, 1, None, None]) & (yy < r[:, 1, None, None])
                filled = (inside[:, None] * values[start:start + 32, :, None, None]).amax(0)
                mask[b] = torch.maximum(mask[b], filled.to(mask.dtype))
    return masks


@torch.no_grad()
def box_evidence(boxes, scores, relevance, states, features, topk=64, conf=0.05):
    """Detached top-K dense candidates, without NMS or rasterization.

    Clip coordinates to image content, normalize as in local heads, and encode
    level as the normalized pyramid index. Padding is explicitly masked.
    """
    boxes = boxes.detach().float().clone()
    finite_boxes = torch.isfinite(boxes).all(-1)
    scores, relevance, states = (x.detach().float() for x in (scores, relevance, states))
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, WIDTH)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(PAD, PAD + CONTENT_HEIGHT)
    coords = boxes / boxes.new_tensor([WIDTH, CONTENT_HEIGHT, WIDTH, CONTENT_HEIGHT])
    coords[..., [1, 3]] -= PAD / CONTENT_HEIGHT
    size = coords[..., 2:] - coords[..., :2]
    log_area = size.prod(-1).clamp_min(1e-12).log().unsqueeze(-1)
    levels = torch.cat([scores.new_full((f.shape[-2] * f.shape[-1],), i / max(1, len(features) - 1))
                        for i, f in enumerate(features)])
    values = torch.cat([coords, log_area, levels[None, :, None].expand(len(boxes), -1, -1),
                        scores[..., None], (scores * relevance)[..., None], scores[..., None] * states], -1)
    valid = finite_boxes & (scores >= conf) & (size > 0).all(-1) & torch.isfinite(values).all(-1)
    evidence = values.new_zeros((len(boxes), topk, 11))
    mask = torch.zeros((len(boxes), topk), device=boxes.device, dtype=torch.bool)
    for b in range(len(boxes)):
        ids = valid[b].nonzero(as_tuple=True)[0]
        k = min(topk, len(ids))
        if not k:
            continue
        chosen = ids[scores[b, ids].topk(k).indices]
        # Cutoff ties use feature values, never input order: invariance also
        # holds when more than K boxes have the same score.
        cutoff = scores[b, chosen].min()
        tied = ids[scores[b, ids] == cutoff]
        higher = chosen[scores[b, chosen] > cutoff]
        if len(tied) > k - len(higher):
            for col in reversed(range(values.shape[-1])):
                tied = tied[values[b, tied, col].argsort(stable=True)]
            chosen = torch.cat([higher, tied[:k - len(higher)]])
        evidence[b, :k] = values[b, chosen]
        mask[b, :k] = True
    return evidence, mask


class SetEvidencePool(nn.Module):
    """Shared 11->128->128 MLP; attention over boxes plus learned NULL."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(11, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.null_token = nn.Parameter(torch.zeros(1, 1, 128))
        nn.init.normal_(self.null_token, std=0.02)
        self.attention = nn.Linear(128, 1)

    def forward(self, evidence, valid):
        tokens = self.mlp(evidence.detach())
        tokens = torch.cat([tokens, self.null_token.to(tokens.dtype).expand(len(tokens), -1, -1)], 1)
        valid = torch.cat([valid, valid.new_ones((len(valid), 1))], 1)
        logits = self.attention(tokens).squeeze(-1).float().masked_fill(~valid, -torch.inf)
        weights = logits.softmax(-1)
        pooled = (tokens.float() * weights[..., None]).sum(1).to(tokens.dtype)
        return pooled, weights


class JointModel(nn.Module):
    def __init__(self, config, pretrained=True):
        super().__init__()
        self.config = config
        architecture = config['model'].get('architecture', 'yolov8n.yaml')
        if architecture not in ('yolov8n.yaml', 'yolov8s.yaml', 'yolov8m.yaml', 'yolov8s-p2.yaml'):
            raise ValueError('Unsupported YOLO architecture')
        self.detector = DetectionModel(architecture, nc=1, verbose=False)
        self.detector.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self.transfer_report = {}
        if pretrained:
            source = YOLO(config["model"]["weights"]).model.float()
            if architecture == 'yolov8s-p2.yaml':
                from .transfer import transfer_p2
                self.transfer_report = transfer_p2(source, self.detector)
            else:
                original, target = source.state_dict(), self.detector.state_dict()
                # Older COCO files have width/depth multipliers, not a 'scale' key.
                # Validate actual backbone/neck tensor shapes, allowing the one-class head to differ.
                head_prefix = f'model.{len(self.detector.model) - 1}.'
                if any(k not in original or original[k].shape != v.shape
                       for k, v in target.items() if not k.startswith(head_prefix)):
                    raise ValueError('Pretrained backbone/neck must match the configured YOLO scale')
                compatible = {k: v for k, v in original.items() if k in target and v.shape == target[k].shape}
                self.detector.load_state_dict(compatible, strict=False)
                copied_outputs = 0
                for old, new in zip(source.model[-1].cv3, self.detector.model[-1].cv3):
                    if old[-1].weight.shape[1:] == new[-1].weight.shape[1:]:
                        with torch.no_grad():
                            new[-1].weight.copy_(old[-1].weight[9:10])
                            new[-1].bias.copy_(old[-1].bias[9:10])
                        copied_outputs += 1
                self.transfer_report = {"compatible_tensors": len(compatible), "total_tensors": len(target), "traffic_light_output_layers_copied": copied_outputs, "note": "Standard nc=1 head widths may differ from COCO; incompatible classification tensors are freshly initialized."}
        channels = [branch[0].conv.in_channels for branch in self.detector.model[-1].cv2]
        self.feature_channels = channels
        if config['model'].get('joint_priors', False):
            raise ValueError('This repository retains the selected five-prior model only')
        # Nonpersistent keeps strict loading of original nano checkpoints compatible.
        self.register_buffer('global_class_weights', torch.ones(3), persistent=False)
        self.global_class_counts = None
        c = config["model"]["channels"]
        self.attr_project = nn.ModuleList([nn.Conv2d(ch, c, 1) for ch in channels])
        self.relevance = nn.ModuleList([nn.Linear(c + 4, 1) for _ in channels])
        self.state = nn.ModuleList([nn.Linear(c + 4, 3) for _ in channels])
        self.global_project = nn.ModuleList([nn.Conv2d(ch, c, 1) for ch in channels])
        self.global_head = config['model'].get('global_head', 'raster')
        if self.global_head == 'raster':
            self.global_conv = nn.ModuleList([nn.Sequential(nn.Conv2d(c + 5, c, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)) for _ in channels])
            self.global_classifier = nn.Linear(len(channels) * c, 3)
        elif self.global_head == 'set_null':
            self.global_topk = config['model'].get('set_topk', 64)
            if not isinstance(self.global_topk, int) or isinstance(self.global_topk, bool) or self.global_topk < 1:
                raise ValueError('set_topk must be a positive integer')
            self.global_set_pool = SetEvidencePool()
            self.global_classifier = nn.Linear(len(channels) * c + 128, 3)
        else:
            raise ValueError('Unknown global_head; expected raster or set_null')
        self.criterion = None

    def configure_global_weights(self, rows):
        labels = torch.tensor([r['global_label'] for r in rows], dtype=torch.long)
        counts = torch.bincount(labels[labels >= 0], minlength=3).float()
        self.global_class_counts = counts.long().tolist()
        mode = self.config['train'].get('global_weighting', 'none')
        if mode not in ('none', 'inverse_sqrt'):
            raise ValueError('Unknown global_weighting')
        weights = torch.ones(3)
        if mode == 'inverse_sqrt':
            if (counts == 0).any():
                raise ValueError('Weighted training requires all three global classes')
            frequencies = counts / counts.sum()
            weights = frequencies.rsqrt()
            weights /= (frequencies * weights).sum()
        self.global_class_weights.copy_(weights)

    def _raw(self, x):
        saved = []
        for m in self.detector.model[:-1]:
            if m.f != -1:
                x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
            x = m(x)
            saved.append(x if m.i in self.detector.save else None)
        head = self.detector.model[-1]
        features = [saved[i] for i in head.f]
        return head.forward_head(features, **head.one2many)

    def forward(self, images):
        if self.criterion is None:
            self.criterion = v8DetectionLoss(self.detector)
        raw = self._raw(images)
        features = raw["feats"]
        points, strides = make_anchors(features, self.detector.model[-1].stride, 0.5)
        boxes = self.criterion.bbox_decode(points.float(), raw["boxes"].permute(0, 2, 1).float()) * strides
        coords = boxes.clone()
        coords[..., [0, 2]] /= WIDTH
        coords[..., [1, 3]] = (coords[..., [1, 3]] - PAD) / CONTENT_HEIGHT
        rel, state, offset = [], [], 0
        for f, proj, rh, sh in zip(features, self.attr_project, self.relevance, self.state):
            v = proj(f).flatten(2).transpose(1, 2)
            z = torch.cat([v, coords[:, offset:offset + v.shape[1]].to(v.dtype)], -1)
            rel.append(rh(z).squeeze(-1))
            state.append(sh(z))
            offset += v.shape[1]
        rel, state = torch.cat(rel, 1), torch.cat(state, 1)
        scores = raw["scores"].squeeze(1).sigmoid()
        mc = self.config["model"]
        extra = {}
        if self.global_head == 'raster':
            masks = rasterize(boxes.detach(), scores.detach(), rel.detach().sigmoid(), state.detach().softmax(-1), features, mc["prior_conf"], mc["nms_iou"], mc["max_det"])
            pooled = [conv(torch.cat([proj(f), mask], 1)).flatten(1) for f, mask, proj, conv in zip(features, masks, self.global_project, self.global_conv)]
        else:
            evidence, valid = box_evidence(boxes, scores, rel.detach().sigmoid(), state.detach().softmax(-1),
                                           features, self.global_topk, mc['prior_conf'])
            tokens, attention = self.global_set_pool(evidence, valid)
            pooled = [F.adaptive_avg_pool2d(proj(f), 1).flatten(1) for f, proj in zip(features, self.global_project)]
            pooled.append(tokens)
            masks = []
            extra = dict(box_evidence=evidence, evidence_valid=valid, evidence_attention=attention)
        return dict(raw=raw, boxes=boxes, scores=scores, relevance_logits=rel, state_logits=state, global_logits=self.global_classifier(torch.cat(pooled, 1)), priors=masks, **extra)

    def loss(self, prediction, batch):
        assigned, det, _ = self.criterion.get_assigned_targets_and_loss(prediction["raw"], batch)
        foreground, target_indices = assigned[:2]
        prediction['_foreground_count'] = foreground.sum().detach()
        batch_size = foreground.shape[0]
        counts = torch.bincount(batch["batch_idx"].long(), minlength=batch_size)
        offsets = counts.cumsum(0) - counts
        if foreground.any():
            ids = (target_indices + offsets[:, None])[foreground]
            rel = F.binary_cross_entropy_with_logits(prediction["relevance_logits"][foreground].float(), batch["relevance"][ids])
            state = F.cross_entropy(prediction["state_logits"][foreground].float(), batch["states"][ids])
        else:
            rel = prediction["relevance_logits"].sum() * 0
            state = prediction["state_logits"].sum() * 0
        valid = batch["global_label"] >= 0
        if valid.any():
            per_image = F.cross_entropy(prediction['global_logits'][valid].float(), batch['global_label'][valid], reduction='none')
            weights = self.global_class_weights.to(per_image.device)[batch['global_label'][valid]]
            glob = (per_image * weights).mean()
        else:
            glob = prediction['global_logits'].sum() * 0
        losses = dict(box=det[0], cls=det[1], dfl=det[2], relevance=rel, state=state, global_loss=glob)
        weights = self.config['train'].get('task_weights', {})
        phase = getattr(self, 'training_phase', 'joint')
        total = sum(value * weights.get('detection' if key in ('box', 'cls', 'dfl') else key, 1.0)
                    * (0.0 if (phase == 'local' and key == 'global_loss') or
                       (phase == 'global' and key != 'global_loss') else 1.0)
                    for key, value in losses.items())
        return total, {k: v.detach() for k, v in losses.items()}

    @torch.no_grad()
    def detections(self, prediction, conf=0.001):
        result = []
        mc = self.config["model"]
        for i in range(len(prediction["boxes"])):
            ids, boxes = select_detections(prediction["boxes"][i], prediction["scores"][i], conf, mc["nms_iou"], mc["max_det"])
            result.append(dict(boxes=boxes, scores=prediction["scores"][i, ids], relevance=prediction["relevance_logits"][i, ids].sigmoid(), states=prediction["state_logits"][i, ids].softmax(-1), candidate_indices=ids, global_probabilities=prediction["global_logits"][i].softmax(-1)))
        return result
