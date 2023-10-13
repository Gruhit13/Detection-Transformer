import torch
from torch import nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from utils.box_ops import generalized_iou, box_cxcywh_xyxy

class HungarianMatcher(nn.Module):
  def __init__(self, cost_class = 1, cost_bbox = 1, cost_giou = 1):
    super(HungarianMatcher, self).__init__()

    self.cost_class = cost_class
    self.cost_bbox = cost_bbox
    self.cost_giou = cost_giou

  def forward(self, preds, targets):
    """
    Preds is a dictonary that contains following keys:
      pred_class: [B, N, num_class]: Predicted class logits
      pred_bbox: [B, N, 4]: Predicted bouding boxes

    Target is a list of dictonary with following keys
      class: [B, N]: True labels
      bbox: [B, N, 4]: True bounding boxes
    """

    bs, num_queries = preds['pred_class'].shape[:2]

    # [B, N, num_class] -> [B*N, num_class]
    pred_class = preds['pred_class'].flatten(0, 1).softmax(-1)
    pred_bbox = preds['pred_bbox'].flatten(0, 1)

    tgt_ids = torch.cat([v['classes'] for v in targets])
    tgt_bbox = torch.cat([v['bboxes'] for v in targets])

    cost_class = -pred_class[:, tgt_ids]

    cost_bbox = torch.cdist(pred_bbox, tgt_bbox, p=1)

    cost_giou = -generalized_iou(box_cxcywh_xyxy(pred_bbox), box_cxcywh_xyxy(tgt_bbox))

    C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

    C = C.view(bs, num_queries, -1).detach().cpu()

    sizes = [len(v['bboxes']) for v in targets]

    indices = []
    for i, c in enumerate(C.split(sizes, -1)):
      indices.append(linear_sum_assignment(c[i]))

    # indices = [linear_sum_assignment(c[i] for i, c in enumerate(C.split(sizes, -1)))]
    return [(torch.as_tensor(i, dtype=torch.int32), torch.as_tensor(j, dtype=torch.int32)) for i, j in indices]

class Criterion(nn.Module):
  def __init__(self, matcher, num_classes, eos_coef):
    super(Criterion, self).__init__()

    self.matcher = matcher
    self.num_classes = num_classes
    self.eos_coef = eos_coef
    empty_weight = torch.ones(self.num_classes + 1, device=device)
    empty_weight[-1] = self.eos_coef
    self.register_buffer('empty_weight', empty_weight)

  def _get_src_permutation_idx(self, indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

  def _get_tgt_pernuration_idx(self, indices):
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

  def get_label_loss(self, outputs, targets, indices):
    pred_class = outputs['pred_class']
    idx = self._get_src_permutation_idx(indices)

    targets_class_o = torch.cat([t['classes'][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(pred_class.shape[:2], self.num_classes, dtype=torch.int64, device=pred_class.device)

    target_classes[idx] = targets_class_o

    loss_ce = F.cross_entropy(pred_class.transpose(1, 2), target_classes, self.empty_weight)

    return loss_ce

  def get_bbox_loss(self, outputs, targets, indices, num_bboxes):

    idx = self._get_src_permutation_idx(indices)
    pred_bbox = pred_bbox = outputs['pred_bbox'][idx]
    tgt_bbox = torch.cat([t['bboxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    loss_bbox = F.l1_loss(pred_bbox, tgt_bbox, reduction='none').sum() / num_bboxes
    giou_bbox = 1 - torch.diag(generalized_iou(
        box_cxcywh_xyxy(pred_bbox),
        box_cxcywh_xyxy(tgt_bbox)
    ))

    giou_loss = giou_bbox.sum() / num_bboxes
    return loss_bbox, giou_loss

  def forward(self, outputs, targets):
    """
    Arguments:
    Outputs is a dictonary containing:
      pred_class: [B, num_queries, num_classes] prediction for class
      pred_bbox: [B, num_queries, 4] prediction for bbox

    targets is a list of dictonary with elements having following keys:
      classes: [B, N] true labels
      bboxes: [B, N, 4] True bounding boxes
    """

    indices = self.matcher(outputs, targets)
    num_bboxes = sum(len(v['bboxes']) for v in targets)
    num_bboxes = torch.as_tensor([num_bboxes], dtype=torch.float, device=outputs['pred_class'].device)

    losses = {}
    l1_bbox_loss, giou_bbox = self.get_bbox_loss(outputs, targets, indices, num_bboxes)

    losses['label_loss'] = self.get_label_loss(outputs, targets, indices)
    losses['l1_loss'] = l1_bbox_loss
    losses['giou_loss'] = giou_bbox

    return losses