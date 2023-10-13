import torch
from torchvision.ops.boxes import box_area
# Get Intersection over union for given bounding boxes
def box_iou(boxes1, boxes2):
  area1 = box_area(boxes1)
  area2 = box_area(boxes2)

  lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
  rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

  wh = (rb - lt).clamp(min=0)
  inter = wh[:, :, 0] * wh[:, :, 1]

  union = area1[:, None] + area2 - inter

  iou = inter/union
  return iou, union

# A Generalized IoU function
def generalized_iou(boxes1, boxes2):
  assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
  assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

  iou, union = box_iou(boxes1, boxes2)

  lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
  rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

  wh = (rb - lt).clamp(min=0)
  area = wh[:, :, 0] * wh[:, :, 1]

  return iou - (area - union) / area

# Converting Boxes coordinate from center system to xy
def box_cxcywh_xyxy(boxes):
  x_c, y_c, w, h = boxes.unbind(-1)

  boxes = [
      (x_c - 0.5*w), (y_c - 0.5*h),
      (x_c + 0.5*w), (y_c + 0.5*h)
  ]

  return torch.stack(boxes, dim=-1)

# Converting bounding boxes from xy co-ordinate to center xy
def box_xyxy_cxcyhw(boxes):
  x1, y1, x2, y2 = boxes.unbind(-1)

  boxes = [
      (x1 + x2)/2, (y1+y2)/2,
      (x2 - x1), (y2 - y1)
  ]

  return torch.stack(boxes, dim=-1)