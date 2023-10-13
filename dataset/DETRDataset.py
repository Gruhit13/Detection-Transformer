from PIL import Image

import torch
from torch.utils.data import Dataset

class DETRDataset(Dataset):
  def __init__(self, dataset, class_to_label, n_queries, transform=None):
    super(Dataset, self).__init__()
    self.filename = dataset['filename']
    self.classes = dataset['class']
    self.bboxes = dataset['bbox']

    self.class_to_label = class_to_label
    self.n_queries = n_queries
    self.no_object = len(self.class_to_label)

    self.transform = transform

  def __len__(self):
    return len(self.filename)

  def __getitem__(self, index):
    fname, classes, bboxes = self.filename[index], self.classes[index], self.bboxes[index]

    img, old_h, old_w = self.load_image(fname, bboxes)
    classes = self.pad_classes(classes)

    new_h, new_w = img.shape[1:]

    # Passing this mask to key_padding_mask so places with True will be ignored for attention calculation
    # And value with False will be used for attention calculation. Hence creating a False padded mask
    mask = torch.zeros((new_h, new_w), dtype=torch.bool)

    bboxes = self.process_bbox(bboxes, (new_h, new_w), (old_h, old_w))

    target = {}
    target['classes'] = classes.to(device)
    target['bboxes'] = bboxes.to(device)

    source = {
        'img': img,
        'mask': mask
    }

    return source, target

  def load_image(self, img_path, bboxes):
    img = Image.open(img_path)
    img_h, img_w = img.size

    if self.transform is not None:
      img = self.transform(img)

    return img, img_h, img_w

  def pad_classes(self, classes):
    return torch.as_tensor(classes[:self.n_queries], dtype=torch.int64, device=device)

  def process_bbox(self, bboxes, new_size, old_size):
    bboxes = torch.as_tensor(bboxes[:self.n_queries], dtype=torch.float32, device=device)

    bboxes = box_xyxy_cxcyhw(bboxes)
    img_height_old, img_width_old = old_size
    old_arr = torch.tensor([img_width_old, img_height_old, img_width_old, img_height_old], dtype=bboxes.dtype, device=device)

    img_height_new, img_width_new = new_size
    new_arr = torch.tensor([img_width_new, img_height_new, img_width_new, img_height_new], dtype=bboxes.dtype, device=device)

    # Resize the box co-ordinates as well
    bboxes = (bboxes * new_arr) / old_arr

    # Normalize the bbox w.r.t image resoluation
    bboxes = bboxes / new_arr

    return bboxes