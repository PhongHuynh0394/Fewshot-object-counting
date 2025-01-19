from torchvision import transforms
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
import numpy as np
import torch

TTensor = transforms.Compose([transforms.ToTensor()])

class TransformVal:

  def __init__(self, MAX_HW=384):
    self.max_hw = MAX_HW

  def __call__(self, sample, num_shot=3):
    image, dots, lines_boxes, m_flag, im_id = sample['image'], sample['dots'], sample['lines_box'], sample['m_flag'], sample['image_name']
    W, H = image.size

    new_H = new_W = self.max_hw
    scale_factor_h = float(new_H) / H
    scale_factor_w = float(new_W) / W
    resized_image = transforms.Resize((new_H, new_W))(image)
    resized_image = TTensor(resized_image)

    # Resize density map
    resized_density = np.zeros((new_H, new_W), dtype='float32')

    for i in range(dots.shape[0]):
        resized_density[min(new_H - 1, int(dots[i][1] * scale_factor_h))] \
                        [min(new_W - 1, int(dots[i][0] * scale_factor_w))] = 1

    # resized_density = gaussian_filter(resized_density, sigma=4, radius=7, order=0)
    resized_density = gaussian_filter(resized_density, sigma=4, radius=7, order=0)
    resized_density = torch.from_numpy(resized_density) * 60

    # Crop bboxes and resize as 64x64
    boxes = list()
    rects = list()
    cnt = 0

    for box in lines_boxes:
        cnt += 1
        if cnt > 3:
            break

        box2 = [int(k) for k in box]
        y1 = int(box2[0] * scale_factor_h)
        x1 = int(box2[1] * scale_factor_w)
        y2 = int(box2[2] * scale_factor_h)
        x2 = int(box2[3] * scale_factor_w)
        rects.append(torch.tensor([y1, x1, y2, x2]))
        bbox = resized_image[:, y1:y2 + 1, x1:x2 + 1]
        bbox = transforms.Resize((64, 64))(bbox)
        boxes.append(bbox)

    boxes = torch.stack(boxes) # Exampler image
    pos = torch.stack(rects) # bounding box rectagle

    # boxes shape [3,3,64,64], image shape [3,384,384], density shape[384,384]
    sample = {
        "image_name": im_id,
        "image": resized_image,
        "boxes": boxes,
        "density_gt": resized_density,
        "pos": pos,
        "m_flag": m_flag
    }

    assert sample['image'].shape[1:] == (384, 384), f"Image not resized correctly, got {sample['image'].shape}"
    assert sample['density_gt'].shape == (384, 384), f"Density map not resized correctly, got {sample['density_gt'].shape}"
    assert sample['boxes'].size(0) == num_shot and sample['boxes'].size(1) == 3 and sample['boxes'].size(2) == 64 and sample['boxes'].size(3) == 64, f"Boxes have inconsistent shape: {sample['boxes'].shape}"

    return sample

class TransformInference:

  def __init__(self, MAX_HW=384):
    self.max_hw = MAX_HW

  def __call__(self, sample, num_shot=3):
    image, lines_boxes = sample['image'], sample['lines_box']
    W, H = image.size

    new_H = new_W = self.max_hw
    scale_factor_h = float(new_H) / H
    scale_factor_w = float(new_W) / W
    resized_image = transforms.Resize((new_H, new_W))(image)
    resized_image = TTensor(resized_image)

    # Crop bboxes and resize as 64x64
    boxes = list()
    rects = list()
    cnt = 0

    for box in lines_boxes:
        cnt += 1
        if cnt > 3:
            break

        box2 = [int(k) for k in box]
        y1 = int(box2[0] * scale_factor_h)
        x1 = int(box2[1] * scale_factor_w)
        y2 = int(box2[2] * scale_factor_h)
        x2 = int(box2[3] * scale_factor_w)
        rects.append(torch.tensor([y1, x1, y2, x2]))
        bbox = resized_image[:, y1:y2 + 1, x1:x2 + 1]
        bbox = transforms.Resize((64, 64))(bbox)
        boxes.append(bbox)


    # boxes shape [num_shot,3,64,64], image shape [3,384,384], 
    sample = {
       "image": resized_image
    }
    if num_shot == 0:
       sample.update({
          'boxes': torch.Tensor([])
       })
    else:
      boxes = torch.stack(boxes) # Exampler image
      pos = torch.stack(rects) # bounding box rectagle
      sample.update({
          "image": resized_image,
          "boxes": boxes,
          "pos": pos,
      })
      assert sample['boxes'].size(0) == num_shot and sample['boxes'].size(1) == 3 and sample['boxes'].size(2) == 64 and sample['boxes'].size(3) == 64, f"Boxes have inconsistent shape: {sample['boxes'].shape}"

    assert sample['image'].shape[1:] == (384, 384), f"Image not resized correctly, got {sample['image'].shape}"

    return sample