import torch
from PIL import Image
import numpy as np

def model_predict(model, sample, num_shot):
    boxes = sample['boxes'].unsqueeze(0) 
    with torch.no_grad():
        y_density = model(sample['image'].unsqueeze(0), boxes, num_shot)
        count_predict = torch.sum(y_density / 60).item()

    # Normalize density map: [min, max] -> [0, 1] -> [0, 255]
    y_density = y_density.squeeze(0).cpu().numpy()  # Get numpy array
    y_density = (y_density - y_density.min()) / (y_density.max() - y_density.min())  # Normalize to [0, 1]
    y_density = (y_density * 255).astype(np.uint8)  # Convert to [0, 255]

    # Convert to image
    y_density_image = Image.fromarray(y_density)

    return y_density_image, count_predict
