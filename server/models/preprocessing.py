from PIL import Image
from .util.transform import TransformInference

def preprocessing(image: Image, boxes: list, **kwargs) -> dict:
    sample = {
        "image": image,
        "lines_box": boxes,
    }

    if kwargs:
        sample.update(kwargs)
    
    num_shot = len(boxes) if len(boxes) <= 3 else 3
    
    transform = TransformInference()
    output = transform(sample, num_shot=num_shot)

    return output, num_shot