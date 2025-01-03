from PIL import Image
from .util.transform import TransformInference

def preprocessing(image: Image, boxes: list, **kwargs) -> dict:
    sample = {
        "image": image,
        "lines_box": boxes,
    }

    if kwargs:
        sample.update(kwargs)
    
    transform = TransformInference()
    output = transform(sample, num_shot=len(sample['lines_box']))

    return output