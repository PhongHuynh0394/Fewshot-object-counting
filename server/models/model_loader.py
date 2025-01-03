from .util.load_checkpoint import load_lightning_checkpoint, load_base
from .CounTR import CountingModel

def load_model(model_path: str = "checkpoint/model.cpkt", lora=False, base=True):
    """
    Load model with checkpoint path
    """
    if lora:
        pass

    
    if base:
        model = CountingModel(pretrained_weight=model_path) 
        return model

    load_lightning_checkpoint(model, checkpoint_path=model_path)
    return model
