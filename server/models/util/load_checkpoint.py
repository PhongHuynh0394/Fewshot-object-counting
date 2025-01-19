import torch

def load_base(countmodel, checkpoint_path, do_resume=False):
  checkpoint = torch.load(checkpoint_path, map_location='cpu')

  # Handle positional embedding mismatch
  if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != \
    countmodel.model.state_dict()['pos_embed'].shape:
      print(f"Removing key pos_embed from pretrained checkpoint")
      del checkpoint['model']['pos_embed']

  # Load model weights
  countmodel.model.load_state_dict(checkpoint['model'], strict=False)
  print("Resume checkpoint %s" % checkpoint_path)

  # Optionally load optimizer state and scaler for resuming training
  if 'optimizer' in checkpoint and 'epoch' in checkpoint and do_resume:
    optimizer_state = checkpoint['optimizer']
    scaler_state = checkpoint.get('scaler', None)

    # Update the optimizer state in Lightning (manually inject state)
    optimizer = countmodel.configure_optimizers()
    if isinstance(optimizer, dict):
        optimizer = optimizer['optimizer']
    optimizer.load_state_dict(optimizer_state)

    if scaler_state and countmodel.loss_scaler is not None:
        countmodel.loss_scaler.load_state_dict(scaler_state)

    # Update starting epoch for training
    start_epoch = checkpoint['epoch'] + 1
    print("With optim & scheduler!")
    return start_epoch



def load_lightning_checkpoint(countmodel, checkpoint_path, do_resume=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        countmodel.model.load_state_dict(state_dict, strict=False)
    else:
        raise KeyError("Checkpoint does not contain 'state_dict' key")

    print("Resume checkpoint %s" % checkpoint_path)

    # Optionally load optimizer state and scaler for resuming training
    if 'optimizer_states' in checkpoint and 'epoch' in checkpoint and do_resume:
        optimizer_state = checkpoint['optimizer_states'][0]  # Assuming optimizer states are in a list
        scaler_state = checkpoint.get('MixedPrecision', None)

        # Update the optimizer state in Lightning (manually inject state)
        optimizer = countmodel.configure_optimizers()
        if isinstance(optimizer, dict):
            optimizer = optimizer['optimizer']
        optimizer.load_state_dict(optimizer_state)

        if scaler_state and countmodel.loss_scaler is not None:
            countmodel.loss_scaler.load_state_dict(scaler_state)

        # Update starting epoch for training
        start_epoch = checkpoint['epoch'] + 1
        print("With optim & scheduler!")
        return start_epoch
