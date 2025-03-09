import models.models_mae_cross as models_mae_cross
import models.util.misc as misc
from .util.misc import NativeScalerWithGradNormCount as NativeScaler
from .util import load_checkpoint

from pytorch_lightning import LightningModule
import torch
import torchvision
from torchmetrics import MeanSquaredError, MeanAbsoluteError
import timm.optim.optim_factory as optim_factory
import random
import numpy as np
import wandb

class CountingModel(LightningModule):

    def __init__(self,
                 model="mae_vit_base_patch16",
                 pretrained_weight=None,
                 blr=1e-3,
                 lr=None,
                 min_lr=0.0,
                 weight_decay=0.05,
                 norm_pix_loss=False,
                 mask_ratio=0.5,
                 batch_size=26,
                 accum_iter=1,
                do_resume=False,):

        super().__init__()

        self.model = models_mae_cross.__dict__[model](norm_pix_loss=norm_pix_loss)
        self.loss_scaler = NativeScaler()

        # Optimization parameters
        self.blr = blr
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.accum_iter = accum_iter
        self.mask_ratio = mask_ratio

        # Load pretrained weights if provided

        if pretrained_weight:
            load_checkpoint.load_base(self, pretrained_weight, do_resume=do_resume)

        self.save_hyperparameters()

        self.mae_metric = MeanAbsoluteError()
        self.rmse_metric = MeanSquaredError(squared=False)


    def forward(self, samples, boxes, shot_num):
        samples = samples.to(torch.float32) # astype from float64 to float32 (if needed)
        return self.model(samples, boxes, shot_num)


    def training_step(self, batch, batch_idx):
        samples, boxes, gt_density, m_flag = batch['image'], batch['boxes'], batch['density_gt'], batch['m_flag']
        device = next(self.parameters()).device

        # If there is at least one image in the batch using Type 2 Mosaic, 0-shot is banned.
        flag = sum(m_flag)
        shot_num = random.randint(1, 3) if flag > 0 else random.randint(0, 3)

        # Forward pass
        output = self(samples, boxes, shot_num)

        # Create mask
        mask = np.random.binomial(n=1, p=0.8, size=(384, 384))
        masks = torch.tensor(mask).to(device).expand_as(output)

        # Compute loss and metrics
        loss, mae, rmse, _ = self.compute_loss_and_metrics(output, gt_density, masks)

        # Log metrics
        self.log_dict({"train/Loss": loss, "train/MAE": mae, "train/RMSE": rmse}, prog_bar=True, sync_dist=True)

        # Only log the media in last batch
        train_dataloader = self.trainer.train_dataloader  # Access the first validation dataloader
        total_batches = len(train_dataloader)
        if batch_idx == total_batches - 1:
        # if batch_idx == len(self.trainer.datamodule.train_dataloader()) - 1:
            # Log images for the last batch
            self.log_image_wandb(output, batch, mode="train")
        # self.log_image_wandb(output, batch, mode="train")

        return loss



    def validation_step(self, batch, batch_idx):
        samples, boxes, gt_density, _ = batch['image'], batch['boxes'], batch['density_gt'], batch['m_flag']
        device = next(self.parameters()).device

        # Forward pass
        shot_num = random.randint(0, 3)
        output = self(samples, boxes, shot_num)

        # Create mask
        mask = np.ones((384, 384))  # No masking during validation

        masks = torch.tensor(mask).to(device).expand_as(output)

        # Compute metrics
        _, mae, rmse, nae = self.compute_loss_and_metrics(output, gt_density, masks)

        # Log metrics
        self.log_dict({"val/MAE": mae, "val/RMSE": rmse, "val/NAE": nae}, prog_bar=True, sync_dist=True)

        # log wandb image
        # Only log the media in last batch
        val_dataloader = self.trainer.val_dataloaders  # Access the first validation dataloader
        total_batches = len(val_dataloader)
        if batch_idx == total_batches - 1:
        # if batch_idx == len(self.trainer.datamodule.val_dataloaders()) - 1:
            # Log images for the last batch
            self.log_image_wandb(output, batch, mode="val")
        # self.log_image_wandb(output, batch, mode="val")


    def test_step(self, batch, batch_idx):
        samples, boxes, gt_density, _ = batch['image'], batch['boxes'], batch['density_gt'], batch['m_flag']
        device = next(self.parameters()).device

        # Forward pass
        shot_num = boxes.shape[1] # the num shot
        output = self(samples, boxes, shot_num)

        # Create mask
        mask = np.ones((384, 384))  # No masking during validation
        masks = torch.tensor(mask).to(device).expand_as(output)

        # Compute metrics
        _, mae, rmse, nae = self.compute_loss_and_metrics(output, gt_density, masks)

        # Log metrics
        self.log_dict({"test/MAE": mae, "test/RMSE": rmse, "test/NAE": nae}, prog_bar=True, sync_dist=True)
        self.log("shot_num", shot_num, prog_bar=True, sync_dist=True)

        # Log image to wandb (if logger is wandb)
        self.log_image_wandb(output, batch, mode="test")


    def compute_loss_and_metrics(self, output, gt_density, masks):
        """Compute loss, MAE, RMSE, and NAE."""
        # Compute loss
        loss = ((output - gt_density) ** 2 * masks / (384 * 384)).sum() / output.shape[0]

        # Metrics: MAE, RMSE, NAE
        pred_cnt = output.view(output.size(0), -1).sum(1) / 60
        gt_cnt = gt_density.view(gt_density.size(0), -1).sum(1) / 60
        mae = self.mae_metric(pred_cnt, gt_cnt)
        rmse = self.rmse_metric(pred_cnt, gt_cnt)
        cnt_err = torch.abs(pred_cnt - gt_cnt)
        nae = (cnt_err / (gt_cnt + 1e-6)).mean()  # Avoid division by zero

        return loss, mae, rmse, nae

    def log_image_wandb(self, output, batch, mode="train"):
        """
        Log visualizations for both training and validation phases.

        Args:
            output: The model's predicted output (for the current batch).
            batch: batch samples
            mode: The mode - "train" or "val" or "test".
        """
        samples, boxes, gt_density, pos, im_names, cnt_gt = batch['image'], batch['boxes'], batch['density_gt'], batch['pos'], batch['image_name'], batch['count']
        if not hasattr(self.logger, "experiment") or not isinstance(self.logger.experiment, wandb.wandb_sdk.wandb_run.Run):
          return

        # Lists to collect visualizations for WandB
        wandb_densities = []
        wandb_bboxes = []
        black = torch.zeros([384, 384], device=output.device)  # Black canvas for overlays

        # Iterate through each sample in the batch
        for i in range(output.shape[0]):
            # Log density maps (ground truth and predicted)
            w_d_map = torch.stack([output[i], black, black])
            gt_map = torch.stack([gt_density[i], black, black])
            box_map = misc.get_box_map(samples[i], pos[i], output.device)
            w_gt_density = samples[i] / 2 + gt_map + box_map
            w_d_map_overlay = samples[i] / 2 + w_d_map
            w_densities = torch.cat([w_gt_density, w_d_map, w_d_map_overlay], dim=2)
            w_densities = torch.clamp(w_densities, 0, 1)  # Ensure values are between 0 and 1

            # Prepare the WandB image and add it to the list
            wandb_densities.append(wandb.Image(torchvision.transforms.ToPILImage()(w_densities),
                                               caption=f"[E#{self.current_epoch}] {im_names[i]} cnt: {cnt_gt[i]} (dens_gt: {torch.sum(gt_density[i] / 60).item():.3f}, pred: {torch.sum(output[i] / 60).item():.3f})"))

            # Log bounding boxes
            w_boxes = torch.cat([boxes[i][x, :, :, :] for x in range(boxes[i].shape[0])], 2)
            wandb_bboxes.append(wandb.Image(torchvision.transforms.ToPILImage()(w_boxes),
                                            caption=f"[E#{self.current_epoch}] {im_names[i]}"))

        # Log the visualizations to WandB (whether train or val phase)
        if mode == "train":
            wandb.log({"train_densities": wandb_densities, "train_bboxes": wandb_bboxes})
        elif mode == "val":
            wandb.log({"val_densities": wandb_densities, "val_bboxes": wandb_bboxes})
        else:
            wandb.log({"test_densities": wandb_densities, "test_bboxes": wandb_bboxes})


    def configure_optimizers(self):

        # Effective batch size
        eff_batch_size = self.batch_size * self.accum_iter
        lr = self.blr * eff_batch_size / 256 if self.lr is None else self.lr

        # Optimizer
        param_groups = optim_factory.add_weight_decay(self.model, self.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        return optimizer

if __name__ == "__main__":
    model = CountingModel()
