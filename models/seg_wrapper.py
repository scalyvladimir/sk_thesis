from torchmetrics.classification import MulticlassJaccardIndex

import pytorch_lightning as pl
from .unet import UNet2D

import torch
import torch.nn.functional as F

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

import surface_distance.metrics as surf_dc

from torchvision.utils import make_grid 

import numpy as np

from .utils import pad_tensor, unpad_tensor

# from dpipe.torch.functional import weighted_cross_entropy_with_logits, dice_loss

def dice_loss(pred_mask, mask, eps=1e-6):
    """ Squared for better convergence, 
        link: https://arxiv.org/abs/1606.04797
    """
        
    num = 2 * torch.sum(mask * pred_mask, dim=(1))
    den = torch.sum(torch.square(mask) + torch.square(pred_mask))
    res = torch.mean(num / (den + eps))
                    
    return 1. - res

# def dice_loss(pred, target, eps=1e-6):

#     target = target.float()

#     inter = torch.dot(pred.reshape(-1), target.reshape(-1))
#     sets_sum = torch.sum(pred) + torch.sum(target)

#     if sets_sum.item() == 0:
#         sets_sum = 2 * inter

#     return (2 * inter + eps) / (sets_sum + eps)

def bce_with_logits_loss(y_pred, y_real):
    
    res = torch.mean(F.relu(y_pred) - y_pred * y_real + torch.log(1. + torch.exp(-torch.abs(y_pred))))
    
    return res

def sdice(pred_mask_tensor, mask_tensor, spacing=(1, 1), tolerance=1.):
    a = pred_mask_tensor.cpu().numpy().astype(bool)
    b = mask_tensor.cpu().numpy().astype(bool)

    surface_distances = surf_dc.compute_surface_distances(a, b, spacing)
    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=1)

class LitSegNet(pl.LightningModule):
    def __init__(self, num_classes, n_epochs, backbone=None):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.n_epochs = n_epochs

        self.backbone_model = backbone

        self.surface_dice_score = sdice
    
    def forward(self, x):
        return self.backbone_model.forward(x)
    
    def segmentation_loss(self, out, mask, bce_ratio=0.4, dice_ratio=0.6):

        mask = mask.flatten(start_dim=1).float()
        out = out.flatten(start_dim=1).float()

        # print(mask.shape, out.shape)

        # l_bce = weighted_cross_entropy_with_logits(out, mask)

        l_dice = dice_loss(torch.sigmoid(out), mask) 
        l_bce = bce_with_logits_loss(out, mask)

        l_total = l_dice * dice_ratio + l_bce * bce_ratio

        return l_total, l_dice, l_bce
    
    def training_step(self, train_batch, batch_idx):

        src_img = train_batch['img']

        out = self.forward(src_img).squeeze()
        pred_mask = (torch.sigmoid(out) > 0.5).long()
        # Bx1xHxW -> BxHxW
        src_mask = train_batch['mask'].squeeze()
        # Bx1xHxW -> BxHxWx2 -> Bx2xHxW
        # src_mask_ohe = F.one_hot(src_mask, self.num_classes).permute(0, 3, 1, 2)

        # print('mask', pred_mask)
        # print('src_mask', src_mask)

        l_total, l_dice, l_bce = self.segmentation_loss(out, src_mask)

        self.log('train Dice loss', l_dice, on_step=False, on_epoch=True)
        self.log('train BCE loss', l_bce, on_step=False, on_epoch=True)
        self.log('train Total loss', l_total, on_step=False, on_epoch=True)

        # print(src_mask, 'train')
        
        with torch.no_grad():
            sds_metric = np.mean([self.surface_dice_score(x, y) for x, y in zip(pred_mask, src_mask)])
            self.log('train Surface Dice', sds_metric, on_step=False, on_epoch=True)

        return l_total

    def validation_step(self, val_batch, batch_idx):

        src_img = torch.tensor(pad_tensor(val_batch['img'].cpu()), requires_grad=True, device=val_batch['img'].device)

        out = unpad_tensor(self.forward(src_img), val_batch['img']).squeeze(1)
        pred_mask = (torch.sigmoid(out) > 0.5).long()
        # Bx1xHxW -> BxHxW
        src_mask = val_batch['mask'].squeeze(1)
        # Bx1xHxW -> BxHxWx2 -> Bx2xHxW

        l_total, l_dice, l_bce = self.segmentation_loss(out, src_mask)
        
        self.log('val Dice loss', l_dice, on_step=False, on_epoch=True)
        self.log('val BCE loss', l_bce, on_step=False, on_epoch=True)
        self.log('val Total loss', l_total, on_step=False, on_epoch=True)

        with torch.no_grad():
            sds_metric = np.mean([self.surface_dice_score(x, y) for x, y in zip(pred_mask, src_mask)])
            self.log('val Surface Dice', sds_metric, on_step=False, on_epoch=True)

            if batch_idx == 0:
                self.log_grid(
                    val_batch['img'],
                    val_batch['mask'].float(),
                    pred_mask.unsqueeze(1),
                    tag='Input vs Mask vs Output',
                    topn=5
                )

        return l_total

    def log_grid(self, t_img, t_mask, t_out, tag='none', topn=1):
        with torch.no_grad():

            grid = make_grid(
                torch.dstack((t_img[:topn].cpu(),
                    t_mask[:topn].cpu(),
                    t_out[:topn].cpu(),
                )),
                nrow=5
            )

            self.logger.log_image(tag, [grid], self.global_step)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        # optimizer = SGD(
        #     self.parameters(),
        #     lr=1e-3,
        #     nesterov=True,
        #     momentum=0.9
        # )
        # scheduler = CosineAnnealingLR(optimizer, eta_min=1e-8, T_max=self.n_epochs)
        # scheduler = StepLR(optimizer, gamma=0.9, step_size=int(self.n_epochs // 10))
        return [optimizer]#, [scheduler]

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     return torch.argmax(self.sm(self(batch)), dim=1)