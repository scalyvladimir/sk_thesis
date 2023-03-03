from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_lightning as pl

from torchmetrics.classification import MulticlassJaccardIndex
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

from .utils import beta_transform

class FDANet(pl.LightningModule):
    def __init__(self, num_classes, n_epochs, beta, ent_lambda, eta):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.n_epochs = n_epochs

        self.model = 

        self.beta = beta
        self.ent_lambda = ent_lambda
        self.eta = eta

        self.train_miou = MulticlassJaccardIndex(num_classes=self.num_classes)
        self.val_miou = MulticlassJaccardIndex(num_classes=self.num_classes)

    def entropy_loss(self, out):
        P = F.softmax(out, dim=1)        # [B, C, H, W]
        logP = F.log_softmax(out, dim=1) # [B, C, H, W]
        PlogP = P * logP                 # [B, C, H, W]
        ent = -1.0 * PlogP.sum(dim=1)    # [B, 1, H, W]
        ent /= torch.log(torch.tensor(self.num_classes))
        # compute robust entropy
        ent = (ent ** 2.0 + 1e-8) ** self.eta

        return ent.mean()
    
    def compute_total_loss(self, src_mask, src_out, tgt_out):

        ent_loss = self.entropy_loss(tgt_out)
        seg_loss = self.model.CrossEntropy2d(src_out, src_mask)

        total_loss = seg_loss + self.ent_lambda * ent_loss

        return total_loss
    
    def forward(self, x):

        _, _, h, w = x.shape
        
        out = self.model(x)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        
        return out
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs//3, eta_min=1e-4)
        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx):

        b_size = train_batch['tgt']['img'].shape[0]

        src_img_t = torch.stack([
            beta_transform(sample, train_batch['tgt']['img'][np.random.randint(b_size)], beta=self.beta)
            for sample in train_batch['src']['img']
        ])
        
        src_out = self.forward(src_img_t)
        tgt_out = self.forward(train_batch['tgt']['img'])
        
        # Bx1xHxW -> BxHxW -> BxHxWxC -> BxCxHxW 
        src_mask = train_batch['src']['mask'].squeeze(1)        
        src_mask_ohe = F.one_hot(src_mask, self.num_classes).permute(0, 3, 1, 2)
   
        self.train_miou(preds=F.softmax(src_out, dim=1), target=src_mask_ohe)
        del src_mask_ohe

        self.log('train_miou', self.train_miou, on_step=False, on_epoch=True)
        
        loss = self.compute_total_loss(src_mask, src_out, tgt_out)
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, val_batch, batch_idx):

        b_size = val_batch['tgt']['img'].shape[0]

        src_img_t = torch.stack([
            beta_transform(sample, val_batch['tgt']['img'][np.random.randint(b_size)], beta=self.beta)
            for sample in val_batch['src']['img']
        ])
        
        src_out = self.forward(src_img_t)
        tgt_out = self.forward(val_batch['tgt']['img'])

        src_mask = val_batch['src']['mask'].squeeze(1)
        src_mask_ohe = F.one_hot(src_mask, self.num_classes).permute(0, 3, 1, 2)

        self.val_miou(preds=F.softmax(src_out, dim=1), target=src_mask_ohe)
        del src_mask_ohe

        self.log('val_miou', self.val_miou, on_step=False, on_epoch=True)
        
        loss = self.compute_total_loss(src_mask, src_out, tgt_out)
        self.log('val_loss', loss)
        
        return loss
        
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     return torch.argmax(self.sm(self(batch)), dim=1)