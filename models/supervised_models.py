from .fda_models.deeplab import ResNet101, Bottleneck

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import pytorch_lightning as pl

from torchvision.utils import make_grid

from torchmetrics.classification import MulticlassJaccardIndex
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

from .utils import beta_transform

class LitDeepLabV2(pl.LightningModule):
    def __init__(self, num_classes, n_epochs, beta, ent_lambda, eta):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.n_epochs = n_epochs

        self.model = ResNet101(
            in_channels=1,
            block=Bottleneck,
            layers=[3, 4, 23, 3],
            num_classes=2
        )

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
    
    # def compute_losses(self, src_mask, src_out, tgt_out):

    #     ent_loss = self.entropy_loss(tgt_out)
    #     seg_loss = self.model.CrossEntropy2d(src_out, src_mask)

    #     return ent_loss, seg_loss
    
    def forward(self, x):

        _, _, h, w = x.shape
        
        out = self.model(x)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        
        return out
    
    def training_step(self, train_batch, batch_idx):

        b_size = train_batch['tgt']['img'].shape[0]

        src_img_t = torch.stack([
            beta_transform(sample, train_batch['tgt']['img'][np.random.randint(b_size)], beta=self.beta)
            for sample in train_batch['src']['img']
        ])
        # src forward
        src_out = self.forward(src_img_t)
        src_mask = train_batch['src']['mask'].squeeze(1)        
        loss_seg_src = self.model.CrossEntropy2d(src_out, src_mask)
        # Bx1xHxW -> BxHxW -> BxHxWxC -> BxCxHxW 
        src_mask_ohe = F.one_hot(src_mask, self.num_classes).permute(0, 3, 1, 2)
        self.train_miou(preds=F.softmax(src_out, dim=1), target=src_mask_ohe)
        del src_out
        del src_mask_ohe

        # tgt forward
        tgt_out = self.forward(train_batch['tgt']['img'])
        # trigger entropy after 50.000th step
        loss_ent_tgt = self.entropy_loss(tgt_out) if self.global_step > 50000 else 0.     

        self.log('train_miou', self.train_miou, on_step=False, on_epoch=True)

        total_loss = loss_seg_src + self.ent_lambda * loss_ent_tgt
        self.log('train_loss', total_loss)
        
        return total_loss

    def validation_step(self, val_batch, batch_idx):

        b_size = val_batch['tgt']['img'].shape[0]

        src_img_t = torch.stack([
            beta_transform(sample, val_batch['tgt']['img'][np.random.randint(b_size)], beta=self.beta)
            for sample in val_batch['src']['img']
        ])
        
        # src forward
        src_out = self.forward(src_img_t)
        src_mask = val_batch['src']['mask'].squeeze(1)        
        loss_seg_src = self.model.CrossEntropy2d(src_out, src_mask)
        
        if batch_idx == 50:
            
            self.log_grid(
                val_batch['src']['img'],
                val_batch['src']['mask'].float(),
                src_out,
                tag='source val',
                topn=3
            )
        del src_out
        
        # tgt forward
        tgt_mask = val_batch['tgt']['mask'].squeeze(1)  

        tgt_out = self.forward(val_batch['tgt']['img'])
        
        # trigger entropy after 50.000th step
        loss_ent_tgt = self.entropy_loss(tgt_out) if self.global_step > 50000 else 0.
        
        # Bx1xHxW -> BxHxW -> BxHxWxC -> BxCxHxW       
        tgt_mask_ohe = F.one_hot(tgt_mask, self.num_classes).permute(0, 3, 1, 2)
        self.val_miou(preds=F.softmax(tgt_out, dim=1), target=tgt_mask_ohe)
        del tgt_mask_ohe

        self.log('val_miou', self.val_miou, on_step=False, on_epoch=True)
        
        # loss_seg_src, _ = self.compute_total_loss(src_mask, src_out, tgt_out)

        total_loss = loss_seg_src + self.ent_lambda * loss_ent_tgt
        self.log('val_loss', total_loss)

        if batch_idx == 50:

            self.log_grid(
                val_batch['tgt']['img'],
                val_batch['tgt']['mask'].float(),
                tgt_out,
                tag='target val',
                topn=3
            )

        return total_loss
    
    def log_grid(self, t_img, t_mask, t_out, tag='none', topn=1):
        with torch.no_grad():

            grid = make_grid(
                torch.dstack((t_img[:topn].cpu(),
                    t_mask[:topn].cpu(),
                    torch.argmax(t_out[:topn], dim=1).unsqueeze(1).cpu(),
                )),
                nrow=3
            )

            self.logger.log_image(tag, [grid], self.global_step)

    def configure_optimizers(self):
        optimizer = SGD(self.model.parameters(), lr=2.5e-4, weight_decay=5e-4)
        scheduler = StepLR(optimizer, gamma=0.9, step_size=self.n_epochs//3)
        return [optimizer], [scheduler]

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     return torch.argmax(self.sm(self(batch)), dim=1)