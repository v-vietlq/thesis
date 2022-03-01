import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.clsnetwork import *
from torchmetrics import Accuracy, F1
from pytorch_lightning.core.lightning import LightningModule
from utils.loss import FocalLoss, AsymmetricLossOptimized
from utils.utils import mAP


class EventModule(LightningModule):
    def __init__(self, args):
        super(EventModule, self).__init__()
        self.args = args
        self.output_weights = [1]

        self.net = EventCnnLstm(encoder_name=self.args.backbone, num_classes=self.args.num_classes)
        
        # self.criterion = []
        # for loss_name in self.args.loss:
        #     if loss_name == "ce":
        #         self.criterion += [(loss_name, nn.CrossEntropyLoss())]
        #     elif loss_name == "kullback":
        #         self.criterion += [(loss_name, nn.KLDivLoss())]
        self.criterion = AsymmetricLossOptimized()

        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, lb = batch

        out = self(img)
        loss = self.criterion(out, lb)
        logs = {'train_loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img, lb = batch
        with torch.no_grad():
            pred = self(img)
        loss = self.criterion(pred, lb)
        pred =  torch.sigmoid(pred)
        logs = {'valid_loss': loss}
        self.log_dict(
            logs,
            on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return lb , pred
        
    def validation_epoch_end(self, outputs):
        targets , preds = [], []
        for out in outputs:
            target, pred = out 
            preds.append(pred)
            targets.append(target)
        mean_ap = mAP(torch.cat(targets).cpu().numpy(), torch.cat(preds).cpu().numpy())
        
        self.log('mAP', mean_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
    def configure_optimizers(self):

        # Create optimizer
        self.optimizer = None
        if self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(), lr=self.args.learning_rate)

        # Create learning rate scheduler
        self.scheduler = None
        if self.args.lr_policy == "exp":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.args.lr_gamma
            )
        elif self.args.lr_policy == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.args.lr_step, gamma=self.args.lr_gamma
            )
        elif self.args.lr_policy == "multi_step":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.args.lr_milestones, gamma=self.args.lr_gamma
            )
        elif self.args.lr_policy == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.trainer.max_epochs, 0)
        elif self.args.lr_policy =='onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_dataloader()), epochs=self.args.max_epoch,
                                        pct_start=0.2)
        return [self.optimizer], [{"scheduler": self.scheduler, "name": "lr"}]
