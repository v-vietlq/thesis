import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.siamesenetwork import *
from torchmetrics import Accuracy, F1
from pytorch_lightning.core.lightning import LightningModule
from utils.loss import *


class ImportanceModule(LightningModule):
    def __init__(self, args):
        super(ImportanceModule, self).__init__()
        self.train_opt = args 
        
        self.net =  SiameseNetwork(backbone='alexnet')
        
        self.criterion = nn.MSELoss()
        
        self.save_hyperparameters()
        
    def forward(self, x,y):
        return self.net(x,y)
    
    def training_step(self, batch, batch_idx):
        img1, img2, score1, score2 ,event = batch
                
        out1, out2 = self(img1, img2)
        out1 = torch.sigmoid(out1)
        out2 = torch.sigmoid(out2)
        total_loss = 0
        loss = self.criterion(out1,score1) + self.criterion(out2, score2)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        total_loss += loss
        
        return total_loss
        
    def validation_step(self, batch, batch_idx):
        img1, img2, score1, score2 ,event = batch
        with torch.no_grad():
            out1, out2 = self(img1, img2)
            out1 = torch.sigmoid(out1)
            out2 = torch.sigmoid(out2)
        
        loss = self.criterion(out1,score1) + self.criterion(out2, score2)
        
        return loss
    
    def validation_epoch_end(self, outputs):
        total_loss = 0
        for output in outputs:
            total_loss += output
            
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)
        
    def configure_optimizers(self):

        # Create optimizer
        self.optimizer = None
        if self.train_opt.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.atrain_optrgs.lr,
                momentum=self.train_opt.momentum,
                weight_decay=self.train_opt.weight_decay,
            )
        elif self.train_opt.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.train_opt.lr,
                weight_decay=self.train_opt.weight_decay,
            )
        elif self.train_opt.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(), lr=self.train_opt.lr)

        # Create learning rate scheduler
        self.scheduler = None
        if self.train_opt.lr_policy == "exp":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.train_opt.lr_gamma
            )
        elif self.train_opt.lr_policy == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.train_opt.lr_step, gamma=self.train_opt.lr_gamma
            )
        elif self.train_opt.lr_policy == "multi_step":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.train_opt.lr_milestones, gamma=self.train_opt.lr_gamma
            )
        elif self.train_opt.lr_policy == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.trainer.max_epochs, 0)
        elif self.train_opt.lr_policy == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.train_opt.lr, steps_per_epoch=len(self.trainer._data_connector._train_dataloader_source.dataloader()), epochs=self.train_opt.max_epoch,
                                                                 pct_start=0.2)
        return [self.optimizer], [{"scheduler": self.scheduler, "name": "lr"}]

        