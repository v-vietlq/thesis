from cProfile import label
from functools import total_ordering
import os
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.clsnetwork import *
from pytorch_lightning.core.lightning import LightningModule
from utils.loss import FocalLoss, AsymmetricLossOptimized, AsymmetricLoss
from utils.utils import AP_partial
from models.models import MTResnetAggregate
class EventModule(LightningModule):
        
    def __init__(self, main_opt, val_opt=None):
        super().__init__()
        if val_opt is None: # test phase 
            self.test_opt = main_opt
            self.save_hyperparameters(vars(main_opt))
            if self.test_opt.use_transformer:
                self.net = MTResnetAggregate(self.test_opt)
            else:
                self.net = EventCnnLstm(main_opt.backbone, main_opt.num_classes)
                
            return

        self.train_opt = main_opt
        self.save_hyperparameters(vars(main_opt))
        self.val_opt = val_opt
        if self.train_opt.use_transformer:
            self.net = MTResnetAggregate(self.train_opt)         
        else:
            self.net = EventCnnLstm(self.train_opt.backbone, self.train_opt.num_classes)
            
        
        self.output_weights = [1]
        self.criterion = []
        class_weight =torch.Tensor([2.21, 1.873, 1, 1.567,
                  2.497, 1, 1.622, 1, 2.161,
                  1, 1.804, 2.027, 2.069,
                  2.069, 2.027, 1.567, 1.804,
                  1 , 1.679, 1 , 1, 1, 1]).cuda()
        for loss_name in self.train_opt.loss:
            if loss_name == "asymmetric":
                self.criterion += [(loss_name, AsymmetricLossOptimized(gamma_neg= 2, gamma_pos= 0,clip=0.0,disable_torch_grad_focal_loss=False,eps=1e-5))]
            elif loss_name == "focal":
                self.criterion += [(loss_name, FocalLoss())]
            elif loss_name == 'multilabelsoftmagin':
                self.criterion += [(loss_name, nn.MultiLabelSoftMarginLoss(weight=class_weight))]

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        image, label = batch
        if self.train_opt.use_transformer:
            batch_size, time_steps, channels, height, width = image.size()
            image = image.view(batch_size * time_steps, channels, height, width)
        outputs = self(image)
        if len(self.output_weights) == 1:
            outputs = [outputs]

        total_loss = 0
        for loss_name, criteria in self.criterion:
            loss = 0
            for output, weight in zip(outputs, self.output_weights):
                # import pdb; pdb.set_trace()
                loss = loss + weight * criteria(output, label)
            total_loss = total_loss + loss
            self.log('loss_'+loss_name, loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss


    def validation_step(self, batch, batch_idx):
        image, label = batch
        if self.train_opt.use_transformer:
            batch_size, time_steps, channels, height, width = image.size()
            image = image.view(batch_size * time_steps, channels, height, width)
        with torch.no_grad():
            outputs = self(image)
        pred = torch.sigmoid(outputs)
        pred[(pred >= self.train_opt.threshold)] = 1
        pred[(pred < self.train_opt.threshold)] = 0
        

        return pred, label

    def validation_epoch_end(self, outputs):
        preds, targs = [], []
        for out in outputs:
            pred, label = out
            preds.append(pred)
            targs.append(label)
        preds,targs = torch.cat(preds).cpu().detach().numpy() , torch.cat(targs).cpu().detach().numpy()
        
        acc = AP_partial(targs, preds)
            
        self.log('mAP', acc, on_step=False, on_epoch=True, sync_dist=True)

        
    def test_step(self, batch, batch_idx):
       return

    def test_epoch_end(self, outputs):
        return

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
        elif self.train_opt.lr_policy =='onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.train_opt.lr, steps_per_epoch=len(self.trainer._data_connector._train_dataloader_source.dataloader()), epochs=self.train_opt.max_epoch,
                                        pct_start=0.2)
        return [self.optimizer], [{"scheduler": self.scheduler, "name": "lr"}]