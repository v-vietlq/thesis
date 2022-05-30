import torch

import numpy as np

from dataset import *
from datasets.samplers import *
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from options.train_options import TrainOptions
from torchvision import transforms as T
from multitaskModule import MultitaskModule
import random
from PIL import ImageDraw
from randaugment import RandAugment


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


if __name__ == '__main__':
    # Parse arguments

    train_opt = TrainOptions().parse()
    np.random.seed(train_opt.seed)
    torch.manual_seed(train_opt.seed)
    torch.cuda.manual_seed(train_opt.seed)
    train_opt.phase = 'train'

    val_opt = TrainOptions().parse()
    val_opt.phase = 'val'
    val_opt.batch_size = train_opt.album_clip_length

    # Create SegModule

    multitaskModule = MultitaskModule(train_opt, val_opt)

    # Load pretrained weight of model (for old version)
    if train_opt.pretrained is not None:
        print("Loading pretrained model from", train_opt.pretrained)
        state_dict = torch.load(train_opt.pretrained, map_location='cpu')
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {k.replace("net.", ""): v for k,
                          v in state_dict.items()}
        multitaskModule.net.load_state_dict(state_dict)

    # Save checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='mAP',
        filename='best-{epoch:02d}-{mAP:.2f}',
        # monitor='metrics_iou',
        # filename='best-{epoch:02d}-{metrics_iou:.2f}',
        save_last=True,
        save_top_k=3,
        verbose=True,
        mode='max'
    )

    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='mAP',
        # monitor='metrics_iou',
        min_delta=0.00,
        patience=train_opt.patience,
        verbose=False,
        mode='max'
    )

    # Logging learning rate callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Create Logger
    logger = TensorBoardLogger(
        save_dir=train_opt.save_dir, name=train_opt.name)

    # Create Trainer
    trainer = pl.Trainer(gpus=train_opt.gpus,
                         replace_sampler_ddp=False,
                         resume_from_checkpoint=train_opt.resume,
                         accelerator=train_opt.accelerator,
                         logger=logger,
                         max_epochs=train_opt.max_epoch,
                         callbacks=[early_stopping_callback, checkpoint_callback, lr_monitor])

    train_transform = T.Compose([
        # T.RandomResizedCrop((224, 224)),
        # T.RandomRotation(degrees=30.),
        # T.RandomPerspective(distortion_scale=0.4),
        T.Resize((224, 224)),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        T.ToTensor(),


    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    train_dataset = CUFEDImportanceDataset(
        train_opt.train_root, train_opt.train_list, transforms=train_transform, args=train_opt)

    train_sampler = OrderSampler(train_dataset, args=train_opt)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_opt.batch_size, shuffle=False, sampler=train_sampler,
        num_workers=train_opt.num_threads, drop_last=True, collate_fn=partial(fast_collate_1, clip_length=train_opt.album_clip_length))

    val_dataset = CUFEDImportanceDataset(
        train_opt.val_root, train_opt.val_list, transforms=val_transform, args=train_opt)

    val_sampler = OrderSampler(val_dataset, args=train_opt)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_opt.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=train_opt.num_threads, drop_last=True, collate_fn=partial(fast_collate_1, clip_length=train_opt.album_clip_length))

    # for i, (img, target, score) in enumerate(train_loader):
    #     print(target.shape)
    #     print(img.shape)
    #     print(score.shape)
    #     break
    trainer.fit(multitaskModule, train_loader, val_loader)
