import torch

import numpy as np

from dataset import DatasetFromList, fast_collate
from datasets.samplers import OrderedSampler
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from options import TrainOptions, train_options
from siameseModule import ImportanceModule
from torchvision import transforms as T
from datasets.augmentations.generate_transforms import generate_validation_transform
from siameseModule import ImportanceModule


if __name__ == '__main__':
    # Parse arguments

    train_opt = TrainOptions().parse()
    np.random.seed(train_opt.seed)
    torch.manual_seed(train_opt.seed)
    torch.cuda.manual_seed(train_opt.seed)
    train_opt.phase = 'train'

    # Create SegModule

    importanceModule = ImportanceModule(train_opt)

    # Load pretrained weight of model (for old version)
    if train_opt.pretrained is not None:
        print("Loading pretrained model from", train_opt.pretrained)
        state_dict = torch.load(train_opt.pretrained, map_location='cpu')
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {k.replace("net.", ""): v for k,
                          v in state_dict.items()}
        importanceModule.net.load_state_dict(state_dict)

    # Save checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_loss:.2f}',
        # monitor='metrics_iou',
        # filename='best-{epoch:02d}-{metrics_iou:.2f}',
        save_last=True,
        save_top_k=3,
        verbose=True,
        mode='min'
    )

    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        # monitor='metrics_iou',
        min_delta=0.00,
        patience=train_opt.patience,
        verbose=False,
        mode='min'
    )

    # Logging learning rate callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Create Logger
    logger = TensorBoardLogger(
        save_dir=train_opt.save_dir, name=train_opt.name)

    # Create Trainer
    trainer = pl.Trainer(gpus=train_opt.gpus,
                         replace_sampler_ddp = False,
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
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

    ])
    

    train_dataset = DatasetFromList(
        train_opt.train_root, train_opt.train_list, transform=train_transform, args=train_opt)
    
    train_sampler = OrderedSampler(train_dataset, args=train_opt)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_opt.batch_size, shuffle=False, pin_memory=True, sampler = train_sampler,
        num_workers=train_opt.num_threads, drop_last=False, collate_fn=partial(fast_collate, clip_length=train_opt.album_clip_length))
    
    val_dataset = DatasetFromList(
        train_opt.train_root, train_opt.val_list, transform=val_transform, args=train_opt)
    
    val_sampler = OrderedSampler(val_dataset, args=train_opt)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=train_opt.batch_size, shuffle=False, pin_memory=True, sampler = val_sampler,
        num_workers=train_opt.num_threads, drop_last=False, collate_fn=partial(fast_collate, clip_length=train_opt.album_clip_length))

    for i, (img, target, score) in enumerate(train_loader):
        print(target.shape)
        print(img.shape)
        print(score.shape)
        break
    # trainer.fit(importanceModule, train_loader, val_loader)