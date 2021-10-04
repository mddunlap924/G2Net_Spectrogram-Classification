import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
import torch.nn as nn
import torchvision.transforms as T
import torchmetrics
from pytorch_lightning import seed_everything
import timm
import random


seed_everything(42)
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


# Image augmentations
class ImgAugTransform:
    def __init__(self, augs, resize, model_cfg):
        self.augs = augs
        self.resize = (int(resize.split('(')[1].split(',')[0]),
                       int(resize.split(')')[0].split(' ')[1]),
                       )
        self.model_cfg = model_cfg

        # Resize Transform
        self.resize = T.Resize(size=self.resize)

        # Train Transform
        rand_erase = [T.RandomErasing(p=1.0,
                                      scale=self.augs['RandomErasing']['scale'],
                                      ratio=self.augs['RandomErasing']['ratio'],
                                      inplace=False)
                      for _ in range(self.augs['RandomErasing']['num'])
                      # for _ in range(1)
                      ]
        self.train = T.Compose(
            [T.Normalize(self.model_cfg['mean'], self.model_cfg['std']),
             T.RandomApply(rand_erase, p=self.augs['RandomErasing']['prob'])],
        )

        # Validation Transform
        self.val = T.Compose([T.Normalize(self.model_cfg['mean'], self.model_cfg['std'])])


# Normalize Images
def norm_images(img):
    norm = 1.0
    for i, x in enumerate(range(3)):
        # img[i] = ((img[i] - img[i].min()) / (img[i].max() - img[i].min()))
        img[i] = (img[i] / img[i].max()) * norm

    # img = img / img.max()
    return img


# Normalize time domain signals using descriptive statistics
def sig_unit_var(sigs, mean, std):
    if sigs.ndim == 1:
        sigs = sigs - sigs.mean()
        sigs = sigs / sigs.std()
    else:
        for i in range(3):
            sig = sigs[i] - sigs[i].mean()
            sigs[i] = sig / sig.std()
    return sigs


# Normalize time domain signals by a factor
def sig_norm_mag(sigs, factor):
    if sigs.ndim == 1:
        sigs = sigs / factor
    else:
        for i in range(3):
            sigs[i] = sigs[i] / factor
    return sigs


# Normalize signals by their maximum
def sig_norm_stats(x, hp):
    x_norm = np.empty(x.shape)
    for i in range(x.shape[0]):
        x_norm[i] = x[i] / hp.max[i]
    return x_norm


# Method to select the normalization technique
def norm_sigs(sigs, norm_info, hpf):
    if norm_info['NORM_SIG']['tech'] == 'unit_variance':
        sigs = sig_unit_var(sigs, mean=1.0, std=1.0)
    elif norm_info['NORM_SIG']['tech'] == 'norm_mag':
        sigs = sig_norm_mag(sigs, factor=float(norm_info['NORM_SIG']['factor']))
    elif norm_info['NORM_SIG']['tech'] == 'norm_stats':
        sigs = sig_norm_stats(sigs, hpf)
    return sigs


# Convert time signals to CQT images
def sigs_to_qtransform(sigs, q_transform):
    sigs = torch.from_numpy(sigs).float()
    img = q_transform(sigs)  # Apply Q-Transform
    return img


# Training and Validation Datasets
class GwTrainDataset(Dataset):
    def __init__(self, x, y, q_transform, data_type, img_transform, hpf, norms, spec_mix):
        self.x = x
        self.y = y
        self.data_type = data_type
        self.img_transform = img_transform
        self.q_transform = q_transform
        self.hpf = hpf
        self.norms = norms
        self.spec_mix = spec_mix

    # Return length of dataset
    def __len__(self):
        return len(self.y)

    # Return (feature, label) pair
    def __getitem__(self, idx):
        x_ = self.x[idx]  # Single instance of data
        x_ = self.hpf.highpass_filter(x_)  # Highpass Filter Raw Signals
        x_ = norm_sigs(x_, norm_info=self.norms, hpf=self.hpf)  # Normalize Signals

        # Training Data
        if self.data_type == 'train':
            # Select Q-transform to use
            if self.spec_mix['apply']:
                if np.random.uniform(0, 1, 1) <= self.spec_mix['prob']:
                    imgs = []
                    for i in range(len(self.q_transform)):
                        img_ = sigs_to_qtransform(x_, self.q_transform[i])
                        img_ = self.img_transform.resize(img_)
                        img_ = norm_images(img_)
                        imgs.append(img_)
                    img = torch.stack(imgs).mean(axis=0)
                    img = self.img_transform.train(img)
                else:
                    q_trans_idx = random.randint(0, len(self.q_transform) - 1)
                    img = sigs_to_qtransform(x_, self.q_transform[q_trans_idx])
                    img = self.img_transform.resize(img)
                    img = norm_images(img)
                    img = self.img_transform.train(img)
            else:
                img = sigs_to_qtransform(x_, self.q_transform[0])
                img = norm_images(img)
                img = self.img_transform.train(img)

        # Validation Data
        if self.data_type == 'val':
            img = sigs_to_qtransform(x_, self.q_transform[0])
            img = norm_images(img)
            img = self.img_transform.val(img)

        y_ = torch.tensor(self.y[idx])
        return (img, y_)


# Test Dataset Loader
class GwTestDataset(Dataset):
    def __init__(self, x, df_idxs, q_transform, img_transform, hpf, norms):
        self.x = x
        self.df_idxs = df_idxs
        self.q_transform = q_transform
        self.img_transform = img_transform
        self.hpf = hpf
        self.norms = norms

    # Return length of dataset
    def __len__(self):
        return len(self.x)

    # Return (feature) only
    def __getitem__(self, idx):
        x_ = self.x[idx]  # Single instance of data
        x_ = self.hpf.highpass_filter(x_)  # Highpass Filter Raw Signals
        x_ = norm_sigs(x_, norm_info=self.norms, hpf=self.hpf)  # Normalize Signals

        imgs = []
        for i in range(len(self.q_transform)):
            img_ = sigs_to_qtransform(x_, self.q_transform[i])
            # img_ = self.img_transform.resize(img_)
            img_ = norm_images(img_)
            imgs.append(img_)
        img = torch.stack(imgs).mean(axis=0)
        img = self.img_transform.val(img)
        df_idxs_ = self.df_idxs[idx]
        return (img, df_idxs_)


# Lightning DataModule
class GwDataModule(pl.LightningDataModule):
    def __init__(self, x_train, y_train, x_val, y_val, batch_size, q_transform, img_transform, hpf, norms, spec_mix):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.q_transform = q_transform
        self.img_transform = img_transform
        self.hpf = hpf
        self.norms = norms
        self.spec_mix = spec_mix

    # Setup
    def setup(self, stage=None):
        # Setup Training Data
        self.train_data = GwTrainDataset(x=self.x_train,
                                         y=self.y_train,
                                         q_transform=self.q_transform,
                                         data_type='train',
                                         img_transform=self.img_transform,
                                         hpf=self.hpf,
                                         norms=self.norms,
                                         spec_mix=self.spec_mix,
                                         )

        # Setup Validation data
        self.val_data = GwTrainDataset(x=self.x_val,
                                       y=self.y_val,
                                       q_transform=self.q_transform,
                                       data_type='val',
                                       img_transform=self.img_transform,
                                       hpf=self.hpf,
                                       norms=self.norms,
                                       spec_mix=self.spec_mix,
                                       )

    # Train DataLoader
    def train_dataloader(self):
        return DataLoader(dataset=self.train_data,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True,
                          )

    # Validation DataLoader
    def val_dataloader(self):
        return DataLoader(dataset=self.val_data,
                          batch_size=self.batch_size,
                          num_workers=8,
                          pin_memory=True,
                          )


# Neural Network from timm package
def get_model(model_name, pretrained):
    # Load pretrained model
    spec_model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
    return spec_model


# Model
class GwModel(LightningModule):
    def __init__(self, *, model_inputs, lr_inputs, batch_size, loss_fn):
        super().__init__()
        self.model_name = model_inputs['name']
        self.scheduler_inputs = lr_inputs
        self.num_neurons = model_inputs['num_neurons']
        self.batch_size = batch_size
        self.loss_fn = loss_fn

        self.save_hyperparameters()

        # Train
        if loss_fn == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_auroc = torchmetrics.AUROC(num_classes=None, pos_label=1)
        self.val_auroc = torchmetrics.AUROC(num_classes=None, pos_label=1)

        # Get a backbone model
        model = get_model(self.model_name, model_inputs['pretrained'])
        for param in model.parameters():
            param.requires_grad = True
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        data = data.float()

        # Could be self(data)
        if target.shape[0] == 1:
            output = self.model(data)[0]
            loss = self.criterion(output, target.float())
        else:
            output = torch.squeeze(self.model(data))
            loss = self.criterion(output, target.float())
        self.train_auroc.update(output.sigmoid(), target)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        data = data.float()

        if target.shape[0] == 1:
            output = self.model(data)[0]
            val_loss = self.criterion(output, target.float())
        else:
            output = torch.squeeze(self.model(data))
            val_loss = self.criterion(output, target.float())
        self.val_auroc.update(output.sigmoid(), target)
        self.log('val/loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        # Unpack learning rate scheduler inputs
        T_0 = self.scheduler_inputs['T_0']
        T_mult = self.scheduler_inputs['T_mult']
        lr_initial = self.scheduler_inputs['lr_initial']
        lr_min = self.scheduler_inputs['lr_min']
        optimizer = torch.optim.Adam(self.parameters(), lr=lr_initial)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=T_0,
                                                                         T_mult=T_mult,
                                                                         eta_min=lr_min,
                                                                         last_epoch=-1,
                                                                         )
        return [optimizer], [scheduler]

    def training_epoch_end(self, _):
        self.log('train/auroc', self.train_auroc.compute(), on_epoch=True, prog_bar=True)
        self.train_auroc.reset()

    def validation_epoch_end(self, _):
        self.log('val/auroc', self.val_auroc.compute(), on_epoch=True, prog_bar=True)
        self.val_auroc.reset()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ckpt_metrics'] = self.trainer.logged_metrics


# Define Checkpoint Save Path
def define_ckpt_save_directory(log_path):
    all_subdirs = [os.path.join(log_path, d) for d in os.listdir(log_path)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    new_subdir = latest_subdir.split('_')[0] + '_' + str(int(latest_subdir.split('_')[1]) + 1)
    return new_subdir
