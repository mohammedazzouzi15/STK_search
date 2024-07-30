import os

import numpy as np

#os.chdir("/home/ma11115/github_folder/STK_search")
#os.chdir("C:\\Users\\ma11115\\OneDrive - Imperial College London\\github_folder\\STK_SEARCH")
os.chdir("/rds/general/user/ma11115/home/STK_Search/STK_search")
# %pip install pytorch-lightning
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader

import wandb
from src.stk_search import Database_utils


# %%
# Load the searched space
def load_data():
    df_path = "data/output/Full_datatset/df_total_new2023_08_20.csv"
    df_precursors_path = "data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl"  #'Data/output/Prescursor_data/calculation_data_precursor_310823_clean.pkl'
    df_total, df_precursors = Database_utils.load_data_from_file(
        df_path, df_precursors_path
    )

    # load and clead slatm data
    slatm_rpr_precursor = np.load(
        "data/output/Prescursor_data/repr_slatm_precursor_310823_clean.pkl.npy",
        allow_pickle=True,
    )
    slatm_name_precursor = np.load(
        "data/output/Prescursor_data/names_slatm_precursor_310823_clean.pkl.npy",
        allow_pickle=True,
    )
    slatm_rpr = np.load(
    "data/output/Full_dataset/repr_df_total_new2023_08_20.npy",
    allow_pickle=True,
    )
    slatm_name = np.load(
        "data/output/Full_dataset/names_df_total_new2023_08_20.npy",
        allow_pickle=True,
    )
    return df_total, df_precursors, slatm_rpr_precursor, slatm_name_precursor, slatm_rpr, slatm_name




class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(25638, 100), nn.ReLU(), nn.Linear(100, 14719)
        )
    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss = self._get_preds_loss_accuracy(batch)

        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        """Used for logging metrics"""
        loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss)
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """Convenience function since train/valid/test steps are similar"""
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        loss = F.mse_loss(z, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def generate_repr(slatm_name_precursor, slatm_rpr_precursor, df_total, slatm_rpr):
    slatm_rpr_precursor = slatm_rpr_precursor[
        :, ~(slatm_rpr_precursor == slatm_rpr_precursor[0, :]).all(0)
    ]
    for i, x in enumerate(slatm_name_precursor):
        x = x.split("/")[1].replace(".xyz", "")
        slatm_name_precursor[i] = x

    slatm_rpr_new = slatm_rpr
    slatm_rpr_new = slatm_rpr_new[
        :, ~(slatm_rpr_new == slatm_rpr_new[0, :]).all(0)
    ]

    # concat the slatm data
    for i in range(6):
        df_total[f"slatm_{i}"] = df_total[f"InChIKey_{i}"].apply(
            lambda x: slatm_rpr_precursor[slatm_name_precursor == x][0]
        )
    init_slatm_rpr = np.stack(df_total[f"slatm_{0}"].values)
    for i in range(1, 6):
        init_slatm_rpr = np.concatenate(
            (init_slatm_rpr, np.stack(df_total[f"slatm_{i}"].values)), axis=1
        )
    return init_slatm_rpr, slatm_rpr_new
# convert into PyTorch tensors


def generate_data_loader(init_slatm_rpr, slatm_rpr_new, batch_size=100):
    X = torch.tensor(init_slatm_rpr, dtype=torch.float32)
    y = torch.tensor(slatm_rpr_new, dtype=torch.float32)
    # create DataLoader, then take one batch
    dataset = list(zip(X,y))
    train_len = int(len(dataset)*0.8)
    val_len = len(dataset)-train_len

    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle = True, num_workers=32)
    validation_loader = DataLoader(val_set, batch_size=batch_size, shuffle = False)
    return train_loader, validation_loader

def train_model():
    wandb.login()
    df_total, df_precursors, slatm_rpr_precursor, slatm_name_precursor, slatm_rpr, slatm_name = load_data()
    init_slatm_rpr, slatm_rpr_new = generate_repr(slatm_name_precursor, slatm_rpr_precursor, df_total,slatm_rpr)
    train_loader, validation_loader = generate_data_loader(init_slatm_rpr, slatm_rpr_new, batch_size=100)
    # model
    autoencoder = LitAutoEncoder(Encoder())
    wandb_logger = WandbLogger(log_model="all")
    # train model
    trainer = pl.Trainer(logger=wandb_logger,fast_dev_run=False, max_epochs=2,val_check_interval=1.0)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=validation_loader)
