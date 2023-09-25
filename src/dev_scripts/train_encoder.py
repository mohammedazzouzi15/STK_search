import numpy as np
import os

#os.chdir("/home/ma11115/github_folder/STK_search")
os.chdir("C:\\Users\\ma11115\\OneDrive - Imperial College London\\github_folder\\STK_SEARCH")
# %pip install pytorch-lightning
import pandas as pd
from src.stk_search import Database_utils
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
import wandb

wandb.login()

# %%
# Load the searched space
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
slatm_rpr_precursor = slatm_rpr_precursor[
    :, ~(slatm_rpr_precursor == slatm_rpr_precursor[0, :]).all(0)
]
for i, x in enumerate(slatm_name_precursor):
    x = x.split("/")[1].replace(".xyz", "")
    slatm_name_precursor[i] = x
slatm_rpr = np.load(
    "data/output/Full_dataset/repr_df_total_new2023_08_20.npy",
    allow_pickle=True,
)
slatm_name = np.load(
    "data/output/Full_dataset/names_df_total_new2023_08_20.npy",
    allow_pickle=True,
)
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
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        loss = F.mse_loss(z, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# convert into PyTorch tensors
X = torch.tensor(init_slatm_rpr, dtype=torch.float32)
y = torch.tensor(slatm_rpr_new, dtype=torch.float32)

# create DataLoader, then take one batch
dataset = list(zip(X,y))
train_len = int(len(dataset)*0.8)
val_len = len(dataset)-train_len

train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
batch_size = 100

train_loader = DataLoader(dataset, batch_size=batch_size, 
                                           num_workers=6,shuffle = True)
validation_loader = DataLoader(dataset, batch_size=batch_size,shuffle = False)

# model
autoencoder = LitAutoEncoder(Encoder())
wandb_logger = WandbLogger(log_model="all")
# train model
trainer = pl.Trainer(logger=wandb_logger,fast_dev_run=False, max_epochs=2,val_check_interval=1.0)
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=validation_loader)
