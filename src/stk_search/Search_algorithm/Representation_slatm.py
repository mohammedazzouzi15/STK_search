import numpy as np
import torch
from torchmetrics.functional import accuracy
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(25638, 100), nn.ReLU(), nn.Linear(100, 14719)
        )

    def forward(self, x):
        return self.l1(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(25638, 100), nn.ReLU(), nn.Linear(100, 50),nn.ReLU(),
            nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, 14719)
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss)
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        # x_hat = self.decoder(z)
        loss = F.mse_loss(z, y)  # F.mse_loss(x_hat, x) #
        return loss


class Representation_slatm:
    def __init__(self):
        self.Precursor_name_loc = "data/output/Prescursor_data/repr_slatm_precursor_310823_clean.pkl.npy"
        self.Precursor_repr_loc = "data/output/Prescursor_data/names_slatm_precursor_310823_clean.pkl.npy"
        self.model_name = 'data/Encoder_models/model_encoder_261023.pt'
        self.Precursor_name = None
        self.Precursor_repr = None
        self.load_frag_slatm()
        self.model = None
        self.load_model()
        
    def __getstate__(self):
        return {'model_name': self.model_name,
                "Precursor_name_loc": self.Precursor_name_loc,
                "Precursor_repr_loc": self.Precursor_repr_loc}
    
    def __setstate__(self, d):        
        self.model_name = d['model_name']
        self.Precursor_name_loc = d['Precursor_name_loc']
        self.Precursor_repr_loc = d['Precursor_repr_loc']
        self.load_frag_slatm()
        self.model = None

    def load_frag_slatm(self):
        slatm_rpr_precursor = np.load(
            self.Precursor_name_loc,
            allow_pickle=True,
        )
        slatm_name_precursor = np.load(
            self.Precursor_repr_loc,
            allow_pickle=True,
        )
        slatm_rpr_precursor = slatm_rpr_precursor[
            :, ~(slatm_rpr_precursor == slatm_rpr_precursor[0, :]).all(0)
        ]
        for i, x in enumerate(slatm_name_precursor):
            x = x.split("/")[1].replace(".xyz", "")
            slatm_name_precursor[i] = x
        self.Precursor_name = slatm_name_precursor
        self.Precursor_repr = slatm_rpr_precursor

    def load_model(self):
        model = torch.jit.load(self.model_name)
        model.eval()
        self.model = model

    def generate_repr(self, elements):
        elements_copy = elements.copy()
        elements_copy = elements_copy.values
        init_slatm_rpr = []
        for elm in elements_copy:
            init_slatm_rpr.append(
                np.concatenate(
                    [
                        self.Precursor_repr[self.Precursor_name == x][0]
                        for x in elm
                    ]
                )
            )
        X_explored_BO = torch.tensor(
            np.array(init_slatm_rpr), dtype=torch.float32
        )
        X_explored_BO = self.model(X_explored_BO).detach()

        return X_explored_BO


class Representation_slatm_org:
    def __init__(self, df_total, df_inchikey=None):
        self.slatm_name = None
        self.slatm_rpr_new = None
        self.load_slatm()
        self.df_total = df_total
        self.df_inchikey = df_inchikey  # pd.read_csv('data/output/Full_dataset/df_total_new2023_08_20.csv')

    def load_slatm(self):
        # load and clead slatm data
        slatm_rpr = np.load(
            "data/output/Full_dataset/repr_df_total_new2023_08_20.npy",
            allow_pickle=True,
        )
        self.slatm_name = np.load(
            "data/output/Full_dataset/names_df_total_new2023_08_20.npy",
            allow_pickle=True,
        )
        for i, x in enumerate(self.slatm_name):
            x = x.split("/")[1].replace(".xyz", "")
            self.slatm_name[i] = x
        slatm_rpr_new = slatm_rpr
        self.slatm_rpr_new = slatm_rpr_new[
            :, ~(slatm_rpr_new == slatm_rpr_new[0, :]).all(0)
        ]

    def find_elem_InchiKey(self, elements):
        InChIKeys = []
        if self.df_inchikey is None:
            for elm in elements:
                df_search = self.df_total.copy()
                for i, x in enumerate(elm):
                    df_search = df_search[df_search[f"InChIKey_{i}"] == x]
                InChIKeys.append(df_search['InChIKey'].values.astype(str))
        else:
            InChIKeys = self.df_inchikey['InChIKey'].values.astype(str)
        return InChIKeys
    
    def generate_repr(self, elements):
        elements_copy = elements.copy()
        elements_copy = elements_copy.values
        InChIKeys = self.find_elem_InchiKey(elements_copy)
        init_slatm_rpr = []
        for x in InChIKeys:
            init_slatm_rpr.append(self.slatm_rpr_new[self.slatm_name==x])
        X_explored_BO = torch.tensor(
            np.array(init_slatm_rpr), dtype=torch.float32
        )
        X_explored_BO = X_explored_BO.squeeze()
        return X_explored_BO
