import numpy as np
import torch
from torchmetrics.functional import accuracy
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(25638, 100), nn.ReLU(), nn.Linear(100, 14719))

    def forward(self, x):
        return self.l1(x)
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss = self._get_preds_loss_accuracy(batch)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_loss', loss)
        return loss
    
    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        #x_hat = self.decoder(z)
        loss = F.mse_loss(z, y) # F.mse_loss(x_hat, x) #
        return loss

class Representation_slatm():
    def __init__(self):
        self.Precursor_name = None
        self.Precursor_repr = None
        self.load_frag_slatm()
        self.model = None
        self.load_model()
    def load_frag_slatm(self):
        slatm_rpr_precursor = np.load('data/output/Prescursor_data/repr_slatm_precursor_310823_clean.pkl.npy',allow_pickle=True)
        slatm_name_precursor = np.load('data/output/Prescursor_data/names_slatm_precursor_310823_clean.pkl.npy',allow_pickle=True)
        slatm_rpr_precursor = slatm_rpr_precursor[:, ~(slatm_rpr_precursor == slatm_rpr_precursor[0,:]).all(0)]
        for i,x in enumerate(slatm_name_precursor):
            x=x.split('/')[1].replace('.xyz','')
            slatm_name_precursor[i]=x
        self.Precursor_name = slatm_name_precursor
        self.Precursor_repr = slatm_rpr_precursor
    def load_model(self):
        checkpoint = torch.load('lightning_logs/f3t61f1q/checkpoints/epoch=99-step=41000.ckpt',map_location=torch.device('cpu'))
        model = LitAutoEncoder(Encoder())
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self.model = model
    def generate_repr(self,elements):
        init_slatm_rpr=[]
        for elm in elements:
            init_slatm_rpr.append(np.concatenate([self.Precursor_repr[self.Precursor_name==x][0] for x in elm]))
        X_explored_BO = torch.tensor(np.array(init_slatm_rpr), dtype=torch.float32)
        X_explored_BO = self.model.encoder(X_explored_BO).detach()
        
        X_explored_BO = (X_explored_BO - X_explored_BO.mean(axis=0)) / (
                X_explored_BO.std(axis=0))
        X_explored_BO = torch.nan_to_num(X_explored_BO)
        return X_explored_BO