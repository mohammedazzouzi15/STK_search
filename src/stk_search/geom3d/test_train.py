"""
script to train the SchNet model on the STK dataset
created by Mohammed Azzouzi
date: 2023-11-14
"""
import stk
import pymongo
import numpy as np
import os
import pandas as pd
import wandb
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl
import torch.nn.functional as Functional
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from stk_search.geom3d.models import SchNet
from stk_search.geom3d.utils import database_utils
from pathlib import Path
from stk_search.geom3d.config_utils import read_config


def main(config_dir):
    config = read_config(config_dir)
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    config["device"] = (
        "cuda" if torch.cuda.is_available() else torch.device("cpu")
    )
    dataset = load_data(config)
    train_loader, val_loader, test_loader = train_val_test_split(
        dataset, config=config
    )
    model_config = config["model"]
    model = SchNet(
        hidden_channels=model_config["emb_dim"],
        num_filters=model_config["SchNet_num_filters"],
        num_interactions=model_config["SchNet_num_interactions"],
        num_gaussians=model_config["SchNet_num_gaussians"],
        cutoff=model_config["SchNet_cutoff"],
        readout=model_config["SchNet_readout"],
        node_class=model_config["node_class"],
    )
    graph_pred_linear = torch.nn.Linear(
        model_config["emb_dim"], model_config["num_tasks"]
    )
    os.chdir(config["running_dir"])
    wandb.login()
    # model
    #check if chkpt exists
    if os.path.exists(config["model_embedding_chkpt"]):
        pymodel_SCHNET = Pymodel.load_from_checkpoint(config["model_embedding_chkpt"])
    else:
        pymodel_SCHNET = Pymodel(model, graph_pred_linear)
    wandb_logger = WandbLogger(log_model="all", project="Geom3D", name=config["name"])
    wandb_logger.log_hyperparams(config)

    # train model
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["name"],
        filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=config["max_epochs"],
        val_check_interval=1.0,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=pymodel_SCHNET,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    wandb.finish()

    # load dataframe with calculated data


def load_data(config):
    
    if config["load_dataset"]:
        if os.path.exists(config["dataset_path"]):
            dataset = torch.load(config["dataset_path"])
            return dataset
        else:
            print("dataset not found")
    df_path = Path(
        config["STK_path"], "data/output/Full_dataset/", config["df_total"]
    )
    df_precursors_path = Path(
        config["STK_path"],
        "data/output/Prescursor_data/",
        config["df_precursor"],
    )
    df_total, df_precursors = database_utils.load_data_from_file(
        df_path, df_precursors_path
    )
    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )

    dataset = generate_dataset(
        df_total,
        df_precursors,
        db,
        number_of_molecules=config["num_molecules"],
    )
    if config["save_dataset"]:
            name = config["name"] + "_frag_" + str(config["number_of_fragement"])
            os.makedirs(name, exist_ok=True)
            torch.save(dataset, name+"/dataset.pt")
    return dataset


class Pymodel(pl.LightningModule):
    def __init__(self, model, graph_pred_linear):
        super().__init__()
        self.save_hyperparameters(ignore=['graph_pred_linear', 'model'])
        self.molecule_3D_repr = model
        self.graph_pred_linear = graph_pred_linear

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss = self._get_preds_loss_accuracy(batch)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss)
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        z = self.molecule_3D_repr(batch.x, batch.positions, batch.batch)
        z = self.graph_pred_linear(z)
        loss = Functional.mse_loss(z, batch.y.unsqueeze(1))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer
    
    def forward(self, batch):
        z = self.molecule_3D_repr(batch.x, batch.positions, batch.batch)
        z = self.graph_pred_linear(z)
        return z


def load_3d_rpr(model, output_model_path):
    saved_model_dict = torch.load(output_model_path)
    model.load_state_dict(saved_model_dict["model"])
    model.eval()
    return model

def load_molecule(InChIKey, target, db):
    polymer = db.get({"InChIKey": InChIKey})
    dat_list = list(polymer.get_atomic_positions())
    positions = np.vstack(dat_list)
    positions = torch.tensor(positions, dtype=torch.float)
    atom_types = list(
        [
            atom.get_atom().get_atomic_number()
            for atom in polymer.get_atom_infos()
        ]
    )
    atom_types = torch.tensor(atom_types, dtype=torch.long)
    y = torch.tensor(target, dtype=torch.float32)

    molecule = Data(x=atom_types, positions=positions, y=y,
        InChIKey=InChIKey)
    return molecule


def generate_dataset(df_total, df_precursors, db, number_of_molecules=1000):
    molecule_index = np.random.choice(
        len(df_total), number_of_molecules, replace=False
    )
    data_list = []
    for i in molecule_index:
        molecule = load_molecule(
            df_total["InChIKey"][i], df_total["target"][i], db
        )
        data_list.append(molecule)
    return data_list



def train_val_test_split(dataset, config, smiles_list=None):
    seed = config["seed"]
    num_mols = len(dataset)
    np.random.seed(seed)
    all_idx = np.random.permutation(num_mols)

    Nmols = num_mols
    Ntrain = int(num_mols * config["train_ratio"])
    Nvalid = int(num_mols * config["valid_ratio"])
    Ntest = Nmols - (Ntrain + Nvalid)

    train_idx = all_idx[:Ntrain]
    valid_idx = all_idx[Ntrain : Ntrain + Nvalid]
    test_idx = all_idx[Ntrain + Nvalid :]

    print("train_idx: ", train_idx)
    print("valid_idx: ", valid_idx)
    print("test_idx: ", test_idx)
    # np.savez("customized_01", train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols
    train_dataset = [dataset[x] for x in train_idx]
    valid_dataset = [dataset[x] for x in valid_idx]
    test_dataset = [dataset[x] for x in test_idx]
    # Set dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    if not smiles_list:
        return train_loader, val_loader, test_loader
    else:
        train_smiles = [smiles_list[i] for i in train_idx]
        valid_smiles = [smiles_list[i] for i in valid_idx]
        test_smiles = [smiles_list[i] for i in test_idx]
        return (
            train_loader,
            val_loader,
            test_loader,
            (train_smiles, valid_smiles, test_smiles),
        )


if __name__ == "__main__":
    from argparse import ArgumentParser
    root = os.getcwd()
    argparser = ArgumentParser()
    argparser.add_argument(
        "--config_dir",
        type=str,
        default="",
        help="directory to config.json",
    )
    args = argparser.parse_args()
    config_dir = root + args.config_dir
    main(config_dir=config_dir)
