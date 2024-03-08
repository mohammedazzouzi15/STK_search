"""
script to train the SchNet model on the STK dataset
created by Mohammed Azzouzi
date: 2023-11-14
"""

import numpy as np
import os
import glob
import time
import wandb
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from stk_search.geom3d.dataloader import (
    load_data,
    train_val_split,
    load_3d_rpr,
)

from stk_search.utils.config_utils import read_config, save_config
from stk_search.geom3d.pl_model import Pymodel, model_setup
from stk_search.geom3d import pl_model


def main(config_dir):
    """Train the model using the given configuration.
    Args:
        config_dir (str): The path to the directory containing the
            configuration file.

    """
    config = read_config(config_dir)
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_data(config)
    train_loader, val_loader = train_val_split(dataset, config=config)

    load_and_run_model_training(config, train_loader, val_loader)
    # load dataframe with calculated data


def load_and_run_model_training(config, train_loader, val_loader):
    """Load the model and train it using the given data loaders.

    Args:
        config (dict): The configuration of the model.
        train_loader (DataLoader): The data loader for the training data.
        val_loader (DataLoader): The data loader for the validation data.
    """
    start_time = time.time()  # Record the start time
    model, graph_pred_linear = pl_model.model_setup(config)
    print("Model loaded: ", config["model_name"])

    if config["model_path"]:
        model = load_3d_rpr(model, config["model_path"])
    os.chdir(config["running_dir"])
    # wandb.login()
    # wandb.init(settings=wandb.Settings(start_method="fork"))
    # model
    # check if chkpt exists
    pymodel = Pymodel(model, graph_pred_linear, config)
    if os.path.exists(config["model_embedding_chkpt"]):
        chkpt_path = config["model_embedding_chkpt"]
        checkpoint = torch.load(chkpt_path, map_location=config["device"])
        print("Model loaded: ", config["model_embedding_chkpt"])
        # Pass the model and graph_pred_linear to the Pymodel constructor
        # Load the state dictionary
        pymodel.load_state_dict(state_dict=checkpoint["state_dict"])    


    wandb_logger = WandbLogger(
        log_model=True,
        project=config["name"].split("__")[0],
        name=config["name"].split("__")[1],
        settings=wandb.Settings(start_method="fork"),
    )
    wandb_logger.log_hyperparams(config)

    # train model
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["running_dir"],
        filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode="min",
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=config["max_epochs"],
        val_check_interval=1.0,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    trainer.fit(
        model=pymodel,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    wandb.finish()
    end_time = time.time()  # Record the end time
    total_time = end_time - start_time
    print(f"Total time taken for model training: {total_time} seconds")


def get_best_embedding_model(config_dir):
    """Get the best model from the given directory.
    Args:
        config_dir (str): The path to the directory containing the
            configuration file.
    Returns:
            dict: The configuration of the best model."""
    config = read_config(config_dir, model_name="SchNet")
    files = glob.glob(config_dir + "/*.ckpt")
    min_val_loss = 1000
    for file in files:
        # get val loss from file name
        val_loss = float(file.split("val_loss=")[1].split("-")[0])
        if val_loss < min_val_loss:
            print(file)
            min_val_loss = val_loss
            config["model_embedding_chkpt"] = file
    save_config(config, config_dir)
    return config, min_val_loss



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
    config_dir = args.config_dir
    main(config_dir=config_dir)
