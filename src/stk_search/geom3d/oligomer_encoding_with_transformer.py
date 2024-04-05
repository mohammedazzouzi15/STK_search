"""
this script is to encode the representation of the oligomer from the representation of the fragments
"""

import numpy as np
import os
import wandb
import torch
import lightning.pytorch as pl
import torch.nn.functional as Functional
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from stk_search.geom3d.transformer_utils import TransformerPredictor
from stk_search.utils.config_utils import  save_config
from torch_geometric.data import Data
from stk_search.geom3d.pl_model import Pymodel, model_setup
import glob



def run_encoding_training(config, train_loader, val_loader):
    """Load the model and train it using the given data loaders.

    Args:
        config (dict): The configuration of the model.
        train_loader (DataLoader): The data loader for the training data.
        val_loader (DataLoader): The data loader for the validation data.
    """
    max_iters = config["max_epochs"] * len(train_loader)
    # model_config = config["model"]
    EncodingModel = initialise_model(config, max_iters)
    print(" config name", config["name"])
    wandb_logger = WandbLogger(
        log_model=True,
        project="encoding_" + config["name"].split("__")[0],
        name=config["name"].split("__")[1],
        settings=wandb.Settings(start_method="fork"),
    )
    wandb_logger.log_hyperparams(config)
    # train model
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["running_dir"] + "/transformer",
        filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
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
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
    )
    trainer.fit(
        model=EncodingModel,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    wandb.finish()
    return EncodingModel


def save_encoding_dataset(dataset, config, dataset_name="",save_folder=""):
    """Save the encoding of the dataset
    Args:
        dataset (DataLoader): The data loader for the training data.
        config (dict): The configuration of the model.
        dataset_name (str): The name of the dataset

    """
    Checkpoint_dir = config["running_dir"] + "/transformer"

    files = glob.glob(Checkpoint_dir + "/*.ckpt")
    min_val_loss = 1000
    for file in files:
        # get val loss from file name
        val_loss = float(file.split("val_loss=")[1].split("-")[0])
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            config["model_transformer_chkpt"] = file
    save_config(config, config["running_dir"])

    EncodingModel = initialise_model(config)
    EncodingModel.eval()
    EncodingModel = EncodingModel.to(config["device"])
    data_list = []
    counter = 0
    for data_input in dataset:
        with torch.no_grad():

            learned_rpr_data = EncodingModel(
                [x.to(config["device"]) for x in data_input]
            )
            molecule_frag = Data(
                learned_rpr=learned_rpr_data[0][0].type(torch.float16),
                rpr_opt=data_input[0].y.type(torch.float16),
                InChIKey=data_input[0].InChIKey,
            )
            data_list.append(molecule_frag.detach().cpu())
        counter += 1
        if counter % 1000 == 0:
            print(counter)
            if save_folder == "":
                torch.save(
                    data_list,
                    config["running_dir"]
                    + f"/transformer/dataset_representation{dataset_name}.pt",
                )
            else:
                torch.save(
                    data_list,
                    save_folder
                    + f"/dataset_representation{dataset_name}.pt",
                )
    if save_folder == "":
        torch.save(
            data_list,
            config["running_dir"]
            + f"/transformer/dataset_representation{dataset_name}.pt",
        )
    else:
        torch.save(
            data_list,
            save_folder
            + f"/dataset_representation{dataset_name}.pt",
        )

    return data_list


def initialise_model(config, max_iters=10):
    """Initialise the model
    Args:
        config (dict): The configuration of the model.
        max_iters (int): The maximum number of iterations.
    Returns:
        Fragment_encoder: The initialised model.
    """
    # config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    model, graph_pred_linear = model_setup(config)
    pymodel = Pymodel(model, graph_pred_linear, config)

    if os.path.exists(config["model_embedding_chkpt"]):
        chkpt_path = config["model_embedding_chkpt"]
        checkpoint = torch.load(chkpt_path, map_location=config["device"])
        print("Model loaded: ", config["model_embedding_chkpt"])
        # Pass the model and graph_pred_linear to the Pymodel constructor
        # Load the state dictionary
        pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
    # model

    max_oligomer_size = 10
    if config["model_name"] == "PaiNN":
        num_classes = config["model"]["n_atom_basis"]
    else:
        num_classes = config["model"]["emb_dim"]
    print("lr", config["lr"])
    EncodingModel = Fragment_encoder(
        input_dim=num_classes * max_oligomer_size,
        model_dim=num_classes,
        num_heads=1,
        num_classes=num_classes,
        num_layers=1,
        dropout=0.0,
        lr=config["lr_encoder"],
        warmup=50,
        max_iters=max_iters,
    )
    EncodingModel.add_encoder(
        pymodel.molecule_3D_repr, batch_size=config["batch_size"]
    )
    EncodingModel.add_graph_pred_linear(pymodel.graph_pred_linear)
    EncodingModel.model_name = config["model_name"]
    if (
        os.path.exists(f"{config['model_transformer_chkpt']}")
        and config["model_transformer_chkpt"] != ""
    ):
        try:
            print(
                "loading model from checkpoint",
                f"{config['model_transformer_chkpt']}",
            )
            state_dict = torch.load(
                f"{config['model_transformer_chkpt']}",
                map_location=config["device"],
            )
            EncodingModel.load_state_dict(state_dict["state_dict"])
        except Exception as e:
            print("loading model from checkpoint failed")
            print(e)
            # delete the checkpoint
            # os.remove(config["model_transformer_chkpt"])
    else:
        print("no checkpoint found for encoding model")
    # print(EncodingModel.hparams.model_dim)
    return EncodingModel


class Fragment_encoder(TransformerPredictor):
    def add_encoder(self, model_encoder, batch_size):
        self.model_encoder = model_encoder
        self.hparams.batch_size = batch_size

    def add_graph_pred_linear(self, graph_pred_linear):
        self.graph_pred_linear = graph_pred_linear
        # set the graph_pred_linear to eval mode
        # self.graph_pred_linear.eval()
        # freeze all the parameters
        for param in self.graph_pred_linear.parameters():
            param.requires_grad = False

    def forward(self, batch, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        if self.model_encoder is not None:
            x = torch.zeros(
                self.hparams.batch_size,
                self.hparams.input_dim,
                device=self.device,
            )
            for i, b in enumerate(batch):
                if self.model_name == "PaiNN":
                    if b.batch is None:
                        b.batch = torch.zeros_like(b.x)
                    x[
                        :,
                        i
                        * self.hparams.num_classes : (i + 1)
                        * self.hparams.num_classes,
                    ] = self.model_encoder(
                        b.x, b.positions, b.radius_edge_index, b.batch
                    )
                else:
                    x[
                        :,
                        i
                        * self.hparams.num_classes : (i + 1)
                        * self.hparams.num_classes,
                    ] = self.model_encoder(b.x, b.positions, b.batch)
        else:
            x = batch.x
        x = x.unsqueeze_(0)
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x

    def _calculate_loss(self, batch, mode="train"):
        """Calculate the loss for the given batch.
        Args:
            batch: The batch of data.
            mode: The mode of the model (train, val, test).
            Returns:
            loss: The loss for the given batch.
        """
        # Fetch data and transform categories to one-hot vectors
        inp_data, labels = batch, batch[0].y.squeeze()

        # inp_data = F.one_hot(inp_data, num_classes=self.hparams.num_classes).float()

        # Perform prediction and calculate loss and accuracy
        preds = self.forward(inp_data, add_positional_encoding=True)
        # normalise the prediction
        preds_target = self.graph_pred_linear(preds.view(-1, preds.size(-1)))
        labels_target_pred = self.graph_pred_linear(labels)
        loss1 = Functional.mse_loss(preds_target, labels_target_pred)
        loss2 = Functional.mse_loss(
            preds.view(-1, preds.size(-1)), labels
        )
        loss = loss1 + 10*loss2 
        # print (labels.shape, preds.argmax(dim=-1).shape)
        # acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logging
        self.log("%s_loss" % mode, loss)
        self.log("%s_loss1" % mode, loss1)
        self.log("%s_loss2" % mode, loss2)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="test")


