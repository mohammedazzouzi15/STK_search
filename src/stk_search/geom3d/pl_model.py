"""PyTorch Lightning model for 3D molecular representation learning."""

import lightning.pytorch as pl
import torch
import torch.nn.functional as Functional
from torch import optim

from stk_search.geom3d.models import (
    DimeNet,
    DimeNetPlusPlus,
    EquiformerEnergy,
    GemNet,
    PaiNN,
    SchNet,
    SphereNet,
)


class PrintLearningRate(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        trainer.optimizers[0].param_groups[0]["lr"]


class Pymodel(pl.LightningModule):
    """PyTorch Lightning model for 3D molecular representation learning.
    The loss function is the mean squared error (MSE) loss.
    The learning rate scheduler can be chosen from CosineAnnealingLR, CosineAnnealingWarmRestarts, and StepLR.
    The initial learning rate and the learning rate scheduler parameters can be set in the configuration file.

    Args:
    ----
    - model (nn.Module): 3D molecular representation learning model
    - graph_pred_linear (nn.Module): linear layer for graph prediction
    - config (dict): dictionary containing the configuration

    """

    def __init__(self, model, graph_pred_linear, config):
        super().__init__()
        self.save_hyperparameters(ignore=["graph_pred_linear", "model"])
        self.molecule_3D_repr = model
        self.graph_pred_linear = graph_pred_linear
        self.config = config

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        with torch.cuda.amp.autocast(
            enabled=self.trainer.precision == 16
        ):  # 16-bit precision for mixed precision training, activated only when self.trainer.precision == 16
            loss = self._get_preds_loss_accuracy(batch)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log("train_loss", loss, batch_size=batch.size(0))
        self.log(
            "lr",
            lr,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch.size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Used for logging metrics."""
        with torch.cuda.amp.autocast(
            enabled=self.trainer.precision == 16
        ):  # 16-bit precision for mixed precision training, activated only when self.trainer.precision == 16
            loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss, batch_size=batch.size(0))
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """Convenience function since train/valid/test steps are similar."""
        batch = batch.to(self.device)
        z = self.forward(batch)

        if self.graph_pred_linear is not None:
            loss = Functional.mse_loss(z, batch.y.unsqueeze(1))
        else:
            loss = Functional.mse_loss(z, batch.y)
        return loss

    def configure_optimizers(self):
        # set up optimizer
        # make sure the optimiser step does not reset the val_loss metrics

        config = self.config
        optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])

        lr_scheduler = None

        if config["lr_scheduler"] == "CosineAnnealingLR":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, config["max_epochs"]
            )
        elif config["lr_scheduler"] == "CosineAnnealingWarmRestarts":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, config["max_epochs"], eta_min=1e-4
            )
        elif config["lr_scheduler"] == "StepLR":
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["lr_decay_step_size"],
                gamma=config["lr_decay_factor"],
            )
        else:
            pass

        return [optimizer], [lr_scheduler]

        # optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        # return optimizer

    def forward(self, batch):
        batch = batch.to(self.device)
        model_name = type(self.molecule_3D_repr).__name__
        if model_name == "EquiformerEnergy":
            model_name = "Equiformer"

        if self.graph_pred_linear is not None:
            if model_name == "PaiNN":
                z = self.molecule_3D_repr(
                    batch.x,
                    batch.positions,
                    batch.radius_edge_index,
                    batch.batch,
                ).squeeze()
                z = self.graph_pred_linear(z)
            else:
                z = self.molecule_3D_repr(
                    batch.x, batch.positions, batch.batch
                )
                z = self.graph_pred_linear(z)
        elif model_name == "GemNet":
            z = self.molecule_3D_repr(
                batch.x, batch.positions, batch
            ).squeeze()
        elif model_name == "Equiformer":
            z = self.molecule_3D_repr(
                node_atom=batch.x, pos=batch.positions, batch=batch.batch
            ).squeeze()
        else:
            z = self.molecule_3D_repr(
                batch.x, batch.positions, batch.batch
            ).squeeze()
        return z


class Pymodel_new(pl.LightningModule):
    """lightning model taking into account two different oligomer representations."""

    def __init__(self, model, graph_pred_linear, config):
        super().__init__()

        self.save_hyperparameters(ignore=["graph_pred_linear", "model"])
        self.molecule_3D_repr = model
        self.graph_pred_linear = graph_pred_linear
        self.config = config
        self.transform_to_opt = torch.nn.Linear(config["emb_dim"], config["emb_dim"])


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        with torch.cuda.amp.autocast(
            enabled=self.trainer.precision == 16
        ):  # 16-bit precision for mixed precision training, activated only when self.trainer.precision == 16
            loss, loss1, loss2 = self._get_preds_loss_accuracy(batch)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train_loss", loss, batch_size=batch.size(0))
        self.log("train_loss_target", loss1, batch_size=batch.size(0))
        self.log("train_loss_repr", loss2, batch_size=batch.size(0))
        self.log(
            "lr",
            lr,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch.size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Used for logging metrics."""
        with torch.cuda.amp.autocast(
            enabled=self.trainer.precision == 16
        ):  # 16-bit precision for mixed precision training, activated only when self.trainer.precision == 16
            loss, loss1, loss2 = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss, batch_size=batch.size(0))
        self.log("val_loss_target", loss1, batch_size=batch.size(0))
        self.log("val_loss_repr", loss2, batch_size=batch.size(0))
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """Convenience function since train/valid/test steps are similar."""
        batch = batch.to(self.device)
        z, z_opt, z_repr, z_repr_opt = self.forward_train(batch)

        if self.graph_pred_linear is not None:
            loss1 = Functional.mse_loss(z_opt, batch.y.unsqueeze(1))
            loss2 = Functional.mse_loss(z_repr, z_repr_opt)
            # loss = loss + Functional.mse_loss(z, batch.y.unsqueeze(1))
            a = torch.tensor(0.5, requires_grad=True)
            loss = a * loss1 + (1 - a) * loss2
        else:
            loss = Functional.mse_loss(z, batch.y)
        return loss, loss1, loss2

    def forward_train(self, batch):

        batch = batch.to(self.device)
        model_name = type(self.molecule_3D_repr).__name__

        def get_Z(x, positions, model_name, batch):
            if model_name == "EquiformerEnergy":
                model_name = "Equiformer"

            if self.graph_pred_linear is not None:
                if model_name == "PaiNN":
                    z_repr = self.molecule_3D_repr(
                        x,
                        positions,
                        batch.radius_edge_index,
                        batch.batch,
                    ).squeeze()
                else:
                    z_repr = self.molecule_3D_repr(x, positions, batch.batch)

                return z_repr
            return None

        z_repr = get_Z(batch.x, batch.positions, model_name, batch)
        z_repr_opt = get_Z(
            batch.x_opt, batch.positions_opt, model_name, batch
        )
        z_repr = self.transform_to_opt(z_repr)
        z = self.graph_pred_linear(z_repr)
        z_opt = self.graph_pred_linear(z_repr_opt)

        return z, z_opt, z_repr, z_repr_opt
    def configure_optimizers(self):
        # set up optimizer
        # make sure the optimiser step does not reset the val_loss metrics

        config = self.config
        optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])

        lr_scheduler = None

        if config["lr_scheduler"] == "CosineAnnealingLR":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, config["max_epochs"]
            )
        elif config["lr_scheduler"] == "CosineAnnealingWarmRestarts":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, config["max_epochs"], eta_min=1e-4
            )
        elif config["lr_scheduler"] == "StepLR":
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["lr_decay_step_size"],
                gamma=config["lr_decay_factor"],
            )
        else:
            pass

        return [optimizer], [lr_scheduler]

        # optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        # return optimizer

    def forward(self, batch):
        batch = batch.to(self.device)
        model_name = type(self.molecule_3D_repr).__name__
        if model_name == "EquiformerEnergy":
            model_name = "Equiformer"

        if self.graph_pred_linear is not None:
            if model_name == "PaiNN":
                z = self.molecule_3D_repr(
                    batch.x,
                    batch.positions,
                    batch.radius_edge_index,
                    batch.batch,
                ).squeeze()
            else:
                z = self.molecule_3D_repr(
                    batch.x, batch.positions, batch.batch
                )
            z = self.transform_to_opt(z)
            z = self.graph_pred_linear(z)
        elif model_name == "GemNet":
            z = self.molecule_3D_repr(
                batch.x, batch.positions, batch
            ).squeeze()
        elif model_name == "Equiformer":
            z = self.molecule_3D_repr(
                node_atom=batch.x, pos=batch.positions, batch=batch.batch
            ).squeeze()
        else:
            z = self.molecule_3D_repr(
                batch.x, batch.positions, batch.batch
            ).squeeze()
        return z


def model_setup(config, trial=None):
    """Setup the model based on the configuration file.

    Args:
    ----
    - config (dict): configuration file
    - trial (optuna.trial): optuna trial object

    Returns:
    -------
    - model (nn.Module): model
    - graph_pred_linear (nn.Module): output layer for the model

    """
    model_config = config["model"]

    if trial:
        config = hyperparameter_setup(config, trial)

    if config["model_name"] == "SchNet":
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

    elif config["model_name"] == "DimeNet":
        model = DimeNet(
            node_class=model_config["node_class"],
            hidden_channels=model_config["hidden_channels"],
            out_channels=model_config["out_channels"],
            num_blocks=model_config["num_blocks"],
            num_bilinear=model_config["num_bilinear"],
            num_spherical=model_config["num_spherical"],
            num_radial=model_config["num_radial"],
            cutoff=model_config["cutoff"],
            envelope_exponent=model_config["envelope_exponent"],
            num_before_skip=model_config["num_before_skip"],
            num_after_skip=model_config["num_after_skip"],
            num_output_layers=model_config["num_output_layers"],
        )
        graph_pred_linear = None

    elif config["model_name"] == "DimeNetPlusPlus":
        model = DimeNetPlusPlus(
            node_class=model_config["node_class"],
            hidden_channels=model_config["hidden_channels"],
            out_channels=model_config["out_channels"],
            num_blocks=model_config["num_blocks"],
            int_emb_size=model_config["int_emb_size"],
            basis_emb_size=model_config["basis_emb_size"],
            out_emb_channels=model_config["out_emb_channels"],
            num_spherical=model_config["num_spherical"],
            num_radial=model_config["num_radial"],
            cutoff=model_config["cutoff"],
            envelope_exponent=model_config["envelope_exponent"],
            num_before_skip=model_config["num_before_skip"],
            num_after_skip=model_config["num_after_skip"],
            num_output_layers=model_config["num_output_layers"],
        )
        graph_pred_linear = None

    elif config["model_name"] == "GemNet":
        model = GemNet(
            node_class=model_config["node_class"],
            num_targets=model_config["num_targets"],
            num_blocks=model_config["num_blocks"],
            emb_size_atom=model_config["emb_size_atom"],
            emb_size_edge=model_config["emb_size_edge"],
            emb_size_trip=model_config["emb_size_trip"],
            emb_size_quad=model_config["emb_size_quad"],
            emb_size_rbf=model_config["emb_size_rbf"],
            emb_size_cbf=model_config["emb_size_cbf"],
            emb_size_sbf=model_config["emb_size_sbf"],
            emb_size_bil_quad=model_config["emb_size_bil_quad"],
            emb_size_bil_trip=model_config["emb_size_bil_trip"],
            num_concat=model_config["num_concat"],
            num_atom=model_config["num_atom"],
            triplets_only=model_config["triplets_only"],
            direct_forces=model_config["direct_forces"],
            extensive=model_config["extensive"],
            forces_coupled=model_config["forces_coupled"],
            cutoff=model_config["cutoff"],
            int_cutoff=model_config["int_cutoff"],
            envelope_exponent=model_config["envelope_exponent"],
            num_spherical=model_config["num_spherical"],
            num_radial=model_config["num_radial"],
            num_before_skip=model_config["num_before_skip"],
            num_after_skip=model_config["num_after_skip"],
        )
        graph_pred_linear = None

    elif config["model_name"] == "SphereNet":
        model = SphereNet(
            energy_and_force=False,
            hidden_channels=model_config["hidden_channels"],
            out_channels=model_config["out_channels"],
            cutoff=model_config["cutoff"],
            num_layers=model_config["num_layers"],
            int_emb_size=model_config["int_emb_size"],
            basis_emb_size_dist=model_config["basis_emb_size_dist"],
            basis_emb_size_angle=model_config["basis_emb_size_angle"],
            basis_emb_size_torsion=model_config["basis_emb_size_torsion"],
            out_emb_channels=model_config["out_emb_channels"],
            num_spherical=model_config["num_spherical"],
            num_radial=model_config["num_radial"],
            envelope_exponent=model_config["envelope_exponent"],
            num_before_skip=model_config["num_before_skip"],
            num_after_skip=model_config["num_after_skip"],
            num_output_layers=model_config["num_output_layers"],
        )
        graph_pred_linear = None

    elif config["model_name"] == "PaiNN":
        model = PaiNN(
            n_atom_basis=model_config["n_atom_basis"],
            n_interactions=model_config["n_interactions"],
            n_rbf=model_config["n_rbf"],
            cutoff=model_config["cutoff"],
            max_z=model_config["max_z"],
            n_out=model_config["n_out"],
            readout=model_config["readout"],
        )
        graph_pred_linear = model.create_output_layers()

    elif config["model_name"] == "Equiformer":
        if config["model"]["Equiformer_hyperparameter"] == 0:
            # This follows the hyper in Equiformer_l2
            model = EquiformerEnergy(
                irreps_in=model_config["Equiformer_irreps_in"],
                max_radius=model_config["Equiformer_radius"],
                node_class=model_config["node_class"],
                number_of_basis=model_config["Equiformer_num_basis"],
                irreps_node_embedding=model_config["irreps_node_embedding"],
                num_layers=6,
                irreps_node_attr="1x0e",
                irreps_sh="1x0e+1x1e+1x2e",
                fc_neurons=[32, 32],
                irreps_feature="256x0e",
                irreps_head="32x0e+16x1e+8x2e",
                num_heads=2,
                irreps_pre_attn=None,
                rescale_degree=False,
                nonlinear_message=False,
                irreps_mlp_mid="192x0e+96x1e+48x2e",
                norm_layer="layer",
                alpha_drop=0.3,
                proj_drop=0.1,
                out_drop=0.1,
                drop_path_rate=0.1,
            )
        elif config["model"]["Equiformer_hyperparameter"] == 1:
            # This follows the hyper in Equiformer_nonlinear_bessel_l2_drop00
            model = EquiformerEnergy(
                irreps_in=model_config["Equiformer_irreps_in"],
                max_radius=model_config["Equiformer_radius"],
                node_class=model_config["node_class"],
                number_of_basis=model_config["Equiformer_num_basis"],
                irreps_node_embedding=model_config["irreps_node_embedding"],
                num_layers=6,
                irreps_node_attr="1x0e",
                irreps_sh="1x0e+1x1e+1x2e",
                fc_neurons=[64, 64],
                basis_type="bessel",
                irreps_feature="512x0e",
                irreps_head="32x0e+16x1e+8x2e",
                num_heads=4,
                irreps_pre_attn=None,
                rescale_degree=False,
                nonlinear_message=True,
                irreps_mlp_mid="384x0e+192x1e+96x2e",
                norm_layer="layer",
                alpha_drop=0.0,
                proj_drop=0.0,
                out_drop=0.0,
                drop_path_rate=0.0,
            )
        graph_pred_linear = None

    else:
        msg = "Invalid model name"
        raise ValueError(msg)

    return model, graph_pred_linear
