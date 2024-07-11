from stk_search.Search_algorithm import Bayesian_Optimisation
from stk_search.Search_algorithm import (
    Represenation_3D,
)
from stk_search.geom3d.train_models import model_setup, Pymodel, read_config
from stk_search.geom3d.utils.config_utils import read_config, save_config
from stk_search.geom3d.utils import database_utils
from stk_search.geom3d.oligomer_encoding_with_transformer import Fragment_encoder, initialise_model
from stk_search.geom3d.models import SchNet
import torch
import stk
import pymongo
from pathlib import Path
import stk_search
from stk_search import SearchedSpace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader


def load_data(
    df_path="data/output/Full_datatset/df_total_new2023_08_20.csv",
    df_precursors_path="data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl",
):
    df_total, df_precursors = database_utils.load_data_from_file(
        df_path, df_precursors_path
    )
    SP = Searched_pace.Searched_Space(
        number_of_fragments=6,
        df=df_precursors,
        features_frag=df_precursors.columns[0:1],
        generation_type="conditional",
    )
    searched_space_df = SP.check_df_for_element_from_SP(df_to_check=df_total)
    fitness_acquired = searched_space_df["target"].values
    searched_space_df_InChIKey = searched_space_df[["InChIKey"]]
    searched_space_df = searched_space_df[[f"InChIKey_{x}" for x in range(6)]]
    return (
        df_total,
        df_precursors,
        searched_space_df,
        fitness_acquired,
        searched_space_df_InChIKey,
    )


def load_models(config, chkpt_path=None):
    EncodingModel = initialise_model(config)
    if chkpt_path is not None:
        # load pymodel
        # to get the try and except start indent here
        checkpoint = torch.load(chkpt_path,map_location=config["device"])
        model, graph_pred_linear = model_setup(config)
        print("Model loaded: ", config["model_name"])
        #config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Pass the model and graph_pred_linear to the Pymodel constructor

        pymodel = Pymodel(model, graph_pred_linear)
        # Load the state dictionary
        pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
        # Set the model to evaluation mode
        pymodel.eval()
        model_embedding = pymodel.molecule_3D_repr
        model_inferrence = pymodel.graph_pred_linear
        return EncodingModel, model_embedding, model_inferrence
    else:
        return EncodingModel


def PredictTargetFromEmbedding(
    data, model_inferecence, device=torch.device("cpu")
):
    data = data.to(device)
    model = model_inferecence.to(device)
    model.eval()
    with torch.no_grad():
        out = model(data)
    return out.squeeze()


def generate_train_val_data(
    dataset,
    EncodingModel,
    model_inferrence,
    config,
    target_name,
    aim,
    df_total,
    model_embedding=None

):
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    X_explored_frag=torch.tensor([], device=config["device"])
    X_explored_org=torch.tensor([], device=config["device"])
    y_explored=torch.tensor([], device=config["device"])
    y_predicted = torch.tensor([], device=config["device"])
    y_predicted_org = torch.tensor([], device=config["device"])
    for x in loader:
        with torch.no_grad():
            EncodingModel.to(config["device"])
            representation = EncodingModel(x)
            representation = representation.squeeze()
            model_inferrence.to(config["device"])
            Y_pred = model_inferrence(representation.to(config["device"]))
            # add y_pred from org representation
            y_pred_org = model_inferrence(x[0].y.to(config["device"]))
            X_explored_frag =torch.cat((X_explored_frag, representation), dim=0)
            if model_embedding is None:
                X_explored_org =torch.cat((X_explored_org, x[0].y), dim=0)
            else:
                X_explored_org =torch.cat((X_explored_org, model_embedding(x)), dim=0)
            y_predicted = torch.cat((y_predicted, Y_pred), dim=0)
            y_predicted_org = torch.cat((y_predicted_org, y_pred_org), dim=0)
        df_dataset = pd.DataFrame(x[0]["InChIKey"], columns=["InChIKey"])
        df_dataset = df_dataset.merge(df_total, on="InChIKey", how="left")
        df_dataset[target_name] = -np.sqrt((df_dataset[target_name] - aim) ** 2)
        y_explored = torch.cat((y_explored, torch.tensor(df_dataset[target_name].values,
            dtype=torch.float32,
            device=config["device"])), dim=0)


    return X_explored_frag, X_explored_org, y_explored, y_predicted,y_predicted_org


def generate_test_val_data(
    dataset,
    df_total,
    df_precursors,
    config,
    EncodingModel,
    model_embedding,
    target_name="target",
    model_inferrence=None,
    aim=0,
):
    client = pymongo.MongoClient(config["pymongo_client"])
    db_poly = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    db_frag = stk.MoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    SP = Searched_pace.Searched_Space(
        number_of_fragments=6,
        df=df_precursors,
        features_frag=df_precursors.columns[0:1],
        generation_type="conditional",
    )
    df_dataset = pd.DataFrame(
        [x["InChIKey"] for x in dataset], columns=["InChIKey"]
    )
    df_dataset = df_dataset.merge(df_total, on="InChIKey", how="left")
    df_dataset[target_name] = -np.sqrt((df_dataset[target_name] - aim) ** 2)
    searched_space_df = SP.check_df_for_element_from_SP(df_to_check=df_dataset)

    y_true = searched_space_df[target_name].values
    searched_space_df = searched_space_df[[f"InChIKey_{x}" for x in range(6)]]
    Representation = Represenation_3D.Representation3DFrag_transformer(
        EncodingModel,
        df_total,
        db_poly=db_poly,
        db_frag=db_frag,
        device=config["device"],
    )
    X_explored_frag = Representation.generate_repr(searched_space_df)
    # generate orginal representation
    Representation_org = Represenation_3D.Representation3D(
        model_embedding,
        df_total,
        data=None,
        db_poly=db_poly,
        device=config["device"],
    )
    X_explored_org = Representation_org.generate_repr(searched_space_df)
    y_explored = torch.tensor(
        y_true, dtype=torch.float32, device=config["device"]
    )
    y_predicted = model_inferrence(X_explored_frag)
    y_predicted_org = model_inferrence(X_explored_org)
    return X_explored_frag, X_explored_org, y_explored, y_predicted,y_predicted_org


# plot train
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def plot_inference_test(y_explored, X_explored_frag, X_explored_org, 
                        model_inferrence, ax):
    y_true = y_explored.cpu().numpy()
    Y_pred = (
        PredictTargetFromEmbedding(
            X_explored_frag, model_inferecence=model_inferrence
        )
        .cpu()
        .numpy()
    )
    Y_pred_org = (
        PredictTargetFromEmbedding(
            X_explored_org, model_inferecence=model_inferrence
        )
        .cpu()
        .numpy()
    )
    ax.scatter(y_true, Y_pred, label="original embedding")
    ax.scatter(y_true, Y_pred_org, label="learned embedding")
    ax.legend()
    ax.set_xlabel("calculated fitness")
    ax.set_ylabel("predicted fitness")
    ax.set_title("training data", fontsize=20)
    score_list = []
    try:
        score = r2_score(y_true, Y_pred_org)
        ax.text(
            0.7,
            0.15,
            f"original R2 score: {score:.2f}",
            transform=ax.transAxes,
        )
        score_list.append(score)
        score = mean_squared_error(y_true, Y_pred_org)
        ax.text(
            0.7, 0.125, f"original MSE: {score:.2f}", transform=ax.transAxes
        )
        score_list.append(score)
        score = mean_absolute_error(y_true, Y_pred_org)
        ax.text(0.7, 0.1, f"original MAE: {score:.2f}", transform=ax.transAxes)
        score_list.append(score)
        score = r2_score(y_true, Y_pred)
        ax.text(
            0.4, 0.15, f"learned R2 score: {score:.2f}", transform=ax.transAxes
        )
        score_list.append(score)
        score = mean_squared_error(y_true, Y_pred)
        ax.text(
            0.4, 0.125, f"learned MSE: {score:.2f}", transform=ax.transAxes
        )
        score_list.append(score)
        score = mean_absolute_error(y_true, Y_pred)
        ax.text(0.4, 0.1, f"learned MAE: {score:.2f}", transform=ax.transAxes)
        score_list.append(score)
        return score_list
    except ValueError as e:
        print(e)
        print("ValueError")
        return []


from gpytorch.likelihoods import GaussianLikelihood
from gpytorch import kernels

from stk_search.Search_algorithm.tanimoto_kernel import TanimotoKernel
from stk_search.Search_algorithm import Bayesian_Optimisation
from botorch.models.gp_regression import SingleTaskGP

# from stk_search.tanimoto_kernel import TanimotoKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean



# We define our custom GP surrogate model using the Tanimoto kernel
class TanimotoGP(SingleTaskGP):
    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MaternKernel(SingleTaskGP):
    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=kernels.MaternKernel()
        )  # kernels.RBFKernel())#)
        self.to(train_X)  # make sure we're on the right device/dtype

    def change_kernel(self, kernel):
        self.covar_module = ScaleKernel(base_kernel=kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def plot_prediction(
    y_pred,
    y_test,
    y_pred_train,
    y_train,
    y_var,
    fig=None,
    axs=None,
    save_plot=False,
    plot_name="prediction.png",
):
    def plot_prediction(y_pred, y_test, axis, label):
        axis.scatter(
            y_test.detach().numpy(),
            y_pred.detach().numpy(),
            marker="x",
            color="red",
        )
        # axis.plot(y_test, y_test, linestyle="--", color="black")
        axis.set_xlabel("True values")
        axis.set_ylabel("Predicted values")
        score_r2 = r2_score(y_test.detach().numpy(), y_pred.detach().numpy())
        score_mse = mean_squared_error(
            y_test.detach().numpy(), y_pred.detach().numpy()
        )
        score_mae = mean_absolute_error(
            y_test.detach().numpy(), y_pred.detach().numpy()
        )
        axis.text(
            0.1, 0.9, f"R2 score: {score_r2:.2f}", transform=axis.transAxes
        )
        axis.text(
            0.1, 0.85, f"MSE score: {score_mse:.2f}", transform=axis.transAxes
        )
        axis.text(
            0.1, 0.8, f"MAE score: {score_mae:.2f}", transform=axis.transAxes
        )

        axis.set_title(label)
        axis.set_xlim(
            min(y_test.detach().numpy().min(), y_pred.detach().numpy().min()),
            max(y_test.detach().numpy().max(), y_pred.detach().numpy().max()),
        )
        axis.set_ylim(
            min(y_test.detach().numpy().min(), y_pred.detach().numpy().min()),
            max(y_test.detach().numpy().max(), y_pred.detach().numpy().max()),
        )
        return {"mae": score_mae, "mse": score_mse, "r2": score_r2}

    scores_test = plot_prediction(y_pred, y_test, axs[0], label="test set")
    scores_train = plot_prediction(
        y_pred_train, y_train, axs[1], label="train set"
    )
    y_var = np.array(y_var.detach().numpy())
    data_plot = []
    spacing_array = np.linspace(y_var.min(), y_var.max(), 30)
    spacing = spacing_array[1] - spacing_array[0]
    for x in spacing_array:
        tes = [y_var > x] and [y_var < x + spacing]
        data_plot.append(
            np.abs(y_pred.detach().numpy() - y_test.detach().numpy())[
                tes[0]
            ].mean()
        )
    # add a ligne witht eh max of the training set in the plot
    max_train_value = y_train.detach().numpy().max()
    axs[0].axhline(max_train_value, color="blue", linestyle="--")

    axs[2].scatter(
        y_var,
        np.abs(y_test.detach().numpy() - y_pred.detach().numpy()),
        marker="x",
        color="red",
        label="error",
    )
    axs[2].plot(
        spacing_array,
        data_plot,
        linestyle="--",
        color="black",
        label="mean error",
    )
    axs[2].set_xlabel("Variance")
    axs[2].set_ylabel("Absolute error")
    axs[2].legend()
    fig.tight_layout()
    if save_plot:
        dir_name = "data/figures/test_BO/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig.savefig(dir_name + plot_name)
    return fig, axs


def run_training_BO_torch(
    BO,
    X_explored_train,
    y_explored_train,
    X_explored_test,
    y_explored_test,
    kernel=MaternKernel,
):
    def normalise_output(y):
        y_mean = y.mean()
        y_std = y.std()
        y = (y - y_mean) / y_std
        return y, y_mean, y_std

    def unnorm_output(y, y_mean, y_std):
        y = y * y_std + y_mean
        return y

    print(kernel)
    BO.kernel = kernel
    X_train = X_explored_train.cpu().type(torch.float64)
    y_train = y_explored_train.cpu().type(torch.float64).reshape(-1, 1)
    X_test = X_explored_test.cpu().type(torch.float64)
    y_test = y_explored_test.cpu().type(torch.float64).reshape(-1, 1)
    y_train, y_mean, y_std = normalise_output(y_train)

    data = {
        "X_train": X_train.cpu(),
        "X_test": X_test.cpu(),
        "y_train": y_train,
        "y_test": y_test,
    }
    (
        y_pred,
        y_var,
        y_pred_train,
        y_var_train,
    ) = BO.test_model_prediction(
        data["X_train"],
        data["y_train"],
        data["X_test"],
    )
    fig_name = (
        f"trainsize_"
        + str(data["X_train"].shape[0])
        + "_testSi_"
        + str(data["X_test"].shape[0])
        + "_botorch.png"
    )
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    y_pred = unnorm_output(y_pred, y_mean, y_std)
    y_pred_train = unnorm_output(y_pred_train, y_mean, y_std)
    y_var = y_var*y_std
    # y_test = unnorm_output(y_test, y_mean, y_std)
    y_train = unnorm_output(y_train, y_mean, y_std)
    fig, axs = plot_prediction(
        y_pred,
        y_test,
        y_pred_train,
        y_train,
        y_var,
        fig,
        axs,
        save_plot=False,
        plot_name=fig_name,
    )
    return fig, axs
