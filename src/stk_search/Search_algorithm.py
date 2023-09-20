# class to define the search algorithm
import numpy as np
import pandas as pd
import torch
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from botorch.models import FixedNoiseGP, SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
import os
import pickle
import stk
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Search_Algorithm:
    def __init__(self):
        pass


class random_search(Search_Algorithm):
    def __init__(self, seed=0):
        np.random.seed(seed)

    def suggest_element(
        self,
        search_space_df: pd.DataFrame = [],
        fitness_acquired: list = [],
        ids_acquired: list = [],
        bad_ids: list = [],
    ):
        # get the element
        search_space_df = search_space_df.drop(ids_acquired)
        searched_space_df = search_space_df.drop(bad_ids)
        # evaluate the element
        return float(np.random.choice(searched_space_df.index, 1, replace=False))

    def initial_suggestion(
        self,
        search_space_df: pd.DataFrame = [],
        num_elem_initialisation: int = 10,
    ):
        # get initial elements
        return list(
            np.random.choice(
                search_space_df.index, num_elem_initialisation, replace=False
            )
        )


class Bayesian_Optimisation(Search_Algorithm):
    def __init__(self):
        self.which_acquisition = "EI"
        self.kernel = TanimotoGP
        self.device = "cpu"
        self.likelihood = ExactMarginalLogLikelihood
        self.model = None
        self.verbose = False

    def prepare_data_for_BO(
        self,
        search_space_df: pd.DataFrame = [],
        fitness_acquired: list = [],
        ids_acquired: list = [],
    ):
        searched_space_df = search_space_df.loc[np.array(ids_acquired)]
        # print("search space df shape", search_space_df.shape)
        numeric_cols = searched_space_df.select_dtypes(
            include=["float64", "int64", "int32", "float32"]
        ).columns
        # put numeric data from dataframe to tensor
        X_explored_BO = torch.tensor(
            searched_space_df[numeric_cols].values,
            dtype=torch.float64,
            device=self.device,
        )
        X_unsqueezed = torch.tensor(
            search_space_df[numeric_cols].values,
            dtype=torch.float64,
            device=self.device,
        )
        X_explored_BO = (X_explored_BO - X_unsqueezed.min(axis=0).values) / (
            X_unsqueezed.max(axis=0).values - X_unsqueezed.min(axis=0).values
        )
        X_unsqueezed = (X_unsqueezed - X_unsqueezed.min(axis=0).values) / (
            X_unsqueezed.max(axis=0).values - X_unsqueezed.min(axis=0).values
        )
        # limit the dataframe to only the numeric data
        y_explored_BO = torch.tensor(
            fitness_acquired, dtype=torch.float64, device=self.device
        )
        # normalise the data
        if y_explored_BO.std() != 0:
            y_explored_BO_norm = (
                y_explored_BO - y_explored_BO.mean()
            ) / y_explored_BO.std()
        else:
            y_explored_BO_norm = y_explored_BO
        y_explored_BO_norm = y_explored_BO_norm.reshape(-1, 1)  # for the GP
        # set up acquisition function
        X_unsqueezed = X_unsqueezed.reshape(-1, 1, X_unsqueezed.shape[1])  # for the GP

        return X_explored_BO, y_explored_BO_norm, X_unsqueezed

    def suggest_element(
        self,
        search_space_df: pd.DataFrame = [],
        fitness_acquired: list = [],
        ids_acquired: list = [],
        bad_ids: list = [],
    ):
        # get the element

        X_explored_BO, y_explored_BO_norm, X_unsqueezed = self.prepare_data_for_BO(
            search_space_df, fitness_acquired, ids_acquired
        )
        # construct and fit GP model
        self.train_model(X_explored_BO, y_explored_BO_norm)
        # get the aquisition values

        acquisition_values = self.get_acquisition_values(
            self.model,
            best_f=y_explored_BO_norm.max().item(),
            X_unsqueezed=X_unsqueezed,
        )
        # select element to acquire with maximal aquisition value, which is not in the acquired set already
        ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
        if self.verbose:
            print("ids_sorted_by_aquisition", ids_sorted_by_aquisition[:5])
            print("ids_acquired", ids_acquired)
            print("bad_ids", bad_ids)
        for id in ids_sorted_by_aquisition:
            if self.verbose:
                print("id", id.item())
            if (
                search_space_df.index[id.item()] not in ids_acquired
                and search_space_df.index[id.item()] not in bad_ids
            ):
                index = id.item()
                return search_space_df.index[index]

    def get_acquisition_values(self, model, best_f, X_unsqueezed):
        # set up acquisition function
        if self.which_acquisition == "EI":
            acquisition_function = ExpectedImprovement(model, best_f=best_f)
            with torch.no_grad():  # to avoid memory issues; we arent using the gradient...
                acquisition_values = acquisition_function.forward(
                    X_unsqueezed
                )  # runs out of memory
        elif self.which_acquisition == "max_y_hat":
            with torch.no_grad():
                acquisition_values = model.posterior(X_unsqueezed).mean.squeeze()
        elif self.which_acquisition == "max_sigma":
            with torch.no_grad():
                acquisition_values = model.posterior(X_unsqueezed).variance.squeeze()
        elif self.which_acquisition == "LOG_EI":
            acquisition_function = LogExpectedImprovement(model, best_f=best_f)
            with torch.no_grad():  # to avoid memory issues; we arent using the gradient...
                acquisition_values = acquisition_function.forward(
                    X_unsqueezed
                )  # runs out of memory
        else:
            with torch.no_grad():
                acquisition_values = model.posterior(X_unsqueezed).variance.squeeze()
        return acquisition_values

    def initial_suggestion(
        self,
        search_space_df: pd.DataFrame = [],
        num_elem_initialisation: int = 10,
    ):
        return list(
            np.random.choice(
                search_space_df.index, num_elem_initialisation, replace=False
            )
        )

    def get_test_train_data_for_BO(
        self,
        search_space_df: pd.DataFrame = [],
        fitness_acquired: list = [],
        test_set_size: float = 0.2,
    ):
        def transform_data(X_train, y_train, X_test, y_test):
            """
            Apply feature scaling, dimensionality reduction to the data. Return the standardised and low-dimensional train and
            test sets together with the scaler object for the target values.

            :param X_train: input train data
            :param y_train: train labels
            :param X_test: input test data
            :param y_test: test labels
            :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
            """
            X_train_scaled = (X_train - X_test.min(axis=0).values) / (
                X_test.max(axis=0).values - X_test.min(axis=0).values
            )
            X_test_scaled = (X_test - X_test.min(axis=0).values) / (
                X_test.max(axis=0).values - X_test.min(axis=0).values
            )
            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train)
            y_test_scaled = y_scaler.transform(y_test)

            return (
                X_train_scaled,
                y_train_scaled,
                X_test_scaled,
                y_test_scaled,
                y_scaler,
            )

        X_explored = torch.tensor(
            search_space_df.values, dtype=torch.float64, device=self.device
        )
        # limit the dataframe to only the numeric data
        y_explored = torch.tensor(
            fitness_acquired, dtype=torch.float64, device=self.device
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_explored, y_explored, test_size=test_set_size, random_state=0
        )

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        #  We standardise the outputs but leave the inputs unchanged

        X_train, y_train, X_test, y_test, y_scaler = transform_data(
            X_train, y_train, X_test, y_test
        )
        return X_train, X_test, y_train, y_test, y_scaler

    def train_model(self, X_train, y_train):
        self.model = self.kernel(
            X_train, torch.tensor(y_train, dtype=torch.float64, device=self.device)
        )
        mll = self.likelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    def test_model_prediction(self, X_train, y_train, X_test, y_test, y_scaler):
        # X_train = torch.nn.functional.normalize(X_train, dim = 0)
        # X_test = torch.nn.functional.normalize(X_test, dim = 0)
        self.train_model(X_train, y_train)
        model = self.model
        y_pred = model.posterior(X_test).mean.tolist()
        y_var = model.posterior(X_test).variance.tolist()
        y_pred_train = model.posterior(X_train).mean.tolist()
        y_var_train = model.posterior(X_train).variance.tolist()
        y_pred = y_scaler.inverse_transform(y_pred)
        y_test = y_scaler.inverse_transform(y_test)
        y_train = y_scaler.inverse_transform(y_train.tolist())
        y_pred_train = y_scaler.inverse_transform(y_pred_train)
        return y_pred, y_var, y_pred_train, y_train, y_test, y_var_train


    def plot_prediction(self, y_pred, y_test, y_pred_train, y_train, y_var):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        def plot_prediction(y_pred, y_test, axis, label):
            axis.scatter(y_test, y_pred, marker="x", color="red")
            # axis.plot(y_test, y_test, linestyle="--", color="black")
            axis.set_xlabel("True values")
            axis.set_ylabel("Predicted values")
            score_r2 = r2_score(y_test, y_pred)
            print(f"R2 score: {score_r2:.2f}")
            score_mse = mean_squared_error(y_test, y_pred)
            print(f"MSE score: {score_mse:.2f}")
            score_mae = mean_absolute_error(y_test, y_pred)
            print(f"MAE score: {score_mae:.2f}")
            axis.set_title(
                label
                + "\n"
                + "R2 score: {:.2f}".format(score_r2)
                + "\n"
                + "MSE score: {:.2f}".format(score_mse)
                + "\n"
                + "MAE score: {:.2f}".format(score_mae)
            )

        plot_prediction(y_pred, y_test, axs[0], label="test set")
        plot_prediction(y_pred_train, y_train, axs[1], label="train set")
        axs[2].scatter(y_var, np.abs(y_test - y_pred), marker="x", color="red")
        axs[2].set_xlabel("Variance")
        axs[2].set_ylabel("Absolute error")
        plt.show()


from stk_search.tanimoto_kernel import TanimotoKernel


# We define our custom GP surrogate model using the Tanimoto kernel
class TanimotoGP(SingleTaskGP):
    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y, GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
