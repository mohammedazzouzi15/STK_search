# class to define the search algorithm
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
)
from botorch.models import SingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stk_search.Search_algorithm.Search_algorithm import Search_Algorithm
from stk_search.Search_space import Search_Space
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


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
        def remov_similar_columns(df, numeric_cols):
            repr_array = df[numeric_cols].values
            repr_array = repr_array[
                :, ~(repr_array == repr_array[0, :]).all(0)
            ]  # remove columns with all the same values
            return repr_array

        searched_space_df = search_space_df.loc[np.array(ids_acquired)]
        # print("search space df shape", search_space_df.shape)
        numeric_cols = searched_space_df.select_dtypes(
            include=["float64", "int64", "int32", "float32", "float16"]
        ).columns

        # put numeric data from dataframe to tensor
        X_explored_BO = torch.tensor(
            remov_similar_columns(searched_space_df, numeric_cols),
            dtype=torch.float64,
            device=self.device,
        )
        X_unsqueezed = torch.tensor(
            remov_similar_columns(search_space_df, numeric_cols),
            dtype=torch.float64,
            device=self.device,
        )
        X_explored_BO = (X_explored_BO - X_explored_BO.min(axis=0).values) / (
            X_explored_BO.max(axis=0).values - X_explored_BO.min(axis=0).values
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
        X_unsqueezed = X_unsqueezed.reshape(
            -1, 1, X_unsqueezed.shape[1]
        )  # for the GP

        return X_explored_BO, y_explored_BO_norm, X_unsqueezed

    def suggest_element(
        self,
        search_space_df,
        fitness_acquired,
        SP : Search_Space,
        benchmark=True,
        df_total: pd.DataFrame = None,
    ):
        df_search = search_space_df.copy()
        fitness_acquired = np.array(fitness_acquired)
        X_rpr = self.Representation.generate_repr(df_search.values)
        X_rpr = X_rpr.double()
        y_explored_BO_norm = torch.tensor(
            fitness_acquired, dtype=torch.float64
        )
        y_explored_BO_norm = (
            y_explored_BO_norm - y_explored_BO_norm.mean(axis=0)
        ) / (y_explored_BO_norm.std(axis=0))
        y_explored_BO_norm = y_explored_BO_norm.reshape(-1, 1)
        self.train_model(X_rpr, y_explored_BO_norm)

        def mutate_element(element):
            elements = []
            for i in range(6):
                for frag in SP.df_precursors.InChIKey:
                    element_new = element.copy()
                    element_new[i] = frag
                    elements.append(element_new)
            return elements

        best_element_arg = fitness_acquired.argsort()[-3:][::-1]
        elements = []
        for element in df_search.loc[best_element_arg].values:
            if len(elements) == 0:
                elements = mutate_element(element)
            else:
                elements = np.append(elements, mutate_element(element), axis=0)

        elements = np.append(elements, df_search.values, axis=0)
        df_elements = pd.DataFrame(
            elements, columns=[f"InChIKey_{x}" for x in range(6)]
        )
        df_elements = SP.check_df_for_element_from_SP(df_to_check=df_elements)
        if benchmark:
            # take only element in df_total
            df_elements = df_elements.merge(
                df_total,
                on=[f"InChIKey_{i}" for i in range(6)],
                how="left",
            )
            df_elements.dropna(subset="target", inplace=True)
            df_elements = df_elements[[f"InChIKey_{i}" for i in range(6)]]
            print(df_elements.shape)
        X_unsqueezed = self.Representation.generate_repr(df_elements.values)
        X_unsqueezed = X_unsqueezed.double()
        X_unsqueezed = X_unsqueezed.reshape(-1, 1, X_unsqueezed.shape[1])
        acquisition_values = self.get_acquisition_values(
            self.model,
            best_f=y_explored_BO_norm.max().item(),
            X_unsqueezed=X_unsqueezed,
        )
        # select element to acquire with maximal aquisition value, which is not in the acquired set already
        ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)

        def add_element(df, element):
            if ~(df == element).all(1).any():
                df.loc[len(df)] = element
                return True
            return False

        print(df_search.shape)
        for id in ids_sorted_by_aquisition:
            if add_element(df_search, df_elements.values[id.item()]):
                print(id.item())
                break
                # index = id.item()
                # return df_search_space_frag
        print(df_search.shape)
        return len(df_search) - 1, df_search

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
                acquisition_values = model.posterior(
                    X_unsqueezed
                ).mean.squeeze()
        elif self.which_acquisition == "max_sigma":
            with torch.no_grad():
                acquisition_values = model.posterior(
                    X_unsqueezed
                ).variance.squeeze()
        elif self.which_acquisition == "LOG_EI":
            acquisition_function = LogExpectedImprovement(model, best_f=best_f)
            with torch.no_grad():  # to avoid memory issues; we arent using the gradient...
                acquisition_values = acquisition_function.forward(
                    X_unsqueezed
                )  # runs out of memory
        else:
            with torch.no_grad():
                acquisition_values = model.posterior(
                    X_unsqueezed
                ).variance.squeeze()
        return acquisition_values

    def initial_suggestion(
        self,
        SP: Search_Space = [],
        num_elem_initialisation: int = 10,
        benchmark=False,
        df_total: pd.DataFrame = None,
    ):
        if benchmark:
            searched_space_df = SP.check_df_for_element_from_SP(df_to_check=df_total)
            searched_space_df= searched_space_df.sample(num_elem_initialisation)
        else:
            searched_space_df = SP.random_generation_df(num_elem_initialisation)
        # reindex the df
        searched_space_df = searched_space_df[['InChIKey_'+str(i) for i in range(SP.number_of_fragments)]] # careful here, this is hard coded
        searched_space_df.index = range(len(searched_space_df))
        return searched_space_df.index.tolist() , searched_space_df


    def get_test_train_data_for_BO(
        self,
        search_space_df: pd.DataFrame = [],
        fitness_acquired: list = [],
        test_set_size: float = 0.2,
    ):
        def transform_data(
            X_train, y_train, X_test, y_test, X_explored, y_explored
        ):
            """
            Apply feature scaling, dimensionality reduction to the data. Return the standardised and low-dimensional train and
            test sets together with the scaler object for the target values.

            :param X_train: input train data
            :param y_train: train labels
            :param X_test: input test data
            :param y_test: test labels
            :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
            """
            X_train_scaled = (X_train - X_explored.min(axis=0).values) / (
                X_explored.max(axis=0).values - X_explored.min(axis=0).values
            )
            X_test_scaled = (X_test - X_explored.min(axis=0).values) / (
                X_explored.max(axis=0).values - X_explored.min(axis=0).values
            )
            y_scaler = StandardScaler()
            y_scaler.fit(y_explored)
            y_train_scaled = y_scaler.transform(y_train)
            y_test_scaled = y_scaler.transform(y_test)

            return (
                X_train_scaled,
                y_train_scaled,
                X_test_scaled,
                y_test_scaled,
                y_scaler,
            )

        repr_array = search_space_df.values
        repr_array = repr_array[
            :, ~(repr_array == repr_array[0, :]).all(0)
        ]  # remove columns with all the same values
        X_explored = torch.tensor(
            repr_array, dtype=torch.float64, device=self.device
        )
        # limit the dataframe to only the numeric data
        y_explored = torch.tensor(
            fitness_acquired, dtype=torch.float64, device=self.device
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_explored, y_explored, test_size=test_set_size, random_state=0
        )

        y_explored = y_explored.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        #  We standardise the outputs but leave the inputs unchanged

        X_train, y_train, X_test, y_test, y_scaler = transform_data(
            X_train, y_train, X_test, y_test, X_explored, y_explored
        )
        return X_train, X_test, y_train, y_test, y_scaler

    def train_model(self, X_train, y_train):
        self.model = self.kernel(
            X_train,
            y_train,
        )
        mll = self.likelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    def test_model_prediction(
        self, X_train, y_train, X_test, y_test, y_scaler
    ):
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

    def plot_prediction(
        self,
        y_pred,
        y_test,
        y_pred_train,
        y_train,
        y_var,
        save_plot=False,
        plot_name="prediction.png",
    ):
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
        y_var = np.array(y_var)
        data_plot = []
        spacing_array = np.linspace(y_var.min(), y_var.max(), 30)
        spacing = spacing_array[1] - spacing_array[0]
        for x in spacing_array:
            tes = [y_var > x] and [y_var < x + spacing]
            data_plot.append(np.abs(y_pred - y_test)[tes[0]].mean())
        axs[2].scatter(
            y_var,
            np.abs(y_test - y_pred),
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
        plt.show()
        if save_plot:
            dir_name = "data/figures/test_BO/"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            fig.savefig(dir_name + plot_name)


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