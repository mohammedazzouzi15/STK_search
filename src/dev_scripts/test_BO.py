import numpy as np
import os
import pandas as pd
from stk_search.utils import database_utils
from botorch.models.gp_regression import SingleTaskGP
from stk_search.tanimoto_kernel import TanimotoKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch import kernels
from stk_search.Search_algorithm import (
    Representation_slatm,
    RepresentationPrecursor,
)
from stk_search.Search_algorithm import Bayesian_Optimisation
from stk_search import Searched_space
from sklearn.decomposition import PCA

import torch


# %%
# Load the searched space
def load_data():
    df_path = "data/output/Full_datatset/df_total_new2023_08_20.csv"
    df_precursors_path = "data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl"  #'Data/output/Prescursor_data/calculation_data_precursor_310823_clean.pkl'
    df_total, df_precursors = database_utils.load_data_from_file(
        df_path, df_precursors_path
    )
    SP = Searched_space.Searched_Space(
        number_of_fragments=6,
        df=df_precursors,
        features_frag=df_precursors.columns[0:1],
        generation_type="conditional",
    )
    searched_space_df = SP.check_df_for_element_from_SP(df_to_check=df_total)
    fitness_acquired = searched_space_df["target"].values
    searched_space_df_InChIKey = searched_space_df[['InChIKey']]
    searched_space_df = searched_space_df[[f"InChIKey_{x}" for x in range(6)]]
    return df_total, df_precursors, searched_space_df, fitness_acquired, searched_space_df_InChIKey


def prepare_run(case, df_precursors, df_total, searched_space_df_InChIKey):
    if case == "slatm":
        BO = Bayesian_Optimisation.Bayesian_Optimisation()
        BO.Representation = Representation_slatm.Representation_slatm()
    elif case == "slatm_org":
        BO = Bayesian_Optimisation.Bayesian_Optimisation()
        BO.Representation = Representation_slatm.Representation_slatm_org(df_total, searched_space_df_InChIKey)
    elif case == "precursor":
        BO = Bayesian_Optimisation.Bayesian_Optimisation()
        frag_properties = []
        frag_properties = df_precursors.columns[1:7]
        frag_properties = frag_properties.append(df_precursors.columns[17:23])
        print(frag_properties)
        BO.Representation = RepresentationPrecursor.RepresentationPrecursor(
            df_precursors, frag_properties
        )
    else:
        raise ValueError("case not recognised")

    BO.verbose = True

    test_set_size = 0.9
    return BO


def run_training_BO_torch(
    BO, kernel_name, test_BO_dict, count, searched_space_df,test_set_size, case, data, pca_N
):
    class test_kernel(SingleTaskGP):
        def __init__(self, train_X, train_Y):
            super().__init__(train_X, train_Y, GaussianLikelihood())
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(
                base_kernel=kernel_name
            )  # kernels.RBFKernel())#)
            self.to(train_X)  # make sure we're on the right device/dtype

        def change_kernel(self, kernel):
            self.covar_module = ScaleKernel(base_kernel=kernel)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    BO.kernel = test_kernel

    # %%
    count += 1
    test_BO_dict[count] = {}
    test_BO_dict[count]["test_set_size"] = test_set_size
    test_BO_dict[count]["case"] = case
    test_BO_dict[count]["kernel"] = kernel_name._get_name()  #'MaternKernel'
    test_BO_dict[count]["model_training"] = "Botorch"
    test_BO_dict[count]["train_size"] = searched_space_df.shape[0] * (
        1 - test_set_size
    )
    test_BO_dict[count]["test_size"] = int(
        searched_space_df.shape[0] * (test_set_size)
    )
    print("Training size", data["X_train"].shape)
    # %%

    (
        y_pred,
        y_var,
        y_pred_train,
        y_train_plot,
        y_test_plot,
        y_var_train,
    ) = BO.test_model_prediction(
        data["X_train"],
        torch.tensor(data["y_train"], dtype=torch.float64),
        data["X_test"],
        torch.tensor(data["y_test"], dtype=torch.float64),
        data['y_scaler'],
    )
    fig_name = (
        f"pca{pca_N}_{case}_trainsize_"
        + str(data["X_train"].shape[0])
        + "_testSi_"
        + str(data["X_test"].shape[0])
        + "_kernel_"
        + kernel_name._get_name()
        + "_botorch.png"
    )
    scores_test, scores_train = BO.plot_prediction(
        y_pred,
        y_test_plot,
        y_pred_train,
        y_train_plot,
        y_var,
        save_plot=True,
        plot_name=fig_name,
    )
    test_BO_dict[count]["scores_test"] = scores_test
    test_BO_dict[count]["scores_train"] = scores_train
    return test_BO_dict, count


def run_training_gpytorch(
    BO, kernel_name, test_BO_dict, count, searched_space_df, test_set_size, case, data, pca_N
):
    class test_kernel(SingleTaskGP):
        def __init__(self, train_X, train_Y):
            super().__init__(train_X, train_Y, GaussianLikelihood())
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(
                base_kernel=kernel_name
            )  # kernels.RBFKernel())#)
            self.to(train_X)  # make sure we're on the right device/dtype

        def change_kernel(self, kernel):
            self.covar_module = ScaleKernel(base_kernel=kernel)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    BO.kernel = test_kernel
    y_train = torch.tensor(data["y_train"], dtype=torch.float64)
    y_test = torch.tensor(data["y_test"], dtype=torch.float64)
    count += 1
    test_BO_dict[count] = {}
    test_BO_dict[count]["test_set_size"] = test_set_size
    test_BO_dict[count]["case"] = case
    test_BO_dict[count]["kernel"] = kernel_name._get_name()  #'MaternKernel'
    test_BO_dict[count]["model_training"] = "gpytorch"
    test_BO_dict[count]["train_size"] = searched_space_df.shape[0] * (
        1 - test_set_size
    )
    test_BO_dict[count]["test_size"] = int(
        searched_space_df.shape[0] * (test_set_size)
    )
    BO.train_model_gpytorch(data["X_train"], y_train, NUM_EPOCHS=1000)
    BO.model.eval()
    fig_name = (
        f"pca{pca_N}_{case}_trainsize_"
        + str(data["X_train"].shape[0])
        + "_testSi_"
        + str(data["X_test"].shape[0])
        + "_kernel_"
        + kernel_name._get_name()
        + "_gpytorch.png"
    )
    with torch.no_grad():
        posterior_train = BO.model.posterior(data["X_train"])
        posterior_test = BO.model.posterior(data["X_test"])

    y_pred = posterior_test.mean.cpu()
    y_pred_train = posterior_train.mean.cpu()
    y_var = posterior_test.variance.cpu()
    y_pred = data['y_scaler'].inverse_transform(y_pred)
    y_pred_train = data['y_scaler'].inverse_transform(y_pred_train)
    y_test = data['y_scaler'].inverse_transform(y_test)
    y_train = data['y_scaler'].inverse_transform(y_train)
    scores_test, scores_train = BO.plot_prediction(
        y_pred,
        y_test,
        y_pred_train,
        y_train,
        y_var,
        save_plot=True,
        plot_name=fig_name,
    )
    test_BO_dict[count]["scores_test"] = scores_test
    test_BO_dict[count]["scores_train"] = scores_train
    return test_BO_dict, count


def prepare_data(case, X_train, X_test,PCA_n=100):
    if case == "slatm":
        pca = PCA(n_components=PCA_n)
        pca.fit(X_test[:5000])
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    elif case == "slatm_org":
        pca = PCA(n_components=PCA_n)
        pca.fit(X_test[:5000])
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    X_train = torch.tensor(X_train, dtype=torch.float64)
    X_test = torch.tensor(X_test, dtype=torch.float64)
    X_train = (X_train - X_test.min(axis=0).values) / (
        X_test.max(axis=0).values - X_test.min(axis=0).values
    )
    X_test = (X_test - X_test.min(axis=0).values) / (
        X_test.max(axis=0).values - X_test.min(axis=0).values
    )
    return X_train, X_test


def save_data(test_BO_dict):
    df = pd.DataFrame.from_dict(test_BO_dict, orient="index")
    df["r2_test"] = df["scores_test"].apply(lambda x: x["r2"])
    df["r2_train"] = df["scores_train"].apply(lambda x: x["r2"])
    df["rmse_test"] = df["scores_test"].apply(lambda x: x["mse"])
    df["rmse_train"] = df["scores_train"].apply(lambda x: x["mse"])

    df.to_csv("data/output/evaluate_BO_org_newEncoder.csv")


def run_full_test():
    df_total, df_precursors, searched_space_df, fitness_acquired, searched_space_df_InChIKey= load_data()
    test_model_dict = {}
    count = 0
    for case in [
        "slatm_org",
        "slatm",
    ]:  # "slatm_org","precursor",,'slatm']:#'precursor','slatm'.'precursor','slatm_PCA100',
        BO = prepare_run(case, df_precursors,df_total,searched_space_df_InChIKey)
        X_explored = BO.Representation.generate_repr(searched_space_df)
            # limit the dataframe to only the numeric data
        y_explored = torch.tensor(
            fitness_acquired, dtype=torch.float32, device=BO.device
        )
        
        for test_set_size in [0.995, 0.99, 0.98]:  # ,0.99,0.98]:

            (
                X_train,
                X_test,
                y_train,
                y_test,
                y_scaler,
            ) = BO.get_test_train_data_for_BO(
                X_explored,
                y_explored,
                test_set_size=test_set_size,
            )
            for pca_N in [10,30,50,100]:
                X_train_PCA, X_test_PCA = prepare_data(case, X_train, X_test,pca_N)
                data = {
                    "X_train": X_train_PCA,
                    "X_test": X_test_PCA,
                    "y_train": y_train,
                    "y_test": y_test,
                    "y_scaler": y_scaler,
                }
                for kernel_name in [
                    kernels.MaternKernel(),
                
                ]: #kernels.RBFKernel(), TanimotoKernel(),
                    test_model_dict, count = run_training_BO_torch(
                        BO, kernel_name, test_model_dict, count,searched_space_df,test_set_size, case, data,pca_N
                    )
                    test_model_dict[count]["PCA_N"] = pca_N
                    save_data(test_model_dict)
                    # %%
                    test_model_dict, count = run_training_gpytorch(
                        BO, kernel_name, test_model_dict, count,searched_space_df,test_set_size, case, data,pca_N
                    )
                    test_model_dict[count]["PCA_N"] = pca_N
                    save_data(test_model_dict)


run_full_test()
