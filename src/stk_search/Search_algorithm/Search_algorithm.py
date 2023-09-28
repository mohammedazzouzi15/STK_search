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

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Search_Algorithm:
    def __init__(self):
        pass
    def suggest_element(
        self,
        search_space_df: pd.DataFrame = [],
        fitness_acquired: list = [],
        ids_acquired: list = [],
        bad_ids: list = [],
    ) -> float:
        pass
    def initial_suggestion(
        self,
        search_space_df: pd.DataFrame = [],
        num_elem_initialisation: int = 10,
    ) -> list:
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
        return float(
            np.random.choice(searched_space_df.index, 1, replace=False)
        )

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

