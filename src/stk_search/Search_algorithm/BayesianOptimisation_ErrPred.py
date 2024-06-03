# class to define the search algorithm
import os
import torch
import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, qKnowledgeGradient
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from stk_search.Search_algorithm import BayesianOptimisation
from stk_search.Search_space import Search_Space
from stk_search.Search_algorithm.Botorch_kernels import (
    TanimotoGP,
    RBFKernel,
    MaternKernel,
)
import itertools

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class BayesianOptimisation_ErrPred(BayesianOptimisation.BayesianOptimisation):
    def suggest_element(
        self,
        search_space_df,
        fitness_acquired,
        ids_acquired,
        SP: Search_Space,
        benchmark=True,
        df_total: pd.DataFrame = None,
    ):
        """Suggest a new element to evaluate.
        Args:
            search_space_df (pd.DataFrame): search space
            fitness_acquired (list): fitness of the acquired elements
            ids_acquired (list): ids of the acquired elements
            SP (Search_Space): search space
            benchmark (bool): if True, the search space is a benchmark
            df_total (pd.DataFrame): dataframe of the total dataset
        Returns:
            int: id of the new element
            pd.DataFrame: updated search space
        """
        df_search = search_space_df.copy()
        fitness_acquired = np.array(fitness_acquired)
        # prepare input for the BO
        if self.pred_model is None:
            raise ValueError(
                "pred_model is None, but it's required for UCB_GNN acquisition"
            )

        X_rpr = self.Representation.generate_repr(
            df_search.loc[ids_acquired, :]
        )
        y_pred_model = self.pred_model(X_rpr.float())
        X_rpr = X_rpr.double()

        X_rpr = self.normalise_input(X_rpr)

        y_explored_BO = torch.tensor(fitness_acquired, dtype=torch.float64)
        y_Err_BO_norm = (
            y_explored_BO - y_pred_model.squeeze().detach().double()
        )
        self.target_normmean = y_Err_BO_norm.mean(axis=0)
        self.target_normstd = y_Err_BO_norm.std(axis=0)
        y_Err_BO_norm = (y_Err_BO_norm - self.target_normmean) / (
            self.target_normstd
        )
        y_Err_BO_norm = y_Err_BO_norm.reshape(-1, 1)
        # train model
        self.train_model(X_rpr, y_Err_BO_norm)
        # optimise the acquisition function
        ids_sorted_by_aquisition, df_elements = (
            self.optimise_acquisition_function(
                best_f=y_Err_BO_norm.max().item(),
                fitness_acquired=fitness_acquired,
                df_search=df_search,
                SP=SP,
                benchmark=benchmark,
                df_total=df_total,
            )
        )

        # add the new element to the search space
        def add_element(df, element):
            if ~(df == element).all(1).any():
                df.loc[len(df)] = element
                return True
            return False

        for element_id in ids_sorted_by_aquisition:
            if add_element(df_search, df_elements.values[element_id.item()]):
                break
        return len(df_search) - 1, df_search

    def get_acquisition_values(self, best_f, Xrpr):
        """Get the acquisition values.
        Args:
            best_f (float): best fitness
            Xrpr (torch.tensor): representation of the element
        Returns:
            torch.tensor: acquisition values"""
        if self.which_acquisition == "UCB_GNN":
            X_unsqueezed = self.normalise_input(Xrpr).double()
            X_unsqueezed = X_unsqueezed.reshape(-1, 1, X_unsqueezed.shape[1])
            if self.pred_model is None:
                raise ValueError(
                    "pred_model is None, but it's required for UCB_GNN acquisition"
                )
            with torch.no_grad():
                acquisition_values = self.pred_model(Xrpr.float())
                predicted_error = (
                    self.model.posterior(X_unsqueezed).mean.squeeze()
                    * self.target_normstd
                    + self.target_normmean
                )
                predicted_error_std = (
                    self.model.posterior(X_unsqueezed).variance.squeeze()
                    * self.target_normstd
                )
                acquisition_values = (
                    acquisition_values.squeeze()
                    + predicted_error
                    + predicted_error_std
                )
            return acquisition_values
        if self.which_acquisition == "EI_GNN":
            X_unsqueezed = self.normalise_input(Xrpr).double()
            X_unsqueezed = X_unsqueezed.reshape(-1, 1, X_unsqueezed.shape[1])
            acquisition_function = ExpectedImprovement_GNN(
                self.model, best_f=best_f
            )
            with torch.no_grad():  # to avoid memory issues; we arent using the gradient...
                GNN_pred = self.pred_model(Xrpr.float()).detach()
                acquisition_values = acquisition_function.forward(
                    X_unsqueezed,
                    GNN_pred.squeeze()
                )  # runs out of memory
            return acquisition_values


from typing import Dict, Optional, Tuple, Union
from torch import Tensor
from botorch.models.model import Model
from botorch.acquisition.objective import PosteriorTransform


class ExpectedImprovement_GNN(ExpectedImprovement):
    r"""Single-outcome Expected Improvement (analytic).

    Computes expected improvement considering the
    """


    def forward(self, X: Tensor,GNN_pred) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        u = _scaled_improvement(
            mean + GNN_pred,
            sigma,
            self.best_f,
            self.maximize,
        )
        return sigma * _ei_helper(u)


def _scaled_improvement(
    mean: Tensor, sigma: Tensor, best_f: Tensor, maximize: bool
) -> Tensor:
    """Returns `u = (mean - best_f) / sigma`, -u if maximize == True."""
    u = (mean - best_f) / sigma
    return u if maximize else -u


from botorch.utils.probability.utils import (
    ndtr as Phi,
    phi,
)


def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)
