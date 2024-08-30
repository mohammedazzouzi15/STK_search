# class to define the search algorithm
import os

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.analytic import (
    ExpectedImprovement,
)
from botorch.utils.probability.utils import (
    ndtr as Phi,
)
from botorch.utils.probability.utils import (
    phi,
)
from torch import Tensor

from stk_search.Search_algorithm import BayesianOptimisation
from stk_search.SearchSpace import SearchSpace

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class BayesianOptimisation_ErrPred(BayesianOptimisation.BayesianOptimisation):
    """Class to define the Bayesian Optimisation with a GNN surrogate model.

    Here we train the Gaussian Process on the error of the GNN model.

    Attributes
    ----------
        pred_model (torch.nn.Module): GNN model
        model (GPyTorch model): Gaussian Process model
        Representation (Representation): Representation of the elements
        target_normmean (float): mean of the target
        target_normstd (float): std of the target
        which_acquisition (str): acquisition function to use

    Functions:
    ----------
        normalise_input: normalise the input
        train_model: train the model
        optimise_acquisition_function: optimise the acquisition function
        suggest_element: suggest a new element to evaluate
        get_acquisition_values: get the acquisition values

    """

    def suggest_element(
        self,
        searchspace_df,
        fitness_acquired,
        ids_acquired,
        sp: SearchSpace,
        benchmark=True,
        df_total: pd.DataFrame = None,
    ):
        """Suggest a new element to evaluate.

        Args:
        ----
            searchspace_df (pd.DataFrame): search space
            fitness_acquired (list): fitness of the acquired elements
            ids_acquired (list): ids of the acquired elements
            sp (SearchSpace): search space
            benchmark (bool): if True, the search space is a benchmark
            df_total (pd.DataFrame): dataframe of the total dataset
        Returns:
            int: id of the new element
            pd.DataFrame: updated search space

        """
        df_search = searchspace_df.copy()
        fitness_acquired = np.array(fitness_acquired)
        # prepare input for the BO
        if self.pred_model is None:
            raise ValueError(
                "pred_model is None, but it's required for UCB_GNN acquisition"
            )

        x_rpr = self.Representation.generate_repr(
            df_search.loc[ids_acquired, :]
        )
        y_pred_model = self.pred_model(x_rpr.float())
        x_rpr = x_rpr.double()

        x_rpr = self.normalise_input(x_rpr)

        y_explored_bo = torch.tensor(fitness_acquired, dtype=torch.float64)
        y_explored_bo_norm = (
            y_explored_bo - y_explored_bo.mean()
        ) / y_explored_bo.std()
        y_err_bo_norm = (
            y_explored_bo - y_pred_model.squeeze().detach().double()
        )
        self.target_normmean = y_err_bo_norm.mean(axis=0)
        self.target_normstd = y_err_bo_norm.std(axis=0)
        y_err_bo_norm = (y_err_bo_norm - self.target_normmean) / (
            self.target_normstd
        )
        y_err_bo_norm = y_err_bo_norm.reshape(-1, 1)
        # train model
        self.train_model(x_rpr, y_err_bo_norm)
        # optimise the acquisition function
        ids_sorted_by_aquisition, df_elements = (
            self.optimise_acquisition_function(
                best_f=y_explored_bo_norm.max().item(),
                fitness_acquired=fitness_acquired,
                df_search=df_search,
                sp=sp,
                benchmark=benchmark,
                df_total=df_total,
            )
        )

        # add the new element to the search space
        def add_element(df, element) -> bool:
            """Add the element to the search space."""
            if ~(df == element).all(1).any():
                df.loc[len(df)] = element
                return True
            return False

        for element_id in ids_sorted_by_aquisition:
            if add_element(
                df_search, df_elements.to_numpy()[element_id.item()]
            ):
                break
        return len(df_search) - 1, df_search

    def get_acquisition_values(self, best_f, xrpr):
        """Get the acquisition values.

        Args:
        ----
            best_f (float):
                best fitness
            xrpr (torch.tensor): 
                Representation of the element
                
        Returns:
        -------
            torch.tensor: acquisition values

        """
        if self.which_acquisition == "UCB_GNN":
            x_unsqueezed = self.normalise_input(xrpr).double()
            x_unsqueezed = x_unsqueezed.reshape(-1, 1, x_unsqueezed.shape[1])
            if self.pred_model is None:
                raise ValueError(
                    "pred_model is None, but it's required for UCB_GNN acquisition"
                )
            with torch.no_grad():
                gnn_pred = self.pred_model(xrpr.float())
                predicted_error = (
                    self.model.posterior(x_unsqueezed).mean.squeeze()
                    * self.target_normstd
                    + self.target_normmean
                )
                predicted_error_std = (
                    self.model.posterior(x_unsqueezed).variance.squeeze()
                    * self.target_normstd
                )
            return gnn_pred.squeeze() + predicted_error + predicted_error_std
        if self.which_acquisition == "EI_GNN":
            x_unsqueezed = self.normalise_input(xrpr).double()
            x_unsqueezed = x_unsqueezed.reshape(-1, 1, x_unsqueezed.shape[1])
            acquisition_function = ExpectedImprovement_GNN(
                self.model, best_f=best_f
            )
            with torch.no_grad():  # to avoid memory issues; we arent using the gradient...
                gnn_pred = (
                    self.pred_model(xrpr.float()).detach()
                    / self.target_normstd
                )  # assuming the same std for the GNN and the data

            return acquisition_function.forward(
                x_unsqueezed, gnn_pred.squeeze()
            )  # runs out of memory
        raise ValueError("No acquisition function selected")


class ExpectedImprovement_GNN(ExpectedImprovement):
    r"""Single-outcome Expected Improvement (analytic).

    Computes expected improvement considering the
    """

    def forward(self, x: Tensor, gnn_pred) -> Tensor:
        """Evaluate Expected Improvement on the candidate set X.

        Args:
        ----
            x: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.
            gnn_pred: A `(b1 x ... bk)`-dim tensor of the GNN predictions at the

        Returns:
        -------
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.

        """
        mean, sigma = self._mean_and_sigma(x)
        u = _scaled_improvement(
            mean + gnn_pred,
            sigma,
            self.best_f,
            self.maximize,
        )
        return sigma * _ei_helper(u)


def _scaled_improvement(
    mean: Tensor, sigma: Tensor, best_f: Tensor, maximize: bool
) -> Tensor:
    """Calculate the scaled improvement.

    Args:
    ----
        mean: The mean of the predictive distribution.
        sigma: The standard deviation of the predictive distribution.
        best_f: The best function value observed so far.
        maximize: If True, the acquisition function is maximized.

    Returns:
    -------
        The scaled improvement.
        `u = (mean - best_f) / sigma`, -u if maximize == True.

    """
    u = (mean - best_f) / sigma
    return u if maximize else -u


def _ei_helper(u: Tensor) -> Tensor:
    """Calculate the Expected Improvement.

    Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.

    Args:
    ----
        u: A `(b1 x ... bk)`-dim tensor of scaled improvements.

    Returns:
    -------
        A `(b1 x ... bk)`-dim tensor of Expected Improvement values.

    """
    return phi(u) + u * Phi(u)
