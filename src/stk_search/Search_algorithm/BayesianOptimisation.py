"""Class to define the Bayesian Optimisation with a GNN surrogate model.

Here teh Bayesian Optimisation search algorithm is defined and is a subclass of Search_Algorithm.
The Bayesian Optimisation search algorithm is used to optimise the acquisition function and suggest the next element to evaluate.
the different step of the algorithm are:

1. Prepare input for the BO
2. Train the model
3. Optimise the acquisition function
4. Generate elements to evaluate
5. Suggest a new element to evaluate

"""

import itertools
import os

import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qKnowledgeGradient
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
)
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from gpytorch.mlls import ExactMarginalLogLikelihood

from stk_search.Search_algorithm.Botorch_kernels import (
    RBFKernel,
)
from stk_search.Search_algorithm.Search_algorithm import evolution_algorithm
from stk_search.SearchSpace import SearchSpace

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class BayesianOptimisation(evolution_algorithm):
    """BAYESIAN OPTIMISATION CLASS.

    Here the Bayesian Optimisation search algorithm is defined and is a subclass of Search_Algorithm.
    The Bayesian Optimisation search algorithm is used to optimise the acquisition function and suggest the next element to evaluate.
    the different step of the algorithm are:
    1. Prepare input for the BO
    2. Train the model
    3. Optimise the acquisition function
    4. Generate elements to evaluate
    5. Suggest a new element to evaluate.

    Attributes
    ----------
    verbose (bool): if True, print the output
    which_acquisition (str): acquisition function to use
    kernel (gpytorch.kernels): kernel to use
    device (str): device to use
    likelihood (gpytorch.likelihoods): likelihood to use

    """

    def __init__(
        self,
        verbose=False,
        which_acquisition="EI",
        kernel=RBFKernel,
        likelihood=ExactMarginalLogLikelihood,
        model=None,
        lim_counter=2,
        representation=None,
    ):
        """Initialise the class.

        Args:
        ----
            verbose (bool): if True, print the output
            PCA_input (bool): if True, use PCA to reduce the dimension of the input
            normalise_input (bool): if True, normalise the input
            which_acquisition (str): acquisition function to use
            kernel (gpytorch.kernels): kernel to use
            likelihood (gpytorch.likelihoods): likelihood to use
            model (gpytorch.models): model to use
            lim_counter (int): max iteration for the acquisition function optimisation
            representation (object): representation of the element

        """
        self.verbose = verbose
        self.which_acquisition = which_acquisition
        self.kernel = kernel
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.likelihood = likelihood
        self.model = model
        self.lim_counter = lim_counter  # max iteration for the acquisition function optimisation
        self.Representation = representation
        self.name = "Bayesian_Optimisation"
        self.pred_model = None
        self.config_dir = ""
        self.multi_fidelity = False
        self.budget = None

    def update_representation(self, representation):
        """Update the representation."""
        self.Representation = representation

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
            searchspace_df (pd.DataFrame):
                search space dataframe.
            fitness_acquired (list):
                fitness of the acquired elements.
            ids_acquired (list):
                ids of the acquired elements.
            sp (SearchSpace):
                search space.
            benchmark (bool):
                if True, the search space is a benchmark.
            df_total (pd.DataFrame):
                dataframe of the total dataset.

        Returns:
        -------
            int: id of the new element.
            pd.DataFrame: updated search space.

        """
        df_search = searchspace_df.copy()
        fitness_acquired = np.array(fitness_acquired)
        # prepare input for the BO
        x_rpr = self.Representation.generate_repr(
            df_search.loc[ids_acquired, :]
        )
        x_rpr = x_rpr.double()
        y_explored_bo = torch.tensor(
            fitness_acquired, dtype=torch.float64
        )
        y_explored_bo = y_explored_bo.reshape(-1, 1)
        # train model
        self.train_model(x_rpr, y_explored_bo)
        # optimise the acquisition function
        ids_sorted_by_aquisition, df_elements = (
            self.optimise_acquisition_function(
                best_f=y_explored_bo.max().item(),
                fitness_acquired=fitness_acquired,
                df_search=df_search,
                sp=sp,
                benchmark=benchmark,
                df_total=df_total,
            )
        )

        # add the new element to the search space
        def add_element(df, element) -> bool:
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

    def normalise_input(self, x_rpr):
        """Normalise the input.

        Args:
        ----
            x_rpr (torch.tensor): Representation of the element.

        Returns:
        -------
            torch.tensor: normalised input

        """
        x_rpr = x_rpr.double()
        # min max scaling the input
        return (x_rpr - x_rpr.min(dim=0)[0]) / (
            x_rpr.max(dim=0)[0] - x_rpr.min(dim=0)[0]
        )

    def optimise_acquisition_function(
        self,
        best_f,
        fitness_acquired,
        df_search,
        sp: SearchSpace,
        benchmark=False,
        df_total=None,
    ):
        """Optimise the acquisition function.

        Args:
        ----
            best_f:
                float: best fitness.
            fitness_acquired:
                list: fitness of the acquired elements.
            df_search :
                pd.DataFrame: search space dataframe.
            sp :
                SearchSpace: search space.
            benchmark :
                bool: if True, the search space is a benchmark.
            df_total :
                pd.DataFrame: dataframe of the total dataset.

        Returns:
        -------
            torch.tensor: acquisition values
            pd.DataFrame: updated search space

        """
        # generate list of element to evaluate using acquistion function
        counter, lim_counter = 0, self.lim_counter
        df_elements = self.Generate_element_to_evaluate(
            fitness_acquired, df_search, sp, benchmark, df_total
        )
        xrpr = self.Representation.generate_repr(df_elements)
        #xrpr = self.normalise_input(xrpr)
        acquisition_values = self.get_acquisition_values(
            self.model,
            best_f=best_f,
            xrpr=xrpr,
        )

        if "dataset_local" in self.Representation.__dict__:
            pass
        # select element to acquire with maximal aquisition value, which is not in the acquired set already
        ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
        max_acquisition_value = acquisition_values.max()
        max_counter, max_optimisation_iteration = 0, 100
        while counter < lim_counter:
            counter += 1
            max_counter += 1
            df_elements = self.Generate_element_to_evaluate(
                acquisition_values.cpu().numpy(),
                df_elements,
                sp,
                benchmark,
                df_total,
            )

            xrpr = self.Representation.generate_repr(df_elements)
            #xrpr = self.normalise_input(xrpr)
            # if benchmark:
            acquisition_values = self.get_acquisition_values(
                self.model,
                best_f=best_f,
                xrpr=xrpr,
            )
            if "dataset_local" in self.Representation.__dict__:
                pass
            # select element to acquire with maximal aquisition value, which is not in the acquired set already
            ids_sorted_by_aquisition = acquisition_values.argsort(
                descending=True
            )
            max_acquisition_value_current = acquisition_values.max()
            if (
                max_acquisition_value_current
                > max_acquisition_value + 0.001 * max_acquisition_value
            ):
                max_acquisition_value = max_acquisition_value_current
                counter = 0
            if max_counter > max_optimisation_iteration:
                break
        return ids_sorted_by_aquisition, df_elements

    def Generate_element_to_evaluate(
        self,
        fitness_acquired,
        df_search,
        sp: SearchSpace,
        benchmark=False,
        df_total=None,
    ):
        """Generate elements to evaluate.

        Args:
        ----
            fitness_acquired (list): fitness of the acquired elements.
            df_search (pd.DataFrame): search space.
            sp (SearchSpace): search space.
            benchmark (bool): if True, the search space is a benchmark.
            df_total (pd.DataFrame): dataframe of the total dataset.

        Returns:
        -------
                pd.DataFrame: elements to evaluate.

        TODO: use the same function as in the EA

        """

        def mutate_element(element)->list:
            elements_val = []
            for i in range(element.shape[0]):
                for frag in sp.df_precursors.InChIKey:
                    element_new = element.copy()
                    element_new[i] = frag
                    elements_val.append(element_new)
            return elements_val

        def cross_element(element1, element2)->list:
            elements_val = []
            for i in range(element.shape[0]):
                element_new = element1.copy()
                element_new[i] = element2[i]
                elements_val.append(element_new)
            return elements_val

        # select the 3 best one and add two random element from the search space
        best_element_arg = fitness_acquired.argsort()[-3:][::-1]
        list_parents = df_search.loc[best_element_arg, :].to_numpy()
        list_parents = np.append(
            list_parents, df_search.sample(2).values, axis=0
        )
        elements = []
        for element in list_parents:
            if len(elements) == 0:
                elements = mutate_element(element)
            else:
                elements = np.append(elements, mutate_element(element), axis=0)
        for element1, element2 in itertools.product(
            list_parents, list_parents
        ):
            if len(elements) == 0:
                elements = cross_element(element1, element2)
            else:
                elements = np.append(
                    elements, cross_element(element1, element2), axis=0
                )
        elements = np.append(elements, df_search.values, axis=0)
        df_elements = pd.DataFrame(
            elements,
            columns=[
                f"InChIKey_{x}" for x in range(elements.shape[1])
            ],  # check this for generalization
        )
        df_elements = sp.check_df_for_element_from_sp(df_to_check=df_elements)
        if benchmark:
            # take only element in df_total
            df_elements = df_elements.merge(
                df_total,
                on=[
                    f"InChIKey_{i}" for i in range(elements.shape[1])
                ],  # check this for generalization
                how="left",
            )
            df_elements = df_elements.dropna(subset="target")
            df_elements = df_elements[
                [f"InChIKey_{i}" for i in range(elements.shape[1])]
            ]  # check this for generalization
            df_elements = df_elements.drop_duplicates()
        if (
            df_elements.shape[0] > 1000
        ):  # limit the number of elements to evaluate each time
            df_elements = df_elements.sample(1000)
        return df_elements.reset_index(drop=True)

    def train_model(self, x_train, y_train):
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        """Train the model.

        Args:
        ----
            x_train (torch.tensor): input.
            y_train (torch.tensor): output.

        """
        self.model = self.kernel(
            x_train,
            y_train,
        )
        mll = self.likelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def get_acquisition_values(self, model, best_f, xrpr):
        """Get the acquisition values.

        Args:
        ----
            model (gpytorch.models): model.
            best_f (float): best fitness.
            xrpr (torch.tensor): Representation of the element.

        Returns:
        -------
        torch.tensor: acquisition values.

        """
        x_unsqueezed = xrpr.double()
        x_unsqueezed = x_unsqueezed.reshape(-1, 1, x_unsqueezed.shape[1])
        # set up acquisition function
        if self.which_acquisition == "EI":
            acquisition_function = ExpectedImprovement(model, best_f=best_f)
            with torch.no_grad():  # to avoid memory issues; we arent using the gradient...
                acquisition_values = acquisition_function.forward(
                    x_unsqueezed
                )  # runs out of memory
        elif self.which_acquisition == "max_y_hat":
            with torch.no_grad():
                acquisition_values = model.posterior(
                    x_unsqueezed
                ).mean.squeeze()
        elif self.which_acquisition == "max_sigma":
            with torch.no_grad():
                acquisition_values = model.posterior(
                    x_unsqueezed
                ).variance.squeeze()
        elif self.which_acquisition == "LOG_EI":
            acquisition_function = LogExpectedImprovement(model, best_f=best_f)
            with torch.no_grad():  # to avoid memory issues; we arent using the gradient...
                acquisition_values = acquisition_function.forward(
                    x_unsqueezed
                )  # runs out of memory
        elif self.which_acquisition == "UCB_GNN":
            if self.pred_model is None:
                msg = "pred_model is None, but it's required for UCB_GNN acquisition"
                raise ValueError(msg)
            with torch.no_grad():
                acquisition_values = self.pred_model(
                    x_unsqueezed.float()
                ).squeeze()
                acquisition_values = (
                    acquisition_values
                    + self.model.posterior(x_unsqueezed).variance.squeeze()
                )
        elif self.which_acquisition == "UCB":
            with torch.no_grad():
                acquisition_values = (
                    model.posterior(x_unsqueezed).mean.squeeze()
                    + self.model.posterior(x_unsqueezed).variance.squeeze()
                )

                acquisition_values = self.pred_model(
                    x_unsqueezed.float()
                ).squeeze()
                acquisition_values = (
                    acquisition_values
                    + self.model.posterior(x_unsqueezed).variance.squeeze()
                )
        elif self.which_acquisition == "KG":
            acquisition_function = qKnowledgeGradient(
                model=model, num_fantasies=5
            )
            bounds = torch.tensor(
                [[0.0] * xrpr.shape[1], [1.0] * xrpr.shape[1]],
                dtype=torch.float64,
            )
            acquisition_values = acquisition_function.evaluate(
                x_unsqueezed, bounds=bounds
            )
        elif self.which_acquisition == "MES":
            bounds = torch.tensor(
                [[0.0] * xrpr.shape[1], [1.0] * xrpr.shape[1]]
            )
            candidate_set = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
                10000, xrpr.shape[1]
            )
            acquisition_function = qMaxValueEntropy(
                model, candidate_set=candidate_set
            )
            acquisition_values = acquisition_function(
                x_unsqueezed,
            ).detach()
        else:
            with torch.no_grad():
                acquisition_values = model.posterior(
                    x_unsqueezed
                ).variance.squeeze()
        return acquisition_values

    def load_representation_model(self):
        """Load the representation model.
        
        Returns
        -------
            representation (object): representation of the element.
            pymodel (object): model.
        
        """
        from stk_search.geom3d import pl_model
        from stk_search.Representation import Representation_poly_3d
        from stk_search.utils.config_utils import read_config

        config_dir = self.config_dir
        config = read_config(config_dir)
        chkpt_path = config["model_embedding_chkpt"]
        checkpoint = torch.load(chkpt_path, map_location=config["device"])
        model, graph_pred_linear = pl_model.model_setup(config)
        print("Model loaded: ", config["model_name"])
        # Pass the model and graph_pred_linear to the Pymodel constructor
        pymodel = pl_model.Pymodel_new(model, graph_pred_linear, config)
        # Load the state dictionary
        pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
        pymodel.to(config["device"])
        representation = Representation_poly_3d.RepresentationPoly3d(
            pymodel,
            mongo_client=config["pymongo_client"],
            database=config["database_name"],
            device=pymodel.device,
        )
        self.pred_model = pymodel.graph_pred_linear
        self.Representation = representation

        return representation, pymodel
