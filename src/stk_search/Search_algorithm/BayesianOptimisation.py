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

import os
import time

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
        super().__init__()
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
        error_counter_lim=10,
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
        # Start timing the function
        total_start_time = time.time()

        # Step 1: Prepare input for the BO
        step_start_time = time.time()
        df_search = searchspace_df.copy()
        fitness_acquired = np.array(fitness_acquired)
        x_rpr = self.Representation.generate_repr(
            df_search.loc[ids_acquired, :]
        )
        x_rpr = x_rpr.double()
        y_explored_bo = torch.tensor(fitness_acquired, dtype=torch.float64)
        y_explored_bo = y_explored_bo.reshape(-1, 1)
        if self.verbose:
            print(
                f"Step 1 (Prepare input for BO) took {time.time() - step_start_time:.4f} seconds"
            )

        # Step 2: Train the model
        step_start_time = time.time()
        self.train_model(x_rpr, y_explored_bo)
        if self.verbose:
            print(
                f"Step 2 (Train the model) took {time.time() - step_start_time:.4f} seconds"
            )

        # Step 3: Optimise the acquisition function
        step_start_time = time.time()

        def add_element(df, element) -> bool:
            if ~(df == element).all(1).any():
                df.loc[len(df)] = element
                return True
            return False

        df_search = searchspace_df
        df_elements = searchspace_df
        leave_loop = False
        error_counter = 0
        while not self._check_new_element_in_SearchSpace(
            df_search, df_elements
        ):
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
            if self.verbose:
                print(
                    f"Step 3 (Optimise acquisition function) took {time.time() - step_start_time:.4f} seconds"
                )
            error_counter = error_counter + 1
            print(
                f"Error counter: {error_counter} / {error_counter_lim}"
            )
            if error_counter > error_counter_lim:
                df_elements = df_elements.drop_duplicates()
                msg = "no new element found"
                raise ValueError(msg)
            # Step 4: Add the new element to the search space
            step_start_time = time.time()
            print(df_elements.shape)

        for element_id in ids_sorted_by_aquisition:
            if add_element(
                df_search, df_elements.to_numpy()[element_id.item()]
            ):
                break

        if self.verbose:
            print(
                f"Step 4 (Add new element to search space) took {time.time() - step_start_time:.4f} seconds"
            )

        # End timing the function
        if self.verbose:
            print(
                f"Total execution time for suggest_element: {time.time() - total_start_time:.4f} seconds"
            )
        

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
        df_elements = self.generate_df_elements_to_choose_from(
            fitness_acquired, df_search, sp, benchmark, df_total
        )
        xrpr = self.Representation.generate_repr(df_elements)
        # xrpr = self.normalise_input(xrpr)
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
        good_df_elements = df_elements.copy()
        # only keep the top 10 elements
        good_df_elements = good_df_elements.iloc[
            ids_sorted_by_aquisition[:100].cpu().numpy()
        ]
        good_acquisition_values = (
            acquisition_values[ids_sorted_by_aquisition[:100]]
            .cpu()
            .numpy()
            .reshape(-1)
        )
        # check if the new element is in the search space
        while counter < lim_counter:
            counter += 1
            max_counter += 1
            df_elements = self.generate_df_elements_to_choose_from(
                acquisition_values.cpu().numpy(),
                df_elements,
                sp,
                benchmark,
                df_total,
            )

            xrpr = self.Representation.generate_repr(df_elements)
            # xrpr = self.normalise_input(xrpr)
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

            # Store top 100 elements and their IDs
            

            if (
                max_acquisition_value_current
                > max_acquisition_value + 0.001 * max_acquisition_value
            ):
                max_acquisition_value = max_acquisition_value_current
                counter = 0
                # add elements better than the current best in the good_df_elements

                good_df_elements = pd.concat(
                    [
                        good_df_elements,
                        df_elements.iloc[
                            ids_sorted_by_aquisition[:100].cpu().numpy()
                        ],
                    ],
                    ignore_index=True,
                )
                good_acquisition_values = np.concatenate(
                    [
                        good_acquisition_values,
                        acquisition_values[ids_sorted_by_aquisition[:100]]
                        .cpu()
                        .numpy()
                        .reshape(-1),
                    ]
                )
            if self.verbose:
                print(
                    f"Acquisition counter: {counter}"
                )
            if max_counter > max_optimisation_iteration:
                break
        good_ids_sorted_by_aquisition = -good_acquisition_values.argsort()

        return good_ids_sorted_by_aquisition, good_df_elements

    def generate_df_elements_to_choose_from(
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
        elements = self.Generate_element_to_evaluate(
            fitness_acquired, df_search, sp
        )
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
        # from botorch.models.transforms.input import Normalize
        # from botorch.models.transforms.outcome import Standardize
        """Train the model.

        Args:
        ----
            x_train (torch.tensor): input.
            y_train (torch.tensor): output.

        """
        #return self.train_model_with_torch(x_train, y_train)
        self.model = self.kernel(
            x_train,
            y_train,
        )
        mll = self.likelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        

    def train_model_with_torch(self, x_train, y_train):
        from torch.optim import SGD

        NUM_EPOCHS = 2000
        self.model = self.kernel(
            x_train,
            y_train,
        )
        mll = self.likelihood(self.model.likelihood, self.model)
        mll = mll.to(x_train)

        self.model.train()
        optimizer = SGD(self.model.parameters(), lr=0.15)

        for epoch in range(NUM_EPOCHS):
            # clear gradients
            optimizer.zero_grad()
            # forward pass through the model to obtain the output MultivariateNormal
            output = self.model(x_train)
            # Compute negative marginal log likelihood
            loss = -mll(output, self.model.train_targets)
            # back prop gradients
            loss.backward()
            # print every 10 iterations
            # print("loss", loss)
            if (epoch + 1) % 1000 == 0:
                print(
                    f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
                    # f"lengthscale: {self.model.covar_module.lengthscale.item():>4.3f} "
                    # f"noise: {self.model.likelihood.noise.item():>4.3f}"
                )
            optimizer.step()

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
