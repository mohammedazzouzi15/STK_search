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
from stk_search.Search_algorithm.Search_algorithm import Search_Algorithm
from stk_search.Search_space import Search_Space
from stk_search.Search_algorithm.Botorch_kernels import (
    TanimotoGP,
    RBFKernel,
    MaternKernel,
)
import itertools

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class BayesianOptimisation(Search_Algorithm):
    def __init__(
        self,
        verbose=False,
        which_acquisition="EI",
        kernel=RBFKernel,
        likelihood=ExactMarginalLogLikelihood,
        model=None,
        lim_counter=2,
        Representation=None,
    ):
        """Initialise the class.
        Args:
            verbose (bool): if True, print the output
            PCA_input (bool): if True, use PCA to reduce the dimension of the input
            normalise_input (bool): if True, normalise the input
            which_acquisition (str): acquisition function to use
            kernel (gpytorch.kernels): kernel to use
            likelihood (gpytorch.likelihoods): likelihood to use
            model (gpytorch.models): model to use
            lim_counter (int): max iteration for the acquisition function optimisation
            Representation (object): representation of the element
        """
        self.verbose = verbose
        # self.normalise_input = normalise_input
        self.which_acquisition = which_acquisition
        self.kernel = kernel
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.likelihood = likelihood
        self.model = model
        self.lim_counter = lim_counter  # max iteration for the acquisition function optimisation
        self.Representation = Representation
        self.name = "Bayesian_Optimisation"
        self.pred_model = None
        self.population_size = 1000

    def update_representation(self, Representation):
        self.Representation = Representation

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
        X_rpr = self.Representation.generate_repr(
            df_search.loc[ids_acquired, :]
        )
        X_rpr = X_rpr.double()
        X_rpr = self.normalise_input(X_rpr)
        y_explored_BO_norm = torch.tensor(
            fitness_acquired, dtype=torch.float64
        )
        y_explored_BO_norm = (
            y_explored_BO_norm - y_explored_BO_norm.mean(axis=0)
        ) / (y_explored_BO_norm.std(axis=0))
        y_explored_BO_norm = y_explored_BO_norm.reshape(-1, 1)
        # train model
        self.train_model(X_rpr, y_explored_BO_norm)
        # optimise the acquisition function
        ids_sorted_by_aquisition, df_elements = (
            self.optimise_acquisition_function(
                best_f=y_explored_BO_norm.max().item(),
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

    def normalise_input(self, X_rpr):
        X_rpr = X_rpr.double()
        # min max scaling the input
        X_rpr = (X_rpr - X_rpr.min(dim=0)[0]) / (
            X_rpr.max(dim=0)[0] - X_rpr.min(dim=0)[0]
        )
        return X_rpr

    def optimise_acquisition_function(
        self,
        best_f,
        fitness_acquired,
        df_search,
        SP,
        benchmark=False,
        df_total=None,
    ):
        """Optimise the acquisition function.
        Args:
            best_f (float): best fitness
            fitness_acquired (list): fitness of the acquired elements
            df_search (pd.DataFrame): search space
            SP (Search_Space): search space
            benchmark (bool): if True, the search space is a benchmark
            df_total (pd.DataFrame): dataframe of the total dataset
        Returns:
            torch.tensor: acquisition values
            pd.DataFrame: updated search space
        """
        # generate list of element to evaluate using acquistion function
        counter, lim_counter = 0, self.lim_counter
        df_elements = self.Generate_element_to_evaluate(
            fitness_acquired, df_search, SP, benchmark, df_total
        )
        Xrpr = self.Representation.generate_repr(df_elements)
        acquisition_values = self.get_acquisition_values(
            best_f=best_f,
            Xrpr=Xrpr,
        )
        df_elements['acquisition_value'] = acquisition_values.detach().numpy()

        if "dataset_local" in self.Representation.__dict__:
            print(
                "size of representation dataset ",
                len(self.Representation.dataset_local),
            )
        # select element to acquire with maximal aquisition value, which is not in the acquired set already
        acquisition_values = acquisition_values.numpy()
        ids_sorted_by_aquisition = -acquisition_values.argsort()
        max_acquisition_value = acquisition_values.max()
        # print('max_acquisition_value is ', max_acquisition_value)
        # print('min_acquisition_value is ', acquisition_values.min())
        max_counter, max_optimisation_iteration = 0, 100
        while counter < lim_counter:
            df_elements_old = df_elements.copy()
            counter += 1
            max_counter += 1
            df_elements = self.Generate_element_to_evaluate(
                acquisition_values,
                df_elements_old.drop(columns="acquisition_value"),
                SP,
                benchmark,
                df_total,
            )
            Xrpr = self.Representation.generate_repr(df_elements)
            acquisition_values = self.get_acquisition_values(
                best_f=best_f,
                Xrpr=Xrpr,
            )
            if "dataset_local" in self.Representation.__dict__:
                print(
                    "size of representation dataset ",
                    len(self.Representation.dataset_local),
                )
            # merge the new elements with the old ones
            df_elements['acquisition_value'] = acquisition_values.detach().numpy()
            df_elements = pd.concat([df_elements_old, df_elements])
            #print('when merged df_elements size is ', df_elements.shape[0])

            df_elements.drop_duplicates(inplace=True)
            df_elements.reset_index(drop=True, inplace=True)
            df_elements = df_elements.sort_values(
                by="acquisition_value", ascending=False
            )
            if df_elements.shape[0] > self.population_size:  
                df_elements = df_elements.loc[:self.population_size]
            acquisition_values = df_elements['acquisition_value'].values
            #print('df_elements size is ', df_elements.shape[0])
            # select element to acquire with maximal aquisition value, which is not in the acquired set already
            ids_sorted_by_aquisition = -acquisition_values.argsort()
            max_acquisition_value_current = acquisition_values.max()
            if (
                max_acquisition_value_current
                > max_acquisition_value + 0.001 * max_acquisition_value
            ):
                max_acquisition_value = max_acquisition_value_current
                # print(
                #   f"counter is {max_counter}, max_acquisition_value is {max_acquisition_value}"
                # )
                counter = 0
            if max_counter > max_optimisation_iteration:
                # print(
                #   f"counter is {max_counter}, max_acquisition_value is {max_acquisition_value}"
                # )
                break
        # print("finished acquisition function optimisation")
        # print(ids_sorted_by_aquisition[:1], df_elements[:1])
        return ids_sorted_by_aquisition, df_elements.drop(columns="acquisition_value")

    def Generate_element_to_evaluate(
        self,
        fitness_acquired,
        df_search,
        SP: Search_Space,
        benchmark=False,
        df_total=None,
    ):
        """Generate elements to evaluate.
        Args:
            fitness_acquired (list): fitness of the acquired elements
            df_search (pd.DataFrame): search space
            SP (Search_Space): search space
            benchmark (bool): if True, the search space is a benchmark
            df_total (pd.DataFrame): dataframe of the total dataset
            Returns:
                pd.DataFrame: elements to evaluate
        """

        #
        def mutate_element(element):
            elements_val = []
            for i in range(element.shape[0]):
                for frag in SP.df_precursors.InChIKey:
                    element_new = element.copy()
                    element_new[i] = frag
                    elements_val.append(element_new)
            return elements_val

        def cross_element(element1, element2):
            elements_val = []
            for i in range(element.shape[0]):
                element_new = element1.copy()
                element_new[i] = element2[i]
                elements_val.append(element_new)
            return elements_val

        # select the 3 best one and add two random element from the search space
        best_element_arg = fitness_acquired.argsort()[-3:][::-1]
        list_parents = df_search.loc[best_element_arg, :].values
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
        df_elements = SP.check_df_for_element_from_SP(df_to_check=df_elements)
        if benchmark:
            # take only element in df_total
            # print("started acquisition function optimisation")
            df_elements = df_elements.merge(
                df_total,
                on=[
                    f"InChIKey_{i}" for i in range(elements.shape[1])
                ],  # check this for generalization
                how="left",
            )
            df_elements.dropna(subset="target", inplace=True)
            df_elements = df_elements[
                [f"InChIKey_{i}" for i in range(elements.shape[1])]
            ]  # check this for generalization
            df_elements.drop_duplicates(inplace=True)
        if (
            df_elements.shape[0] > self.population_size
        ):  # limit the number of elements to evaluate each time
            df_elements = df_elements.sample(1000)
        df_elements.reset_index(drop=True, inplace=True)
        return df_elements

    def train_model(self, X_train, y_train):
        """Train the model.
        Args:
            X_train (torch.tensor): input
            y_train (torch.tensor): output
        """
        self.model = self.kernel(
            X_train,
            y_train,
        )
        mll = self.likelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def get_acquisition_values(self, best_f, Xrpr):
        """Get the acquisition values.
        Args:
            model (gpytorch.models): model
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
                acquisition_values = self.pred_model(
                    Xrpr.float()
                ).squeeze()
                acquisition_values = (
                    acquisition_values
                    + self.model.posterior(X_unsqueezed).variance.squeeze()
                )
            return acquisition_values
        Xrpr = self.normalise_input(Xrpr)
        X_unsqueezed = Xrpr.double()
        X_unsqueezed = X_unsqueezed.reshape(-1, 1, X_unsqueezed.shape[1])
        # set up acquisition function
        if self.which_acquisition == "EI":
            acquisition_function = ExpectedImprovement(self.model, best_f=best_f)
            with torch.no_grad():  # to avoid memory issues; we arent using the gradient...
                acquisition_values = acquisition_function.forward(
                    X_unsqueezed
                )  # runs out of memory
        elif self.which_acquisition == "max_y_hat":
            with torch.no_grad():
                acquisition_values = self.model.posterior(
                    X_unsqueezed
                ).mean.squeeze()
        elif self.which_acquisition == "max_sigma":
            with torch.no_grad():
                acquisition_values = self.model.posterior(
                    X_unsqueezed
                ).variance.squeeze()
        elif self.which_acquisition == "LOG_EI":
            acquisition_function = LogExpectedImprovement(self.model, best_f=best_f)
            with torch.no_grad():  # to avoid memory issues; we arent using the gradient...
                acquisition_values = acquisition_function.forward(
                    X_unsqueezed
                )  # runs out of memory
        
        elif self.which_acquisition == "UCB":
            with torch.no_grad():
                acquisition_values = acquisition_values = (
                    model.posterior(X_unsqueezed).mean.squeeze()
                    + self.model.posterior(X_unsqueezed).variance.squeeze()
                )

                acquisition_values = self.pred_model(X_unsqueezed.float()).squeeze()
                acquisition_values = acquisition_values + self.model.posterior(
                                X_unsqueezed
                            ).variance.squeeze()
        elif self.which_acquisition == "KG":
            acquisition_function = qKnowledgeGradient(model=self.model,num_fantasies= 50)
            bounds = torch.tensor([[0.0] * Xrpr.shape[1], [1.0] * Xrpr.shape[1]], dtype=torch.float64)   
            acquisition_values = acquisition_function.evaluate(
                X_unsqueezed,
                bounds= bounds
            ).detach() 
        else:
            with torch.no_grad():
                acquisition_values = model.posterior(
                    X_unsqueezed
                ).variance.squeeze()
        return acquisition_values
