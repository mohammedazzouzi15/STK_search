# class to define the search algorithm
import os
import torch
import numpy as np
import pandas as pd
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from stk_search.Search_algorithm.Search_algorithm import Search_Algorithm
from stk_search.Search_space import Search_Space
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.acquisition import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.analytic import ExpectedImprovement

import itertools


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class MultifidelityBayesianOptimisation(Search_Algorithm):
    def __init__(
        self,
        verbose=False,
        which_acquisition="KG",
        kernel=SingleTaskMultiFidelityGP,
        likelihood=ExactMarginalLogLikelihood,
        model=None,
        lim_counter=2,
        Representation=None,
        fidelity_col=72,
        budget=None
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
        self.which_acquisition=which_acquisition
        #self.normalise_input = normalise_input
        self.kernel = kernel
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.likelihood = likelihood
        self.model = model
        self.lim_counter = lim_counter  # max iteration for the acquisition function optimisation
        self.Representation = Representation
        self.name = "Multifidelity_Bayesian_Optimisation"
        self.pred_model = None
        self.fidelity_col = fidelity_col
        self.multiFidelity = True
        self.budget = budget

    def initial_suggestion(
        self,
        SP: Search_Space = [],
        num_elem_initialisation: int = 10,
        benchmark=False,
        df_total: pd.DataFrame = None,
    ):
        if benchmark:
            searched_space_df = SP.check_df_for_element_from_SP(
                df_to_check=df_total
            )
            searched_space_df = searched_space_df.sample(
                num_elem_initialisation
            )
        else:
            if df_total is not None:
                searched_space_df = SP.check_df_for_element_from_SP(
                    df_to_check=df_total
                )
                # add top elements from the search space
                searched_space_df = searched_space_df.sort_values(
                    by="target", ascending=False
                )
                searched_space_df = pd.concat(
                    [
                        searched_space_df.sample(num_elem_initialisation - 10),
                        searched_space_df[:10],
                    ]
                )

            else:
                searched_space_df = SP.random_generation_df(
                    num_elem_initialisation
                )
        # reindex the df
        filtered_cols = ["InChIKey_" + str(i) for i in range(SP.number_of_fragments)]
        searched_space_df = searched_space_df[
            filtered_cols
        ]  # careful here, this is hard coded
        # This ensures we take both hf and lf values for intitial sample 
        searched_space_df = searched_space_df.merge(
            df_total,
            on=filtered_cols,  # check this for generalization
            how="left",
        )

        filtered_cols.append("fidelity")
        searched_space_df = searched_space_df[filtered_cols]
        searched_space_df.index = range(len(searched_space_df))
        if self.budget is not None:
            self.budget -= searched_space_df['fidelity'].sum()
        return searched_space_df.index.tolist(), searched_space_df
    
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
        repr = df_search.loc[ids_acquired, :]

        X_rpr = self.generate_rep_with_fidelity(repr)
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

        for element_id in ids_sorted_by_aquisition:
            if self.add_element(df_search, df_elements.values[element_id.item()]):
                break
        return len(df_search) - 1, df_search

    # Add the new element to the search space. It checks if the element is already in the
    # df which is usually the df_search (i.e. those elts formally evaulated by the OF).
    def add_element(self, df, element):
        if ~(df == element).all(1).any():
            if self.budget is not None:
                self.budget -= element[-1]
            df.loc[len(df)] = element
            return True
        return False

    def normalise_input(self, X_rpr):
        X_rpr = X_rpr.double()
        # min max scaling the input
        X_rpr = (X_rpr - X_rpr.min(dim=0)[0]) / (X_rpr.max(dim=0)[0] - X_rpr.min(dim=0)[0])
        return torch.tensor(pd.DataFrame(X_rpr).fillna(0.5).to_numpy())

# This should be edited since how we evaluate the generated elements needs to change. -EJ
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
        df_elements = self.generate_element_to_evaluate(
            fitness_acquired, df_search.iloc[:,:-1], SP, benchmark, df_total
        )

        Xrpr = self.generate_rep_with_fidelity(df_elements)
        
        acquisition_values = self.get_acquisition_values(
            self.model,
            Xrpr=Xrpr,
            best_f = best_f
        )

        if "dataset_local" in self.Representation.__dict__:
            print(
                "size of representation dataset ",
                len(self.Representation.dataset_local),
            )
        # select element to acquire with maximal aquisition value, which is not in the acquired set already
        ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
        max_acquisition_value = acquisition_values.max()

        max_counter, max_optimisation_iteration = 0, 100
        while counter < lim_counter:
            counter += 1
            max_counter += 1
            df_elements = self.generate_element_to_evaluate(
                acquisition_values.detach().numpy(), df_elements.iloc[:,:-1], SP, benchmark, df_total
            )

            Xrpr = self.generate_rep_with_fidelity(df_elements)
            
            acquisition_values = self.get_acquisition_values(
                self.model,
                best_f=best_f,
                Xrpr=Xrpr,
            )
            if "dataset_local" in self.Representation.__dict__:
                print("size of representation dataset ", len(self.Representation.dataset_local),)
            # select element to acquire with maximal aquisition value, which is not in the acquired set already
            ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
            max_acquisition_value_current = acquisition_values.max()
            if (
                max_acquisition_value_current
                > max_acquisition_value + 0.001 * max_acquisition_value
            ):
                max_acquisition_value = max_acquisition_value_current
                #print(
                 #   f"counter is {max_counter}, max_acquisition_value is {max_acquisition_value}"
                #)
                counter = 0
            if max_counter > max_optimisation_iteration:
                break
        return ids_sorted_by_aquisition, df_elements

# Similar to the BO case, except when the elements are generated at the end we add the fidelity
# data as well -EJ
    def generate_element_to_evaluate(
        self,
        fitness_acquired,
        df_search,
        SP: Search_Space,
        benchmark=False,
        df_total=None,
    ):
        """ Generate elements to evaluate.
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
            df_elements = df_elements.merge(
                df_total,
                on=[
                    f"InChIKey_{i}" for i in range(elements.shape[1])
                ],  # check this for generalization
                how="left",
            )
            df_elements.dropna(subset="target", inplace=True)
            columns = [f"InChIKey_{i}" for i in range(elements.shape[1])]
            columns.append('fidelity')
            df_elements = df_elements[
                columns
            ]  # check this for generalization
            df_elements.drop_duplicates(inplace=True)
        if (
            df_elements.shape[0] > 10
        ):  # limit the number of elements to evaluate each time
            df_elements = df_elements.sample(10)
        df_elements.reset_index(drop=True, inplace=True)
        return df_elements

# Hardcoded values for the columns at the moment for the data-fidelities - EJ
    def train_model(self, X_train, y_train):
        """Train the model.
        Args:
            X_train (torch.tensor): input
            y_train (torch.tensor): output
        """
        self.model = self.kernel(
            X_train,
            y_train,
            data_fidelity=self.fidelity_col
        )
        mll = self.likelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def get_acquisition_values(self, model, best_f, Xrpr):
        """Get the acquisition values.
        Args:
            model (gpytorch.models): model
            best_f (float): best fitness
            Xrpr (torch.tensor): representation of the element
        Returns:
            torch.tensor: acquisition values"""
        
        X_unsqueezed = Xrpr.double()
        X_unsqueezed = X_unsqueezed.reshape(-1, 1, X_unsqueezed.shape[1])
        # set up acquisition function
        if self.which_acquisition == "KG":
            bounds = torch.tensor([[0.0] * Xrpr.shape[1], [1.0] * Xrpr.shape[1]], dtype=torch.float64)   
            target_fidelities = {self.fidelity_col:1}
            cost_model = AffineFidelityCostModel(fidelity_weights=target_fidelities, fixed_cost=1.0)
            cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

            curr_val_acqf = FixedFeatureAcquisitionFunction(
                    acq_function=PosteriorMean(model),
                    d=Xrpr.shape[1],
                    columns=[Xrpr.shape[1]-1],
                    values=[1],
                )                
            _, current_value = optimize_acqf(
                    acq_function=curr_val_acqf,
                    bounds=bounds[:,:-1],
                    q=1,
                    num_restarts= 2,
                    raw_samples=4
            )
            acquisition_function = qMultiFidelityKnowledgeGradient(
                model=model,
                num_fantasies= 50,
                cost_aware_utility=cost_aware_utility,
                project=lambda x: project_to_target_fidelity(X=x, target_fidelities=target_fidelities),
                current_value=current_value
            )
            # with torch.no_grad():  # to avoid memory issues; we arent using the gradient...
            acquisition_values = acquisition_function.evaluate(
                        X_unsqueezed,
                        bounds=bounds
                    ).detach()  # runs out of memory
            
        elif self.which_acquisition == "MES":
            acquisition_values=self.MES( model, Xrpr, X_unsqueezed)
        elif self.which_acquisition == "TVR":
            acquisition_values = self.TVR( model, Xrpr, best_f)

        elif self.which_acquisition == "custom":
            mes = self.MES(model, Xrpr, X_unsqueezed)
            normalized_mes= mes / torch.sqrt(torch.sum(mes**2))
            tvr = self.TVR(model, Xrpr, best_f)
            normalized_tvr = tvr / torch.sqrt(torch.sum(tvr**2))
            acquisition_values= normalized_mes + normalized_tvr
        else:
            # with torch.no_grad():
            acquisition_values = model.posterior(
                    X_unsqueezed
                ).variance.squeeze()
        return acquisition_values
   
    def TVR(self, model, Xrpr, best_f):
        Xrpr_hf = Xrpr[np.where(Xrpr[:,-1]==1)]

        acquisition = ExpectedImprovement( model=model, best_f= best_f)

        acquisition_scores = acquisition.forward(Xrpr_hf.reshape(-1,1, Xrpr_hf.shape[1]) ).detach()
        max_hf_ind = acquisition_scores.argmax()

        index_in_xrpr = Xrpr.tolist().index(Xrpr_hf[max_hf_ind].tolist())

        posterior = model.posterior(Xrpr)

        pcov = posterior.distribution.covariance_matrix
        p_var = posterior.variance
        hf_max_cov = pcov[index_in_xrpr]
        hf_max_var = hf_max_cov[index_in_xrpr]
        cost = Xrpr[:, -1]

        return hf_max_cov ** 2 / (p_var.reshape(-1) * hf_max_var * cost)
    
    def MES(self, model, Xrpr, X_unsqueezed):
        bounds = torch.tensor([[0.0] * Xrpr.shape[1], [1.0] * Xrpr.shape[1]], dtype=torch.float64)
        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(10000, 1)   
        target_fidelities = {self.fidelity_col:1}
        cost_model = AffineFidelityCostModel(fidelity_weights=target_fidelities, fixed_cost=1.0)
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        acquisition_function = qMultiFidelityMaxValueEntropy(
            model=model,
            cost_aware_utility=cost_aware_utility,
            project=lambda x: project_to_target_fidelity(X=x, target_fidelities=target_fidelities),
            candidate_set=candidate_set,
        )
        # with torch.no_grad():  # to avoid memory issues; we arent using the gradient...
        return acquisition_function(
                    X_unsqueezed,
                ).detach()  # runs out of memory
        
    def generate_rep_with_fidelity(self, df_elements):
        repr = df_elements.drop(columns = df_elements.columns[-1])
        Xrpr = self.Representation.generate_repr(repr)
        Xrpr = self.normalise_input(Xrpr)
        fid = torch.tensor(df_elements[["fidelity"]].to_numpy(), dtype=torch.float64)
        return torch.concat([Xrpr, fid], dim=1)