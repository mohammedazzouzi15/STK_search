"""Contains the class Ea_surrogate."""
import os

import pandas as pd
import torch

from stk_search.geom3d import pl_model
from stk_search.Representation import Representation_poly_3d
from stk_search.Search_algorithm.Search_algorithm import evolution_algorithm
from stk_search.SearchSpace import SearchSpace
from stk_search.utils.config_utils import read_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Ea_surrogate(evolution_algorithm):
    """EA with surrogate model.

    Compared to the EA, here we need a surrogate model and a molecule Representation to run the search
    the surrogate model applied on the molecule Representation is used to select a new molecule to evaluate.
    the generation of offspring is the same as in the EA


    Attributes
    ----------
    device: str
        the device to use
    model: torch.nn.Module
        the surrogate model
    verbose: bool
        if True, print more information
    pred_model: torch.nn.Module
        the prediction model
    Representation: Representation
        the Representation of the molecules
    name: str
        the name of the search algorithm
    selection_method_mutation: str
        the selection method for the mutation
    selection_method_cross: str
        the selection method for the crossover
    number_of_parents: int
        the number of parents
    multi_fidelity: bool
        if True, the search is multi-fidelity
    budget: int
        the budget of the search
    config_dir: str
        the directory of the config file for the model

    Functions
    ---------
    suggest_element(searchspace_df, fitness_acquired, ids_acquired, sp, benchmark=True, df_total=None)
        suggest a new element to evaluate
    load_representation_model()
        load the Representation and the prediction model
    
    """

    def __init__(self):
        """Initialise the class."""
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.verbose = False
        self.pred_model = None
        self.Representation = None
        self.name = "Surrogate_EA"
        self.selection_method_mutation = "top"
        self.selection_method_cross = "top"
        self.number_of_parents = 5
        self.multi_fidelity = False
        self.budget = None
        self.config_dir = ""

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

        The element is selected using the surrogate model and the Representation of the molecules.
        here we try 10 time to select a new element, if we can't find a new element, we raise an error.

        Args:
        ----
            searchspace_df: pd.DataFrame
                the search space
            fitness_acquired: np.array
                the fitness of the acquired elements
            ids_acquired: np.array
                the ids of the acquired elements
            sp: SearchSpace
                the search space object
            benchmark: bool
                if True, the benchmark is used to evaluate the fitness
            df_total: pd.DataFrame
                the total dataframe containing the data

        Returns:
        -------
            int: the id of the element to evaluate
            pd.DataFrame: the dataframe containing the element to evaluate

        """
        for _ in range(10):
            df_elements, df_search = self.generate_df_elements_to_choose_from(
                searchspace_df,
                fitness_acquired,
                sp,
                benchmark,
                df_total,
            )
            # get the best using the surrogate model
            x_unsqueezed = self.Representation.generate_repr(df_elements)
            if self.verbose:
                pass
            # get model prediction
            # make sure that the model and the data have the same dtype
            x_unsqueezed = x_unsqueezed.to(self.device)
            model_dtype = next(self.pred_model.parameters()).dtype
            if x_unsqueezed.dtype != model_dtype:
                x_unsqueezed = x_unsqueezed.type(model_dtype)
            acquisition_values = (
                self.pred_model(x_unsqueezed).squeeze().cpu().detach().numpy()
            )
            # select element to acquire with maximal aquisition value, which is not in the acquired set already
            ids_sorted_by_aquisition = (-acquisition_values).argsort()
            if self.verbose:
                pass

            def add_element(df, element) -> bool:
                if ~(df == element).all(1).any():
                    df.loc[len(df)] = element
                    return True
                return False

            for elem_id in ids_sorted_by_aquisition:
                element = df_elements.to_numpy()[elem_id.item()]
                if add_element(df_search, element):
                    return len(df_search) - 1, df_search
        # if all elements are already in the acquired set, return the last one
        raise ValueError("All elements are already in the acquired set")


    def load_representation_model(self):
        """Load the Representation and the prediction model.

        uses the config file to load the model and the Representation.
        
        Returns
        -------
            Representation: the Representation of the molecules
            Pymodel: the prediction model

        """
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
        self.Representation = Representation_poly_3d.RepresentationPoly3d(
            pymodel,
            mongo_client=config["pymongo_client"],
            database=config["database_name"],
            device=pymodel.device,
        )
        self.pred_model = pymodel.graph_pred_linear
        return self.Representation, pymodel
