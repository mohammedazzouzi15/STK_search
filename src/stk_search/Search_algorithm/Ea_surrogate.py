# class to define the search algorithm
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stk_search.Search_algorithm.Search_algorithm import evolution_algorithm
from stk_search.SearchSpace import SearchSpace
from stk_search.geom3d import pl_model
import torch.nn.functional as Functional
from stk_search.geom3d import train_models
from stk_search.Representation import Representation_poly_3d
from stk_search.utils.config_utils import read_config


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Ea_surrogate(evolution_algorithm):
    """

    Class to run the surrogate EA algorithm
    Compared to the EA, here we need a surrogate model and a molecule representation to run the search
    the surrogate model applied on the molecule representation is used to select a new molecule to evaluate.

    the generation of offspring is the same as in the EA

    Args

    """

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.verbose = False
        self.pred_model = None
        self.representation = None
        self.name = "Surrogate_EA"
        self.selection_method_mutation = "top"
        self.selection_method_cross = "top"
        self.number_of_parents = 5
        self.multiFidelity = False
        self.budget = None
        self.config_dir =""

    def suggest_element(
        self,
        search_space_df,
        fitness_acquired,
        ids_acquired,
        SP: SearchSpace,
        benchmark=True,
        df_total: pd.DataFrame = None,
    ):
        df_elements, df_search = self.generate_df_elements_to_choose_from(
            search_space_df,
            fitness_acquired,
            SP,
            benchmark,
            df_total,
        )
        # get the best using the surrogate model
        X_unsqueezed = self.Representation.generate_repr(df_elements)
        if self.verbose:
            print("X_unsqueezed shape is ", X_unsqueezed.shape)
        # get model prediction
        # make sure that the model and the data have the same dtype
        X_unsqueezed = X_unsqueezed.to(self.device)
        model_dtype = next(self.pred_model.parameters()).dtype
        if X_unsqueezed.dtype != model_dtype:
            X_unsqueezed = X_unsqueezed.type(model_dtype)
        acquisition_values = (
            self.pred_model(X_unsqueezed).squeeze().cpu().detach().numpy()
        )
        # select element to acquire with maximal aquisition value, which is not in the acquired set already
        ids_sorted_by_aquisition = (-acquisition_values).argsort()
        if self.verbose:
            print(
                "max acquisition value is ",
                acquisition_values[ids_sorted_by_aquisition[0]],
            )

        def add_element(df, element):
            if ~(df == element).all(1).any():
                df.loc[len(df)] = element
                return True
            return False

        print("new_element_df shape is ", df_elements.shape)
        for elem_id in ids_sorted_by_aquisition:
            element = df_elements.values[elem_id.item()]
            if add_element(df_search, element):
                print(elem_id.item())
                break
                # index = id.item()
                # return df_search_space_frag
        return len(df_search) - 1, df_search

    def load_representation_model(self):
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
        # pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
        pymodel.to(config["device"])
        Representation = Representation_poly_3d.Representation_poly_3d(
            pymodel,
            mongo_client=config["pymongo_client"],
            database=config["database_name"],
            device=pymodel.device,
        )
        self.pred_model = pymodel.graph_pred_linear
        self.Representation = Representation

        return Representation, pymodel
