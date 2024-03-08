# class to define the search algorithm
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stk_search.Search_algorithm.Search_algorithm import Search_Algorithm
from stk_search.Search_space import Search_Space


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Ea_surrogate(Search_Algorithm):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.verbose = False
        self.pred_model = None
        self.name = "Surrogate_EA"


    def suggest_element(
        self,
        search_space_df,
        fitness_acquired,
        ids_acquired,
        SP: Search_Space,
        benchmark=True,
        df_total: pd.DataFrame = None,
    ):
        df_search = search_space_df.copy()
        fitness_acquired = np.array(fitness_acquired)
        # generate list of element to evaluate using acquistion function
        elements = self.Generate_element_to_evaluate(
            fitness_acquired, df_search.loc[ids_acquired, :], SP
        )
        elements = np.append(elements, df_search.values, axis=0)
        df_elements = pd.DataFrame(
            elements, columns=[f"InChIKey_{x}" for x in range(elements.shape[1])]# check this for generalization
        )
        df_elements = SP.check_df_for_element_from_SP(df_to_check=df_elements)
        if benchmark:
            # take only element in df_total
            df_elements = df_elements.merge(
                df_total,
                on=[f"InChIKey_{i}" for i in range(elements.shape[1])],# check this for generalization
                how="left",
            )
            df_elements.drop_duplicates(
                subset=[f"InChIKey_{i}" for i in range(elements.shape[1])], inplace=True # check this for generalization
            )
            df_elements.dropna(subset="target", inplace=True)
            df_elements = df_elements[[f"InChIKey_{i}" for i in range(elements.shape[1])]]# check this for generalization
            if self.verbose:
                print("df_elements shape is ", df_elements.shape)
        X_unsqueezed = self.Representation.generate_repr(df_elements)
        if self.verbose:
            print("X_unsqueezed shape is ", X_unsqueezed.shape)
        # get model prediction
        #make sure that the model and the data have the same dtype
        X_unsqueezed = X_unsqueezed.to(self.device)
        model_dtype = next(self.pred_model.parameters()).dtype
        if X_unsqueezed.dtype != model_dtype:
            X_unsqueezed = X_unsqueezed.type(model_dtype)
        acquisition_values = self.pred_model(X_unsqueezed).squeeze().detach().numpy()
        # select element to acquire with maximal aquisition value, which is not in the acquired set already
        ids_sorted_by_aquisition = (-acquisition_values).argsort()
        if self.verbose:
            print("max acquisition value is ", acquisition_values[ids_sorted_by_aquisition[0]])

        def add_element(df, element):
            if ~(df == element).all(1).any():
                df.loc[len(df)] = element
                return True
            return False
        print('new_element_df shape is ', df_elements.shape)
        for id in ids_sorted_by_aquisition:
            if add_element(df_search, df_elements.values[id.item()]):
                print(id.item())
                break
                # index = id.item()
                # return df_search_space_frag
        return len(df_search) - 1, df_search

    def Generate_element_to_evaluate(
        self, fitness_acquired, df_search, SP: Search_Space
    ):
        import itertools

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
        list_parents = np.append(list_parents, df_search.sample(2).values, axis=0)
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
        return elements
    

