# class to define the search algorithm
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stk_search.Search_space import Search_Space
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
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def suggest_element(
        self,
        search_space_df,
        fitness_acquired,
        SP: Search_Space,
        benchmark=True,
        df_total: pd.DataFrame = None,
    ):
        df_search = search_space_df.copy()

        def add_element(df, element):
            if ~(df == element).all(1).any():
                df.loc[len(df)] = element
                return True
            return False

        while True:
            leav_loop = False
            if benchmark:
                df_elements = SP.check_df_for_element_from_SP(
                    df_to_check=df_total
                )
                df_elements = df_elements.sample(10)
            else:
                df_elements = SP.random_generation_df(10)
            df_elements=df_elements[["InChIKey_" + str(i) for i in range(SP.number_of_fragments)]]
            for id in df_elements.values:
                if add_element(df_search, id):
                    print(id)
                    leav_loop = True
                    break
            if leav_loop:
                break

        print(df_search.shape)
        return len(df_search) - 1, df_search

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
            searched_space_df = SP.random_generation_df(
                num_elem_initialisation
            )
        # reindex the df
        searched_space_df = searched_space_df[
            ["InChIKey_" + str(i) for i in range(SP.number_of_fragments)]
        ]  # careful here, this is hard coded
        searched_space_df.index = range(len(searched_space_df))
        return searched_space_df.index.tolist(), searched_space_df


class evolution_algorithm(Search_Algorithm):
    def __init__(self):
        pass
    def suggest_element(
        self,
        search_space_df,
        fitness_acquired,
        SP: Search_Space,
        benchmark=True,
        df_total: pd.DataFrame = None,
    ):
        df_search = search_space_df.copy()
        fitness_acquired = np.array(fitness_acquired)
        def mutate_element(element):
            elements = []
            for i in range(6):
                for frag in SP.df_precursors.InChIKey:
                    element_new = element.copy()
                    element_new[i] = frag
                    elements.append(element_new)
            return elements

        best_element_arg = fitness_acquired.argsort()[-3:][::-1]
        elements = []
        for element in df_search.loc[best_element_arg].values:
            if len(elements) == 0:
                elements = mutate_element(element)
            else:
                elements = np.append(elements, mutate_element(element), axis=0)
        
        elements = np.append(elements, df_search.values, axis=0)
        df_elements = pd.DataFrame(
            elements, columns=[f"InChIKey_{x}" for x in range(6)]
        )
        df_elements = SP.check_df_for_element_from_SP(df_to_check=df_elements)
        if benchmark:
            # take only element in df_total
            df_elements = df_elements.merge(
                df_total,
                on=[f"InChIKey_{i}" for i in range(6)],
                how="left",
            )
            df_elements.dropna(subset="target", inplace=True)
            df_elements = df_elements[[f"InChIKey_{i}" for i in range(6)]]
            print(df_elements.shape)

        def add_element(df, element):
            if ~(df == element).all(1).any():
                df.loc[len(df)] = element
                return True
            return False

        while True:
            id = df_elements.values[np.random.randint(len(df_elements))]
            if add_element(df_search, id):
                print(id)
                break

        print(df_search.shape)
        return len(df_search) - 1, df_search

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
            searched_space_df = SP.random_generation_df(
                num_elem_initialisation
            )
        # reindex the df
        searched_space_df = searched_space_df[
            ["InChIKey_" + str(i) for i in range(SP.number_of_fragments)]
        ]  # careful here, this is hard coded
        searched_space_df.index = range(len(searched_space_df))
        return searched_space_df.index.tolist(), searched_space_df