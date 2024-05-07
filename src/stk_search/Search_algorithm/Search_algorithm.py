# class to define the search algorithm
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stk_search.Search_space import Search_Space

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Search_Algorithm:
    def __init__(self):
        self.name = "default"
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
        SP: Search_Space = [],
        num_elem_initialisation: int = 10,
        benchmark=False,
        df_total: pd.DataFrame = None,
    ):
        """Initial suggestion of the search space
        Args:
            SP (Search_Space): search space
            num_elem_initialisation (int): number of element to initialise
            benchmark (bool): if the search is a benchmark
            df_total (pd.DataFrame): dataframe containing the results
        Returns:
            list: list of index of the elements
            pd.DataFrame: dataframe containing the elements"""
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
        searched_space_df = searched_space_df[
            ["InChIKey_" + str(i) for i in range(SP.number_of_fragments)]
        ]  # careful here, this is hard coded
        searched_space_df.index = range(len(searched_space_df))
        return searched_space_df.index.tolist(), searched_space_df


class random_search(Search_Algorithm):
    def __init__(self, seed=None):
        self.name = "Random"
        if seed is not None:
            np.random.seed(seed)

    def suggest_element(
        self,
        search_space_df,
        ids_acquired,
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
            df_elements = df_elements[
                ["InChIKey_" + str(i) for i in range(SP.number_of_fragments)]
            ]
            for id in df_elements.values:
                if add_element(df_search, id):
                    print(id)
                    leav_loop = True
                    break
            if leav_loop:
                break

        print(df_search.shape)
        return len(df_search) - 1, df_search


class evolution_algorithm(Search_Algorithm):
    def __init__(self):
        self.name = "Evolution_algorithm"
        pass

    def Generate_element_to_evaluate(
        self, fitness_acquired, df_search, SP: Search_Space
    ):
        import itertools

        def mutate_element(element):
            elements = []
            for i in range(element.shape[0]):  # check this for generalization
                for frag in SP.df_precursors.InChIKey:
                    element_new = element.copy()
                    element_new[i] = frag
                    elements.append(element_new)
            return elements

        def cross_element(element1, element2):
            elements = []
            for i in range(element.shape[0]):  # check this for generalization
                element_new = element1.copy()
                element_new[i] = element2[i]
                elements.append(element_new)
            return elements

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
        return elements

    def suggest_element(
        self,
        search_space_df,
        ids_acquired,
        fitness_acquired,
        SP: Search_Space,
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

        def add_element(df, element):
            if ~(df == element).all(1).any():
                df.loc[len(df)] = element
                return True
            return False

        for _ in range(len(df_elements)):
            elem_id = df_elements.values[np.random.randint(len(df_elements))]
            if add_element(df_search, elem_id):
                print(elem_id)
                break

        return len(df_search) - 1, df_search

    def generate_df_elements_to_choose_from(
        self,
        search_space_df,
        fitness_acquired,
        SP: Search_Space,
        benchmark=True,
        df_total: pd.DataFrame = None,
    ):
        df_search = search_space_df.copy()
        fitness_acquired = np.array(fitness_acquired)
        elements = self.Generate_element_to_evaluate(
            fitness_acquired, df_search, SP
        )
        #elements = np.append(elements, df_search.values, axis=0)
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
            df_elements = df_elements[
                [f"InChIKey_{i}" for i in range(elements.shape[1])]
            ]  # check this for generalization
            # print(df_elements.shape)
        return df_elements, df_search
