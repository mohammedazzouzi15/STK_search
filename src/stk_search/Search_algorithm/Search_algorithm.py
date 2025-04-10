"""Search algorithm base class.

Class to define the search algorithm
Here we define the base class for the search algorithm. The search algorithm is used to suggest a new element to evaluate from the search space.
We also define the random search and evolution algorithm classes.
"""

import itertools
import os
from typing import Optional

import numpy as np
import pandas as pd

from stk_search.SearchSpace import SearchSpace

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Search_Algorithm:
    """Search algorithm  base class.

    Class to define the search algorithm


    Attributes
    ----------
        name (str): name of the search algorithm
        multi_fidelity (bool): if the search is multi_fidelity
        budget (int): budget of the search



    Functions:

        suggest_element(searchspace_df, fitness_acquired, ids_acquired, bad_ids, SP):
            Suggest an element to evaluate
        initial_suggestion(SP, num_elem_initialisation, benchmark, df_total):
            Initial suggestion of the search space




    """

    def __init__(self):
        """Initialise the search algorithm."""
        self.name = "default"
        self.multi_fidelity = False
        self.budget = None

    def suggest_element(
        self,
        sp: SearchSpace,
        searchspace_df: pd.DataFrame = None,
        fitness_acquired: Optional[list] = None,
        ids_acquired: Optional[list] = None,
        bad_ids: Optional[list] = None,
    ) -> float:
        """Suggest an element to evaluate.

        Args:
        ----
            searchspace_df (pd.DataFrame): dataframe containing the searched space.

            fitness_acquired (list): list of the fitness of the elements
            ids_acquired (list): list of the ids of the elements
            bad_ids (list): list of the ids of the bad elements
            sp (SearchSpace): search space

        Returns:
        -------
            float: id of the element to evaluate

        """

    def initial_suggestion(
        self,
        sp: SearchSpace = None,
        num_elem_initialisation: int = 10,
        benchmark=False,
        df_total: pd.DataFrame = None,
    ):
        """Sugest Initial population of the search space.

        Args:
        ----
            sp (SearchSpace): search space
            num_elem_initialisation (int): number of element to initialise
            benchmark (bool): if the search is a benchmark
            df_total (pd.DataFrame): dataframe containing the results

        Returns:
        -------
            list: list of index of the elements
        pd.DataFrame: dataframe containing the elements.

        """
        if sp is None:
            sp = []
        if benchmark:
            searched_space_df = sp.check_df_for_element_from_sp(
                df_to_check=df_total
            )
            searched_space_df = searched_space_df.sample(
                num_elem_initialisation
            )
        elif df_total is not None:
            searched_space_df = sp.check_df_for_element_from_sp(
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
            searched_space_df = sp.random_generation_df(
                num_elem_initialisation
            )
        # reindex the df
        searched_space_df = searched_space_df[
            ["InChIKey_" + str(i) for i in range(sp.number_of_fragments)]
        ]  # careful here, this is hard coded
        searched_space_df.index = range(len(searched_space_df))
        return searched_space_df.index.tolist(), searched_space_df


class random_search(Search_Algorithm):
    """Random search algorithm.

    Class to define the random search algorithm
    Suggest a random molecule to evalutate from the search space

    Attributes
    ----------
        seed (int): seed for the random search
        name (str): name of the search algorithm
        multi_fidelity (bool): if the search is multi_fidelity
        budget (int): budget of the search

    Functions:

        suggest_element(searchspace_df, fitness_acquired, ids_acquired, bad_ids, SP):
             ramdomly Suggest an element to evaluate from the search space
        initial_suggestion(SP, num_elem_initialisation, benchmark, df_total):
            Initial suggestion of the search space


    """

    def __init__(self, seed=None):
        """Initialise the random search algorithm."""
        self.name = "Random"
        self.seed = seed
        self.multi_fidelity = False
        self.budget = None
        if seed is not None:
            np.random.seed(seed)

    def suggest_element(
        self,
        searchspace_df,
        ids_acquired,
        fitness_acquired,
        sp: SearchSpace,
        benchmark=True,
        df_total: pd.DataFrame = None,
    ):
        """Suggest an element to evaluate.

        Suggest a random element to evaluate from the search space

        Args:
        ----
            searchspace_df (pd.DataFrame): dataframe containing the searched space
            ids_acquired (list): list of the ids of the acquired elements
            fitness_acquired (list): list of the fitness of the elements
            sp (SearchSpace): search space
            benchmark (bool): if the search is a benchmark
            df_total (pd.DataFrame): dataframe containing the results

        Returns:
        -------
            int: id of the element to evaluate
            pd.DataFrame: dataframe containing the element to evaluate

        """
        df_search = searchspace_df.copy()

        def add_element(df, element) -> bool:
            if ~(df == element).all(1).any():
                df.loc[len(df)] = element
                return True
            return False

        while True:
            leav_loop = False
            if benchmark:
                df_elements = sp.check_df_for_element_from_sp(
                    df_to_check=df_total
                )
                df_elements = df_elements.sample(100)
            else:
                df_elements = sp.random_generation_df(10)
            df_elements = df_elements[
                ["InChIKey_" + str(i) for i in range(sp.number_of_fragments)]
            ]
            for _id in df_elements.to_numpy():
                if add_element(df_search, _id):
                    leav_loop = True
                    break
            if leav_loop:
                break

        return len(df_search) - 1, df_search


class evolution_algorithm(Search_Algorithm):
    """Evolution algorithm.

    Class to define the evolution algorithm
    Suggest a molecule to evaluate from the search space using an evolution algorithm

    Attributes
    ----------
        name (str): name of the search algorithm
        selection_method_mutation (str): selection method for the mutation
        selection_method_cross (str): selection method for the cross
        number_of_parents (int): number of parents
        multi_fidelity (bool): if the search is multi_fidelity
        budget (int): budget of the search


    Functions:

        suggest_element(searchspace_df, fitness_acquired, ids_acquired, bad_ids, SP):
            Suggest an element to evaluate
        generate_df_elements_to_choose_from(searchspace_df, fitness_acquired, SP, benchmark, df_total):
            Generate the dataframe of the elements to choose from
        mutate_element(element, SP):
            Mutate an element
        cross_element(element1, element2):
            Cross two elements
        roulette_wheel_selection(fitness_acquired, df_search, size):
            Roulette wheel selection
        tournament_selection(fitness_acquired, df_search, size):
            Tournament selection
        rank_selection(fitness_acquired, df_search, size):
            Rank selection
        top_selection(fitness_acquired, df_search, size, number_of_random):
            Top selection
        run_selection_method(selection_method, fitness_acquired, df_search):
            Run the selection method
        Generate_element_to_evaluate(fitness_acquired, df_search, SP):
            Generate the elements to evaluate


    """

    def __init__(self):
        """Initialise the evolution algorithm."""
        self.name = "Evolution_algorithm"
        self.selection_method_mutation = "top"
        self.selection_method_cross = "top"
        self.number_of_parents = 5
        self.number_of_random = 2
        self.multi_fidelity = False
        self.budget = None
        self.num_added_random = 0

    def suggest_element(
        self,
        searchspace_df,
        ids_acquired,
        fitness_acquired,
        sp: SearchSpace,
        benchmark=True,
        df_total: pd.DataFrame = None,
        error_counter_lim=10,
    ):
        """Suggest an element to evaluate.

        Start the algorithm by generating a list of offspring from the parents
        the list of offspring is generated by mutation and cross-over of the parents
        the selection of the parents is done using the selection method over the list of molecules in the searched space
        the selection method is defined by the user
        the selection method can be "roulette", "tournament", "rank" or "top".

        Args:
        ----
            searchspace_df (pd.DataFrame): dataframe containing the searched space
            ids_acquired (list): list of the ids of the acquired elements
            fitness_acquired (list): list of the fitness of the elements
            sp (SearchSpace): search space
            benchmark (bool): if the search is a benchmark
            df_total (pd.DataFrame): dataframe containing the results
            error_counter_lim (int): limit of the error counter

        Returns:
        -------
            int: id of the element to evaluate
            pd.DataFrame: dataframe containing the element to evaluate

        """
        df_search = searchspace_df
        df_elements = searchspace_df
        error_counter = 0
        while not self._check_new_element_in_SearchSpace(
            df_search, df_elements
        ):
            df_elements, df_search = self.generate_df_elements_to_choose_from(
                searchspace_df,
                fitness_acquired,
                sp,
                benchmark,
                df_total,
            )
            error_counter = error_counter + 1
            if error_counter > error_counter_lim:
                df_elements = df_elements.drop_duplicates()
                msg = "no new element found"
                raise ValueError(msg)

        def add_element(df, element) -> bool:
            if ~(df == element).all(1).any():
                df.loc[len(df)] = element
                return True
            return False

        df_elements_shuffled = df_elements.sample(frac=1).reset_index(
            drop=True
        )
        for element in df_elements_shuffled.to_numpy():
            if add_element(df_search, element):
                break

        return len(df_search) - 1, df_search

    def _check_new_element_in_SearchSpace(
        self, df_search, df_elements
    ) -> bool:
        """Check if the element is already in the search space.

        Args:
        ----
            df_search (pd.DataFrame): dataframe containing the searched space
            df_elements (pd.DataFrame): dataframe containing the elements to choose from

        Returns:
        -------
            bool: True if the element is not in the search space, False otherwise

        """
        df_search_copy = df_search.copy()
        df_search_copy = df_search_copy[df_elements.columns]
        all_df = df_elements.merge(
            df_search_copy, how="left", indicator="exists"
        )
        all_df["exists"] = np.where(all_df.exists == "both", True, False)
        return all_df[~all_df["exists"]].shape[0] > 0

    def generate_df_elements_to_choose_from(
        self,
        searchspace_df,
        fitness_acquired,
        sp: SearchSpace,
        benchmark=True,
        df_total: pd.DataFrame = None,
    ):
        """Generate the dataframe of the elements to choose from.

        Args:
        ----
            searchspace_df (pd.DataFrame): dataframe containing the searched space
            fitness_acquired (list): list of the fitness of the elements
            sp (SearchSpace): search space
            benchmark (bool): if the search is a benchmark
            df_total (pd.DataFrame): dataframe containing the results

        Returns:
        -------
            pd.DataFrame: dataframe containing the elements to choose from
            pd.DataFrame: dataframe containing the searched space

        """
        df_search = searchspace_df.copy()
        fitness_acquired = np.array(fitness_acquired)
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
            print("df_elements", df_elements.shape)
            print("df_total", df_total.shape)
            print(
                "df_total in searchspace_df",
                sp.check_df_for_element_from_sp(df_to_check=df_total).shape,
            )

            df_elements = df_elements.merge(
                df_total,
                on=[
                    f"InChIKey_{i}" for i in range(elements.shape[1])
                ],  # check this for generalization
                how="left",
            )

            df_elements = df_elements.dropna(subset="target")
            print(
                "df_total in searchspace_df",
                max(df_elements["target"]),
                np.mean(df_elements["target"]),
            )
            df_elements = df_elements[
                [f"InChIKey_{i}" for i in range(elements.shape[1])]
            ]  # check this for generalization
            print("df_elements", df_elements.shape)
        return df_elements, df_search

    def mutate_element(self, element, sp: SearchSpace):
        """Mutate an element.

        Change one fragment of the element with all the other from the search space and return the list of elements

        Args:
        ----
            element (np.array): element to mutate
            sp (SearchSpace): search space

        Returns:
        -------
            list: list of elements

        """
        elements = []
        for i in range(element.shape[0]):  # check this for generalization
            for frag in sp.df_precursors.InChIKey:
                element_new = element.copy()
                element_new[i] = frag
                elements.append(element_new)

        return elements

    def cross_element(self, element1, element2):
        """Cross two elements.

        Cross two elements by complementing one part of the first element with the rest from the other element

        Args:
        ----
            element1 (np.array): element 1
            element2 (np.array): element 2

        Returns:
        -------
            list: list of elements

        """
        elements = []
        for i in range(element1.shape[0]):  # check this for generalization
            element_new = element1.copy()
            element_new[i] = element2[i]
            elements.append(element_new)
        return elements

    def roulette_wheel_selection(self, fitness_acquired, df_search, size=3):
        """Roulette wheel selection.

        Select parents based on the roulette wheel selection

        Args:
        ----
            fitness_acquired (list): fitness of the acquired elements
            df_search (pd.DataFrame): dataframe containing the searched space
            size (int): number of parents

        Returns:
        -------
            list: list of parents

        """
        total_fitness = np.sum(fitness_acquired)
        selection_probs = fitness_acquired / total_fitness
        selected_indices = np.random.choice(
            df_search.index, size=size, p=selection_probs
        )
        return df_search.iloc[selected_indices].to_numpy()

    def tournament_selection(self, fitness_acquired, df_search, size=3):
        """Tournament selection.

        Select parents based on the tournament selection

        Args:
        ----
            fitness_acquired (list): fitness of the acquired elements
            df_search (pd.DataFrame): dataframe containing the searched space
            size (int): number of parents

        Returns:
        -------
            list: list of parents

        """
        selected_indices = []
        for _ in range(size):
            tournament_indices = np.random.choice(df_search.index, size=2)
            tournament_fitness = fitness_acquired[tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices.append(winner_index)
        return df_search.iloc[selected_indices].to_numpy()

    def rank_selection(self, fitness_acquired, df_search, size=3):
        """Rank selection.

        Select parents based on the rank selection

        Args:
        ----
            fitness_acquired (list): fitness of the acquired elements
            df_search (pd.DataFrame): dataframe containing the searched space
            size (int): number of parents

        Returns:
        -------
            list: list of parents

        """
        ranks = np.argsort(np.argsort(fitness_acquired))
        total_ranks = np.sum(ranks)
        selection_probs = ranks / total_ranks
        selected_indices = np.random.choice(
            df_search.index, size=size, p=selection_probs
        )
        return df_search.iloc[selected_indices].to_numpy()

    def top_selection(
        self, fitness_acquired, df_search, size=5, number_of_random=2
    ):
        """Top selection.

        Select parents based on the top selection

        Args:
        ----
            fitness_acquired (list): fitness of the acquired elements
            df_search (pd.DataFrame): dataframe containing the searched space
            size (int): number of parents
            number_of_random (int): number of random elements to add

        Returns:
        -------
            list: list of parents

        """
        fitness_acquired = np.argsort(np.array(fitness_acquired))
        top_indices = fitness_acquired[-size + number_of_random :]
        random_indices = np.random.choice(
            df_search.shape[0], size=number_of_random
        )

        indices_considered = np.append(top_indices, random_indices)
        # print(df_search.iloc[top_indices].to_numpy()[0])
        return df_search.iloc[indices_considered].to_numpy()

    def run_selection_method(
        self, selection_method, fitness_acquired, df_search
    ):
        """Run the selection method.

        Select parents based on the chosen selection method

        Args:
        ----
            selection_method (str): selection method
            fitness_acquired (list): fitness of the acquired elements
            df_search (pd.DataFrame): dataframe containing the searched space

        Returns:
        -------
            list: list of parents

        """
        # Select parents based on the chosen selection method
        if selection_method == "roulette":
            list_parents = self.roulette_wheel_selection(
                fitness_acquired, df_search
            )
        elif selection_method == "tournament":
            list_parents = self.tournament_selection(
                fitness_acquired, df_search, self.number_of_parents
            )
        elif selection_method == "rank":
            list_parents = self.rank_selection(
                fitness_acquired, df_search, self.number_of_parents
            )
        elif selection_method == "top":
            list_parents = self.top_selection(
                fitness_acquired,
                df_search,
                self.number_of_parents,
                self.number_of_random,
            )
        else:
            msg = f"Unknown selection method: {selection_method}"
            raise ValueError(msg)
        return list_parents

    def Generate_element_to_evaluate(
        self, fitness_acquired, df_search, sp: SearchSpace
    ):
        """Generate the elements to evaluate.

        Generate the elements to evaluate by mutation and cross-over of the parents

        Args:
        ----
            fitness_acquired (list): fitness of the acquired elements
            df_search (pd.DataFrame): dataframe containing the searched space
            sp (SearchSpace): search space

        Returns:
        -------
            list: list of elements

        """
        elements = []
        elements_to_mutate = self.run_selection_method(
            self.selection_method_mutation, fitness_acquired, df_search
        )
        elements_to_cross = self.run_selection_method(
            self.selection_method_cross, fitness_acquired, df_search
        )
        for element in elements_to_mutate:
            if len(elements) == 0:
                elements = self.mutate_element(element, sp)
            else:
                elements = np.append(
                    elements, self.mutate_element(element, sp), axis=0
                )
        for element1, element2 in itertools.product(
            elements_to_cross, elements_to_cross
        ):
            if len(elements) == 0:
                elements = self.cross_element(element1, element2)
            else:
                elements = np.append(
                    elements, self.cross_element(element1, element2), axis=0
                )
    
        searched_space_df = sp.random_generation_df(self.num_added_random)
        elements = np.append(elements, searched_space_df.to_numpy(), axis=0)
        # print("shape of elements", elements.shape)
        return elements
