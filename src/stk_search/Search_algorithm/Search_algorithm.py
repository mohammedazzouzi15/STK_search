# class to define the search algorithm
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stk_search.SearchSpace import SearchSpace
import itertools

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Search_Algorithm:
    """
    Search algorithm  base class
    
    
    Class to define the search algorithm
    
    
    Attributes:
        name (str): name of the search algorithm
        multiFidelity (bool): if the search is multiFidelity
        budget (int): budget of the search
        
        
    
    Functions:

        suggest_element(search_space_df, fitness_acquired, ids_acquired, bad_ids, SP):
            Suggest an element to evaluate
        initial_suggestion(SP, num_elem_initialisation, benchmark, df_total):
            Initial suggestion of the search space

            
    
    
    """


    def __init__(self):
        self.name = "default"
        self.multiFidelity = False
        self.budget = None
        pass

    def suggest_element(
        self,
        SP: SearchSpace,
        search_space_df: pd.DataFrame = [],
        fitness_acquired: list = [],
        ids_acquired: list = [],
        bad_ids: list = [],
    ) -> float:
        """Suggest an element to evaluate
        Args:
            search_space_df (pd.DataFrame): dataframe containing the searched space

            fitness_acquired (list): list of the fitness of the elements
            ids_acquired (list): list of the ids of the elements
            bad_ids (list): list of the ids of the bad elements
            SP (Search_Space): search space

        Returns:
            float: id of the element to evaluate
        """
        pass

    def initial_suggestion(
        self,
        SP: SearchSpace = [],
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
    """
    Random search algorithm
    
    Class to define the random search algorithm
    Suggest a random molecule to evalutate from the search space

    Attributes:
        seed (int): seed for the random search
        name (str): name of the search algorithm
        multiFidelity (bool): if the search is multiFidelity
        budget (int): budget of the search

    Functions:

        suggest_element(search_space_df, fitness_acquired, ids_acquired, bad_ids, SP):
             ramdomly Suggest an element to evaluate from the search space
        initial_suggestion(SP, num_elem_initialisation, benchmark, df_total):
            Initial suggestion of the search space


    """
    def __init__(self, seed=None):
        self.name = "Random"
        self.seed = seed
        self.multiFidelity = False
        self.budget = None
        if seed is not None:
            np.random.seed(seed)

    def suggest_element(
        self,
        search_space_df,
        ids_acquired,
        fitness_acquired,
        SP: SearchSpace,
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

    """
    
    Evolution algorithm
    
    Class to define the evolution algorithm
    Suggest a molecule to evaluate from the search space using an evolution algorithm
    
    Attributes:
        name (str): name of the search algorithm
        selection_method_mutation (str): selection method for the mutation
        selection_method_cross (str): selection method for the cross
        number_of_parents (int): number of parents
        multiFidelity (bool): if the search is multiFidelity
        budget (int): budget of the search
        
        
    Functions:
    
        suggest_element(search_space_df, fitness_acquired, ids_acquired, bad_ids, SP):
            Suggest an element to evaluate
        generate_df_elements_to_choose_from(search_space_df, fitness_acquired, SP, benchmark, df_total):
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
        self.name = "Evolution_algorithm"
        self.selection_method_mutation = "top"
        self.selection_method_cross = "top"
        self.number_of_parents = 5
        self.multiFidelity = False
        self.budget = None
        pass

    def suggest_element(
        self,
        search_space_df,
        ids_acquired,
        fitness_acquired,
        SP: SearchSpace,
        benchmark=True,
        df_total: pd.DataFrame = None,
    ):  
        """
        Suggest an element to evaluate
        Start the algorithm by generating a list of offspring from the parents
        the list of offspring is generated by mutation and cross-over of the parents
        the selection of the parents is done using the selection method over the list of molecules in the searched space
        the selection method is defined by the user
        the selection method can be "roulette", "tournament", "rank" or "top"
        """
        import time
        random_seed = int(time.time()*1000) - int(time.time())*1000
        np.random.seed(random_seed)
        df_search = search_space_df
        df_elements = search_space_df
        error_counter = 0
        while not self._check_new_element_in_search_space(df_search, df_elements):
            df_elements, df_search = self.generate_df_elements_to_choose_from(
                search_space_df,
                fitness_acquired,
                SP,
                benchmark,
                df_total,
            )
            error_counter =error_counter+1
            if error_counter>10:
                df_elements.drop_duplicates(inplace=True)
                print(df_elements.shape)
                raise ValueError('no new element found')

        def add_element(df, element):
            if ~(df == element).all(1).any():
                df.loc[len(df)] = element
                return True
            return False
        for element in df_elements.sample(frac=1).values:
            if add_element(df_search, element):
                break
            
        return len(df_search) - 1, df_search


    def _check_new_element_in_search_space(self, df_search, df_elements):
        """
        Check if the element is already in the search space
        """
        df_search_copy = df_search.copy()
        df_search_copy = df_search_copy[df_elements.columns]  
        all_df = pd.merge(df_elements,df_search_copy, how="left",indicator='exists')
        all_df['exists'] = np.where(all_df.exists == 'both', True, False)

        return all_df[all_df.exists == False].shape[0] > 0

    def generate_df_elements_to_choose_from(
        self,
        search_space_df,
        fitness_acquired,
        SP: SearchSpace,
        benchmark=True,
        df_total: pd.DataFrame = None,
    ):
        """
        
        Generate the dataframe of the elements to choose from   


        Args:
            search_space_df (pd.DataFrame): dataframe containing the searched space
            fitness_acquired (list): list of the fitness of the elements
            SP (Search_Space): search space
            benchmark (bool): if the search is a benchmark
            df_total (pd.DataFrame): dataframe containing the results

        Returns:
            pd.DataFrame: dataframe containing the elements to choose from
            pd.DataFrame: dataframe containing the searched space

        """
        df_search = search_space_df.copy()
        fitness_acquired = np.array(fitness_acquired)
        elements = self.Generate_element_to_evaluate(
            fitness_acquired, df_search, SP
        )
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

    def mutate_element(self, element, SP: SearchSpace):
        elements = []
        for i in range(element.shape[0]):  # check this for generalization
            for frag in SP.df_precursors.InChIKey:
                element_new = element.copy()
                element_new[i] = frag
                elements.append(element_new)

        return elements

    def cross_element(self, element1, element2):
        elements = []
        for i in range(element1.shape[0]):  # check this for generalization
            element_new = element1.copy()
            element_new[i] = element2[i]
            elements.append(element_new)
        return elements

    def roulette_wheel_selection(self, fitness_acquired, df_search, size=3):
        total_fitness = np.sum(fitness_acquired)
        selection_probs = fitness_acquired / total_fitness
        selected_indices = np.random.choice(
            df_search.index, size=size, p=selection_probs
        )
        return df_search.loc[selected_indices].values

    def tournament_selection(self, fitness_acquired, df_search, size=3):
        selected_indices = []
        for _ in range(size):
            tournament_indices = np.random.choice(df_search.index, size=2)
            tournament_fitness = fitness_acquired[tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices.append(winner_index)
        return df_search.loc[selected_indices].values

    def rank_selection(self, fitness_acquired, df_search, size=3):
        ranks = np.argsort(np.argsort(fitness_acquired))
        total_ranks = np.sum(ranks)
        selection_probs = ranks / total_ranks
        selected_indices = np.random.choice(
            df_search.index, size=size, p=selection_probs
        )
        return df_search.loc[selected_indices].values

    def top_selection(
        self, fitness_acquired, df_search, size=3, number_of_random=2
    ):
        top_indices = np.argsort(fitness_acquired)[-size + number_of_random :]
        random_indices = np.random.choice(df_search.shape[0], size=number_of_random)
        indices_considered = np.append(top_indices, random_indices)
        return df_search.loc[indices_considered].values

    def run_selection_method(
        self, selection_method, fitness_acquired, df_search
    ):
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
                fitness_acquired, df_search, self.number_of_parents
            )
        else:
            raise ValueError(
                "Unknown selection method: {}".format(selection_method)
            )
        return list_parents

    def Generate_element_to_evaluate(
        self, fitness_acquired, df_search, SP: SearchSpace
    ):
        elements = []
        elements_to_mutate = self.run_selection_method(
            self.selection_method_mutation, fitness_acquired, df_search
        )
        elements_to_cross = self.run_selection_method(
            self.selection_method_cross, fitness_acquired, df_search
        )
        for element in elements_to_mutate:
            if len(elements) == 0:
                elements = self.mutate_element(element, SP)
            else:
                elements = np.append(
                    elements, self.mutate_element(element, SP), axis=0
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
        return elements
