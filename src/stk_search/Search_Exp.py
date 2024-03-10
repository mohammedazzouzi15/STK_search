# class to setup and run a search experiment
import os
import pickle
from datetime import datetime

# from Scripts.Search_algorithm import Search_Algorithm
from stk_search.Objective_function import Objective_Function
import uuid


class Search_exp:
    def __init__(
        self,
        search_space_loc,
        search_algorithm,
        objective_function,
        number_of_iterations,
        verbose=False,
    ):
        self.search_space_loc = search_space_loc
        self.search_algorithm = (
            search_algorithm  # add a name to the search algorithm
        )
        self.objective_function = objective_function
        self.number_of_iterations = number_of_iterations
        self.output_folder = "Data/search_experiment"
        self.search_space_folder = "Data/search_experiment/search_space"
        self.num_elem_initialisation = 10
        self.search_space = None
        self.df_search_space = None
        self.ids_acquired = []
        self.fitness_acquired = []
        self.InchiKey_acquired = []
        self.bad_ids = []
        self.verbose = verbose
        self.benchmark = False
        self.df_total = None
        self.date = datetime.now().strftime("%Y%m%d")
        self.search_exp_name = uuid.uuid4().hex

    def initialise_search_space(self):
        # load the search space
        self.search_space = pickle.load(open(self.search_space_loc, "rb"))

    def run_seach(self):
        # save the search experiment
        # if not self.benchmark:
        #   self.save_search_experiment()
        # initialise the search space
        self.initialise_search_space()
        # get initial elements
        ids_acquired, df_search_space = (
            self.search_algorithm.initial_suggestion(
                SP=self.search_space,
                num_elem_initialisation=self.num_elem_initialisation,
                benchmark=self.benchmark,
                df_total=self.df_total,
            )
        )
        self.df_search_space = df_search_space
        for id in range(self.num_elem_initialisation):
            # evaluate the element
            self.evaluate_element(
                element_id=ids_acquired[id],
                objective_function=self.objective_function,
            )
        if self.verbose:
            print(f"max fitness acquired: {max(self.fitness_acquired)}")
            print(f"min fitness acquired: {min(self.fitness_acquired)}")
        # run the search
        for id in range(self.number_of_iterations):
            # suggest the next element
            ids_acquired, df_search_space = (
                self.search_algorithm.suggest_element(
                    search_space_df=self.df_search_space,
                    ids_acquired=self.ids_acquired,
                    fitness_acquired=self.fitness_acquired,
                    SP=self.search_space,
                    benchmark=self.benchmark,
                    df_total=self.df_total,
                )
            )
            self.df_search_space = df_search_space
            # evaluate the element
            # if self.verbose:
            # print(f"element id suggested: {ids_acquired}, inchikey suggested: {self.df_search_space.loc[ids_acquired]}")
            self.evaluate_element(
                element_id=ids_acquired,
                objective_function=self.objective_function,
            )
            # self.fitness_acquired.append(Eval)
            # self.InchiKey_acquired.append(InchiKey)
            # save the results
            self.save_results()
            if self.verbose:
                print(f"iteration {id} completed")
                print(f"max fitness acquired: {max(self.fitness_acquired)}")
                print(f"min fitness acquired: {min(self.fitness_acquired)}")
                # print(f"ids acquired: {self.ids_acquired}")
                print(f"new fitness acquired: {self.fitness_acquired[-1]}")
        # save the results
        results_dict = self.save_results()
        return results_dict

    def evaluate_element(
        self,
        element_id: int,
        objective_function: Objective_Function = None,
    ):
        # get the element
        element = self.df_search_space.loc[[element_id], :]
        # evaluate the element
        try:
            Eval, InchiKey = objective_function.evaluate_element(
                element=element
            )
            if self.verbose:
                print(f"element Inchikey suggested: {InchiKey}, Eval: {Eval}")
            if Eval is None:
                self.bad_ids.append(element_id)
                print(f"element {element_id} failed")

                return None, None
            self.fitness_acquired.append(Eval)
            self.InchiKey_acquired.append(InchiKey)
            self.ids_acquired.append(element_id)
            return Eval, InchiKey
        except Exception as e:
            self.bad_ids.append(element_id)
            print(f"element {element_id} failed")
            print(e)
            return None, None

    def save_search_experiment(self):
        # save the search experiment
        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_now = datetime.now().strftime("%Y%m%d")
        os.makedirs(self.output_folder + f"/{date_now}", exist_ok=True)
        with open(
            self.output_folder
            + f"/{date_now}"
            + f"/search_experiment_{self.search_exp_name}.pkl",
            "wb",
        ) as f:
            pickle.dump(self, f)

    def save_results(self):
        # save the results
        # time_now = datetime.now().strftime("%Y%m%d_%H")

        
        resutls_dict = {
            "ids_acquired": self.ids_acquired,
            "searched_space_df": self.df_search_space.loc[self.ids_acquired],
            "fitness_acquired": self.fitness_acquired,
            "InchiKey_acquired": self.InchiKey_acquired,
        }

        path = self.output_folder + f"/{self.date}"
        os.makedirs(path, exist_ok=True)
        with open(path + f"/results_{self.search_exp_name}.pkl", "wb") as f:

            pickle.dump(resutls_dict, f)
        return resutls_dict
