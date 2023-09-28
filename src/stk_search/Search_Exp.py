# class to setup and run a search experiment
import os
import pickle
from datetime import datetime

# from Scripts.Search_algorithm import Search_Algorithm
from stk_search.Objective_function import Objective_Function


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
        self.search_algorithm = search_algorithm
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

    def initialise_search_space(self):
        # load the search space
        self.search_space = pickle.load(open(self.search_space_loc, "rb"))
        self.df_search_space = self.search_space.redefine_search_space()
        if self.benchmark:
            if self.df_total is None:
                print("you need to load the benchmark data first")
            else:
                self.df_searched_space = (
                    self.search_space.check_df_for_element_from_SP(
                        df_to_check=self.df_total
                    )
                )
                list_columns = [
                    f"InChIKey_{i}" for i in range(6)
                ]  # carful here, this is hard coded
                list_columns.append("target")
                if self.df_search_space is not None:
                    self.df_search_space = self.df_search_space.merge(
                        self.df_searched_space[list_columns],
                        on=[f"InChIKey_{i}" for i in range(6)],
                        how="left",
                    )
                    self.df_search_space.dropna(
                        subset=["target"], inplace=True
                    )
                    self.df_search_space.drop(columns=["target"], inplace=True)
                else:
                    columns_name = []
                    for i in range(self.search_space.number_of_fragments):
                        columns_name = columns_name + [
                            x + f"_{i}"
                            for x in self.search_space.features_frag
                        ]
                    self.df_total.dropna(subset=["target"], inplace=True)
                    self.df_search_space = self.df_total[columns_name]

    def run_seach(self):
        # save the search experiment
        self.save_search_experiment()
        # initialise the search space
        self.initialise_search_space()
        # get initial elements
        ids_acquired = self.search_algorithm.initial_suggestion(
            search_space_df=self.df_search_space,
            num_elem_initialisation=self.num_elem_initialisation,
        )
        for id in range(self.num_elem_initialisation):
            # evaluate the element
            self.evaluate_element(
                element_id=ids_acquired[id],
                objective_function=self.objective_function,
            )

        # run the search
        for id in range(self.number_of_iterations):
            # suggest the next element
            ids_acquired, df_search_space = self.search_algorithm.suggest_element(
                search_space_df=self.df_search_space,
                fitness_acquired=self.fitness_acquired,
                ids_acquired=self.ids_acquired,
                bad_ids=self.bad_ids,
            )
            self.df_search_space = df_search_space
            # evaluate the element
            if self.verbose:
                print(f"element id suggested: {ids_acquired}")
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
                print(f"fitness acquired: {self.fitness_acquired}")
                print(f"InchiKey acquired: {self.InchiKey_acquired}")
                print(f"ids acquired: {self.ids_acquired}")
        # save the results
        self.save_results()

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
        os.makedirs(self.output_folder, exist_ok=True)
        # save the search experiment
        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(
            self.output_folder + f"/search_experiment_{time_now}.pkl", "wb"
        ) as f:
            pickle.dump(self, f)

    def save_results(self):
        # save the results
        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        resutls_dict = {
            "ids_acquired": self.ids_acquired,
            "searched_space_df": self.df_search_space.loc[self.ids_acquired],
            "fitness_acquired": self.fitness_acquired,
            "InchiKey_acquired": self.InchiKey_acquired,
        }
        with open(self.output_folder + f"/results_{time_now}.pkl", "wb") as f:
            pickle.dump(resutls_dict, f)
