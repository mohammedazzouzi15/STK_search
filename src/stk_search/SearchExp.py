# class to setup and run a search experiment
import os
import pickle
from datetime import datetime
from stk_search.SearchSpace import SearchSpace

# from Scripts.Search_algorithm import Search_Algorithm
from stk_search.Objective_function import Objective_Function
import uuid


class SearchExp:

    """
    Class to setup and run a search experiment

    Parameters
    ----------
    searchspace : SearchSpace
        The search space to be used in the search experiment
    search_algorithm : Search_Algorithm
        The search algorithm to be used in the search experiment
    objective_function : Objective_Function
        The objective function to be used in the search experiment
    number_of_iterations : int
        The number of iterations to run the search experiment
    verbose : bool
        Whether to print the progress of the search experiment

    Attributes
    ----------
    search_space : SearchSpace
        The search space to be used in the search experiment. this is defined as a class
        of stk_search.SearchSpace.SearchSpace
    search_algorithm : Search_Algorithm
        The search algorithm to be used in the search experiment.
        this is defined as a class of stk_search.Search_algorithm.Search_Algorithm
    objective_function : Objective_Function 
        The objective function to be used in the search experiment
        this is defined as a class of stk_search.Objective_function.Objective_Function
    number_of_iterations : int
        The number of iterations to run the search experiment
    output_folder : str
        The folder to save the search experiment
    search_space_folder : str
        The folder to save the search space
    num_elem_initialisation : int
        The number of elements to initialise the search space
    df_search_space : pd.DataFrame  
        The search space as a pandas dataframe
        the dataframe will host the data corresponding to the molecules considered in the search algorithm, \
        the columns will be the "InchiKey_{i}" of the molecules with i the number of building blocks in the molecule
    ids_acquired : list
        The ids of the elements acquired during the search experiment
        the ids in the df_search_space
    fitness_acquired : list
        The fitness of the elements acquired during the search experiment
    InchiKey_acquired : list
        The InchiKey of the elements acquired during the search experiment
    bad_ids : list
        The ids of the elements that failed during the evaluation using the objective function
    time_calc : list
        The time it took to calculate the fitness of each element
    overall_time : list
        The time at the end of each iteration
    verbose : bool
        Whether to print the progress of the search experiment
    benchmark : bool
        Whether the search experiment is a benchmark
    df_total : pd.DataFrame
        The total dataframe of the search space
        The dataframe with all the data used for the benchmark
        not needed for the normal search experiment
    date : str
        The date of the search experiment
    search_exp_name : str
        The name of the search experiment

    Methods
    -------
    run_seach()
        Run the search experiment
        the search experiment will initialise the search space, get the initial elements, evaluate the elements, \
        run the search algorithm and suggest the next element to evaluate

    evaluate_element()
        Evaluate the element
        Evaluate the element using the objective function

    save_search_experiment()
        Save the search experiment

    save_results()
        Save the results


    """
    def __init__(
        self,
        searchspace: SearchSpace,
        search_algorithm,
        objective_function,
        number_of_iterations,
        verbose=False,
    ):
        self.search_space = searchspace
        self.search_algorithm = (
            search_algorithm  # add a name to the search algorithm
        )
        self.objective_function = objective_function
        self.number_of_iterations = number_of_iterations
        self.output_folder = "Data/search_experiment"
        self.search_space_folder = "Data/search_experiment/search_space"
        self.num_elem_initialisation = 10
        self.df_search_space = None
        self.ids_acquired = []
        self.fitness_acquired = []
        self.InchiKey_acquired = []
        self.bad_ids = []
        self.time_calc = []
        self.overall_time = []
        self.verbose = verbose
        self.benchmark = False
        self.df_total = None
        self.date = datetime.now().strftime("%Y%m%d")
        self.search_exp_name = uuid.uuid4().hex


    def run_seach(self):
        """
        Run the search experiment
        the search experiment will initialise the search space, get the initial elements, evaluate the elements, \
            run the search algorithm and suggest the next element to evaluate
            for the moment we cannot rerun a same search experiment
            
            Returns
            -------
            results_dict : dict
                The results of the search experiment    
        """
        # get initial elements
        if self.ids_acquired ==[]:
            ids_acquired, df_search_space = (
                self.search_algorithm.initial_suggestion(
                    SP=self.search_space,
                    num_elem_initialisation=self.num_elem_initialisation,
                    benchmark=self.benchmark,
                    df_total=self.df_total,
                )
            )

            if (self.search_algorithm.budget is not None) and (
                self.search_algorithm.budget < 0
            ):
                raise Exception("Budget exhausted by Initial Sample")

            self.df_search_space = df_search_space
            for id in range(len(ids_acquired)):
                # evaluate the element
                self.evaluate_element(
                    element_id=ids_acquired[id],
                    objective_function=self.objective_function,
                )
            if self.verbose:
                print(f"max fitness acquired: {max(self.fitness_acquired)}")
                print(f"min fitness acquired: {min(self.fitness_acquired)}")
        # run the search
        number_of_iterations_run = len(self.ids_acquired)-self.num_elem_initialisation
        if number_of_iterations_run > self.number_of_iterations:
            print( ' number of iteration max already run')
            return None
        for id in range(number_of_iterations_run, self.number_of_iterations):
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
            if (self.search_algorithm.budget is not None) and (
                self.search_algorithm.budget < 0
            ):
                break
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
        time_calc = datetime.now()
        # evaluate the element
        try:
            Eval, InchiKey = objective_function.evaluate_element(
                element=element,
                multiFidelity=self.search_algorithm.multiFidelity,
            )
            if self.verbose:
                print(f"element Inchikey suggested: {InchiKey}, Eval: {Eval}")
            if self.search_algorithm.multiFidelity:
                print(f"fitness suggested: {element['fidelity']}")
            if Eval is None:
                self.bad_ids.append(element_id)
                print(f"element {element_id} failed")

                return None, None
            self.fitness_acquired.append(Eval)
            self.InchiKey_acquired.append(InchiKey)
            self.ids_acquired.append(element_id)
            self.time_calc.append(datetime.now() - time_calc)
            self.overall_time.append(datetime.now())
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
            "overall_time": self.overall_time,
            "time_calc": self.time_calc,
        }

        path = self.output_folder + f"/{self.date}"
        os.makedirs(path, exist_ok=True)
        with open(path + f"/results_{self.search_exp_name}.pkl", "wb") as f:

            pickle.dump(resutls_dict, f)
        return resutls_dict
