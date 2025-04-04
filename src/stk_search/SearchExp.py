"""Class to setup and run a search experiment.

In the search experiment, we will initialise the search space, get the initial elements, evaluate the elements, \
run the search algorithm and suggest the next element to evaluate

It takes as input the search space, the search algorithm, the objective function and the number of iterations to run the search experiment

The search experiment will save the results in the output folder

"""

import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from stk_search.SearchSpace import SearchSpace


class SearchExp:
    """Class to setup and run a search experiment.

    Parameters
    ----------
    SearchSpace : SearchSpace
        The search space to be used in the search experiment
    search_algorithm : Search_Algorithm
        The search algorithm to be used in the search experiment
    ObjectiveFunction : ObjectiveFunction
        The objective function to be used in the search experiment
    number_of_iterations : int
        The number of iterations to run the search experiment
    verbose : bool
        Whether to print the progress of the search experiment

    Attributes
    ----------
    SearchSpace : SearchSpace
        The search space to be used in the search experiment. this is defined as a class
        of stk_search.SearchSpace.SearchSpace
    search_algorithm : Search_Algorithm
        The search algorithm to be used in the search experiment.
        this is defined as a class of stk_search.Search_algorithm.Search_Algorithm
    ObjectiveFunction : ObjectiveFunction 
        The objective function to be used in the search experiment
        this is defined as a class of stk_search.ObjectiveFunction.ObjectiveFunction
    number_of_iterations : int
        The number of iterations to run the search experiment
    output_folder : str
        The folder to save the search experiment
    SearchSpace_folder : str
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
        search_space: SearchSpace,
        search_algorithm,
        objective_function,
        number_of_iterations,
        verbose=False,
    ):
        """Initialize the search experiment.

        Parameters
        ----------
        search_space : SearchSpace
            The search space to be used in the search experiment
        search_algorithm : Search_Algorithm
            The search algorithm to be used in the search experiment
        objective_function : ObjectiveFunction
            The objective function to be used in the search experiment
        number_of_iterations : int
            The number of iterations to run the search experiment
        verbose : bool
            Whether to print the progress of the search experiment

        """
        self.SearchSpace = search_space
        self.search_algorithm = (
            search_algorithm  # add a name to the search algorithm
        )
        self.ObjectiveFunction = objective_function
        self.number_of_iterations = number_of_iterations
        self.output_folder = "Data/search_experiment"
        self.SearchSpace_folder = "Data/search_experiment/SearchSpace"
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
        self.date = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        self.search_exp_name = uuid.uuid4().hex
        self.set_save_folder()

    def run_seach(self):
        """Run the search experiment.

        the search experiment will initialise the search space, get the initial elements, evaluate the elements, \
            run the search algorithm and suggest the next element to evaluate
            for the moment we cannot rerun a same search experiment.
            
        Returns
        -------
            results_dict : dict
                The results of the search experiment    

        """
        # get initial elements
        if self.ids_acquired == []:
            ids_acquired, df_search_space = (
                self.search_algorithm.initial_suggestion(
                    sp=self.SearchSpace,
                    num_elem_initialisation=self.num_elem_initialisation,
                    benchmark=self.benchmark,
                    df_total=self.df_total,
                )
            )

            if (self.search_algorithm.budget is not None) and (
                self.search_algorithm.budget < 0
            ):
                msg = "Budget exhausted by Initial Sample"
                raise Exception(msg)

            self.df_search_space = df_search_space
            for id_acquired in range(len(ids_acquired)):
                self.evaluate_element(
                    element_id=ids_acquired[id_acquired],
                )
            if self.verbose:
                pass
        # run the search
        number_of_iterations_run = (
            len(self.ids_acquired) - self.num_elem_initialisation
        )
        if number_of_iterations_run > self.number_of_iterations:
            return None
        for _id in range(number_of_iterations_run, self.number_of_iterations):
            # suggest the next element
            ids_acquired, df_search_space = (
                self.search_algorithm.suggest_element(
                    searchspace_df=self.df_search_space,
                    ids_acquired=self.ids_acquired,
                    fitness_acquired=self.fitness_acquired,
                    sp=self.SearchSpace,
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
            self.evaluate_element(
                element_id=ids_acquired,
            )
            # save the results
            self.save_results()
            if self.verbose:
                print(f"iteration {_id} done")
                print(f"fitness acquired: {self.fitness_acquired[-1]}")
                print(f"InchiKey acquired: {self.InchiKey_acquired[-1]}")
                print(f"max fitness: {max(self.fitness_acquired)}")
                print(f"mean fitness: {sum(self.fitness_acquired[self.num_elem_initialisation-1:-1])/len(self.fitness_acquired[self.num_elem_initialisation-1:-1])}")
            # clear GPU memory
            if hasattr(self.search_algorithm,"device") and self.search_algorithm.device == "cuda":
                torch.cuda.empty_cache()

        # save the results
        return self.save_results()

    def evaluate_element(
        self,
        element_id: int,
    ):
        """Evaluate the element.

        Trie to evaluate the element using the objective function. if it fails it will add the element to the bad_ids list and return None, None

        Args:
        ----
        element_id : int
            The id of the element to evaluate in the df of the search space

        Returns:
        -------
        Eval : float
            The fitness of the element
        InchiKey : str
            The InchiKey of the element

        """
        element = self.df_search_space.loc[[element_id], :]
        time_calc = datetime.now(tz=timezone.utc)
        # evaluate the element
        try:
            eval_value, InchiKey = self.ObjectiveFunction.evaluate_element(
                element=element,
            )
        except Exception as e:
            print(f"Element {element_id} failed")
            print(e)
            self.bad_ids.append(element_id)
            return None, None
        if self.search_algorithm.multi_fidelity:
            pass
        if eval_value is None:
            self.bad_ids.append(element_id)
            return None, None
        self.fitness_acquired.append(eval_value)
        self.InchiKey_acquired.append(InchiKey)
        self.ids_acquired.append(element_id)
        self.time_calc.append(datetime.now(tz=timezone.utc) - time_calc)
        self.overall_time.append(datetime.now(tz=timezone.utc))
        return eval_value, InchiKey

    def save_search_experiment(self):
        """Save the search experiment.

        Save the search experiment in the output folder with the name search_experiment_{self.search_exp_name}.pkl
        """
        # save the search experiment
        datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        date_now = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        Path(self.output_folder + f"/{date_now}").mkdir(
            parents=True, exist_ok=True
        )
        with Path(
            self.output_folder
            + f"/{date_now}"
            + f"/search_experiment_{self.search_exp_name}.pkl",
        ).open("wb") as f:
            pickle.dump(self, f)
    
    def set_save_folder(self):
        path = self.output_folder + f"/{self.date}"
        Path(path).mkdir(parents=True, exist_ok=True)
        self.save_path = path + f"/results_{self.search_exp_name}.pkl"



    def save_results(self):
        """Save the results.

        Save the results in the output folder with the name results_{self.search_exp_name}.pkl
        the results will be saved in a dictionary with the following keys:
            - ids_acquired
            - searched_space_df
            - fitness_acquired
            - InchiKey_acquired
            - overall_time
            - time_calc
        """
        # save the results
        resutls_dict = {
            "ids_acquired": self.ids_acquired,
            "searched_space_df": self.df_search_space.loc[self.ids_acquired],
            "fitness_acquired": self.fitness_acquired,
            "InchiKey_acquired": self.InchiKey_acquired,
            "overall_time": self.overall_time,
            "time_calc": self.time_calc,
        }

        
        with Path(self.save_path).open(
            "wb"
        ) as f:
            pickle.dump(resutls_dict, f)
        return resutls_dict

    def load_results(self, filepath: str):
        """Load a saved search experiment and rerun it for more iterations.

        Parameters
        ----------
        filepath : str
            The path to the saved search experiment file (.pkl).
        additional_iterations : int
            The number of additional iterations to run.

        Returns
        -------
        results_dict : dict
            The updated results of the search experiment.
        """
        # Load the saved search experiment
        with Path(filepath).open("rb") as f:
            resutls_dict = pickle.load(f)

        self.ids_acquired = resutls_dict["ids_acquired"]
        self.df_search_space = resutls_dict["searched_space_df"]
        self.fitness_acquired = resutls_dict["fitness_acquired"]
        self.InchiKey_acquired = resutls_dict["InchiKey_acquired"]
        self.overall_time = resutls_dict["overall_time"]
        self.time_calc = resutls_dict["time_calc"]
        self.ids_acquired = list(set(self.ids_acquired))
        return 0