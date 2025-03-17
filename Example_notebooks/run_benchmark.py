import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import swifter  # noqa: F401
import torch
from stk_search import SearchExp, SearchSpace
from stk_search.geom3d import train_models
from stk_search.ObjectiveFunctions.ObjectiveFunction import LookUpTable
from stk_search.Representation import (
    Representation_from_fragment,
)
from stk_search.Search_algorithm import (
    BayesianOptimisation,
    Ea_surrogate,
    Search_algorithm,
)


def set_benchmark_params():
    exp_name = "Benchmark_30k_dataset"
    num_elem_initialisation = 50
    number_of_repeats = 10
    num_iteration = 1000
    target = "target"
    aim = "maximise"
    case_name_list = [
        "BO_Mord",
        "BO_learned",
        "evolution_algorithm",
        "random",
        "ea_surrogate",
        "BO_precursor",
    ]
    config_dir = "/media/mohammed/Work/STK_search/Example_notebooks/data_example/representation_learning/splitrand-nummol20000"
    return (
        exp_name,
        num_elem_initialisation,
        num_iteration,
        target,
        aim,
        case_name_list,
        config_dir,
        number_of_repeats,
    )


def get_dataframes(
    df_total_path_bench="/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/30K_benchmark_150524.csv",
    df_precursor_path="/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/precursor_with_prop.pkl",
    df_precursor_Mordred_path="/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/df_PCA_mordred_descriptor_290224.pkl",
    SearchSpace_loc="/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/SearchSpace_6_frag_full.pkl",
):
    # run a search experiment with the new target

    SearchSpace = pd.read_pickle(SearchSpace_loc)
    df_Benchmark = pd.read_csv(df_total_path_bench)
    df_precursor = pd.read_pickle(df_precursor_path)
    df_precursor_Mordred = pd.read_pickle(df_precursor_Mordred_path)
    SearchSpace.df_precursors = df_precursor_Mordred
    print("shape of df_Benchmark:", df_Benchmark.shape)
    print("shape of df_precursors:", df_precursor.shape)
    print("shape of df_precursor_Mordred:", df_precursor_Mordred.shape)
    print(
        df_Benchmark[df_Benchmark["target"] > 0].shape[0]
        / df_Benchmark.shape[0]
    )
    return df_Benchmark, df_precursor, df_precursor_Mordred, SearchSpace


def initialise_search_algorithm(
    df_precursor_Mordred, df_precursors, config_dir
):
    # initialise search algorithm

    which_acquisition = "EI"
    lim_counter = 10
    BO_learned = BayesianOptimisation.BayesianOptimisation(
        which_acquisition=which_acquisition, lim_counter=lim_counter
    )
    EA = Search_algorithm.evolution_algorithm()
    RAND = Search_algorithm.random_search()
    SUEA = Ea_surrogate.Ea_surrogate()
    BO_Mord = BayesianOptimisation.BayesianOptimisation(
        which_acquisition=which_acquisition, lim_counter=lim_counter
    )
    BO_prop = BayesianOptimisation.BayesianOptimisation(
        which_acquisition=which_acquisition, lim_counter=lim_counter
    )
    # load the Representation and the model

    config, min_val_loss = train_models.get_best_embedding_model(config_dir)
    SUEA = Ea_surrogate.Ea_surrogate()
    SUEA.config_dir = config_dir
    SUEA.load_representation_model()
    BO_learned.config_dir = config_dir
    BO_learned.load_representation_model()

    frag_properties = df_precursor_Mordred.select_dtypes(
        include=[np.number]
    ).columns
    BO_Mord.Representation = (
        Representation_from_fragment.RepresentationFromFragment(
            df_precursor_Mordred, frag_properties
        )
    )
    frag_properties = []
    frag_properties = df_precursors.columns[1:7]
    frag_properties = frag_properties.append(df_precursors.columns[17:23])
    BO_prop.Representation = (
        Representation_from_fragment.RepresentationFromFragment(
            df_precursors, frag_properties
        )
    )
    return BO_learned, EA, SUEA, BO_Mord, BO_prop, RAND


# set experiment conditions
def define_and_run_search(
    search_algorithm,
    num_elem_initialisation=50,
    number_of_iterations=100,
    verbose=True,
    case_name="test",
    search_space: SearchSpace = SearchSpace,
    ObjectiveFunction=None,
    df_representation_path=None,
    df_total_path=None,
    oligomer_size=6,
    frag_properties="",
    which_acquisition="EI",
    config_dir=None,
    df_total=None,
    SearchSpace_loc=None,
):
    """Define and run a search experiment.

    Args:
    ----
        search_algorithm (Search_algorithm): The search algorithm to use.
        num_elem_initialisation (int): The number of elements to initialise.
        number_of_iterations (int): The number of iterations to run the search for.
        verbose (bool): Whether to print the search progress.
        case_name (str): The name of the search experiment.

    Returns:
    -------
        int: The maximum id acquired.

    """
    s_exp = SearchExp.SearchExp(
        search_space,
        search_algorithm,
        ObjectiveFunction,
        number_of_iterations,
        verbose=verbose,
    )
    benchmark = True
    s_exp.output_folder = (
        f"output/search_experiment_benchmark/{oligomer_size}_frag/" + case_name
    )
    s_exp.num_elem_initialisation = num_elem_initialisation
    s_exp.benchmark = benchmark
    s_exp.df_total = df_total
    # Save search inputs

    def save_run_search_inputs(
        inputs, save_path="run_search_new_inputs.json"
    ) -> int:
        """Save the inputs to a file.

        Args:
        ----
            inputs (dict): The inputs to save.
            save_path (str): The path to save the inputs to.

        """
        # Get the current git version
        git_version = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )

        # Add the git version to the inputs
        inputs["git_version"] = git_version

        # Save the inputs to a file
        with Path(save_path).open("w") as f:
            json.dump(inputs, f)

        print("Inputs saved.")
        return 0

    input_json = {}
    input_json["SearchSpace_loc"] = SearchSpace_loc
    input_json["search_algorithm"] = search_algorithm.name
    input_json["ObjectiveFunction"] = ObjectiveFunction.__str__()
    input_json["number_of_iterations"] = number_of_iterations
    input_json["verbose"] = verbose
    input_json["num_elem_initialisation"] = num_elem_initialisation
    input_json["benchmark"] = benchmark
    input_json["df_total"] = df_total_path
    input_json["df_representation"] = df_representation_path
    if "Representation" in search_algorithm.__dict__:
        input_json["representation"] = search_algorithm.Representation.name
    else:
        input_json["representation"] = None
    input_json["frag_properties"] = list(frag_properties)
    input_json["which_acquisition"] = which_acquisition
    input_json["run_search_name"] = s_exp.search_exp_name
    input_json["search_output_folder"] = s_exp.output_folder
    input_json["date"] = s_exp.date
    input_json["oligomer_size"] = oligomer_size
    input_json["config_dir"] = config_dir
    input_json["case_name"] = case_name
    save_path = f"output/search_experiment_benchmark/search_exp_database/{s_exp.search_exp_name}.json"
    Path("output/search_experiment_benchmark/search_exp_database").mkdir(
        parents=True, exist_ok=True
    )

    save_run_search_inputs(input_json, save_path)
    s_exp.run_seach()
    return max(s_exp.ids_acquired)


def test_representations(
    df_Benchmark, SearchSpace, BO_learned, EA, SUEA, BO_Mord, BO_prop, RAND
):
    # test representation

    molecule_id = np.random.randint(0, df_Benchmark.shape[0])
    oligomer_size = 6
    molecule_properties = SearchSpace.check_df_for_element_from_sp(
        df_Benchmark
    ).iloc[[molecule_id]]
    print(molecule_properties["InChIKey"])
    X_rpr = SUEA.Representation.generate_repr(
        molecule_properties[[f"InChIKey_{x}" for x in range(oligomer_size)]]
    )
    print("representation for SUEA", X_rpr)
    X_rpr = BO_learned.Representation.generate_repr(
        molecule_properties[[f"InChIKey_{x}" for x in range(oligomer_size)]]
    )
    print("representation for BO_learned", X_rpr)

    X_rpr = BO_Mord.Representation.generate_repr(
        molecule_properties[[f"InChIKey_{x}" for x in range(oligomer_size)]]
    )
    print("representation for BO_Mord", X_rpr)
    X_rpr = BO_prop.Representation.generate_repr(
        molecule_properties[[f"InChIKey_{x}" for x in range(oligomer_size)]]
    )
    print("representation for BO_prop", X_rpr)
    


def define_objective_function(target):
    df_total_path = "/media/mohammed/Work/STK_search/Example_notebooks/data_example/Molecule_database/58K_200524.csv"  # "/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/30K_benchmark_150524.csv"
    df_total = pd.read_csv(df_total_path)
    oligomer_size = 6
    target_name = "target"
    aim = "maximise"
    df_total["target"] = (
        -np.abs(df_total["ES1"] - 3)
        - np.abs(df_total["ionisation potential (eV)"] - 5.5)
        + np.log10(df_total["fosc1"])
    )
    df_total.to_csv(df_total_path, index=False)
    # define the evaluation function
    ObjectiveFunction = LookUpTable(
        df_total, oligomer_size, target_name=target_name, aim=aim
    )
    return ObjectiveFunction


def run_benchmark():
    # run a search experiment with the new target
    df_total_path_bench = "/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/30K_benchmark_150524.csv"  # "/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/30K_benchmark_150524.csv"
    df_precursor_path = "/media/mohammed/Work/STK_search/Example_notebooks/data_example/precursor/df_properties.pkl"  # "/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/precursor_with_prop.pkl"
    df_precursor_Mordred_path = "/media/mohammed/Work/STK_search/Example_notebooks/data_example/precursor/df_mordred_24072024.pkl"  # s "/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/df_PCA_mordred_descriptor_290224.pkl"
    SearchSpace_loc = "/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/SearchSpace_6_frag_full.pkl"
    df_Benchmark, df_precursor, df_precursor_Mordred, SearchSpace = (
        get_dataframes(
            df_total_path_bench,
            df_precursor_path,
            df_precursor_Mordred_path,
            SearchSpace_loc,
        )
    )
    (
        exp_name,
        num_elem_initialisation,
        num_iteration,
        target,
        aim,
        case_name_list,
        config_dir,
        number_of_repeats,
    ) = set_benchmark_params()
    BO_learned, EA, SUEA, BO_Mord, BO_prop, RAND = initialise_search_algorithm(
        df_precursor_Mordred, df_precursor, config_dir
    )
    objective_function = define_objective_function(target)
    for case_name in case_name_list:
        df_representation_path = ""
        frag_properties = ""

        if case_name == "evolution_algorithm":
            search_algorithm = EA
        elif case_name == "random":
            search_algorithm = RAND
        elif case_name == "BO_precursor":
            search_algorithm = BO_prop
            df_representation_path = df_precursor_path
            frag_properties = []
            frag_properties = df_precursor.columns[1:7]
            frag_properties = frag_properties.append(
                df_precursor.columns[17:23]
            )
        elif case_name == "ea_surrogate":
            search_algorithm = SUEA
        elif case_name == "BO_learned":
            search_algorithm = BO_learned
        elif case_name == "BO_Mord":
            search_algorithm = BO_Mord
            df_representation_path = df_precursor_Mordred_path
            frag_properties = df_precursor_Mordred.select_dtypes(
                include=[np.number]
            ).columns
        else:
            raise ValueError("case name not recognised")
        for _ in range(number_of_repeats):
            define_and_run_search(
                search_algorithm=search_algorithm,
                case_name=case_name,
                num_elem_initialisation=num_elem_initialisation,
                number_of_iterations=num_iteration,
                search_space=SearchSpace,
                config_dir=config_dir,
                df_total_path="",
                df_total=df_Benchmark,
                df_representation_path=df_representation_path,
                ObjectiveFunction=objective_function,
                frag_properties=frag_properties,
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run_benchmark()
