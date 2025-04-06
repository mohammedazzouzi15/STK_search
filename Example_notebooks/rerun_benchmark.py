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




def get_results_length(x):
    try:

        return max(pd.read_pickle(x)["ids_acquired"])
    except Exception:
        return 0

def get_max_fitness(x):
    try:
        return max(pd.read_pickle(x)["fitness_acquired"])
    except Exception:
        return 0

def get_dataframes(
    df_total_path_bench="/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/30K_benchmark_150525.csv",
    df_precursor_path="/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/precursor_with_prop.pkl",
    df_precursor_Mordred_path="/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/df_PCA_mordred_descriptor_290224.pkl",
    SearchSpace_loc="/media/mohammed/Work/STK_search/Example_notebooks/SearchSpace/SearchSpace_test.pkl",
):
    # run a search experiment with the new target

    sp = pd.read_pickle(SearchSpace_loc)
    df_Benchmark = pd.read_csv(df_total_path_bench)
    df_precursor = pd.read_pickle(df_precursor_path)
    df_precursor_Mordred = pd.read_pickle(df_precursor_Mordred_path)

    sp.df_precursors = df_precursor
    sp.update()
    num_of_elem_in_SP = sp.check_df_for_element_from_sp(
        df_Benchmark
    ).shape
    print(
        "number of elements in the benchmark in the search space:",
        num_of_elem_in_SP,
    )
    print("shape of df_Benchmark:", df_Benchmark.shape)
    print("shape of df_precursors:", df_precursor.shape)
    print("shape of df_precursor_Mordred:", df_precursor_Mordred.shape)
    print("shape of SearchSpace:", sp.df_precursors.shape)
    print(
        " Inchikey 'PTRYVXYLNANOHT-WTKPLQERSA-N' in df_precursor: ",
        sp.df_precursors[
            sp.df_precursors["InChIKey"]
            == "PTRYVXYLNANOHT-WTKPLQERSA-N"
        ].shape[0],
    )
    print(
        df_Benchmark[df_Benchmark["target"] > 0].shape[0]
        / df_Benchmark.shape[0]
    )
    return df_Benchmark, df_precursor, df_precursor_Mordred, sp




# set experiment conditions
def load_and_run_search(
    search_algorithm,
    num_elem_initialisation=50,
    number_of_iterations=100,
    verbose=True,
    case_name="test",
    search_space: SearchSpace = SearchSpace,
    ObjectiveFunction=None,
    oligomer_size=6,
    df_total=None,
    save_path="run_search_new_inputs.json",
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
        f"{save_path}/{oligomer_size}_frag/" + case_name
    )
    s_exp.num_elem_initialisation = num_elem_initialisation
    s_exp.benchmark = benchmark
    s_exp.save_path = save_path 
    s_exp.df_total = df_total
    s_exp.search_exp_name = save_path.split("/")[-1].split(".")[0]
    # Save search inputs
    s_exp.load_results(save_path)
    s_exp.run_seach()
    return max(s_exp.ids_acquired)



def define_objective_function(df_total_path):
    df_total = pd.read_csv(df_total_path)
    oligomer_size = 6
    target_name = "target"
    aim = "maximise"
    df_total["target"] = (
        -np.abs(df_total["ES1"] - 3)
        - np.abs(df_total["ionisation potential (eV)"] - 5.5)
        + np.log10(df_total["fosc1"])
    )
    # define the evaluation function
    ObjectiveFunction = LookUpTable(
        df_total, oligomer_size, target_name=target_name, aim=aim
    )
    return ObjectiveFunction

def normalise_df(df):
    for col in df.columns:
        if df[col].dtype == "object":
            continue
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df

def get_search_algorithm(case_name, df_precursor_path, df_precursor_Mordred_path,EA, RAND, SUEA, BO_Mord, BO_prop,BO_learned,df_precursor,df_precursor_Mordred):
    df_representation_path, frag_properties = "", []
    if case_name == "evolution_algorithm":
        search_algorithm = EA
    elif case_name == "random":
        search_algorithm = RAND
    elif case_name == "BO_precursor":
        search_algorithm = BO_prop
        df_representation_path = df_precursor_path
        frag_properties = df_precursor.columns[2:7]
        frag_properties = frag_properties.append(
            df_precursor.columns[[17,18, 19,20, 21,22]]
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
    return search_algorithm,df_representation_path,frag_properties


def initialise_search_algorithm(
    df_precursor_Mordred, df_precursors, config_dir
):
    # initialise search algorithm

    which_acquisition = "EI"
    lim_counter = 5
    BO_learned = BayesianOptimisation.BayesianOptimisation(
        which_acquisition=which_acquisition, lim_counter=lim_counter
    )
    BO_learned.verbose = True
    EA = Search_algorithm.evolution_algorithm()
    EA.number_of_parents = 5
    EA.num_added_random = 0
    EA.number_of_random = 2
    EA.selection_method_mutation = "top"
    EA.selection_method_cross = "top"
    RAND = Search_algorithm.random_search()
    SUEA = Ea_surrogate.Ea_surrogate()
    SUEA.number_of_parents = 5
    SUEA.num_added_random = 0
    SUEA.number_of_random = 2
    BO_Mord = BayesianOptimisation.BayesianOptimisation(
        which_acquisition=which_acquisition, lim_counter=lim_counter
    )
    BO_prop = BayesianOptimisation.BayesianOptimisation(
        which_acquisition=which_acquisition, lim_counter=lim_counter
    )
    BO_Mord.verbose = True
    BO_prop.verbose = True
    # load the Representation and the model

    config, min_val_loss = train_models.get_best_embedding_model(config_dir)
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
    frag_properties = df_precursors.columns[2:7]
    frag_properties = frag_properties.append(df_precursors.columns[[17,18,19,20,21,22]])
    #frag_properties = df_precursors.columns[[17,19,21]]
    print(frag_properties)
    BO_prop.Representation = (
        Representation_from_fragment.RepresentationFromFragment(
            df_precursors, frag_properties
        )
    )
    BO_learned.number_of_parents = 5
    BO_learned.number_of_random = 2
    BO_prop.number_of_parents = 5
    BO_prop.number_of_random = 2
    BO_Mord.number_of_parents = 5
    BO_Mord.number_of_random = 2
    return BO_learned, EA, SUEA, BO_Mord, BO_prop, RAND





def run_benchmark():
    # run a search experiment with the new target
    df_total_path_bench = "data_example/data_benchmark/30K_benchmark_150524.csv"
    SearchSpace_loc = "data_example/data_benchmark/SearchSpace_6_frag_full.pkl"
    df_precursor_path = "data_example/data_benchmark/df_properties.pkl"
    df_precursor_Mordred_path = "data_example/precursor/df_PCA_mordred_descriptor_290224.pkl" #"data_example/data_benchmark/precursor_with_mordred_descriptor.pkl"
    config_dir = "data_example/representation_learning/splitrand-nummol20000"
    num_elem_initialisation = 50
    num_iteration = 200

    df_Benchmark, df_precursor, df_precursor_Mordred, sp = (
        get_dataframes(
            df_total_path_bench,
            df_precursor_path,
            df_precursor_Mordred_path,
            SearchSpace_loc,
        )
    )
    df_precursor_Mordred = normalise_df(df_precursor_Mordred)
    df_precursor = normalise_df(df_precursor)
    
    BO_learned, EA, SUEA, BO_Mord, BO_prop, RAND = initialise_search_algorithm(
        df_precursor_Mordred, df_precursor, config_dir
    )
    objective_function = define_objective_function(df_total_path_bench)
                                                                                                             

    save_path_list = ["data_example/data_benchmark/runs10/6_frag/BO_learned/20250405/results_942c3296e9b7434ea98d3c579c8b059f.pkl",                    ]
    for save_path in save_path_list:
        case_name = save_path.split("/")[-3]
        search_algorithm,df_representation_path,frag_properties = get_search_algorithm(
            case_name,
            df_precursor_path,
            df_precursor_Mordred_path,
            EA,
            RAND,
            SUEA,
            BO_Mord,
            BO_prop,
            BO_learned,
            df_precursor,
            df_precursor_Mordred,
        )  

        
        print("case_name:", case_name)
        print("ids acquired:", get_results_length(save_path))
        print("max fitness acquired:", get_max_fitness(save_path))
        load_and_run_search(
            search_algorithm,
            num_elem_initialisation,
            num_iteration,
            case_name=case_name,
            search_space=sp,
            ObjectiveFunction=objective_function,
            df_total=df_Benchmark,
            save_path=save_path,
        )


if __name__ == "__main__":
    run_benchmark()
