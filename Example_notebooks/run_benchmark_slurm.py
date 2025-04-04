import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stk_search import SearchExp, SearchSpace
from stk_search.geom3d import train_models
from stk_search.ObjectiveFunctions.ObjectiveFunction import LookUpTable
from stk_search.Representation import Representation_from_fragment
from stk_search.Search_algorithm import (
    BayesianOptimisation,
    Ea_surrogate,
    Search_algorithm,
)


def get_dataframes(
    df_total_path_bench="/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/30K_benchmark_150525.csv",
    df_precursor_path="/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/precursor_with_prop.pkl",
    df_precursor_Mordred_path="/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/df_PCA_mordred_descriptor_290224.pkl",
    SearchSpace_loc="/media/mohammed/Work/STK_search/Example_notebooks/SearchSpace/SearchSpace_test.pkl",
):
    sp = pd.read_pickle(SearchSpace_loc)
    df_Benchmark = pd.read_csv(df_total_path_bench)
    df_precursor = pd.read_pickle(df_precursor_path)
    df_precursor_Mordred = pd.read_pickle(df_precursor_Mordred_path)

    sp.df_precursors = df_precursor
    sp.update()
    num_of_elem_in_SP = sp.check_df_for_element_from_sp(df_Benchmark).shape
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
            sp.df_precursors["InChIKey"] == "PTRYVXYLNANOHT-WTKPLQERSA-N"
        ].shape[0],
    )
    print(
        df_Benchmark[df_Benchmark["target"] > 0].shape[0]
        / df_Benchmark.shape[0]
    )
    return df_Benchmark, df_precursor, df_precursor_Mordred, sp


def normalise_df(df):
    for col in df.columns:
        if df[col].dtype == "object":
            continue
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def initialise_search_algorithm(
    df_precursor_Mordred, df_precursors, config_dir
):
    which_acquisition = "EI"
    lim_counter = 5
    BO_learned = BayesianOptimisation.BayesianOptimisation(
        which_acquisition=which_acquisition, lim_counter=lim_counter
    )
    BO_learned.verbose = True
    EA = Search_algorithm.evolution_algorithm()
    EA.number_of_parents = 5
    # EA.num_added_random = 10000
    EA.number_of_random = 2
    EA.selection_method_mutation = "top"
    EA.selection_method_cross = "top"
    RAND = Search_algorithm.random_search()
    SUEA = Ea_surrogate.Ea_surrogate()
    SUEA.number_of_parents = 5
    SUEA.num_added_random = 10000
    SUEA.number_of_random = 2
    BO_Mord = BayesianOptimisation.BayesianOptimisation(
        which_acquisition=which_acquisition, lim_counter=lim_counter
    )
    BO_prop = BayesianOptimisation.BayesianOptimisation(
        which_acquisition=which_acquisition, lim_counter=lim_counter
    )
    BO_Mord.verbose = True
    BO_prop.verbose = True

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
    frag_properties = df_precursors.columns[2:7].append(
        df_precursors.columns[[17, 18, 19, 20, 21, 22]]
    )
    BO_prop.Representation = (
        Representation_from_fragment.RepresentationFromFragment(
            df_precursors, frag_properties
        )
    )
    BO_learned.number_of_parents = 10
    BO_learned.number_of_random = 5
    BO_prop.number_of_parents = 10
    BO_prop.number_of_random = 5
    BO_Mord.number_of_parents = 10
    BO_Mord.number_of_random = 5
    return BO_learned, EA, SUEA, BO_Mord, BO_prop, RAND


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
    ObjectiveFunction = LookUpTable(
        df_total, oligomer_size, target_name=target_name, aim=aim
    )
    return ObjectiveFunction


def set_benchmark_params(config_dir):
    exp_name = "Benchmark_30k_dataset"
    num_elem_initialisation = 50
    number_of_repeats = 1
    num_iteration = 800
    target = "target"
    aim = "maximise"
    case_name_list = [
        "BO_precursor",
        #"BO_Mord",
        "evolution_algorithm",
        "random",
        "ea_surrogate",
        "BO_learned",
    ]
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
    save_path="run_search_new_inputs.json",
):
    s_exp = SearchExp.SearchExp(
        search_space,
        search_algorithm,
        ObjectiveFunction,
        number_of_iterations,
        verbose=verbose,
    )
    benchmark = True
    s_exp.output_folder = f"{save_path}/{oligomer_size}_frag/" + case_name

    s_exp.num_elem_initialisation = num_elem_initialisation
    s_exp.benchmark = benchmark
    s_exp.df_total = df_total

    def save_run_search_inputs(
        inputs, save_path="run_search_new_inputs.json"
    ) -> int:
        git_version = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        inputs["git_version"] = git_version
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
    Path(f"{save_path}/database").mkdir(parents=True, exist_ok=True)
    save_path_fin = f"{save_path}/database/{s_exp.search_exp_name}.json"

    save_run_search_inputs(input_json, save_path_fin)
    s_exp.set_save_folder()
    s_exp.run_seach()
    return max(s_exp.ids_acquired)


def test_representations(
    df_Benchmark, SearchSpace, BO_learned, EA, SUEA, BO_Mord, BO_prop, RAND
):
    molecule_id = np.random.randint(
        0, SearchSpace.check_df_for_element_from_sp(df_Benchmark).shape[0]
    )
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


def run_benchmark():
    df_total_path_bench = (
        "data_example/data_benchmark/30K_benchmark_150524.csv"
    )
    SearchSpace_loc = "data_example/data_benchmark/SearchSpace_6_frag_full.pkl"
    df_precursor_path = "data_example/data_benchmark/df_properties.pkl"
    df_precursor_Mordred_path = (
        "data_example/precursor/df_PCA_mordred_descriptor_290224.pkl"
    )
    save_path = "data_example/data_benchmark/runs9"
    config_dir = "data_example/representation_learning/splitrand-nummol20000"

    df_Benchmark, df_precursor, df_precursor_Mordred, sp = get_dataframes(
        df_total_path_bench,
        df_precursor_path,
        df_precursor_Mordred_path,
        SearchSpace_loc,
    )
    df_precursor_Mordred = normalise_df(df_precursor_Mordred)
    df_precursor = normalise_df(df_precursor)
    (
        exp_name,
        num_elem_initialisation,
        num_iteration,
        target,
        aim,
        case_name_list,
        config_dir,
        number_of_repeats,
    ) = set_benchmark_params(config_dir)
    BO_learned, EA, SUEA, BO_Mord, BO_prop, RAND = initialise_search_algorithm(
        df_precursor_Mordred, df_precursor, config_dir
    )
    objective_function = define_objective_function(df_total_path_bench)
    test_representations(
        df_Benchmark, sp, BO_learned, EA, SUEA, BO_Mord, BO_prop, RAND
    )
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
            frag_properties = df_precursor.columns[2:7]
            frag_properties = frag_properties.append(
                df_precursor.columns[[17, 18, 19, 20, 21, 22]]
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
        for i in range(number_of_repeats):
            print(f"running {case_name} repeat {i}")
            define_and_run_search(
                search_algorithm=search_algorithm,
                case_name=case_name,
                num_elem_initialisation=num_elem_initialisation,
                number_of_iterations=num_iteration,
                search_space=sp,
                config_dir=config_dir,
                df_total_path="",
                df_total=df_Benchmark,
                df_representation_path=df_representation_path,
                ObjectiveFunction=objective_function,
                frag_properties=frag_properties,
                save_path=save_path,
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-s", "--seed", type=int, default=42, help="Random seed"
    )
    # Set a global random seed for reproducibility
    SEED = argparser.parse_args().seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    run_benchmark()
