from stk_search import Search_Exp
from stk_search.Search_algorithm import Search_algorithm
from stk_search.Search_algorithm import Bayesian_Optimisation
from stk_search.Search_algorithm import (
    Representation_slatm,
    RepresentationPrecursor,
)
from stk_search.Objective_function import IP_ES1_fosc
import pandas as pd
from stk_search import Database_utils
from stk_search import Searched_space


# %%
def main(num_iteration, num_elem_initialisation, test_name="test", case="slatm",search_space_loc = "data/input/search_space/test/search_space1.pkl"):
    # Load the searched space
    df_path = 'data/output/Full_dataset/df_total_2023_10_30.csv'
    df_precursors_path = "data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl"  #'Data/output/Prescursor_data/calculation_data_precursor_310823_clean.pkl'
    df_total, df_precursors = Database_utils.load_data_from_file(
        df_path, df_precursors_path
    )

    # get initial elements
    objective_function = IP_ES1_fosc(oligomer_size=6)

    if case == "slatm":
        BO = Bayesian_Optimisation.Bayesian_Optimisation()
        BO.Representation = Representation_slatm.Representation_slatm()
        BO.PCA_input = True
        search_algorithm = BO
    elif case == "slatm_org":
        BO = Bayesian_Optimisation.Bayesian_Optimisation()
        BO.Representation = Representation_slatm.Representation_slatm_org(df_total)
        BO.PCA_input = True
        search_algorithm = BO
    elif case == "precursor":
        BO = Bayesian_Optimisation.Bayesian_Optimisation()
        frag_properties = []
        frag_properties = df_precursors.columns[1:7]
        frag_properties = frag_properties.append(df_precursors.columns[17:23])
        print(frag_properties)
        BO.Representation = RepresentationPrecursor.RepresentationPrecursor(
            df_precursors, frag_properties
        )
        search_algorithm = BO
    elif case == 'random':
        search_algorithm = Search_algorithm.random_search()
    elif case == 'evolution_algorithm':
        search_algorithm = Search_algorithm.evolution_algorithm()
    else:
        raise ValueError("case not recognised")
    
    number_of_iterations = num_iteration
    verbose = True
    num_elem_initialisation = num_elem_initialisation
    S_exp = Search_Exp.Search_exp(
        search_space_loc,
        search_algorithm,
        objective_function,
        number_of_iterations,
        verbose=verbose,
    )
    search_space_name = search_space_loc.split("/")[-1].replace(".pkl", "")
    S_exp.output_folder = "data/output/search_experiment/"+"exp111_"+search_space_name+"/" + test_name
    S_exp.num_elem_initialisation = num_elem_initialisation
    S_exp.benchmark = False
    S_exp.df_total = df_total
    S_exp.run_seach()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num_iteration", type=int, default=100)
    parser.add_argument("--num_elem_initialisation", type=int, default=10)
    parser.add_argument("--test_name", type=str, default="Exp1")
    parser.add_argument("--case", type=str, default="slatm")
    parser.add_argument("--search_space_loc", type=str, default="data/input/search_space/test/search_space1.pkl")
    args = parser.parse_args()
    main(args.num_iteration, args.num_elem_initialisation, args.test_name,args.case,args.search_space_loc)
