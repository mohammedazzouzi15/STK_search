import Search_Exp
import numpy as np
import os
import sys
sys.path.append("/rds/general/user/ma11115/home/BO_polymers")
import Search_algorithm
import Objective_function
from argparse import ArgumentParser

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main(
    nb_oligomer_initialization: int,
    input_file_folder: str,
    seach_space_num: int,
    num_iteration: int,
    which_acquisition: str,
    verbose: bool = True,
):
    search_space_loc = input_file_folder + f"search_space_{seach_space_num}.pkl"
    objective_function = Objective_function.IP_ES1_fosc(oligomer_size=6)
    search_algorithm = Search_algorithm.Bayesian_Optimisation()
    search_algorithm.which_acquisition = which_acquisition  #'LOG_EI'
    number_of_iterations = num_iteration

    S_exp = Search_Exp.Search_exp(
        search_space_loc,
        search_algorithm,
        objective_function,
        number_of_iterations,
        verbose=verbose,
    )
    S_exp.num_elem_initialisation = num_iteration
    S_exp.run_seach()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nb_oligomer_initialization", type=int, default=5)
    parser.add_argument(
        "--input_file_folder", type=str, default="Inputs/exp1_2023_09_05_14_47_02/"
    )
    parser.add_argument("--seach_space_num", type=int, default=0)
    parser.add_argument("--num_iteration", type=int, default=100)
    parser.add_argument("--which_acquisition", type=str, default="EI")
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()
    main(
        args.nb_oligomer_initialization,
        args.input_file_folder,
        args.seach_space_num,
        args.num_iteration,
        args.which_acquisition,
        verbose=args.verbose,
    )
