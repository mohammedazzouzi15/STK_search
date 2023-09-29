from stk_search import Search_Exp
import importlib
from stk_search.Search_algorithm import Bayesian_Optimisation
from stk_search.Search_algorithm import Representation_slatm
from stk_search.Objective_function import Look_up_table
import pandas as pd
from stk_search import Database_utils
from stk_search import Searched_space

#%% 
# Load the searched space
df_path = 'data/output/Full_datatset/df_total_new2023_08_20.csv'
df_precursors_path = 'data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl'#'Data/output/Prescursor_data/calculation_data_precursor_310823_clean.pkl'
df_total, df_precursors = Database_utils.load_data_from_file(df_path, df_precursors_path)
importlib.reload(Search_Exp)
importlib.reload(Bayesian_Optimisation)

search_space_loc = "data/input/search_space/test/search_space1.pkl"
# get initial elements
objective_function=Look_up_table(df_total,6)

BO = Bayesian_Optimisation.Bayesian_Optimisation()
BO.Representation = Representation_slatm.Representation_slatm()
search_algorithm = BO
num_iteration = 5
number_of_iterations = num_iteration
verbose = True
num_elem_initialisation = 50
S_exp = Search_Exp.Search_exp(
    search_space_loc,
    search_algorithm,
    objective_function,
    number_of_iterations,
    verbose=verbose,
)
S_exp.num_elem_initialisation = num_elem_initialisation
S_exp.benchmark = True
S_exp.df_total = df_total
S_exp.run_seach()

