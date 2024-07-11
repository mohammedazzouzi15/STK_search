import unittest
from run_search import run_search
from stk_search import SearchExp
from stk_search.Search_algorithm import Search_algorithm
from stk_search.Search_algorithm import Bayesian_Optimisation
from stk_search.Search_algorithm import (
    Representation_slatm,
    RepresentationPrecursor,
    Represenation_3D,
)
from stk_search.Objective_function import Look_up_table
import pandas as pd
from stk_search.utils import database_utils
from stk_search import Searched_space
import stk
import torch
import pymongo
from stk_search.geom3d.test_train import read_config, Pymodel
from stk_search.geom3d.frag_encoding_with_transformer import Fragment_encoder
from stk_search.geom3d.models import SchNet
# %%
def main(num_iteration, num_elem_initialisation, test_name="test",case="slatm"):
    # Load the searched space
    df_path = "data/output/Full_dataset/df_total_subset_16_11_23.csv"
    df_precursors_path = "data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl"  #'Data/output/Prescursor_data/calculation_data_precursor_310823_clean.pkl'

    df_total, df_precursors = database_utils.load_data_from_file(
        df_path, df_precursors_path
    )

    search_space_loc = "data/input/search_space/test/search_space1.pkl"
    # get initial elements
    objective_function = Look_up_table(df_total, 6)

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
    elif case == "graph_frag":
        config_dir = "/rds/general/user/ma11115/home/Geom3D/Geom3D/training/SchNet_Trans_80K"
        config = read_config(config_dir)
        model_config = config["model"]
        model = SchNet(
            hidden_channels=model_config["emb_dim"],
            num_filters=model_config["SchNet_num_filters"],
            num_interactions=model_config["SchNet_num_interactions"],
            num_gaussians=model_config["SchNet_num_gaussians"],
            cutoff=model_config["SchNet_cutoff"],
            readout=model_config["SchNet_readout"],
            node_class=model_config["node_class"],
        )
        EncodingModel = Fragment_encoder(
            input_dim = config["emb_dim"]*config["number_of_fragement"],
            model_dim=config["emb_dim"],
            num_heads=1,
            num_classes=model_config["emb_dim"],
            num_layers=1,
            dropout=0.0,
            lr=5e-4,
            warmup=50,
            max_iters=config["max_epochs"] ,
        )

        EncodingModel.add_encoder(model)
        state_dict = torch.load(config["model_transformer_chkpt"],map_location=torch.device('cpu'))
        EncodingModel.load_state_dict(state_dict['state_dict'])
        BO = Bayesian_Optimisation.Bayesian_Optimisation()
        client = pymongo.MongoClient(config["pymongo_client"])
        db_poly = stk.ConstructedMoleculeMongoDb(
            client,
            database=config["database_name"],
        )
        db_frag = stk.MoleculeMongoDb(
            client,
            database=config["database_name"],
        )
        BO.verbose = True
        BO.normalise_input = False
        BO.device = "cpu"#"cuda:0" if torch.cuda.is_available() else "cpu"
        BO.Representation = Represenation_3D.Representation3DFrag_transformer(EncodingModel,df_total,db_poly=db_poly,db_frag=db_frag,device=BO.device)

        search_algorithm = BO
        #X_explored_frag = BO.Representation.generate_repr(searched_space_df.iloc[inds])
    elif case == "graph_geom":
        config_dir = "/rds/general/user/ma11115/home/Geom3D/Geom3D/training/SchNet_Trans_80K"
        config = read_config(config_dir)
        pymodel = Pymodel.load_from_checkpoint(config["model_embedding_chkpt"])
        config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        pymodel.to(config["device"] )
        model_embedding = pymodel.molecule_3D_repr
        BO = Bayesian_Optimisation.Bayesian_Optimisation()
        client = pymongo.MongoClient(config["pymongo_client"])
        db_poly = stk.ConstructedMoleculeMongoDb(
            client,
            database=config["database_name"],
        )
        BO.Representation = Represenation_3D.Representation3D(model_embedding,df_total,db_poly=db_poly)
        BO.verbose = True
        BO.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        BO.normalise_input = False
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
    S_exp = SearchExp.Search_exp(
        search_space_loc,
        search_algorithm,
        objective_function,
        number_of_iterations,
        verbose=verbose,
    )
    S_exp.output_folder = "data/output/search_experiment/benchmark/" + test_name
    S_exp.num_elem_initialisation = num_elem_initialisation
    S_exp.benchmark = True
    S_exp.df_total = df_total
    S_exp.run_seach()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num_iteration", type=int, default=100)
    parser.add_argument("--num_elem_initialisation", type=int, default=10)
    parser.add_argument("--test_name", type=str, default="test")
    parser.add_argument("--case", type=str, default="slatm")
    args = parser.parse_args()
    main(args.num_iteration, args.num_elem_initialisation, args.test_name,args.case)

class TestStkSearch(unittest.TestCase):
    def test_run_search(self):
        try:
            run_search(num_iteration=10, num_elem_initialisation=5, test_name="Test")
        except Exception as e:
            self.fail(f"run_search raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()