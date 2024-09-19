import pandas as pd
import pymongo
import stk
import torch
from stk_search import SearchExp
from stk_search.geom3d.frag_encoding_with_transformer import Fragment_encoder
from stk_search.geom3d.models import SchNet
from stk_search.geom3d.test_train import Pymodel, read_config
from stk_search.ObjectiveFunctions.IpEs1Fosc import IpEs1Fosc
from stk_search.Search_algorithm import (
    Bayesian_Optimisation,
    Represenation_3D,
    Representation_slatm,
    RepresentationPrecursor,
    Search_algorithm,
)
from stk_search.utils import database_utils


# %%
def main(num_iteration, num_elem_initialisation, test_name="test", case="slatm",SearchSpace_loc = "data/input/SearchSpace/test/SearchSpace1.pkl"):
    # Load the searched space
    df_path = "data/output/Full_dataset/df_total_2023_11_09.csv"
    df_precursors_path = "data/output/Prescursor_data/calculation_data_precursor_071123_clean.pkl"  #'Data/output/Prescursor_data/calculation_data_precursor_310823_clean.pkl'
    df_total, df_precursors = database_utils.load_data_from_file(
        df_path, df_precursors_path
    )

    # get initial elements
    ObjectiveFunction = IpEs1Fosc(oligomer_size=6)

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
        df_precursors_old_path = "data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl"
        df_precursors_old = pd.read_pickle(df_precursors_old_path)
        frag_properties = []
        frag_properties = df_precursors_old.columns[1:7]
        frag_properties = frag_properties.append(df_precursors_old.columns[17:23])
        print(frag_properties)
        BO.Representation = RepresentationPrecursor.RepresentationPrecursor(
            df_precursors_old, frag_properties
        )
        search_algorithm = BO
    elif case == "random":
        search_algorithm = Search_algorithm.random_search()
    elif case == "evolution_algorithm":
        search_algorithm = Search_algorithm.evolution_algorithm()
    elif case == "graph_frag":
        ## load model
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
        state_dict = torch.load(config["model_transformer_chkpt"],map_location=torch.device("cpu"))
        EncodingModel.load_state_dict(state_dict["state_dict"])
        ## load search algorithm
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
    elif case == "ea_surrogate":
        from stk_search.Search_algorithm import Ea_surrogate
        ## load model
        config_dir = "/rds/general/user/ma11115/home/Geom3D/Geom3D/training/SchNet_Trans_80K"
        config = read_config(config_dir)

        model_config = config["model"]
        graph_pred_linear = torch.nn.Linear(
                model_config["emb_dim"], model_config["num_tasks"]
            )
        model = SchNet(
            hidden_channels=model_config["emb_dim"],
            num_filters=model_config["SchNet_num_filters"],
            num_interactions=model_config["SchNet_num_interactions"],
            num_gaussians=model_config["SchNet_num_gaussians"],
            cutoff=model_config["SchNet_cutoff"],
            readout=model_config["SchNet_readout"],
            node_class=model_config["node_class"],
        )
        pymodel = Pymodel(model, graph_pred_linear)
        state_dict = torch.load(config["model_embedding_chkpt"],map_location=torch.device("cpu"))
        pymodel.load_state_dict(state_dict["state_dict"])
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
        state_dict = torch.load(config["model_transformer_chkpt"],map_location=torch.device("cpu"))
        EncodingModel.load_state_dict(state_dict["state_dict"])
        ## load search algorithm
        ea_surrogate = Ea_surrogate.Ea_surrogate()
        client = pymongo.MongoClient(config["pymongo_client"])
        db_poly = stk.ConstructedMoleculeMongoDb(
            client,
            database=config["database_name"],
        )
        db_frag = stk.MoleculeMongoDb(
            client,
            database=config["database_name"],
        )
        ea_surrogate.verbose = True
        ea_surrogate.device = "cpu"#"cuda:0" if torch.cuda.is_available() else "cpu"
        ea_surrogate.Representation = Represenation_3D.Representation3DFrag_transformer(EncodingModel,df_total,db_poly=db_poly,db_frag=db_frag,device=ea_surrogate.device)
        pymodel.graph_pred_linear.eval()
        ea_surrogate.pred_model = pymodel.graph_pred_linear
        search_algorithm = ea_surrogate
    else:
        raise ValueError("case not recognised")

    number_of_iterations = num_iteration
    verbose = True
    num_elem_initialisation = num_elem_initialisation
    S_exp = SearchExp.Search_exp(
        SearchSpace_loc,
        search_algorithm,
        ObjectiveFunction,
        number_of_iterations,
        verbose=verbose,
    )
    SearchSpace_name = SearchSpace_loc.split("/")[-1].replace(".pkl", "")
    S_exp.output_folder = "data/output/search_experiment/"+"exp4_"+SearchSpace_name+"/" + test_name
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
    parser.add_argument("--SearchSpace_loc", type=str, default="data/input/SearchSpace/test/SearchSpace1.pkl")
    args = parser.parse_args()
    main(args.num_iteration, args.num_elem_initialisation, args.test_name,args.case,args.SearchSpace_loc)
