from stk_search import Search_Exp
from stk_search.Search_algorithm import Search_algorithm
from stk_search.Search_algorithm import BayesianOptimisation
from stk_search.Representation import Representation_from_fragment,Representation_3d_from_fragment

from stk_search.Search_algorithm import (
    Representation_slatm,
    RepresentationPrecursor,
    Represenation_3D,
)
from stk_search.Objective_function import IP_ES1_fosc, Look_up_table
import pandas as pd
from stk_search.utils import database_utils
from stk_search import Searched_space
import stk
import torch
import pymongo
from stk_search.geom3d.pl_model import Pymodel
from stk_search.utils.config_utils import read_config
from stk_search.geom3d.oligomer_encoding_with_transformer import (
    initialise_model,
)
from stk_search.geom3d.models import SchNet
from stk_search.geom3d.dataloader import load_data_frag
from stk_search.Search_algorithm import Ea_surrogate
import os
import numpy as np


# %%
def main(
    num_iteration,
    num_elem_initialisation,
    test_name="test",
    case="random",
    target_name="target",
    config_dir="",
    aim=0,
    which_acquisition="EI",
    lim_counter=10,
    search_space_loc="data/input/search_space/test/search_space1.pkl",
    oligomer_size=6,
    df_path="",
    df_precursors_path="data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl",
    benchmark=False,
    dataset_representation_path="",
    frag_properties="all",
):
    # Load the searched space
    print(" number of fragment", oligomer_size)
    print(benchmark, "benchmark")
    df_total = pd.read_csv(df_path)
    df_precursors = pd.read_pickle(df_precursors_path)
    #df_total, df_precursors = database_utils.load_data_from_file(
    #    df_path, df_precursors_path, num_fragm=oligomer_size
    #)
    # get initial elements
    if benchmark:
        objective_function = Look_up_table(
            df_total, oligomer_size, target_name=target_name, aim=aim
        )
    else:
        objective_function = IP_ES1_fosc(oligomer_size)
        print("objective function")
        dataset_representation_path = (
            ""  # the dataset representation is only used for the benchmark
        )
    print(case, "  case  ")
    if case == "BO_precursor":
        BO = BayesianOptimisation.BayesianOptimisation()
        if frag_properties == "selected":
            frag_properties = []
            frag_properties = df_precursors.columns[1:7]
            frag_properties = frag_properties.append(
                df_precursors.columns[17:23]
            )
        else:
            frag_properties = df_precursors.select_dtypes(
                include=[np.number]
            ).columns
        print(frag_properties)
        BO.Representation = Representation_from_fragment.Representation_from_fragment(
            df_precursors, frag_properties
        )
        search_algorithm = BO
    elif case == "BO_learned":
        BO = BayesianOptimisation.BayesianOptimisation(
            which_acquisition=which_acquisition, lim_counter=lim_counter
        )
        BO.verbose = True
        #BO.normalise_input = False
        BO.device = "cpu"  # "cuda:0" if torch.cuda.is_available() else "cpu"
        BO.Representation = load_representation_BO_graph_frag(
            config_dir, df_total, dataset_path=dataset_representation_path
        )
        search_algorithm = BO

    elif case == "random":
        search_algorithm = Search_algorithm.random_search()
    elif case == "evolution_algorithm":
        search_algorithm = Search_algorithm.evolution_algorithm()
    elif case == "ea_surrogate":
        ## load model
        ## load search algorithm
        ea_surrogate = Ea_surrogate.Ea_surrogate()
        pymodel, ea_surrogate.Representation = load_representation_model_SUEA(
            config_dir, df_total, dataset_path=dataset_representation_path
        )
        ea_surrogate.verbose = True
        ea_surrogate.device = (
            "cpu"  # "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        ea_surrogate.pred_model = pymodel.graph_pred_linear
        search_algorithm = ea_surrogate
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
    S_exp.output_folder = (
        f"data/output/search_experiment/{oligomer_size}_frag/" + test_name
    )
    S_exp.num_elem_initialisation = num_elem_initialisation
    S_exp.benchmark = benchmark
    S_exp.df_total = df_total
    S_exp.run_seach()
    save_represention = False
    if save_represention == True:
        if case == "BO_learned" or case == "ea_surrogate":
            try:
                save_represention_dataset(
                    config_dir, search_algorithm.Representation
                )
            except:
                print("representation not saved")


def save_represention_dataset(config_dir, representation):
    import datetime

    config = read_config(config_dir)
    representation_dir = (
        f"{config['running_dir']}/{config['name']}" + "/transformer"
    )
    now = datetime.datetime.now().strftime("_%Y_%m_%d")
    torch.save(
        representation.dataset,
        representation_dir
        + f"/dataset_representation_{config['number_of_fragement']}_{now}.pt",
    )


def load_representation_BO_graph_frag(config_dir, df_total, dataset_path=""):
    config = read_config(config_dir)
    print(config["model_transformer_chkpt"])
    if os.path.exists(dataset_path):
        print('loading representation from ', dataset_path)
        data_list = torch.load(dataset_path)
        print("size of data list", len(data_list))
    else:
        print('no dataset found')
        data_list = None
    EncodingModel = initialise_model(config)
    BO = BayesianOptimisation.BayesianOptimisation()
    client = pymongo.MongoClient(config["pymongo_client"])
    db_poly = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    db_frag = stk.MoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    Representation = Representation_3d_from_fragment.Representation_3d_from_fragment(
        EncodingModel,
        df_total,
        data=data_list,
        db_poly=db_poly,
        db_frag=db_frag,
        device=BO.device,
    )
    return Representation


def load_representation_model_SUEA(
    config_dir, df_total, dataset_path="", device="cpu"
):
    config = read_config(config_dir)
    print(config["device"])
    if os.path.exists(dataset_path):
        data_list = torch.load(dataset_path)
        print("size of data list", len(data_list))
    else:
        data_list = None
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
    state_dict = torch.load(
        config["model_embedding_chkpt"], map_location=torch.device(device)
    )
    pymodel.load_state_dict(state_dict["state_dict"])
    EncodingModel = initialise_model(config)
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
    pymodel.graph_pred_linear.eval()
    Representation = Represenation_3D.Representation3DFrag_transformer(
        EncodingModel,
        df_total,
        data=data_list,
        db_poly=db_poly,
        db_frag=db_frag,
        device=ea_surrogate.device,
    )
    return pymodel, Representation


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num_iteration", type=int, default=100)
    parser.add_argument("--num_elem_initialisation", type=int, default=10)
    parser.add_argument("--test_name", type=str, default="test")
    parser.add_argument("--case", type=str, default="slatm")
    parser.add_argument("--target_name", type=str, default="target")
    parser.add_argument("--config_dir", type=str, default="")
    parser.add_argument("--aim", type=float, default=0.0)
    parser.add_argument("--which_acquisition", type=str, default="EI")
    parser.add_argument("--lim_counter", type=int, default=10)
    parser.add_argument("--benchmark", type=bool, default=False)
    parser.add_argument(
        "--search_space_loc",
        type=str,
        default="data/input/search_space/test/search_space1.pkl",
    )
    parser.add_argument("--df_path", type=str, default="")
    parser.add_argument(
        "--df_precursors_path",
        type=str,
        default="data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl",
    )
    parser.add_argument("--dataset_representation_path", type=str, default="")
    parser.add_argument("--oligomer_size", type=int, default=6)
    args = parser.parse_args()
    main(
        args.num_iteration,
        args.num_elem_initialisation,
        args.test_name,
        args.case,
        args.target_name,
        args.config_dir,
        args.aim,
        args.which_acquisition,
        args.lim_counter,
        args.search_space_loc,
        args.oligomer_size,
        args.df_path,
        args.df_precursors_path,
        args.benchmark,
        args.dataset_representation_path,
    )
