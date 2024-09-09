import json
import os
import pickle
import subprocess

import numpy as np
import pandas as pd
import pymongo
import stk
import torch

from stk_search import SearchExp
from stk_search.geom3d.models import SchNet
from stk_search.geom3d.oligomer_encoding_with_transformer import (
    initialise_model,
)
from stk_search.geom3d.pl_model import Pymodel
from stk_search.ObjectiveFunctions.ObjectiveFunction import LookUpTable
from stk_search.ObjectiveFunctions.IpEs1Fosc import IpEs1Fosc

from stk_search.Representation import (
    Representation_3d_from_fragment,
    Representation_from_fragment,
)
from stk_search.Search_algorithm import (
    BayesianOptimisation,
    Ea_surrogate,
    MultifidelityBayesianOptimisation,
    Search_algorithm,
)
from stk_search.utils.config_utils import read_config


# %%
def main(
    num_iteration,
    num_elem_initialisation,
    test_name="test",
    case="random",
    target_name="target",
    config_dir="",
    aim="maximise",
    which_acquisition="EI",
    lim_counter=10,
    SearchSpace_loc="data/input/SearchSpace/test/SearchSpace1.pkl",
    oligomer_size=6,
    df_path="",
    df_representation_path="data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl",
    benchmark=False,
    dataset_representation_path="",
    frag_properties="all",
    budget=None,
):
    input_json = locals()
    SearchSpace_loc = "data/input/STK_SearchSpace/SearchSpace_test.pkl"
    with open(SearchSpace_loc, "rb") as f:
        SearchSpace = pickle.load(f)
    # Load the searched space
    print(" number of fragment", oligomer_size)
    print(benchmark, "benchmark")
    df_total = pd.read_csv(df_path)
    df_Representation = pd.read_pickle(df_representation_path)

    # get initial elements
    if benchmark:
        ObjectiveFunction = LookUpTable(
            df_total, oligomer_size, target_name=target_name, aim=aim
        )
    else:
        ObjectiveFunction = IpEs1Fosc(oligomer_size)
        print("objective function")
        dataset_representation_path = (
            ""  # the dataset Representation is only used for the benchmark
        )
    print(case, "  case  ")
    if case == "BO_precursor":
        BO = BayesianOptimisation.BayesianOptimisation(
            which_acquisition=which_acquisition, lim_counter=lim_counter
        )
        if frag_properties == "selected":
            frag_properties = []
            frag_properties = df_representation.columns[1:7]
            frag_properties = frag_properties.append(
                df_representation.columns[17:23]
            )
        else:
            frag_properties = df_representation.select_dtypes(
                include=[np.number]
            ).columns
        print(frag_properties)
        BO.Representation = (
            Representation_from_fragment.RepresentationFromFragment(
                df_representation, frag_properties
            )
        )
        search_algorithm = BO

    elif case == "BO_learned":
        BO = BayesianOptimisation.BayesianOptimisation(
            which_acquisition=which_acquisition, lim_counter=lim_counter
        )
        BO.verbose = True
        # BO.normalise_input = False
        BO.device = "cpu"  # "cuda:0" if torch.cuda.is_available() else "cpu"
        BO.Representation = load_representation_BO_graph_frag(
            config_dir, df_total, dataset_path=dataset_representation_path
        )
        search_algorithm = BO

    elif case == "BO_learned_new":
        BO = BayesianOptimisation.BayesianOptimisation(
            which_acquisition=which_acquisition, lim_counter=lim_counter
        )
        BO.verbose = True
        # BO.normalise_input = False
        BO.device = "cpu"  # "cuda:0" if torch.cuda.is_available() else "cpu"
        BO.Representation, pymodel = load_representation_model(config_dir)
        BO.pred_model = pymodel.graph_pred_linear
        search_algorithm = BO

    elif case == "MFBO":
        MFBO = MultifidelityBayesianOptimisation.MultifidelityBayesianOptimisation(
            budget=budget,
            which_acquisition=which_acquisition,
            lim_counter=lim_counter,
        )
        if frag_properties == "selected":
            frag_properties = []
            frag_properties = df_representation.columns[17:23]
        else:
            frag_properties = df_representation.select_dtypes(
                include=[np.number]
            ).columns
        print(frag_properties)
        MFBO.fidelity_col = len(frag_properties) * oligomer_size
        MFBO.Representation = (
            Representation_from_fragment.RepresentationFromFragment(
                df_representation, frag_properties
            )
        )
        search_algorithm = MFBO
    elif case == "ea_surrogate_new":
        ea_surrogate = Ea_surrogate.Ea_surrogate()
        ea_surrogate.verbose = True
        # BO.normalise_input = False
        ea_surrogate.device = (
            "cpu"  # "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        ea_surrogate.Representation, pymodel = load_representation_model(
            config_dir
        )
        ea_surrogate.pred_model = pymodel.graph_pred_linear
        search_algorithm = ea_surrogate

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
    S_exp = SearchExp.SearchExp(
        SearchSpace,
        search_algorithm,
        ObjectiveFunction,
        number_of_iterations,
        verbose=verbose,
    )
    S_exp.output_folder = (
        f"data/output/search_experiment/{oligomer_size}_frag/" + test_name
    )
    S_exp.num_elem_initialisation = num_elem_initialisation
    S_exp.benchmark = benchmark
    S_exp.df_total = df_total
    # S_exp.df_precur

    input_json["run_search_name"] = S_exp.search_exp_name
    input_json["search_output_folder"] = S_exp.output_folder
    input_json["date"] = S_exp.date
    save_path = f"data/output/search_experiment/search_exp_database/{S_exp.search_exp_name}.json"
    save_run_search_inputs(input_json, save_path)
    S_exp.run_seach()


def save_run_search_inputs(inputs, save_path="run_search_new_inputs.json"):
    # Get the current git version
    git_version = (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .strip()
        .decode("utf-8")
    )

    # Add the git version to the inputs
    inputs["git_version"] = git_version

    # Save the inputs to a file
    with open(save_path, "w") as f:
        json.dump(inputs, f)

    print("Inputs saved.")


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
    import uuid

    repr_id = str(uuid.uuid4())
    config = read_config(config_dir)
    print(config["model_transformer_chkpt"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if os.path.exists(dataset_path):
        print("loading Representation from ", dataset_path)
        data_list = torch.load(dataset_path, map_location=device)
        print("size of data list", len(data_list))
    else:
        print("no dataset found")
        data_list = None
        name = config["name"]
        ephemeral_dir = config["ephemeral_path"] + f"/{name.replace('_','/')}/"
        os.makedirs(ephemeral_dir + "/local_dataset", exist_ok=True)
        save_dataset_path = (
            ephemeral_dir + f"/local_dataset/local_dataset_new{repr_id}.pt"
        )
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
    Representation = (
        Representation_3d_from_fragment.Representation3dFromFragment(
            EncodingModel,
            df_total,
            data=data_list,
            db_poly=db_poly,
            db_frag=db_frag,
            device=BO.device,
        )
    )
    if not os.path.exists(dataset_path):
        Representation.save_dataset_path = save_dataset_path
        Representation.db_name = config["name"]
    return Representation


def load_representation_model_SUEA(
    config_dir, df_total, dataset_path="", device="cpu"
):
    import uuid

    repr_id = str(uuid.uuid4())
    config = read_config(config_dir)
    print(config["device"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if os.path.exists(dataset_path):
        data_list = torch.load(dataset_path, map_location=device)
        print("size of data list", len(data_list))
    else:
        data_list = None
        name = config["name"]
        ephemeral_dir = config["ephemeral_path"] + f"/{name.replace('_','/')}/"
        os.makedirs(ephemeral_dir + "/local_dataset", exist_ok=True)

        save_dataset_path = (
            ephemeral_dir + f"/local_dataset/local_dataset_new{repr_id}.pt"
        )
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
    pymodel = Pymodel(model, graph_pred_linear, config)
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
    Representation = (
        Representation_3d_from_fragment.Representation3dFromFragment(
            EncodingModel,
            df_total,
            data=data_list,
            db_poly=db_poly,
            db_frag=db_frag,
            device=ea_surrogate.device,
        )
    )
    if ~os.path.exists(dataset_path):
        Representation.save_dataset_path = save_dataset_path
        Representation.db_name = config["name"]
    return pymodel, Representation


def load_representation_model(config_dir):
    """New model Representation for the search algorithm
    Args:
        config_dir: str
            path to the config file
            Returns:
            representation: RepresentationPoly3d

    pymodel: Pymodel
    """
    from stk_search.geom3d import pl_model
    from stk_search.Representation import Representation_poly_3d

    config = read_config(config_dir)
    chkpt_path = config["model_embedding_chkpt"]
    checkpoint = torch.load(chkpt_path, map_location=config["device"])
    model, graph_pred_linear = pl_model.model_setup(config)
    print("Model loaded: ", config["model_name"])
    # Pass the model and graph_pred_linear to the Pymodel constructor
    pymodel = pl_model.Pymodel_new(model, graph_pred_linear, config)
    # Load the state dictionary
    pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
    # pymodel.load_state_dict(state_dict=checkpoint["state_dict"])
    pymodel.to(config["device"])
    Representation = Representation_poly_3d.RepresentationPoly3d(
        pymodel, device="cpu"
    )
    return representation, pymodel


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num_iteration", type=int, default=100)
    parser.add_argument("--num_elem_initialisation", type=int, default=10)
    parser.add_argument("--test_name", type=str, default="test")
    parser.add_argument("--case", type=str, default="slatm")
    parser.add_argument("--target_name", type=str, default="target")
    parser.add_argument("--config_dir", type=str, default="")
    parser.add_argument("--aim")
    parser.add_argument("--which_acquisition", type=str, default="EI")
    parser.add_argument("--lim_counter", type=int, default=10)
    parser.add_argument("--benchmark", type=bool, default=False)
    parser.add_argument(
        "--SearchSpace_loc",
        type=str,
        default="data/input/SearchSpace/test/SearchSpace1.pkl",
    )
    parser.add_argument("--df_path", type=str, default="")
    parser.add_argument(
        "--df_representation_path",
        type=str,
        default="data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl",
    )
    parser.add_argument("--dataset_representation_path", type=str, default="")
    parser.add_argument("--oligomer_size", type=int, default=6)
    parser.add_argument("--frag_properties", type=str, default="all")
    parser.add_argument(
        "--budget",
        type=lambda x: None if x == "None" else int(x),
        nargs="?",
        default=None,
    )
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
        args.SearchSpace_loc,
        args.oligomer_size,
        args.df_path,
        args.df_representation_path,
        args.benchmark,
        args.dataset_representation_path,
        args.frag_properties,
        args.budget,
    )
