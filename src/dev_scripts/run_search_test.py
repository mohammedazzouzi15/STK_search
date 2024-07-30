import pymongo
import stk
import torch
from stk_search import SearchExp
from stk_search.geom3d.models import SchNet
from stk_search.geom3d.oligomer_encoding_with_transformer import (
    initialise_model,
)
from stk_search.geom3d.pl_model import Pymodel
from stk_search.geom3d.utils.config_utils import read_config
from stk_search.Objective_function import Look_up_table
from stk_search.Search_algorithm import (
    Bayesian_Optimisation,
    Ea_surrogate,
    Represenation_3D,
    Representation_slatm,
    RepresentationPrecursor,
    Search_algorithm,
)
from stk_search.utils import database_utils


# %%
def main(num_iteration, num_elem_initialisation, test_name="test",case="slatm",target_name="target",config_dir ="",aim=0,which_acquisition="EI",
                                                         lim_counter=2):
    # Load the searched space
    df_path = "data/output/Full_dataset/df_total_2024-01-05.csv"
    df_precursors_path = "data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl"  #'Data/output/Prescursor_data/calculation_data_precursor_310823_clean.pkl'

    df_total, df_precursors = database_utils.load_data_from_file(
        df_path, df_precursors_path
    )

    search_space_loc = "data/input/search_space/test/search_space1.pkl"
    # get initial elements
    objective_function = Look_up_table(df_total, 6,target_name=target_name, aim=aim)

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
        #config_dir = "/rds/general/user/ma11115/home/Geom3D/Geom3D/training/SchNet_Trans_80K"
        BO = Bayesian_Optimisation.Bayesian_Optimisation(which_acquisition=which_acquisition,
                                                         lim_counter=lim_counter)

        BO.verbose = True
        BO.normalise_input = False
        BO.device = "cpu"#"cuda:0" if torch.cuda.is_available() else "cpu"
        BO.Representation =  load_representation_BO_graph_frag(config_dir,df_total)
        search_algorithm = BO
        #X_explored_frag = BO.Representation.generate_repr(searched_space_df.iloc[inds])
    elif case == "graph_geom":
        #config_dir = "/rds/general/user/ma11115/home/Geom3D/Geom3D/training/SchNet_Trans_80K"
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
    elif case == "random":
        search_algorithm = Search_algorithm.random_search()
    elif case == "evolution_algorithm":
        search_algorithm = Search_algorithm.evolution_algorithm()
    elif case == "ea_surrogate":
        ## load model
        ## load search algorithm
        ea_surrogate = Ea_surrogate.Ea_surrogate()
        pymodel,ea_surrogate.Representation = load_representation_model_SUEA(config_dir,df_total)
        ea_surrogate.verbose = True
        ea_surrogate.device = "cpu"#"cuda:0" if torch.cuda.is_available() else "cpu"
        ea_surrogate.pred_model = pymodel.graph_pred_linear
        search_algorithm = ea_surrogate
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
    if case == "graph_geom" or case == "graph_frag" or case == "ea_surrogate":
        save_represention_dataset(config_dir,search_algorithm.Representation)


def save_represention_dataset(config_dir,representataion):
    config = read_config(config_dir)
    representation_dir = f"{config['running_dir']}/{config['name']}" + "_frag_" + str(config["number_of_fragement"])
    torch.save(representataion.dataset,representation_dir + "/dataset_representation.pt")


def load_representation_BO_graph_frag(config_dir,df_total):
    config = read_config(config_dir)
    print(config["model_transformer_chkpt"])
    representation_dir = f"{config['running_dir']}/{config['name']}" + "_frag_" + str(config["number_of_fragement"])
    print("load representation from ",representation_dir)
    data_list = torch.load( representation_dir + "/dataset_representation.pt")
    print("size of data list",len(data_list))
    EncodingModel = initialise_model(config)
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
    Representation = Represenation_3D.Representation3DFrag_transformer(EncodingModel,df_total,data=data_list,db_poly=db_poly,db_frag=db_frag,device=BO.device)
    return Representation

def load_representation_model_SUEA(config_dir,df_total):
    config = read_config(config_dir)
    representation_dir = f"{config['running_dir']}/{config['name']}" + "_frag_" + str(config["number_of_fragement"])
    data_list = torch.load( representation_dir + "/dataset_representation.pt")
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
    Representation=Represenation_3D.Representation3DFrag_transformer(EncodingModel,df_total,data=data_list,db_poly=db_poly,db_frag=db_frag,device=ea_surrogate.device)
    return pymodel,Representation


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
    parser.add_argument("--lim_counter", type=int, default=1)
    args = parser.parse_args()
    main(args.num_iteration, args.num_elem_initialisation, args.test_name,args.case,args.target_name,args.config_dir,args.aim,args.which_acquisition,args.lim_counter)
