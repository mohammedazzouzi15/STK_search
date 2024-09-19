import datetime
import json
import os

import torch


def read_config(dir, model_name=""):
    if os.path.exists(dir + "/config.json"):
        config = load_config(dir)
    else:
        # Set parameters
        config = {}
        config["seed"] = 42
        config["save_dataset"] = False
        config["name"] = "SchNet_target_1K_TEST_5e4lr"
        config["pymongo_client"] = "mongodb://129.31.66.201/"
        config["database_name"] = "stk_mohammed_BO"
        config["STK_path"] = "/rds/general/user/cb1319/home/GEOM3D/STK_path/"
        config["running_dir"] = (
            "/rds/general/user/cb1319/home/GEOM3D/Geom3D/training/"
        )
        config["batch_size"] = 128
        config["df_total"] = ""
        config["df_precursor"] = "calculation_data_precursor_071123_clean.pkl"
        config["num_molecules"] = 100
        config["num_workers"] = 0
        config["num_tasks"] = 1
        config["emb_dim"] = 128
        config["max_epochs"] = 3
        config["train_ratio"] = 0.8
        config["valid_ratio"] = 0.1
        config["number_of_fragement"] = 6
        config["model_path"] = ""
        config["pl_model_chkpt"] = ""
        config["load_dataset"] = False
        config["dataset_path"] = ""
        config["dataset_path_frag"] = ""
        config["target_name"] = "target"

        # prompt the user to enter model name
        if model_name == "":
            config["model_name"] = input("Enter model name: ")
        else:
            config["model_name"] = model_name

        if config["model_name"] == "SchNet":
            config["model"] = {}
            config["model"]["node_class"] = 119
            config["model"]["edge_class"] = 5
            config["model"]["num_tasks"] = 1
            config["model"]["emb_dim"] = 128
            config["model"]["SchNet_num_filters"] = 128
            config["model"]["SchNet_num_interactions"] = 8
            config["model"]["SchNet_num_gaussians"] = 51
            config["model"]["SchNet_cutoff"] = 6
            config["model"]["SchNet_readout"] = "mean"
            config["batch_size"] = 128

        elif config["model_name"] == "DimeNet":
            config["model"] = {}
            config["model"]["node_class"] = 119
            config["model"]["hidden_channels"] = 300
            config["model"]["out_channels"] = 1
            config["model"]["num_blocks"] = 6
            config["model"]["num_bilinear"] = 8
            config["model"]["num_spherical"] = 7
            config["model"]["num_radial"] = 6
            config["model"]["cutoff"] = 10.0
            config["model"]["envelope_exponent"] = 5
            config["model"]["num_before_skip"] = 1
            config["model"]["num_after_skip"] = 2
            config["model"]["num_output_layers"] = 3

        elif config["model_name"] == "DimeNetPlusPlus":
            config["model"] = {}
            config["model"]["node_class"] = 119
            config["model"]["hidden_channels"] = 300
            config["model"]["out_channels"] = 1
            config["model"]["num_blocks"] = 6
            config["model"]["int_emb_size"] = 64
            config["model"]["basis_emb_size"] = 8
            config["model"]["out_emb_channels"] = 64
            config["model"]["num_spherical"] = 7
            config["model"]["num_radial"] = 6
            config["model"]["cutoff"] = 10.0
            config["model"]["envelope_exponent"] = 5
            config["model"]["num_before_skip"] = 1
            config["model"]["num_after_skip"] = 2
            config["model"]["num_output_layers"] = 3

        elif config["model_name"] == "GemNet":
            config["model"] = {}
            config["model"]["node_class"] = 119
            config["model"]["num_spherical"] = 7
            config["model"]["num_radial"] = 6
            config["model"]["num_blocks"] = 4
            config["model"]["emb_size_atom"] = 64
            config["model"]["emb_size_edge"] = 64
            config["model"]["emb_size_trip"] = 64
            config["model"]["emb_size_quad"] = 32
            config["model"]["emb_size_rbf"] = 16
            config["model"]["emb_size_cbf"] = 16
            config["model"]["emb_size_sbf"] = 32
            config["model"]["emb_size_bil_quad"] = 32
            config["model"]["emb_size_bil_trip"] = 64
            config["model"]["num_before_skip"] = 1
            config["model"]["num_after_skip"] = 1
            config["model"]["num_concat"] = 1
            config["model"]["num_atom"] = 2
            config["model"]["cutoff"] = 5.0
            config["model"]["int_cutoff"] = 10.0
            config["model"]["triplets_only"] = 1
            config["model"]["direct_forces"] = 0
            config["model"]["envelope_exponent"] = 5
            config["model"]["extensive"] = 1
            config["model"]["forces_coupled"] = 0
            config["model"]["num_targets"] = 1

        elif config["model_name"] == "Equiformer":
            config["model"] = {}
            config["model"]["Equiformer_radius"] = 4.0
            config["model"]["Equiformer_irreps_in"] = "5x0e"
            config["model"]["Equiformer_num_basis"] = 32
            config["model"]["Equiformer_hyperparameter"] = 0
            config["model"]["Equiformer_num_layers"] = 3
            config["model"]["node_class"] = 64
            config["model"]["irreps_node_embedding"] = "64x0e+32x1e+16x2e"

        elif config["model_name"] == "PaiNN":
            config["model"] = {}
            config["model"]["n_atom_basis"] = 64
            config["model"]["n_interactions"] = 6
            config["model"]["n_rbf"] = 20
            config["model"]["cutoff"] = 4.0
            config["model"]["max_z"] = 93
            config["model"]["n_out"] = 1
            config["model"]["readout"] = "add"
            config["batch_size"] = 16

        elif config["model_name"] == "SphereNet":
            config["model"] = {}
            config["model"]["hidden_channels"] = 128
            config["model"]["out_channels"] = 1
            config["model"]["cutoff"] = 5.0
            config["model"]["num_layers"] = 4
            config["model"]["int_emb_size"] = 64
            config["model"]["basis_emb_size_dist"] = 8
            config["model"]["basis_emb_size_angle"] = 8
            config["model"]["basis_emb_size_torsion"] = 8
            config["model"]["out_emb_channels"] = 256
            config["model"]["num_spherical"] = 7
            config["model"]["num_radial"] = 6
            config["model"]["envelope_exponent"] = 5
            config["model"]["num_before_skip"] = 1
            config["model"]["num_after_skip"] = 2
            config["model"]["num_output_layers"] = 3
        else:
            msg = "Model name not recognised"
            raise ValueError(msg)
        save_config(config, dir)

    return config


def save_config(config, dir):
    os.makedirs(dir, exist_ok=True)
    # save config to json
    with open(dir + "/config.json", "w") as f:
        json.dump(config, f, indent=4, separators=(",", ": "), sort_keys=True)


def load_config(dir):
    # load config from json
    with open(dir + "/config.json") as f:
        config = json.load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def read_search_config(config_search_dir):
    if os.path.exists(config_search_dir + "/config_search.json"):
        config = load_search_config(config_search_dir)
    else:
        config = {}
        config["num_iteration"] = 100
        config["num_elem_initialisation"] = 10
        config["test_name"] = "test"
        config["case"] = "slatm"
        config["target_name"] = "target"
        config["config_dir"] = ""
        config["aim"] = 0.0
        config["which_acquisition"] = "EI"
        config["lim_counter"] = 1
        config["benchmark"] = False
        config["SearchSpace_loc"] = (
            "data/input/SearchSpace/test/SearchSpace1.pkl"
        )
        config["df_path"] = ""
        config["df_precursors_path"] = (
            "data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl"
        )
        config["oligomer_size"] = 6
        save_search_config(config, config_search_dir)
    return config


def save_search_config(config, dir):
    os.makedirs(dir, exist_ok=True)
    # save config to json
    with open(dir + "/config_search.json", "w") as f:
        json.dump(config, f, indent=4, separators=(",", ": "), sort_keys=True)


def load_search_config(dir):
    # load config from json
    with open(dir + "/config_search.json") as f:
        return json.load(f)


def generate_config(
    target_name="target",
    aim=0.0,
    num_molecules=20000,
    max_epochs=100,
    running_dir="/rds/general/user/ma11115/home/STK_Search/STK_search/data/representation_learning",
    num_fragment=6,
    df_path="",
    model_name="SchNet",
    split_type="rand",
):
    # get config and set it up
    date_now = datetime.datetime.now().strftime("%y%m%d")
    name = f"{num_fragment}-frag_{target_name}_{date_now}__{model_name}_split{split_type}-nummol{num_molecules}"
    config_dir = running_dir + f"/{name.replace('_','/')}/"
    config = read_config(config_dir, model_name=model_name)
    config["number_of_fragement"] = num_fragment
    config["STK_path"] = "/rds/general/user/ma11115/home/STK_Search/STK_search"
    config["ephemeral_path"] = (
        "/rds/general/ephemeral/user/ma11115/ephemeral/home/STK_Search/STK_search/data/representation_learning"
    )
    config["df_precursor"] = (
        "/rds/general/user/ma11115/home/STK_Search/STK_search/data/output/Prescursor_data/calculation_data_precursor_190923_clean.pkl"
    )
    config["max_epochs"] = max_epochs
    config["ephemeral_path"] = (
        "/rds/general/ephemeral/user/ma11115/ephemeral/STK_search/data/representation_learning"
    )
    config["load_dataset"] = False
    config["df_total"] = df_path
    config["save_dataset"] = False
    config["num_molecules"] = num_molecules
    config["running_dir"] = config_dir
    config["train_ratio"] = 0.9
    config["save_dataset_frag"] = True
    config["name"] = name
    config["target_name"] = target_name
    if "model_embedding_chkpt" not in config:
        config["model_embedding_chkpt"] = ""
    if "model_transformer_chkpt" not in config:
        config["model_transformer_chkpt"] = ""
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["lr"] = 5e-4
    config["lr_scheduler"] = "CosineAnnealingLR"
    config["lr_encoder"] = 2e-4
    config["split_type"] = split_type
    config["dataset_all_path"] = (
        "/rds/general/ephemeral/user/ma11115/ephemeral/STK_search/data/representation_learning/6-frag/target"
        + "/dataset_all_schnet.pth"
    )
    config["dataset_all_frag_path"] = (
        "/rds/general/ephemeral/user/ma11115/ephemeral/STK_search/data/representation_learning/6-frag/target"
        + "/dataset_all_frag_schnet.pth"
    )
    save_config(config, config_dir)
    return config
