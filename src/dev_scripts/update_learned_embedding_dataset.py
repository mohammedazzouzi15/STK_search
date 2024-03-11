import os  #
from pathlib import Path  #
import pandas as pd
import torch
import numpy as np
from stk_search.utils.config_utils import read_config, save_config
from stk_search.utils import update_datasets
from argparse import ArgumentParser


def main(config_dir, dataset_all_frag_path=None, df_total_path=None):
    """
    check the input dataset for the learned embedding model and add missing molecules to the dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = read_config(config_dir)
    if dataset_all_frag_path is None:
        dataset_all_frag_path = config["dataset_all_frag_path"]
    if df_total_path is None:
        df_total_path = config["df_total_path"]
    dataset_all_frag = torch.load(dataset_all_frag_path, map_location=device)
    df_total = pd.read_csv(df_total_path)
    dataset_learned_embedding_update = (
        update_datasets.update_dataset_learned_embedding(
            df_total, dataset_all_frag, config, extension="all"
        )
    )
    print(
        "length of new dataset learned embedding",
        len(dataset_learned_embedding_update),
    )
    config[f"dataset_learned_embedding_all"] = (
        config["ephemeral_path"]
        + f"/{config['name'].replace('_','/')}/dataset_representation_all.pt"
    )
    torch.save(
        dataset_learned_embedding_update,
        config[f"dataset_learned_embedding_all"],
    )
    save_config(config, config_dir)
    return config


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--config_dir", type=str, help="config file directory")
    args.add_argument(
        "--dataset_all_frag_path",
        type=str,
        help="path to the dataset_all_frag",
        default="/rds/general/ephemeral/user/ma11115/ephemeral/STK_search/data/representation_learning/6-frag/target//dataset_all_frag_schnet.pth",
    )
    args.add_argument(
        "--df_total_path",
        type=str,
        help="path to the df_total",
        default="/rds/general/user/ma11115/home/STK_Search/STK_search/data/representation_learning/df_total_290224.csv",
    )
    args = args.parse_args()
    main(args.config_dir, args.dataset_all_frag_path, args.df_total_path)
    #
