from stk_search.geom3d import oligomer_encoding_with_transformer
from stk_search.utils.config_utils import save_config, read_config
from stk_search.geom3d import train_models
import glob
import os
import argparse


def run_training(
    target_name="target",
    aim=0.0,
    num_molecules=1000,
    max_epochs=5,
    running_dir="/rds/general/user/ma11115/home/STK_Search/STK_search/data/representation_learning",
    training=True,
    load_dataset=0,
    num_fragment=3,
    df_path="",
):
    # get config and set it up
    config_dir = (
        running_dir
        + f"/SchNet_frag/{target_name}_{num_molecules}_{aim}_frag_{num_fragment}/"
    )
    config = read_config(config_dir, model_name="SchNet")
    config["number_of_fragement"] = num_fragment
    config["STK_path"] = "/rds/general/user/ma11115/home/STK_Search/STK_search"
    config["df_precursor"] = "calculation_data_precursor_190923_clean.pkl"
    config["max_epochs"] = max_epochs
    if load_dataset == 1:
        config["load_dataset"] = True
    else:
        config["load_dataset"] = False
    config["dataset_path"] = ""
    config["dataset_path_frag"] = ""
    config["df_total"] = df_path
    config["save_dataset"] = False
    config["num_molecules"] = num_molecules
    config["running_dir"] = running_dir + f"/SchNet_frag/"
    config["train_ratio"] = 0.8
    config["test_dataset_path"] = ""
    config["save_dataset_frag"] = True
    config["name"] = f"{target_name}_{num_molecules}_{aim}_frag_{num_fragment}"
    config["target_name"] = target_name
    config["model_embedding_chkpt"] = ""
    config["model_transformer_chkpt"] = ""
    config["dataset_path_frag"] = ""
    save_config(config, config_dir)
    # train GNN model
    train_models.main(config_dir)
    config = read_config(config_dir, model_name="SchNet")
    files = glob.glob(config_dir + "/*.ckpt")
    min_val_loss = 1000
    for file in files:
        # get val loss from file name
        val_loss = float(file.split("val_loss=")[1].split("-")[0])
        if val_loss < min_val_loss:
            print(file)
            min_val_loss = val_loss
            config["model_embedding_chkpt"] = file
    save_config(config, config_dir)
    # run training for transformer encoding
    training = True
    config_dir_transformer = (
        running_dir
        + f"/SchNet_frag/{target_name}_{num_molecules}_{aim}_frag_{num_fragment}/transformer/"
    )

    config["save_dataset_frag"] = True

    config["model_transformer_chkpt"] = ""
    config["dataset_path_frag"] = (
        running_dir
        + f"/SchNet_frag/{target_name}_{num_molecules}_{aim}_frag_{num_fragment}/transformer/dataset_frag.pt"
    )
    save_config(config, config_dir_transformer)
    oligomer_encoding_with_transformer.main(config_dir_transformer, training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="fosc1")
    parser.add_argument("--aim", type=float, default=0)
    parser.add_argument("--num_molecules", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--load_dataset", type=int, default=1)
    parser.add_argument("--num_fragment", type=int, default=6)
    parser.add_argument("--df_path", type=str, default="")
    args = parser.parse_args()
    run_training(
        target_name=args.target,
        aim=args.aim,
        num_molecules=args.num_molecules,
        max_epochs=args.max_epochs,
        load_dataset=args.load_dataset,
        num_fragment=args.num_fragment,
        df_path=args.df_path,
    )
