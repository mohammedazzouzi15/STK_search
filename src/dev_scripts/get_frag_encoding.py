"""Script to train the model and evaluate it on the datasets.

Evaluate the model on the datasets and save the results in the ephemeral folder.

Args:
----
    config_dir (str): The path to the directory containing the
        configuration file.

to run it from the command line:
python get_frag_encoding.py --config_dir config_dir.

"""



import os
import pathlib import Path

import numpy as np
import pandas as pd
import Path
import torch

from stk_search.geom3d import (
    dataloader,
    oligomer_encoding_with_transformer,
    train_models,
)
from stk_search.utils.config_utils import read_config, save_config


def main():  # noqa: PLR0915
    """Script to train the model and evaluate it on the datasets.

    Evaluate the model on the datasets and save the results in the ephemeral folder.

    Args:
    ----
        config_dir (str): The path to the directory containing the
            configuration file.

    """
    config = read_config(config_dir)
    (
        _,
        _,
        _,
        dataset_train,
        dataset_val,
        dataset_test,
    ) = dataloader.generate_dataset_and_dataloader(
        config,
    )
    if os.path.isfile(config["dataset_all_path"]):
        dataset_all = torch.load(
            config["dataset_all_path"], map_location=config["device"]
        )
    else:
        dataset_all = torch.tensor([])
    # save the dataset in the ephemeral folder
    config = save_datasets(
        config, dataset_train, dataset_val, dataset_test, dataset_all
    )

    # train the model
    config, min_val_loss = train_models.get_best_embedding_model(config_dir)
    output_file = config_dir + "/info.txt"
    with Path(output_file).open(mode="a") as file:
        file.write(f"Best model: {config['model_embedding_chkpt']}\n")
        file.write(f"Best model val loss: {min_val_loss}\n")
    # get frag dataset
    dataset_train_frag, pymodel = dataloader.load_data_frag(
        config, dataset_opt=dataset_train, dataset_name="train"
    )
    dataset_val_frag, pymodel = dataloader.load_data_frag(
        config, dataset_opt=dataset_val, dataset_name="val"
    )
    dataset_test_frag, pymodel = dataloader.load_data_frag(
        config, dataset_opt=dataset_test, dataset_name="test"
    )
    if Path(config["dataset_all_frag_path"]).is_file():
        dataset_all_frag = torch.load(
            config["dataset_all_frag_path"], map_location=config["device"]
        )
        dataset_all_frag = dataloader.updata_frag_dataset(
            dataset_all_frag,
            dataset_all,
            pymodel.molecule_3D_repr,
            config["model_name"],
        )
    dataset_test_frag = dataloader.updata_frag_dataset(
        dataset_test_frag,
        dataset_test,
        pymodel.molecule_3D_repr,
        config["model_name"],
    )
    dataset_train_frag = dataloader.updata_frag_dataset(
        dataset_train_frag,
        dataset_train,
        pymodel.molecule_3D_repr,
        config["model_name"],
    )
    dataset_val_frag = dataloader.updata_frag_dataset(
        dataset_val_frag,
        dataset_val,
        pymodel.molecule_3D_repr,
        config["model_name"],
    )
    train_loader_frag = dataloader.get_data_loader(dataset_train_frag, config)
    val_loader_frag = dataloader.get_data_loader(dataset_val_frag, config)
    test_loader_frag = dataloader.get_data_loader(dataset_test_frag, config)

    config = save_datasets_frag(
        config,
        dataset_train_frag,
        dataset_val_frag,
        dataset_test_frag,
        dataset_all_frag,
    )
    # evaluate model
    df_train_pred = evaluate_model(
        pymodel,
        train_loader_frag,
        name_df="train",
        config_dir=config["running_dir"],
    )
    df_val_pred = evaluate_model(
        pymodel,
        val_loader_frag,
        name_df="val",
        config_dir=config["running_dir"],
    )
    df_test_pred = evaluate_model(
        pymodel,
        test_loader_frag,
        name_df="test",
        config_dir=config["running_dir"],
    )
    mae_train, mse_train, r2_train = evaluale_model_performance(df_train_pred)
    mae_val, mse_val, r2_val = evaluale_model_performance(df_val_pred)
    mae_test, mse_test, r2_test = evaluale_model_performance(df_test_pred)
    with Path(output_file).open(mode="a") as file:
        file.write(
            f"MAE train: {mae_train:.2f}, MSE train: {mse_train:.2f}, R2 train: {r2_train:.2f}\n"
        )
        file.write(
            f"MAE val: {mae_val:.2f}, MSE val: {mse_val:.2f}, R2 val: {r2_val:.2f}\n"
        )
        file.write(
            f"MAE test: {mae_test:.2f}, MSE test: {mse_test:.2f}, R2 test: {r2_test:.2f}\n"
        )

    # run encoding training
    ephemeral_dir = (
        config["ephemeral_path"] + f"/{config['name'].replace('_','/')}/"
    )
    encoding_dataset_train = (
        oligomer_encoding_with_transformer.save_encoding_dataset(
            dataset_train_frag,
            config,
            dataset_name="_train",
            save_folder=ephemeral_dir,
        )
    )
    encoding_dataset_val = (
        oligomer_encoding_with_transformer.save_encoding_dataset(
            dataset_val_frag,
            config,
            dataset_name="_val",
            save_folder=ephemeral_dir,
        )
    )
    encoding_dataset_test = (
        oligomer_encoding_with_transformer.save_encoding_dataset(
            dataset_test_frag,
            config,
            dataset_name="_test",
            save_folder=ephemeral_dir,
        )
    )
    (
        oligomer_encoding_with_transformer.save_encoding_dataset(
            dataset_all_frag,
            config,
            dataset_name="_all",
            save_folder=ephemeral_dir,
        )
    )

    df_train_pred_learned = evaluate_model_learned(
        pymodel,
        encoding_dataset_train,
        name_df="train",
        target_name=config["target_name"],
        config_dir=config["running_dir"],
    )
    df_val_pred_learned = evaluate_model_learned(
        pymodel,
        encoding_dataset_val,
        name_df="val",
        target_name=config["target_name"],
        config_dir=config["running_dir"],
    )
    df_test_pred_learned = evaluate_model_learned(
        pymodel,
        encoding_dataset_test,
        name_df="test",
        target_name=config["target_name"],
        config_dir=config["running_dir"],
    )

    mae_train, mse_train, r2_train = evaluale_model_performance_learned(
        df_train_pred_learned, target_name=config["target_name"]
    )
    mae_val, mse_val, r2_val = evaluale_model_performance_learned(
        df_val_pred_learned, target_name=config["target_name"]
    )
    mae_test, mse_test, r2_test = evaluale_model_performance_learned(
        df_test_pred_learned, target_name=config["target_name"]
    )
    with Path(output_file).open(mode="a") as file:
        file.write(" Perfomance with learned embedding\n")
        file.write(
            f"MAE train: {mae_train:.2f}, MSE train: {mse_train:.2f}, R2 train: {r2_train:.2f}\n"
        )
        file.write(
            f"MAE val: {mae_val:.2f}, MSE val: {mse_val:.2f}, R2 val: {r2_val:.2f}\n"
        )
        file.write(
            f"MAE test: {mae_test:.2f}, MSE test: {mse_test:.2f}, R2 test: {r2_test:.2f}\n"
        )


def save_datasets(
    config, dataset_train, dataset_val, dataset_test, dataset_all
):
    """Save the datasets in the ephemeral folder.

    Args:
    ----
        config (dict): The configuration dictionary.
        dataset_train (torch.utils.data.Dataset): The training dataset.
        dataset_val (torch.utils.data.Dataset): The validation dataset.
        dataset_test (torch.utils.data.Dataset): The test dataset.
        dataset_all (torch.utils.data.Dataset): The all dataset.

    Returns:
    -------
        dict: The updated configuration dictionary.

    """
    name = config["name"]
    ephemeral_dir = config["ephemeral_path"] + f"/{name.replace('_','/')}/"
    Path(ephemeral_dir).mkdir(parents=True, exist_ok=True)

    torch.save(dataset_train, ephemeral_dir + "dataset_train.pth")
    torch.save(dataset_val, ephemeral_dir + "dataset_val.pth")
    torch.save(dataset_test, ephemeral_dir + "dataset_test.pth")
    torch.save(dataset_all, ephemeral_dir + "dataset_all.pth")
    config["dataset_path" + "_train"] = ephemeral_dir + "dataset_train.pth"
    config["dataset_path" + "_val"] = ephemeral_dir + "dataset_val.pth"
    config["dataset_path" + "_test"] = ephemeral_dir + "dataset_test.pth"
    config["dataset_path" + "_all"] = ephemeral_dir + "dataset_all.pth"
    save_config(config, config_dir)
    return config


def save_datasets_frag(
    config, dataset_train, dataset_val, dataset_test, dataset_all_frag
):
    name = config["name"]
    ephemeral_dir = (
        config["ephemeral_path"] + f"/{name.replace('_','/')}/transformer/"
    )
    Path(ephemeral_dir).mkdir(parents=True, exist_ok=True)

    torch.save(dataset_train, ephemeral_dir + "frag_dataset_train.pth")
    torch.save(dataset_val, ephemeral_dir + "frag_dataset_val.pth")
    torch.save(dataset_test, ephemeral_dir + "frag_dataset_test.pth")
    torch.save(dataset_all_frag, ephemeral_dir + "frag_dataset_all.pth")
    config["frag_dataset_path" + "_train"] = (
        ephemeral_dir + "frag_dataset_train.pth"
    )
    config["frag_dataset_path" + "_val"] = (
        ephemeral_dir + "frag_dataset_val.pth"
    )
    config["frag_dataset_path" + "_test"] = (
        ephemeral_dir + "frag_dataset_test.pth"
    )
    config["dataset_all_frag_path"] = ephemeral_dir + "frag_dataset_all.pth"
    save_config(config, config_dir)
    return config


def evaluate_model(
    pymodel, loader, name_df="train", target_name="target", config_dir=""
):
    InChIKeys = []
    predicted_target = []
    for batch in loader:
        with torch.no_grad():
            InChIKeys.append(batch[0].InChIKey)
            predicted_target.append(
                pymodel.graph_pred_linear(batch[0].y)
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
            # print("embedding size", batch[0].y.shape)
    df_pred = pd.DataFrame(
        {
            "InChIKey": np.concatenate(InChIKeys),
            f"predicted_{target_name}": np.concatenate(predicted_target),
        }
    )
    df_original = pd.read_csv(f"{config_dir}/df_{name_df}.csv")
    df_original = df_original.merge(df_pred, on="InChIKey")
    df_original.to_csv(f"{config_dir}/df_{name_df}_pred.csv", index=False)

    return df_original


def evaluale_model_performance(df_pred, aim=0.0, target_name="target"):
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    mae = mean_absolute_error(
        df_pred[f"{target_name}"], df_pred[f"predicted_{target_name}"]
    )
    mse = mean_squared_error(
        df_pred[f"{target_name}"], df_pred[f"predicted_{target_name}"]
    )
    r2 = r2_score(
        df_pred[f"{target_name}"], df_pred[f"predicted_{target_name}"]
    )
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

    return mae, mse, r2


def evaluate_model_learned(
    pymodel, datase_frag, name_df="train", target_name="target", config_dir=""
):
    InChIKeys = []
    predicted_target = []
    # get model device
    device = next(pymodel.parameters()).device
    for data in datase_frag:
        with torch.no_grad():
            InChIKeys.append(data.InChIKey)
            predicted_target.append(
                pymodel.graph_pred_linear(
                    data.learned_rpr.type(torch.float32).to(device)
                )
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
    df_pred = pd.DataFrame(
        {
            "InChIKey": InChIKeys,
            f"predicted_{target_name}_learned_embedding": predicted_target,
        }
    )
    df_original = pd.read_csv(
        f"{config_dir.replace('transformer','')}/df_{name_df}_pred.csv"
    )
    df_original = df_original.merge(df_pred, on="InChIKey")
    df_original.to_csv(
        f"{config_dir}/df_{name_df}_pred_learned.csv", index=False
    )

    return df_original


def evaluale_model_performance_learned(df_pred, target_name="target"):
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    mae = mean_absolute_error(
        df_pred[f"{target_name}"],
        df_pred[f"predicted_{target_name}_learned_embedding"],
    )
    mse = mean_squared_error(
        df_pred[f"{target_name}"],
        df_pred[f"predicted_{target_name}_learned_embedding"],
    )
    r2 = r2_score(
        df_pred[f"{target_name}"],
        df_pred[f"predicted_{target_name}_learned_embedding"],
    )
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

    return mae, mse, r2


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_dir", type=str, default="config_dir")
    args = argparser.parse_args()
    config_dir = args.config_dir
    main(config_dir)
