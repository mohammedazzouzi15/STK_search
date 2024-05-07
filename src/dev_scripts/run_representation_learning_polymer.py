from stk_search.geom3d import polymer_GNN_architecture_utils
from stk_search.utils.config_utils import read_config, save_config
from stk_search.geom3d import pl_model
import torch.nn.functional as Functional
from stk_search.geom3d import train_models
import torch
import pandas as pd
import os

def main(config_dir):
    config = read_config(config_dir)
    bbs_dict = polymer_GNN_architecture_utils.get_bbs_dict(
        "mongodb://ch-atarzia.ch.ic.ac.uk/", "stk_mohammed_new"
    )

    (
        train_loader,
        val_loader,
        test_loader,
        dataset_train,
        dataset_val,
        dataset_test,
    ) = polymer_GNN_architecture_utils.generate_dataset_and_dataloader(
        config, bbs_dict
    )

    config = polymer_GNN_architecture_utils.save_datasets(
        config, dataset_train, dataset_val, dataset_test
    )
    config_dir = config["running_dir"]
    output_file = config_dir + "info.txt"
    with open(output_file, "w") as file:
        file.write(f"Model: {config['name']}\n")
        file.write(f"Target: {config['target_name']}\n")
        file.write("spliting function: topk_split\n")
        file.write(f"Number of molecules: {config['num_molecules']}\n")
        file.write(f"Number of fragment: {config['number_of_fragement']}\n")
        file.write(f"Number of training data: {len(dataset_train)}\n")
        file.write(f"Number of validation data: {len(dataset_val)}\n")
        file.write(f"Number of test data: {len(dataset_test)}\n")
        file.write(f"Number of training data loader: {len(train_loader)}\n")
        file.write(f"Number of validation data loader: {len(val_loader)}\n")
        file.write(f"Number of test data loader: {len(test_loader)}\n")
    train_models.load_and_run_model_training(
        config, train_loader, val_loader, pl_model.Pymodel_new
    )
    pymodel = load_model(config_dir, config)
    pymodel.eval()
    print("Model loaded")
    with open(output_file, "a") as file:
        file.write(f"Model loaded\n")
    df_train_pred = evaluate_model_prediction(
        train_loader, pymodel, config_dir, name_df="train"
    )
    df_val_pred = evaluate_model_prediction(
        val_loader, pymodel, config_dir, name_df="val"
    )
    df_test_pred = evaluate_model_prediction(
        test_loader, pymodel, config_dir, name_df="test"
    )
    print("Model evaluation done")
    with open(output_file, "a") as file:
        file.write(f"Model evaluation done\n")

    mae_train, mse_train, r2_train = evaluale_model_performance(df_train_pred)
    mae_val, mse_val, r2_val = evaluale_model_performance(df_val_pred)
    mae_test, mse_test, r2_test = evaluale_model_performance(df_test_pred)
    with open(output_file, "a") as file:
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


def load_model(config_dir, config):
    config, min_val_loss = train_models.get_best_embedding_model(config_dir)
    output_file = config_dir + "info.txt"
    with open(output_file, "a") as file:
        file.write(f"Best model: {config['model_embedding_chkpt']}\n")
        file.write(f"Best model val loss: {min_val_loss}\n")

    model, graph_pred_linear = pl_model.model_setup(config)
    print("Model loaded: ", config["model_name"])
    # Pass the model and graph_pred_linear to the Pymodel constructor
    pymodel = pl_model.Pymodel_new(model, graph_pred_linear, config)
    # Load the state dictionary
    chkpt_path = config["model_embedding_chkpt"]
    if os.path.exists(chkpt_path):
        checkpoint = torch.load(chkpt_path, map_location=config["device"])
        pymodel.load_state_dict(state_dict=checkpoint["state_dict"])

    return pymodel


def evaluate_pymodel(data, pymodel, device):
    with torch.no_grad():
        z = pymodel.molecule_3D_repr(
            data.x.to(device), data.positions.to(device), data.batch.to(device)
        )
        z_opt = pymodel.molecule_3D_repr(
            data.x_opt.to(device),
            data.positions_opt.to(device),
            data.batch.to(device),
        )
        z = pymodel.transform_to_opt(z)
        z = pymodel.graph_pred_linear(z)
        z_opt = pymodel.graph_pred_linear(z_opt)
    # print(z,z_opt,data.y)
    return z, z_opt, data.y


def evaluate_model_prediction(loader, pymodel, config_dir, name_df="train"):
    y_pred_list, y_pred_opt_list, y_list = [], [], []
    Inchikey_list = []
    for batch in loader:
        y_pred, y_pred_opt, y = evaluate_pymodel(
            batch, pymodel, pymodel.device
        )
        y_pred_list.extend(y_pred.detach().squeeze().numpy())
        y_pred_opt_list.extend(y_pred_opt.detach().squeeze().numpy())
        y_list.extend(y.detach().squeeze().numpy())
        Inchikey_list.extend(batch.InChIKey)
    df_original = pd.DataFrame(
        {
            "InChIKey": Inchikey_list,
            "y": y_list,
            "y_pred": y_pred_list,
            "y_pred_opt": y_pred_opt_list,
        }
    )
    df_original.to_csv(f"{config_dir}/df_{name_df}_pred.csv", index=False)
    return df_original


def evaluale_model_performance(df_pred):
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    mae = mean_absolute_error(df_pred["y"], df_pred["y_pred"])
    mse = mean_squared_error(df_pred["y"], df_pred["y_pred"])
    r2 = r2_score(df_pred["y"], df_pred["y_pred"])
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

    return mae, mse, r2


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_dir", type=str, default="config_dir")
    args = argparser.parse_args()
    config_dir = args.config_dir
    main(config_dir)
