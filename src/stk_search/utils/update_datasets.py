# plot the performance of the prediction model on the new molecules
from stk_search.geom3d import dataloader
from stk_search.geom3d import oligomer_encoding_with_transformer
import pymongo
import stk
import torch


def get_dataset_from_df(dataset_all, df, config):
    """check the input dataset for the oligomer embeddiing model and add missing molecules to the dataset

    Args:
        dataset_all: list of dictionaries
            list of dictionaries containing the information of the molecules in the dataset
        df: pandas dataframe
            dataframe containing the information of the molecules
        config: dictionary
            dictionary containing the configuration of the model
    Returns:
        dataset: list of dictionaries
    """
    dataset_all_dict = {data["InChIKey"]: data for data in dataset_all}
    dataset = []
    missing_inchikey = []
    for Inchikey in df["InChIKey"]:
        if Inchikey in dataset_all_dict.keys():
            dataset.append(dataset_all_dict[Inchikey])
        else:
            missing_inchikey.append(Inchikey)
    df_missing = df[df["InChIKey"].isin(missing_inchikey)].copy()
    df_missing.reset_index(drop=True, inplace=True)
    print(f"Missing {len(missing_inchikey)} Inchikey in the dataset")
    client = pymongo.MongoClient(config["pymongo_client"])
    db = stk.ConstructedMoleculeMongoDb(
        client,
        database=config["database_name"],
    )
    radius = config["model"]["cutoff"] if "cutoff" in config["model"] else 0.1
    dataset_missing = dataloader.generate_dataset(
        df_missing,
        db,
        number_of_molecules=df_missing.shape[0],
        model_name=config["model_name"],
        radius=radius,
    )
    dataset.extend(dataset_missing)
    return dataset, dataset_missing


# save the dataset for the transformer model if already calculated dataset all exists
def get_dataset_frag_from_df(dataset_all_frag, df, config):
    """
    check the input dataset for the oligomer encoding model and add missing molecules to the dataset

    Args:
        dataset_all_frag: list of dictionaries
            list of dictionaries containing the information of the molecules in the dataset
        df: pandas dataframe
            dataframe containing the information of the molecules
        config: dictionary
            dictionary containing the configuration of the model
    Returns:
        dataset: list of dictionaries"""
    dataset_all_dict = {data[0]["InChIKey"]: data for data in dataset_all_frag}
    # dataset = [dataset_all_dict[Inchikey] for Inchikey in df['InChIKey']]
    dataset = []
    missing_inchikey = []
    for Inchikey in df["InChIKey"]:
        if Inchikey in dataset_all_dict.keys():
            dataset.append(dataset_all_dict[Inchikey])
        else:
            missing_inchikey.append(Inchikey)
    df_missing = df[df["InChIKey"].isin(missing_inchikey)].copy()
    df_missing.reset_index(drop=True, inplace=True)
    print(
        f"Missing {len(missing_inchikey)} Inchikey in the dataset of frag input"
    )

    dataset_missing, _ = dataloader.load_data_frag(
        config, df_total=df_missing, dataset_name="missing"
    )
    dataset.extend(dataset_missing)
    return dataset


def update_dataset_learned_embedding(
    df, dataset_all_frag, config, extension="all"
):
    """
    check the input dataset for the learned embedding model and add missing molecules to the dataset

    Args:
        df: pandas dataframe
            dataframe containing the information of the molecules
        config: dictionary
            dictionary containing the configuration of the model
    Returns:
        dataset: list of dictionaries
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_learned_embedding = torch.load(
        config[f"dataset_learned_embedding_{extension}"], map_location=device
    )
    dataset_all_dict = {
        data["InChIKey"]: data for data in dataset_learned_embedding
    }
    dataset_learned_embedding_update = []
    missing_inchikey = []
    for Inchikey in df["InChIKey"]:
        if Inchikey in dataset_all_dict.keys():
            dataset_learned_embedding_update.append(dataset_all_dict[Inchikey])
        else:
            missing_inchikey.append(Inchikey)
    df_missing = df[df["InChIKey"].isin(missing_inchikey)].copy()
    df_missing.reset_index(drop=True, inplace=True)
    print(
        f"Missing {len(missing_inchikey)} Inchikey in the dataset of learned embedding"
    )
    dataset_frag_dict = {
        data[0]["InChIKey"]: data for data in dataset_all_frag
    }
    dataset_frag_missing = []
    for Inchikey in missing_inchikey:
        if Inchikey in dataset_frag_dict.keys():
            dataset_frag_missing.append(dataset_frag_dict[Inchikey])
    ephemeral_dir = (
        config["ephemeral_path"] + f"/{config['name'].replace('_','/')}/"
    )
    dataset_learned_embedding_missing = (
        oligomer_encoding_with_transformer.save_encoding_dataset(
            dataset_frag_missing,
            config,
            dataset_name="_missing",
            save_folder=ephemeral_dir,
        )
    )
    dataset_learned_embedding_update.extend(dataset_learned_embedding_missing)
    print(
        " length of new dataset learned embedding",
        len(dataset_learned_embedding_update),
    )

    return dataset_learned_embedding_update


def update_dataset_oligomer(dataset_path, df_total, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(dataset_path, map_location=device)
    dataset, dataset_missing = get_dataset_from_df(dataset, df_total, config)
    torch.save(dataset, dataset_path)
    return dataset, dataset_missing


def update_dataset_frag(dataset_path, df_total, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(dataset_path, map_location=device)
    dataset = get_dataset_frag_from_df(dataset, df_total, config)
    torch.save(dataset, dataset_path)
    return dataset
