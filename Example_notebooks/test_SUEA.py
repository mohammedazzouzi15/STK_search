# get the best using the surrogate model


import numpy as np
import pandas as pd
import swifter  # noqa: F401
from stk_search.geom3d import train_models
from stk_search.Representation import (
    Representation_from_fragment,
)
from stk_search.Search_algorithm import (
    BayesianOptimisation,
    Ea_surrogate,
    Search_algorithm,
)
from matplotlib import pyplot as plt


def initialise_search_algorithm(
    df_precursor_Mordred, df_precursors, config_dir
):
    # initialise search algorithm

    which_acquisition = "EI"
    lim_counter = 5
    BO_learned = BayesianOptimisation.BayesianOptimisation(
        which_acquisition=which_acquisition, lim_counter=lim_counter
    )
    BO_learned.verbose = True
    EA = Search_algorithm.evolution_algorithm()
    EA.number_of_parents = 5
    EA.number_of_random = 2
    EA.selection_method_mutation = "top"
    EA.selection_method_cross = "top"
    RAND = Search_algorithm.random_search()
    SUEA = Ea_surrogate.Ea_surrogate()
    SUEA.number_of_parents = 5
    SUEA.number_of_random = 2
    BO_Mord = BayesianOptimisation.BayesianOptimisation(
        which_acquisition=which_acquisition, lim_counter=lim_counter
    )
    BO_prop = BayesianOptimisation.BayesianOptimisation(
        which_acquisition=which_acquisition, lim_counter=lim_counter
    )
    BO_Mord.verbose = True
    BO_prop.verbose = True
    # load the Representation and the model

    config, min_val_loss = train_models.get_best_embedding_model(config_dir)
    SUEA = Ea_surrogate.Ea_surrogate()
    SUEA.config_dir = config_dir
    SUEA.load_representation_model()
    BO_learned.config_dir = config_dir
    BO_learned.load_representation_model()

    frag_properties = df_precursor_Mordred.select_dtypes(
        include=[np.number]
    ).columns
    BO_Mord.Representation = (
        Representation_from_fragment.RepresentationFromFragment(
            df_precursor_Mordred, frag_properties
        )
    )
    frag_properties = []
    frag_properties = df_precursors.columns[2:7]
    frag_properties = frag_properties.append(
        df_precursors.columns[[17, 18, 19, 20, 21, 22]]
    )
    # frag_properties = df_precursors.columns[[17,19,21]]
    print(frag_properties)
    BO_prop.Representation = (
        Representation_from_fragment.RepresentationFromFragment(
            df_precursors, frag_properties
        )
    )
    BO_learned.number_of_parents = 10
    BO_learned.number_of_random = 5
    BO_prop.number_of_parents = 10
    BO_prop.number_of_random = 5
    BO_Mord.number_of_parents = 10
    BO_Mord.number_of_random = 5
    return BO_learned, EA, SUEA, BO_Mord, BO_prop, RAND


def get_predictions(suea, df_elements):
    x_unsqueezed = suea.Representation.generate_repr(df_elements)
    if suea.verbose:
        pass
    # get model prediction
    # make sure that the model and the data have the same dtype
    x_unsqueezed = x_unsqueezed.to(suea.device)
    model_dtype = next(suea.pred_model.parameters()).dtype
    if x_unsqueezed.dtype != model_dtype:
        x_unsqueezed = x_unsqueezed.type(model_dtype)
    acquisition_values = (
        suea.pred_model(x_unsqueezed).squeeze().cpu().detach().numpy()
    )
    return acquisition_values


def main():
    df_total_path_bench = (
        "data_example/data_benchmark/30K_benchmark_150524.csv"
    )
    df_precursor_path = "data_example/data_benchmark/df_properties.pkl"
    df_precursor_Mordred_path = "data_example/precursor/df_PCA_mordred_descriptor_290224.pkl"  # "data_example/data_benchmark/precursor_with_mordred_descriptor.pkl"
    config_dir = "data_example/representation_learning/splitrand-nummol20000"
    df_precursor_Mordred = pd.read_pickle(df_precursor_path)
    df_precursors = pd.read_pickle(df_precursor_Mordred_path)
    BO_learned, EA, SUEA, BO_Mord, BO_prop, RAND = initialise_search_algorithm(
        df_precursor_Mordred, df_precursors, config_dir
    )
    # initialise search space
    top_mol_count = 1000
    df_total = pd.read_csv(df_total_path_bench)
    df_target = df_total.sort_values("target", ascending=False).reset_index(
        drop=True
    )
    df_target = df_target[:top_mol_count]

    df_target["pred"] = get_predictions(SUEA, df_target)
    df_target["rank_target"] = df_target["target"].rank(ascending=False)
    df_target["rank_pred"] = df_target["pred"].rank(ascending=False)
    fig, ax = plt.subplots()    

    ax.scatter(df_target["target"], df_target["pred"],c=df_target["target"], label="pred")
    ax.legend()
    ax.set_ylabel("prediction")
    ax.set_xlabel("target")
    fig.savefig("rank_pred_vs_rank_target.png")
    #print(get_predictions(SUEA, df_target))

if __name__ == "__main__":
    main()
