# plot max fitness
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from stk_search.utils import tanimoto_similarity_utils


def plot_metric(
    df_plot,
    plot_function_list,
    results_dict,
    df_list_dict,
    nb_iterations=300,
    target_name="target",
    nb_initialisation=50,
    number_cols=3,
    df_total=None,
    num_results_min =500
):

    number_rows = int(np.ceil((len(plot_function_list)+2) / number_cols))
    fig, axes = plt.subplots(
        number_rows, number_cols, figsize=(5 * number_rows, 5 * number_cols)
    )
    axes = axes.flatten()
    color_list = sns.color_palette("tab10", len(df_plot))
    df_plot["color"] = color_list

    keys = df_plot["key"]
    metric_dict_res = {}
    for key in keys.values:
        res = results_dict[key][:num_results_min]
        color = df_plot[df_plot["key"] == key]["color"].iloc[0]
        case_name = df_plot[df_plot["key"] == key]["case_name"].iloc[0]
        if df_total is None:
            df_total = pd.read_csv(
                df_plot[df_plot["key"] == key]["df_path"].iloc[0], low_memory=False
            )

        metric_dict = {}

        for ii, plot_function in enumerate(plot_function_list):
            ax = axes[ii]
            metric = plot_function(
                res,
                nb_iterations=nb_iterations,
                axs=ax,
                color=color,
                label=case_name,
                target_name=target_name,
                df_total=df_total,
                nb_initialisation=0,
            )
            metric_dict[plot_function.__name__] = metric
            ax.axvspan(0, nb_initialisation, alpha=0.1, color="grey")
        metric_dict_res[key] = metric_dict
    # ax.set_ylabel("Max Fitness")
    # ax.set_ylim(-6, 0)

    axes[0].legend(
        loc="upper left", bbox_to_anchor=(-0.1, 1.4), ncol=3, fontsize=12
    )
    return fig, axes, metric_dict_res


def plot_metric_mae(metric_dict_res, metric, df_list_dict):
    mae_list, metric_list = [], []
    r2_list = []
    for key in metric_dict_res:
        try:
            path = (
                df_list_dict[key]["config_dir"]
                .iloc[0]
                .split("/transformer/")[0]
            )
            df_results = pd.read_csv(path + "/df_test_pred_learned.csv")
            df_results["predicted_target_learned_embedding"] = df_results[
                "predicted_target_learned_embedding"
            ].apply(lambda x: eval(x)[0])
            mae = mean_absolute_error(
                df_results["target"].values,
                df_results["predicted_target_learned_embedding"].values,
            )
            mae_list.append(mae)
            metric_list.append(metric_dict_res[key][metric])
            r2_list.append(
                r2_score(
                    df_results["target"].values,
                    df_results["predicted_target_learned_embedding"].values,
                )
            )
        except:
            pass
    fig, ax = plt.subplots()
    key = "evolution_algorithm_total"
    ax.scatter(mae_list, metric_list)
    ax.hlines(metric_dict_res[key][metric], 0, 10, color="red")
    ax.set_ylabel(metric)
    ax.set_xlabel("MAE on test set")
    return fig, ax


def add_similarity_plots(axes, df_plot, df_mol_dict, results_dict,
                         nb_iterations=250,  nb_initialisation=50):
    keys = df_plot["key"].values
    ax = axes.flatten()
    for key in keys:
        res = results_dict[key]
        color = df_plot[df_plot["key"] == key]["color"].iloc[0]
        _, df_mol_dict = (
            tanimoto_similarity_utils.plot_similarity_results_elem_suggested_to_initial(
                res[:10],
                nb_iterations=nb_iterations,
                nb_initialisation=nb_initialisation,
                ax=ax[-2],
                color=color,
                label=key,
                df_mol_dict=df_mol_dict,
            )
        )
        _, df_mol_dict = (
            tanimoto_similarity_utils.plot_similarity_results_elem_suggested_df(
                res[:10],
                nb_iterations=nb_iterations,
                nb_initialisation=nb_initialisation,
                ax=ax[-1],
                color=color,
                label=key,
                df_mol_dict=df_mol_dict,
            )
        )
    return df_mol_dict
    # ax[0].legend(loc='upper left', bbox_to_anchor=(-0.1, 1.3), ncol=3)


def save_mol_dict(df_mol_dict):
    df = pd.DataFrame(df_mol_dict, index=[0]).T
    df.columns = ["mol"]
    df["InChIKey"] = df.index

    df.to_pickle(
        "data/output/search_experiment/mol_dict.pkl"
    )


def load_mol_dict():
    df = pd.read_pickle(
        "data/output/search_experiment/mol_dict.pkl"
    )
    return df.T.loc["mol"].to_dict()
