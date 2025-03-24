import copy
import datetime
import glob
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from stk_search.utils import Search_results_plot


def join_name(x):
    return "_".join(x)


def get_results_length(x):
    try:
        return len(pd.read_pickle(x)["fitness_acquired"])
    except Exception:
        return 0

def define_color_dict():
    color_dict = {
        "BO_learned": sns.color_palette("tab10")[0],
        "BO_Mord": sns.color_palette("tab10")[1],
        "BO_precursor": sns.color_palette("tab10")[2],
        "ea_surrogate": sns.color_palette("tab10")[3],
        "evolution_algorithm": sns.color_palette("tab10")[4],
        "random": sns.color_palette("tab10")[5],
    }
    return color_dict

def generate_datafame_from_search_results(search_results, max_iteration, num_initialisation):
    """Generate a dataframe from the search results."""
    for dict_org in search_results:
        dict_copy = dict_org.copy()
        dict_copy.pop("searched_space_df")
        df = pd.DataFrame.from_records(dict_copy)
        df = df[(df["ids_acquired"] < max_iteration) & (df["ids_acquired"] > num_initialisation)]
        df = df.drop_duplicates(subset=["ids_acquired"], keep="last")
        df["max_fitness_acquired"] = df["ids_acquired"].apply(
            lambda x: df[df["ids_acquired"] <= x]["fitness_acquired"].max()
        )
        df["mean_fitness_acquired"] = df["ids_acquired"].apply(
            lambda x: df[df["ids_acquired"] <= x]["fitness_acquired"].mean()
        )
        yield df


def plot_simple_regret_stuff(df_summary_1, results_dict, num_results_min, max_iteration, num_initialisation,color_dict):
    df_plot = df_summary_1.copy()
    keys = set(df_plot["key"].to_numpy()).intersection(set(results_dict.keys()))
    print("Keys:", keys)
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.2, 1, 0.2])
    gs.update(wspace=0.05, hspace=0.05)
    ax1, ax2, ax3, ax4 = [plt.subplot(gs[i]) for i in range(4)]
    df_total_bench = pd.read_csv(df_plot["df_path"].iloc[0], low_memory=False)
    keys = ["BO-learned", "BO-Mord", "BO-Prop", "SUEA", "EA", "Rand"]
    keys = ["BO_learned", "BO_Mord", "BO_precursor", "ea_surrogate", "evolution_algorithm", "random"]
    keys = [x for x in keys if x in results_dict.keys()]
    #keys = set(keys).intersection(set(results_dict.keys()))
    print("Keys:", keys)
    for color_num, key in enumerate(keys):
        res = results_dict[key][:num_results_min]
        df_results = pd.concat(
            generate_datafame_from_search_results(res, max_iteration=max_iteration, num_initialisation=num_initialisation)
        )
        sns.lineplot(x="ids_acquired", y="max_fitness_acquired", data=df_results, ax=ax1, label=key, color=color_dict[key])
        sns.lineplot(x="ids_acquired", y="mean_fitness_acquired", data=df_results, ax=ax3, color=color_dict[key])
        df_results.to_csv(f"figures/df_results_{key}.csv")

    sns.histplot(data=df_total_bench, y="target", bins=100, ax=ax2, color="black", alpha=0.5)
    sns.histplot(data=df_total_bench, y="target", bins=50, ax=ax4, color="black", alpha=0.5)
    return fig, df_plot


def plot_metric(
    df_plot,
    plot_function_list,
    results_dict,
    nb_iterations=300,
    target_name="target",
    nb_initialisation=50,
    number_cols=3,
    df_total=None,
    num_results_min=500,
    color_dict=None,
):
    number_rows = int(np.ceil((len(plot_function_list) + 2) / number_cols))
    fig, axes = plt.subplots(
        number_rows, number_cols, figsize=(5 * number_rows, 5 * number_cols)
    )
    axes = axes.flatten()
    #color_list = sns.color_palette("tab10", len(df_plot))
    #df_plot["color"] = color_list

    keys = set(df_plot["key"].to_numpy())
    keys = keys.intersection(set(results_dict.keys()))
    metric_dict_res = {}
    print("Keys:", keys)
    #reorder the keys
    keys = ["BO_learned", "BO_Mord", "BO_precursor", "ea_surrogate", "evolution_algorithm", "random"]
    #keys = set(keys).intersection(set(results_dict.keys()))
    print("Keys:", keys)

    for key in keys:
        res = results_dict[key][:num_results_min]
        #color = df_plot[df_plot["key"] == key]["color"].iloc[0]
        color = color_dict[key]
        case_name = df_plot[df_plot["key"] == key]["case_name"].iloc[0]
        if df_total is None:
            df_total = pd.read_csv(
                df_plot[df_plot["key"] == key]["df_path"].iloc[0],
                low_memory=False,
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
    axes[0].legend(
        loc="upper left", bbox_to_anchor=(-0.1, 1.4), ncol=3, fontsize=12
    )
    return fig, axes, metric_dict_res


def load_search_list(row):
    search_list = []
    search_list.append(pickle.load(open(row["search_exp_file"], "rb")))
    return search_list


def get_dataframe_of_searches(
    save_path="/media/mohammed/Work/STK_search/Example_notebooks/output/search_experiment_benchmark/search_exp_database",
):
    print(save_path)
    json_files = glob.glob(f"{save_path}/*.json")
    list_json = []
    for json_file in json_files:
        with open(json_file) as f:
            list_json.append(json.load(f))
        f.close()
    print(len(list_json))
    df = pd.DataFrame(list_json)
    df["search_exp_file"] = (
        df["search_output_folder"]
        + "/"
        + df["date"]
        + "/results_"
        + df["run_search_name"]
        + ".pkl"
    )
    df["results_lenght"] = df["search_exp_file"].apply(
        lambda x: get_results_length(x)
    )
    return df


def load_search_dict(df, min_num_iteration):
    results_dict = {}
    for _, row in df.iterrows():
        if get_results_length(row["search_exp_file"]) >= min_num_iteration:
            results_dict.setdefault(row["key"], []).append(pickle.load(open(row["search_exp_file"], "rb")))
    return results_dict


def generate_datafame_from_search_results_best_mol(
    search_results, max_iteration=200, num_initialisation=0, min_value=0
):
    """Generate a dataframe from the search results."""
    for dict_org in search_results:
        dict_copy = dict_org.copy()
        dict_copy.pop("searched_space_df")
        df = pd.DataFrame.from_records(dict_copy)
        df = df[(df["ids_acquired"] < max_iteration) & (df["ids_acquired"] > num_initialisation)]
        df = df.drop_duplicates(subset=["ids_acquired"], keep="last")

        df["number_of_top_mol_found"] = df["ids_acquired"].apply(
            lambda x: len(
                df[
                    (df["ids_acquired"] <= x)
                    & (df["fitness_acquired"] >= min_value)
                ]
            )
        )
        df["rate_of_discovery"] = df["ids_acquired"].apply(
            lambda x: len(
                df[
                    (df["ids_acquired"] <= x)
                    & (df["fitness_acquired"] >= min_value)
                ]
            )
            / df[df["ids_acquired"] <= x].shape[0]
        )
        yield df


def plot_top_mols_acquired(results_dict, df_plot, fig, num_results_min, max_iteration, num_initialisation, top_mol_count,color_dict):
    cool_colors = sns.color_palette("tab10", len(df_plot))
    keys = set(df_plot["key"].to_numpy()).intersection(set(results_dict.keys()))
    print("Keys:", keys)
    df_total_bench = pd.read_csv(df_plot["df_path"].iloc[0], low_memory=False)
    min_value = df_total_bench["target"].sort_values().iloc[-top_mol_count]
    keys = ["BO_learned", "BO_Mord", "BO_precursor", "ea_surrogate", "evolution_algorithm", "random"]
    keys = [x for x in keys if x in results_dict.keys()]
    #keys = set(keys).intersection(set(results_dict.keys()))
    print("Keys:", keys)

    for color_num, key in enumerate(keys):
        res = results_dict[key][:num_results_min]
        df_results = pd.concat(
            generate_datafame_from_search_results_best_mol(
                res, max_iteration=max_iteration, num_initialisation=num_initialisation, min_value=min_value
            )
        )
        sns.lineplot(x="ids_acquired", y="number_of_top_mol_found", data=df_results, ax=fig.axes[4], label=key, color=color_dict[key])
        sns.lineplot(x="ids_acquired", y="rate_of_discovery", data=df_results, ax=fig.axes[5],color=color_dict[key])


def modify_figure(fig_org, legend_list, x_limits, y_limits, tick_labels):
    """
    Modify the figure layout and scales.

    Args:
    ----
        fig_org: Original figure object.
        legend_list: List of legend labels.
        x_limits: Dictionary with x-axis limits for each subplot (e.g., {0: (50, 450)}).
        y_limits: Dictionary with y-axis limits for each subplot (e.g., {4: (0, 38)}).
        tick_labels: Dictionary with tick labels for each subplot (e.g., {4: [0, 50, 100]}).
    """
    fig = copy.deepcopy(fig_org)
    fig.set_size_inches(20, 13)

    lg = fig.axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(0.1, 1.15),
        ncol=6,
        fontsize=20,
    )
    for i in range(len(lg.texts)):
        lg.texts[i].set_text(legend_list[i])
    fig.axes[4].legend().set_visible(False)

    # Apply axis limits and tick labels
    for ax_index, ax in enumerate(fig.axes):
        if ax_index in x_limits:
            ax.set_xlim(*x_limits[ax_index])
        if ax_index in y_limits:
            ax.set_ylim(*y_limits[ax_index])
        if ax_index in tick_labels:
            ax.xaxis.set_ticklabels(tick_labels[ax_index])

    fig.axes[4].xaxis.set_major_locator(MaxNLocator(nbins=8))
    fig.axes[5].xaxis.set_major_locator(MaxNLocator(nbins=8))
    fig.axes[0].xaxis.set_major_locator(MaxNLocator(nbins=8))
    fig.axes[2].xaxis.set_major_locator(MaxNLocator(nbins=8))

    for ax_index in [0, 2, 4, 5]:
        for spine in fig.axes[ax_index].spines.values():
            spine.set_visible(True)

    # Adjust subplot positions
    fig.axes[0].set_position([0.1, 0.5, 0.3, 0.4])
    fig.axes[1].set_position([0.405, 0.5, 0.05, 0.4])
    fig.axes[2].set_position([0.525, 0.5, 0.3, 0.4])
    fig.axes[3].set_position([0.83, 0.5, 0.05, 0.4])
    fig.axes[4].set_position([0.1, 0.1, 0.3, 0.4])
    fig.axes[5].set_position([0.525, 0.1, 0.3, 0.4])

    # Set axis labels
    fig.axes[0].set_xlabel("")
    fig.axes[2].set_xlabel("")
    fig.axes[4].set_xlabel("Number of iterations")
    fig.axes[5].set_xlabel("Number of iterations")
    fig.axes[0].set_ylabel("Max $F_{comb}$ \n of molecules found (a.u.)")
    fig.axes[2].set_ylabel("Mean $F_{comb}$ \n of molecules found (a.u.)")
    fig.axes[4].set_ylabel("Number of top \nmolecules found")
    fig.axes[5].set_ylabel(" Rate of discovery \nof top molecules")

    # Add subplot labels
    fig.text(0.05, 0.9, "a)", fontsize=23, fontweight="bold", va="top")
    fig.text(0.47, 0.9, "b)", fontsize=23, fontweight="bold", va="top")
    fig.text(0.05, 0.5, "c)", fontsize=23, fontweight="bold", va="top")
    fig.text(0.47, 0.5, "d)", fontsize=23, fontweight="bold", va="top")
    fig.text(0.88, 0.51, "x100", fontsize=16, va="top")

    return fig


def modify_figure__layout_simple(fig, legend_list, x_limits, y_limits):
    """
    Modify the figure layout and scales for a simpler layout.

    Args:
    ----
        fig: Figure object.
        legend_list: List of legend labels.
        x_limits: Dictionary with x-axis limits for each subplot.
        y_limits: Dictionary with y-axis limits for each subplot.
    """
    from matplotlib import gridspec

    gs = gridspec.GridSpec(2, 3, figure=fig)
    fig.set_size_inches(20, 8)

    for i, ax in enumerate(fig.axes[:3]):
        ax.set_subplotspec(gs[0, i])
    if len(fig.axes) < 6:
        fig.add_subplot(gs[1, 0])
        fig.add_subplot(gs[1, 1])
    fig.axes[4].set_subplotspec(gs[1, 0])
    fig.axes[5].set_subplotspec(gs[1, 1])
    gs.update(wspace=0.1, hspace=0.1)

    for ax in fig.get_axes():
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)

    # Apply axis limits
    for ax_index, ax in enumerate(fig.axes):
        if ax_index in x_limits:
            ax.set_xlim(*x_limits[ax_index])
        if ax_index in y_limits:
            ax.set_ylim(*y_limits[ax_index])

    fig.axes[0].legend().set_visible(False)
    fig.tight_layout()
    lg = fig.axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(0.1, 1.12),
        ncol=6,
        fontsize=20,
        prop={"style": "italic"},
    )

    for i in range(len(lg.texts)):
        lg.texts[i].set_text(legend_list[i])

    plt.show()
    return fig


def main():
    run_name = "runs5"
    save_path = f"/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/{run_name}"
    print(save_path)
    color_dict = define_color_dict()
    # Configurable parameters
    min_num_iteration = 250
    num_results_min = min_num_iteration
    max_iteration = min_num_iteration
    num_initialisation = 50
    top_mol_count = 300
    x_limits = {0: (50, min_num_iteration), 2: (50, min_num_iteration), 4: (50, min_num_iteration), 5: (50, min_num_iteration), 1: (0, 80), 3: (0, 3100)}
    y_limits = {0:(0,0.7),1:(0,0.7),2:(-2,1),3:(-2,1),4: (0, 38), 5: (0, 0.43)}
    tick_labels = {4: np.arange(0, min_num_iteration, 50),5: np.arange(0, min_num_iteration, 50), 3: [0,15,30]}

    df = get_dataframe_of_searches(save_path)
    df = df[df["benchmark"]]
    df_all = df.copy()
    print(df_all.head())
    df_all["key"] = df.apply(lambda x: join_name([x["search_output_folder"].split("/")[-1]]), axis=1)
    df_all["df_path"] = "/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/30K_benchmark_150524.csv"
    df_all.to_csv(f"figures/df_all_{run_name}.csv")
    results_dict = load_search_dict(df_all, min_num_iteration)

    #fig, axes, metric_dict_res = plot_metric(
     #   df_all, plot_function_list_single, results_dict, nb_iterations=min_num_iteration,color_dict=color_dict
    #)
    #fig.tight_layout()
    #fig.savefig("figures/single.png")
    fig, df_plot = plot_simple_regret_stuff(df_all, results_dict, num_results_min, max_iteration, num_initialisation,color_dict=color_dict)
    fig.savefig(f"figures/single2_{run_name}.png")
    modify_figure__layout_simple(
        fig,
        legend_list=["BO-learned", "BO-Mord", "BO-Prop", "SUEA", "EA", "Rand"],
        x_limits=x_limits,
        y_limits=y_limits,
    )

    fig_org = copy.deepcopy(fig)
    legend_list = [
        "BO-learned",
        "BO-Mord",
        "BO-Prop",
        "SUEA",
        "EA",
        "Rand",
    ]
    plot_top_mols_acquired(results_dict, df_plot, fig_org, num_results_min, max_iteration, num_initialisation, top_mol_count,color_dict=color_dict)

    fig = modify_figure(fig_org, legend_list, x_limits, y_limits, tick_labels)
    fig.tight_layout()
    fig.savefig(f"figures/single3_{run_name}.png")
    with open(f"figures/fig_{run_name}.pkl", "wb") as f:
        pickle.dump(fig, f)


if __name__ == "__main__":
    main()
