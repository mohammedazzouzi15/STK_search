import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

import os

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torch
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.interpolate import splrep, BSpline
from stk_search.utils import oligomer_scaffold_split
import pandas as pd
from sklearn.cluster import HDBSCAN
import glob
import pickle


plt.matplotlib.style.use(
    "https://gist.githubusercontent.com/JonnyCBB/c464d302fefce4722fe6cf5f461114ea/raw/64a78942d3f7b4b5054902f2cee84213eaff872f/matplotlibrc"
)
cool_colors = [
    "#00BEFF",
    "#D4CA3A",
    "#FF6DAE",
    "#67E1B5",
    "#EBACFA",
    "#9E9E9E",
    "#F1988E",
    "#5DB15A",
    "#E28544",
    "#52B8AA",
]
cool_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

plt.rcParams.update({"font.size": 14})

search_to_color = {
    "BO": cool_colors[0],
    "random": cool_colors[1],
    "evolutionary": cool_colors[2],
    "RF": cool_colors[3],
    "RF (div)": cool_colors[5],
}


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "same") / w


def plot_simple_regret(
    res,
    nb_iterations=100,
    axs=None,
    color=search_to_color["BO"],
    label="BO",
    target_name="target",
    df_total=[],
    nb_initialisation=0,
):
    """
    Plot the maximum value acquired up to this point

    Args:
        res (list): The results of the search
        nb_iterations (int): The number of iterations
        axs (matplotlib.pyplot): The axis to plot
        color (str): The color of the plot
    """
    target_value = np.max(df_total[target_name].values)
    nb_iterations_range = np.arange(nb_iterations) + 1
    y_max_mu = -10 * np.ones(nb_iterations)
    y_max_sig_bot = -10 * np.ones(nb_iterations)
    y_max_sig_top = -10 * np.ones(nb_iterations)
    nb_runs = len(res)
    for i in range(1, nb_iterations + 1):
        # max value acquired up to this point
        y_maxes = [
            np.max(
                res[r]["fitness_acquired"][
                    nb_initialisation : nb_initialisation + i
                ]
                - target_value
            )
            for r in range(nb_runs)
        ]
        y_maxes = np.array(y_maxes)
        assert np.size(y_maxes) == nb_runs
        y_max_mu[i - 1] = np.mean(y_maxes)
        y_max_sig_bot[i - 1] = np.std(y_maxes[y_maxes < y_max_mu[i - 1]])
        y_max_sig_top[i - 1] = np.std(y_maxes[y_maxes > y_max_mu[i - 1]])
    axs.plot(nb_iterations_range, y_max_mu, label=label, color=color)
    axs.fill_between(
        nb_iterations_range,
        y_max_mu - y_max_sig_bot,
        y_max_mu + y_max_sig_top,  #
        alpha=0.2,
        ec="None",
        color=color,
    )
    axs.set_xlabel("number of evaluated oligomers")
    axs.set_ylabel("simple regret ")
    axs.set_ylim([df_total[target_name].min(), df_total[target_name].max()])
    return max(y_max_mu)


def plot_inst_regret(
    res,
    nb_iterations=100,
    axs=None,
    color=search_to_color["BO"],
    label="BO",
    target_name="target",
    df_total=[],
    nb_initialisation=0,
):
    """
    Plot the maximum value acquired up to this point

    Args:
        res (list): The results of the search
        nb_iterations (int): The number of iterations
        axs (matplotlib.pyplot): The axis to plot
        color (str): The color of the plot
    """
    target_value = np.max(df_total[target_name].values)
    nb_iterations_range = np.arange(nb_iterations) + 1
    y_max_mu = -10 * np.ones(nb_iterations)
    y_max_sig_bot = -10 * np.ones(nb_iterations)
    y_max_sig_top = -10 * np.ones(nb_iterations)
    nb_runs = len(res)
    for i in range(1, nb_iterations + 1):
        # max value acquired up to this point
        y_maxes = [
            res[r]["fitness_acquired"][i - 1] - target_value
            for r in range(nb_runs) if len(res[r]["fitness_acquired"]) > i - 1
        ]
        y_maxes = np.array(y_maxes)
        #assert np.size(y_maxes) == nb_runs
        y_max_mu[i - 1] = np.mean(y_maxes)
        y_max_sig_bot[i - 1] = np.std(y_maxes)
        y_max_sig_top[i - 1] = np.std(y_maxes)
    axs.plot(nb_iterations_range, y_max_mu, label=label, color=color)
    axs.fill_between(
        nb_iterations_range,
        y_max_mu - y_max_sig_bot,
        y_max_mu + y_max_sig_top,  #
        alpha=0.2,
        ec="None",
        color=color,
    )
    axs.set_xlabel("number of evaluated oligomers")
    axs.set_ylabel("instatnious regret ")
    axs.set_ylim([df_total[target_name].min(), df_total[target_name].max()])
    return max(y_max_mu)


def plot_cumulative_regret(
    res,
    nb_iterations=100,
    axs=None,
    color=search_to_color["BO"],
    label="BO",
    target_name="target",
    df_total=[],
    nb_initialisation=0,
):
    """
    Plot the maximum value acquired up to this point

    Args:
        res (list): The results of the search
        nb_iterations (int): The number of iterations
        axs (matplotlib.pyplot): The axis to plot
        color (str): The color of the plot
    """
    target_value = np.max(df_total[target_name].values)
    nb_iterations_range = np.arange(nb_iterations) + 1
    y_max_mu = -10 * np.ones(nb_iterations)
    y_max_sig_bot = -10 * np.ones(nb_iterations)
    y_max_sig_top = -10 * np.ones(nb_iterations)
    nb_runs = len(res)
    for i in range(1, nb_iterations + 1):
        # max value acquired up to this point
        y_maxes = [
            np.sum(
                res[r]["fitness_acquired"][
                    nb_initialisation : nb_initialisation + i
                ]
                - target_value
            )
            for r in range(nb_runs)
        ]

        y_maxes = -np.array(y_maxes)
        assert np.size(y_maxes) == nb_runs
        y_max_mu[i - 1] = np.mean(y_maxes)
        y_max_sig_bot[i - 1] = np.std(y_maxes[y_maxes < y_max_mu[i - 1]])
        y_max_sig_top[i - 1] = np.std(y_maxes[y_maxes > y_max_mu[i - 1]])
    axs.plot(nb_iterations_range, y_max_mu, label=label, color=color)
    axs.fill_between(
        nb_iterations_range,
        y_max_mu - y_max_sig_bot,
        y_max_mu + y_max_sig_top,  #
        alpha=0.2,
        ec="None",
        color=color,
    )
    axs.set_xlabel("number of evaluated oligomers")
    axs.set_ylabel("cumulative regret ")
    return max(y_max_mu)


def plot_y_max(
    res,
    nb_iterations=100,
    axs=None,
    color=search_to_color["BO"],
    label="BO",
    target_name="target",
    df_total=[],
    nb_initialisation=0,
):
    """
    Plot the maximum value acquired up to this point

    Args:
        res (list): The results of the search
        nb_iterations (int): The number of iterations
        axs (matplotlib.pyplot): The axis to plot
        color (str): The color of the plot
    """
    nb_iterations_range = np.arange(nb_iterations) + 1
    y_max_mu = -10 * np.ones(nb_iterations)
    y_max_sig_bot = -10 * np.ones(nb_iterations)
    y_max_sig_top = -10 * np.ones(nb_iterations)
    nb_runs = len(res)
    for i in range(1, nb_iterations + 1):
        # max value acquired up to this point

        y_maxes = np.array(
            [
                np.max(
                    res[r]["fitness_acquired"][
                        nb_initialisation : nb_initialisation + i
                    ]
                )
                for r in range(nb_runs)
            ]
        )  # among runs
        assert np.size(y_maxes) == nb_runs
        y_max_mu[i - 1] = np.mean(y_maxes)
        y_max_sig_bot[i - 1] = np.std(y_maxes[y_maxes < y_max_mu[i - 1]])
        y_max_sig_top[i - 1] = np.std(y_maxes[y_maxes > y_max_mu[i - 1]])
    axs.plot(nb_iterations_range, y_max_mu, label=label, color=color)
    axs.fill_between(
        nb_iterations_range,
        y_max_mu - y_max_sig_bot,
        y_max_mu + y_max_sig_top,  #
        alpha=0.2,
        ec="None",
        color=color,
    )
    axs.set_xlabel("number of evaluated oligomers")
    axs.set_ylabel("maximum fitness  \n acquired up to iteration")
    axs.set_ylim([df_total[target_name].min(), df_total[target_name].max()])

    axs.axhline(
        y=np.max(df_total[target_name].values),
        color="k",
        linestyle="--",
        zorder=0,
    )
    return max(y_max_mu)


def plot_all_y_max(
    res,
    nb_iterations=100,
    axs=None,
    color=search_to_color["BO"],
    label="BO",
    target_name="target",
    df_total=[],
    nb_initialisation=0,
):
    nb_iterations_range = np.arange(nb_iterations) + 1
    y_maxes = np.ones((len(res), nb_iterations))
    y_max_mu = -10 * np.ones(nb_iterations)
    y_max_sig_bot = -10 * np.ones(nb_iterations)
    y_max_sig_top = -10 * np.ones(nb_iterations)
    y_max = -10 * np.ones(nb_iterations)
    y_min = -10 * np.ones(nb_iterations)
    nb_runs = len(res)
    for i in range(1, nb_iterations + 1):
        # max value acquired up to this point
        y_maxes[:, i - 1] = np.array(
            [
                np.max(
                    res[r]["fitness_acquired"][
                        nb_initialisation : nb_initialisation + i
                    ]
                )
                for r in range(nb_runs)
            ]
        )  # among runs
        y_max_mu[i - 1] = np.mean(y_maxes[:, i - 1])
        y_max[i - 1] = np.max(y_maxes[:, i - 1])
        y_min[i - 1] = np.min(y_maxes[:, i - 1])
        y_max_sig_bot[i - 1] = np.std(y_maxes[:, i - 1])
        y_max_sig_top[i - 1] = np.std(y_maxes[:, i - 1])
    for r in range(len(res)):
        axs.plot(nb_iterations_range, y_maxes[r, :], color=color, alpha=0.3)
    axs.plot(nb_iterations_range, y_min, color=color, linestyle="--")
    axs.plot(nb_iterations_range, y_max, color=color, linestyle="--")
    axs.plot(nb_iterations_range, y_max_mu, label=label, color=color)

    axs.fill_between(
        nb_iterations_range,
        y_max_mu - y_max_sig_bot,
        y_max_mu + y_max_sig_top,  #
        alpha=0.2,
        ec="None",
        color=color,
    )
    axs.set_xlabel("number of evaluated oligomers")
    axs.set_ylabel("maximum fitness  \n acquired up to iteration")
    axs.set_ylim([df_total[target_name].min(), df_total[target_name].max()])

    axs.axhline(
        y=np.max(df_total[target_name].values),
        color="k",
        linestyle="--",
        zorder=0,
    )
    return max(y_max_mu)


def plot_all_y_max_diff(
    res,
    nb_iterations=100,
    axs=None,
    color=search_to_color["BO"],
    label="BO",
    nb_initialisation=0,
    target_name="target",
    df_total=[],
):
    nb_iterations_range = np.arange(nb_iterations) + 1
    y_maxes = np.ones((len(res), nb_iterations))
    y_max_mu = -10 * np.ones(nb_iterations)
    y_max_sig_bot = -10 * np.ones(nb_iterations)
    y_max_sig_top = -10 * np.ones(nb_iterations)
    y_max = -10 * np.ones(nb_iterations)
    y_min = -10 * np.ones(nb_iterations)
    nb_runs = len(res)
    for i in range(1, nb_iterations + 1):
        # max value acquired up to this point
        y_maxes[:, i - 1] = (
            np.array(
                [
                    np.max(
                        res[r]["fitness_acquired"][0 : nb_initialisation + i]
                    )
                    for r in range(nb_runs)
                ]
            )
            - y_maxes[:, nb_initialisation]
        )  # among runs
        y_max_mu[i - 1] = np.mean(y_maxes[:, i - 1])
        y_max[i - 1] = np.max(y_maxes[:, i - 1])
        y_min[i - 1] = np.min(y_maxes[:, i - 1])
        y_max_sig_bot[i - 1] = np.std(y_maxes[:, i - 1])
        y_max_sig_top[i - 1] = np.std(y_maxes[:, i - 1])
    for r in range(len(res)):
        axs.plot(nb_iterations_range, y_maxes[r, :], color=color, alpha=0.3)
    axs.plot(nb_iterations_range, y_min, color=color, linestyle="--")
    axs.plot(nb_iterations_range, y_max, color=color, linestyle="--")
    axs.plot(nb_iterations_range, y_max_mu, label=label, color=color)

    axs.fill_between(
        nb_iterations_range,
        y_max_mu - y_max_sig_bot ,
        y_max_mu + y_max_sig_top ,  #
        alpha=0.2,
        ec="None",
        color=color,
    )
    axs.set_xlabel("number of evaluated oligomers")
    axs.set_ylabel("maximum fitness  \n acquired up to iteration")
    return max(y_max_mu)


def plot_y_mean(
    res,
    nb_iterations=100,
    axs=None,
    color=search_to_color["BO"],
    label="BO",
    target_name="target",
    df_total=[],
    nb_initialisation=0,
):

    nb_iterations_range = np.arange(nb_iterations) + 1
    y_mean_mu_BO = -10 * np.ones(nb_iterations)
    nb_runs = len(res)
    for i in range(1, nb_iterations + 1):
        # max value acquired up to this point
        y_maxes = np.array(
            [
                res[r]["fitness_acquired"][nb_initialisation + i - 1]
                for r in range(nb_runs)
                if len(res[r]["fitness_acquired"]) > nb_initialisation + i - 1
            ]
        )  # among runs
        if len(y_maxes) == 0:
            break
        y_mean_mu_BO[i - 1] = np.mean(y_maxes)
    y_mean_mov_av = moving_average(np.array(y_mean_mu_BO), 5)

    axs.plot(nb_iterations_range, y_mean_mov_av, label=label, color=color)

    axs.set_xlabel("# evaluated oligomers")
    axs.set_ylabel("mean fitness acquired at iteration")
    axs.set_ylim([df_total[target_name].min(), df_total[target_name].max()])
    axs.axhline(
        y=np.max(df_total[target_name].values),
        color="k",
        linestyle="--",
        zorder=0,
    )
    return y_mean_mu_BO


def plot_number_of_molecule_discovered(
    res,
    nb_iterations=100,
    topKmol=1000,
    axs=None,
    color=search_to_color["BO"],
    label="BO",
    df_total=[],
    nb_initialisation=0,
    target_name="target",
):
    min_target = -np.sort(-df_total[target_name].values)[topKmol]
    nb_iterations_range = np.arange(nb_iterations) + 1
    y_elm = -10 * np.ones(nb_iterations)
    y_elm_sig_bot = -10 * np.ones(nb_iterations)
    y_elm_sig_top = -10 * np.ones(nb_iterations)
    nb_runs = len(res)
    for i in range(1, nb_iterations + 1):
        # max value acquired up to this point

        y_maxes = np.array(
            [
                np.array(
                    res[r]["fitness_acquired"][
                        nb_initialisation : nb_initialisation + i
                    ]
                )
                > min_target
                for r in range(nb_runs)
                if len(res[r]["fitness_acquired"]) > nb_initialisation + i - 1
            ]
        ).sum(
            axis=1
        )  # /topKmol*100 # among runs
        y_elm[i - 1] = np.mean(y_maxes)
        y_elm_sig_bot[i - 1] = np.std(y_maxes[y_maxes < y_elm[i - 1]])
        y_elm_sig_top[i - 1] = np.std(y_maxes[y_maxes > y_elm[i - 1]])
    axs.plot(nb_iterations_range, y_elm, label=label, color=color)
    axs.fill_between(
        nb_iterations_range,
        y_elm - y_elm_sig_bot,
        y_elm + y_elm_sig_top,  #
        alpha=0.2,
        ec="None",
        color=color,
    )
    axs.set_xlabel("# evaluated oligomers")
    axs.set_ylabel(
        f"Top {topKmol/df_total.shape[0]*100:1.2f}%  of oligomers\n ({topKmol} molecules) "
    )
    return max(y_elm)


def plot_number_of_molecule_discovered_sum(
    res,
    nb_iterations=100,
    topKmol=1000,
    axs=None,
    color=search_to_color["BO"],
    label="BO",
    df_total=[],
    nb_initialisation=0,
    target_name="target",
    number_of_results=25,
    min_target=0,
):  
    if topKmol is not None:
        min_target = -np.sort(-df_total[target_name].values)[topKmol]
    df_results = pd.concat(
        list(
            generate_datafame_from_search_results(
                res[:number_of_results],
                max_iteration=nb_iterations,
                num_initialisation=nb_initialisation,
            )
        )
    )

    df_results["InChIKey"] = df_results["InchiKey_acquired"]
    df_results.drop_duplicates(subset=["InChIKey"], inplace=True)
    df_max = df_total[df_total[target_name] > min_target].copy()

    df_max_found = df_max.merge(df_results, on="InChIKey", how="inner")
    df_results = df_total.merge(df_results, on="InChIKey", how="inner")
    total_found = np.zeros(nb_iterations)
    max_mol_found = np.zeros(nb_iterations)
    for num_iter in range(nb_iterations):
        total_found[num_iter] = df_results[
            df_results["ids_acquired"] < num_iter
        ].shape[0]
        max_mol_found[num_iter] = df_max_found[
            df_max_found["ids_acquired"] < num_iter
        ].shape[0]
    axs.plot(np.arange(nb_iterations), max_mol_found, label=label, color=color)
    axs.set_xlabel("# evaluated oligomers")
    axs.set_ylabel(f"Number of unique top \n" + f" {topKmol} molecules found")
    return max(max_mol_found)

def plot_simple_regret_batch(
    res,
    nb_iterations=100,
    topKmol=1000,
    axs=None,
    color=search_to_color["BO"],
    label="BO",
    df_total=[],
    nb_initialisation=0,
    target_name="target",
    number_of_results=25,
    min_target=0,
):  
    if topKmol is not None:
        min_target = -np.sort(-df_total[target_name].values)[topKmol]
    df_results = pd.concat(
        list(
            generate_datafame_from_search_results(
                res[:number_of_results],
                max_iteration=nb_iterations,
                num_initialisation=nb_initialisation,
            )
        )
    )

    df_results["InChIKey"] = df_results["InchiKey_acquired"]
    df_results.drop_duplicates(subset=["InChIKey"], inplace=True)
    df_max = df_total[df_total[target_name] > min_target].copy()

    df_max_found = df_max.merge(df_results, on="InChIKey", how="inner")
    df_results = df_total.merge(df_results, on="InChIKey", how="inner")
    max_mol_found = np.zeros(nb_iterations)
    for num_iter in range(nb_iterations):
        max_mol_found[num_iter] = df_max_found[
            df_max_found["ids_acquired"] < num_iter
        ][target_name].max() - df_max_found[target_name].max()
    axs.plot(np.arange(nb_iterations), max_mol_found, label=label, color=color)
    axs.set_xlabel("# evaluated oligomers")
    axs.set_ylabel(f"Simple regret")
    return max(max_mol_found)


def plot_rate_of_discovery_old(
    res,
    nb_iterations=100,
    topKmol=1000,
    axs=None,
    color=search_to_color["BO"],
    label="BO",
    df_total=[],
    nb_initialisation=0,
    target_name="target",
    number_of_results=25,
    min_target=0,
):
    if topKmol is not None:
        min_target = -np.sort(-df_total[target_name].values)[topKmol]
    print( 'min_target is ', min_target)
    df_results = pd.concat(
        list(
            generate_datafame_from_search_results(
                res[:number_of_results],
                max_iteration=nb_iterations,
                num_initialisation=nb_initialisation,
            )
        )
    )
    df_results["InChIKey"] = df_results["InchiKey_acquired"]
    df_results.drop_duplicates(subset=["InChIKey"], inplace=True)
    df_max = df_total[df_total[target_name] > min_target].copy()
    df_max_found = df_max.merge(df_results, on="InChIKey", how="inner")
    df_results = df_total.merge(df_results, on="InChIKey", how="inner")
    total_found = np.ones(nb_iterations)
    max_mol_found = np.zeros(nb_iterations)
    for num_iter in range(1, nb_iterations):
        total_found[num_iter] = df_results[
            df_results["ids_acquired"] < num_iter
        ].shape[0]
        max_mol_found[num_iter] = df_max_found[
            df_max_found["ids_acquired"] < num_iter
        ].shape[0]
    axs.plot(
        np.arange(nb_iterations),
        max_mol_found / (total_found+1) * 100,
        label=label,
        color=color,
    )
    axs.set_xlabel("# evaluated oligomers")
    axs.set_ylabel(f"Rate of discovery of \n top molecules for batch (%)")
    return max(max_mol_found / (total_found+1) * 100)


def plot_rate_of_discovery(
    res,
    nb_iterations=350,
    topKmol=1000,
    axs=None,
    color="C0",
    label="BO",
    df_total=[],
    target_name="target",
    min_target=0,
    nb_initialisation=50,
    prop=1,
):

    if topKmol is not None:
        min_target = -np.sort(-df_total[target_name].values)[topKmol]
    y_results = np.zeros((nb_iterations + nb_initialisation, len(res)))
    top_mol_inchikey_list = df_total[df_total[target_name] > min_target][
        "InChIKey"
    ].values
    nb_iterations_range = np.arange(0, nb_iterations)
    for ii, results in enumerate(res):
        y_results[:, ii] = (
            np.array(
                [
                    x in top_mol_inchikey_list
                    for x in results["InchiKey_acquired"][:nb_iterations]
                ]
            ).cumsum()
            / (1 + np.arange(0, nb_iterations))
            * 100
            / prop
        )
    y_elm = np.mean(y_results, axis=1)
    # get bottom  variance of data
    #y_elm_sig_bot = np.std(y_results[y_results < y_elm], axis=1)
    #y_elm_sig_top = np.std(y_results[y_results > y_elm], axis=1)
    y_elm_sig_bot = np.array([np.std(row[row < y_elm[i]]) for i, row in enumerate(y_results)])
    y_elm_sig_top = np.array([np.std(row[row > y_elm[i]]) for i, row in enumerate(y_results)])  

    nb_iterations_range = np.arange(0, nb_iterations)
    axs.plot(nb_iterations_range, y_elm, label=label, color=color)
    axs.fill_between(
        nb_iterations_range,
        y_elm - y_elm_sig_bot,
        y_elm + y_elm_sig_top,  #
        alpha=0.2,
        ec="None",
        color=color,
    )
    axs.set_xlabel("# evaluated oligomers")
    axs.set_ylabel(f"Rate of discovery \n of top molecules (%)")
    return max(y_elm)


def plot_hist_mol_found(
    search_results,
    target_name,
    df_total,
    num_elem_initialisation=100,
    axs=None,
    color=search_to_color["BO"],
):
    INchikeys_found = []
    for search_result in search_results:
        INchikeys_found.append(
            search_result["InchiKey_acquired"][num_elem_initialisation:]
        )
    INchikeys_found = np.concatenate(INchikeys_found)
    df_total_found = df_total[df_total["InChIKey"].isin(INchikeys_found)]
    print("mol_found", df_total_found.shape[0])
    df_total_found[target_name].hist(
        ax=axs,
        bins=20,
        orientation="horizontal",
        color=color,
        alpha=0.5,
        density=True,
    )
    axs.set_ylim([df_total[target_name].min(), df_total[target_name].max()])
    # axs.set_xscale('log')
    # axs.set_xlim([0.9,1e4])


def plot_exploration_evolution(
    BOresults,
    df_total_org,
    nb_initialisation,
    nb_iteration=100,
    axs=None,
    color=search_to_color["BO"],
    label="BO",
    operation=np.max,
    target_name="target",
    aim=5.5,
    topKmol=1000,
):

    df_total = df_total_org.copy()
    df_total[target_name] = df_total[target_name].apply(
        lambda x: -np.sqrt((x - aim) ** 2)
    )
    min_target_out_of_database = -np.sort(-df_total[target_name].values)[
        topKmol
    ]
    y_elm, y_elm_sig_bot, y_elm_sig_top = plot_element_above_min(
        BOresults,
        min_target_out_of_database,
        nb_iterations=nb_iteration,
        topKmol=topKmol,
        axs=axs[2],
        color=color,
        label=label,
        df_total=df_total,
        nb_initialisation=0,
    )
    y_mean_mu_BO = plot_y_mean(
        BOresults,
        nb_iterations=nb_iteration,
        axs=axs[1],
        color=color,
        label=label,
        target_name=target_name,
        df_total=df_total,
        nb_initialisation=0,
    )
    y_max_mu_BO, y_max_sig_bot_BO, y_max_sig_top_BO = plot_y_max(
        BOresults,
        nb_iterations=nb_iteration,
        axs=axs[0],
        color=color,
        label=label,
        df_total=df_total,
        operation=operation,
        target_name=target_name,
        nb_initialisation=0,
    )
    # df_total[target_name].hist(ax=axs[3], bins=20, orientation="horizontal", color=search_to_color['BO'])
    # axs[3].set_ylim([df_total[target_name].min(),df_total[target_name].max()])
    min_target_out_of_database = -np.sort(-df_total[target_name].values)[100]
    y_elm, y_elm_sig_bot, y_elm_sig_top = plot_element_above_min(
        BOresults,
        min_target_out_of_database,
        nb_iterations=nb_iteration,
        topKmol=100,
        axs=axs[3],
        color=color,
        label=label,
        df_total=df_total,
        nb_initialisation=0,
    )


import glob
import pickle


def load_search_data(search_type, date, test_name, min_eval=100):
    files = glob.glob(
        f"data/output/search_experiment/{test_name}/"
        + search_type
        + "/"
        + date
        + "/*.pkl"
    )
    BOresults = []
    max_num_eval = 0
    for file in files:

        with open(file, "rb") as f:
            results = pickle.load(f)
            if len(results["fitness_acquired"]) > min_eval:
                BOresults.append(results)
                max_num_eval = max(
                    max_num_eval, len(results["fitness_acquired"])
                )
    print(len(BOresults), max_num_eval)
    return BOresults


def generate_datafame_from_search_results(
    search_results, max_iteration=200, num_initialisation=0
):
    """Generate a dataframe from the search results

    Args:
        search_results (list): A list of search results
        max_iteration (int): The maximum number of iterations
        num_initialisation (int): The number of initialisation
    """
    for dict_org in search_results:
        dict = dict_org.copy()

        dict.pop("searched_space_df")
        df = pd.DataFrame.from_records(dict)
        df = df[df["ids_acquired"] < max_iteration]
        df = df[df["ids_acquired"] > num_initialisation]
        yield df


def get_df_max_found(
    search_results,
    df_total,
    max_iteration=200,
    topKmol=1000,
    num_initialisation=0,
    target_name="target",
):
    """Get the dataframe of the maximum found molecules

    Args:
        search_results (list): The search results
        df_total (pd.DataFrame): The dataframe of the total molecules
        max_iteration (int): The maximum number of iterations
        topKmol (int): The number of top molecules
        num_initialisation (int): The number of initialisation
    """

    df_results = pd.concat(
        list(
            generate_datafame_from_search_results(
                search_results,
                max_iteration=max_iteration,
                num_initialisation=num_initialisation,
            )
        )
    )
    df_results["InChIKey"] = df_results["InchiKey_acquired"]
    df_results.drop_duplicates(subset=["InChIKey"], inplace=True)
    df_total.sort_values(target_name, inplace=True, ascending=False)
    df_max = df_total.iloc[0:topKmol]
    df_max_found = df_max.merge(df_results, on="InChIKey", how="inner")
    df_results = df_total.merge(df_results, on="InChIKey", how="inner")
    return df_results, df_max_found


def get_df_max_target_found(
    search_results,
    df_total,
    max_iteration=200,
    min_target=0,
    num_initialisation=0,
    target_name="target",
):
    """Get the dataframe of the maximum found molecules

    Args:
        search_results (list): The search results
        df_total (pd.DataFrame): The dataframe of the total molecules
        max_iteration (int): The maximum number of iterations
        topKmol (int): The number of top molecules
        num_initialisation (int): The number of initialisation
    """

    df_results = pd.concat(
        list(
            generate_datafame_from_search_results(
                search_results,
                max_iteration=max_iteration,
                num_initialisation=num_initialisation,
            )
        )
    )
    df_results["InChIKey"] = df_results["InchiKey_acquired"]
    df_results.drop_duplicates(subset=["InChIKey"], inplace=True)
    df_total.sort_values(target_name, inplace=True, ascending=False)
    df_max = df_total[df_total[target_name] > min_target].copy()
    df_max_found = df_max.merge(df_results, on="InChIKey", how="inner")
    df_results = df_total.merge(df_results, on="InChIKey", how="inner")
    return df_results, df_max_found


def get_clusters(df):
    hdb_model = HDBSCAN(min_cluster_size=1000, min_samples=100)
    # Fit the model to the average PCA scores
    cluster_labels = hdb_model.fit_predict(df[["PCA1", "PCA2"]])
    df["Cluster"] = cluster_labels
    return df


def plot_space_with_target(df, target_name="target", ax=None, size_of_bin=1):

    list_target_splits = [
        df[df[target_name] > i][target_name].values
        for i in range(
            int(df[target_name].min()), int(df[target_name].max()), size_of_bin
        )
    ]

    color_list = sns.color_palette("viridis", len(list_target_splits))
    for i, target_split in enumerate(list_target_splits):
        df_plot = df[df[target_name].isin(target_split)]
        ax.scatter(
            df_plot["PCA1"],
            df_plot["PCA2"],
            color=color_list[i],
            s=3,
        )
    return df


def oligomer_cluster_plot(
    df,
    target_name="target",
    size_of_bin=10,
):
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    if "Cluster" not in df.columns:
        df = get_clusters(df)
    for cluster in df["Cluster"].unique():
        df_total_cluster = df[df["Cluster"] == cluster]
        print(f"Cluster {cluster} has {df_total_cluster.shape[0]} molecules")
        df_total_cluster.hist(
            column=target_name,
            bins=100,
            ax=ax[0],
            alpha=0.5,
            label=f"Cluster {cluster}",
        )  # ,color='C'+str(cluster))
        ax[1].scatter(
            df_total_cluster["PCA1"],
            df_total_cluster["PCA2"],
            alpha=0.5,
        )  # ,color='C'+str(cluster))
    ax[0].set_xlabel("target")
    ax[0].set_ylabel("count")
    ax[0].legend()
    plot_space_with_target(df, target_name, ax[2], size_of_bin=size_of_bin)
    # add colorbar
    mappable = ax[2].collections[0]
    # set colorbar scale
    mappable.set_clim(df[target_name].min(), df[target_name].max())
    # set colorbar ticks
    ax[2].figure.colorbar(mappable, ax=ax[2]).set_ticks(
        np.arange(df[target_name].min(), df[target_name].max(), 2)
    )
    ax[2].set_title("Chemical space by target")
    ax[1].set_title("Chemical space by clusters")

    return df, fig, ax


# helper functions to generate the pca of the search space


def load_search_data(search_type, date, test_name, min_eval=100):
    files = glob.glob(
        f"data/output/search_experiment/{test_name}/"
        + search_type
        + "/"
        + date
        + "/*.pkl"
    )
    BOresults = []
    max_num_eval = 0
    for file in files:

        with open(file, "rb") as f:
            results = pickle.load(f)
            if len(results["fitness_acquired"]) > min_eval:
                BOresults.append(results)
                max_num_eval = max(
                    max_num_eval, len(results["fitness_acquired"])
                )
    print(len(BOresults), max_num_eval)
    return BOresults


def plot_BO_results_in_space(BOresults, ax, title_label, df_rep):
    bo = BOresults.copy()
    bo.pop("searched_space_df")
    pd_results = pd.DataFrame(bo)
    pd_results["InChIKey"] = pd_results["InchiKey_acquired"]
    df_plot_results = pd_results.merge(df_rep, on="InChIKey", how="left")

    def plot_pca_space_results(
        df_tot_plot, df_plot_results, property_name, added_text=""
    ):

        cax = ax.scatter(
            df_tot_plot["2D PCA 1"],
            df_tot_plot["2D PCA 2"],
            c=-df_tot_plot["target"],
            cmap="viridis",
            s=20,
            alpha=0.3,
        )

        ax.scatter(
            df_plot_results["2D PCA 1"],
            df_plot_results["2D PCA 2"],
            c=df_plot_results["ids_acquired"],
            cmap="Oranges",
            s=20,
            alpha=0.9,
            label="EA results",
        )
        ax.set_title(f"Chemical space , {added_text}")

        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        return cax

    cax = plot_pca_space_results(
        df_rep, df_plot_results, "target", "Morgan fingerprints"
    )
    ax.set_title(title_label)
    return df_plot_results


def plot_df_results_in_space(pd_results, ax, title_label, df_rep):

    df_plot_results = pd_results.merge(
        df_rep, on="InChIKey", how="left", suffixes=("_acquired", "")
    )

    def plot_pca_space_results(
        df_tot_plot, df_plot_results, property_name, added_text=""
    ):

        cax = ax.scatter(
            df_tot_plot["PCA1"],
            df_tot_plot["PCA2"],
            c=df_tot_plot["target"],
            cmap="viridis",
            s=2,
            alpha=0.3,
        )

        ax.scatter(
            df_plot_results["PCA1"],
            df_plot_results["PCA2"],
            c=df_plot_results["ids_acquired"],
            cmap="Oranges",
            s=5,
            alpha=0.9,
            label="EA results",
        )
        ax.set_title(f"Chemical space , {added_text}")

        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        return cax

    cax = plot_pca_space_results(
        df_rep, df_plot_results, "target", "Morgan fingerprints"
    )
    ax.set_title(title_label)
    return df_plot_results


def plot_df_max_in_space(pd_results, ax, title_label, df_rep):

    df_plot_results = pd_results.merge(
        df_rep, on="InChIKey", how="left", suffixes=("_acquired", "")
    )

    def plot_pca_space_results(
        df_tot_plot, df_plot_results, property_name, added_text=""
    ):

        # ax.scatter(df_all_plot_morgan['PCA1'], df_all_plot_morgan['PCA2'], c='black', s=20, alpha=0.9, label='random generation')

        ax.scatter(
            df_tot_plot["PCA1"], df_tot_plot["PCA2"], c="black", s=2, alpha=0.3
        )

        ax.scatter(
            df_plot_results["PCA1"],
            df_plot_results["PCA2"],
            c="orange",
            s=5,
            alpha=0.9,
            label="EA results",
        )
        ax.set_title(f"Chemical space , {added_text}")

        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    plot_pca_space_results(
        df_rep, df_plot_results, "target", "Morgan fingerprints"
    )
    ax.set_title(title_label)
    return df_plot_results


def plot_pca_space(df_tot_plot, property_name, added_text="", ax=None):

    # ax.scatter(df_all_plot_morgan['PCA1'], df_all_plot_morgan['PCA2'], c='black', s=20, alpha=0.9, label='random generation')

    cax = ax.scatter(
        df_tot_plot["PCA1"],
        df_tot_plot["PCA2"],
        c=df_tot_plot[property_name],
        cmap="viridis",
        s=1,
        alpha=0.8,
        
    )
    ax.set_title(f"Chemical space , {added_text}")

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return cax


def get_rep_pca(InChIKey, dataset_dict, pca_kernel):
    learned_rpr = (
        dataset_dict[InChIKey].learned_rpr.detach().numpy().reshape(1, -1)
    )
    return pca_kernel.transform(learned_rpr)[0]
