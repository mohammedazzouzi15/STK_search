# # post analysis of the specific experiment
import pickle

import ipywidgets as widgets
import pandas as pd
from ipywidgets import (
    VBox,
    interactive,
)


# plot the results of the experiment
def plot_exp_batch_results(
    exp_name="Inputs/exp1_2023_09_05_14_47_02/",
    df_total: pd.DataFrame | None = None,
):
    df_list = []
    label_list = []

    for i in [3, 4, 5, 6]:
        search_space_loc = exp_name + f"search_space_{i}.pkl"
        SP = pickle.load(open(search_space_loc, "rb"))
        df_eval = SP.check_df_for_element_from_SP(df_to_check=df_total)
        df_list.append(df_eval)
        top5_percent_length = int(df_total.shape[0] * 0.05)
        min_target_5percent = (
            df_total["target"].nlargest(n=top5_percent_length).min()
        )
        ration_top_5 = str(
            int(
                df_list[-1][df_list[-1]["target"] > min_target_5percent].shape[
                    0
                ]
                / df_list[-1].shape[0]
                * 100
            )
        )
        label_list.append(
            f"space {i}" + f"({len(df_eval)})" + ration_top_5 + " %"
        )

    widge_plot = interactive(
        SP.plot_hist_compare,
        df_all=widgets.fixed(df_total),
        df_list=widgets.fixed(df_list),
        label_list=widgets.fixed(label_list),
    )

    columns_dropdown_2 = widgets.Dropdown(
        options=[
            x.replace("_0", "")
            for x in df_total.select_dtypes(include=["int", "float"]).columns
            if "_0" in x
        ],
        description="Value:",
    )
    hist_widget_plot = interactive(
        SP.plot_histogram_fragment,
        column_name=columns_dropdown_2,
        df_list=widgets.fixed(df_list),
        df_total=widgets.fixed(df_total),
        number_of_fragments=widgets.fixed(SP.number_of_fragments),
        label_list=widgets.fixed(label_list),
    )
    vbox_layout = widgets.Layout(
        display="flex",
        flex_flow="row",
        align_items="stretch",
        width="100%",
    )

    vb = VBox(
        [widge_plot, hist_widget_plot],
        layout=vbox_layout,
    )
    display(vb)
