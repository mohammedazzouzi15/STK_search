""" define a class to store the searched space
    this is a sub class of the search space class"""
import os
import numpy as np
import pandas as pd
from stk_search.Search_space import Search_Space
from ipywidgets import interact, widgets, Layout, VBox, interactive
from IPython.display import display
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt


class Searched_Space(Search_Space):
    def plot_hist_compare(self, df_all, df_list, label_list):
        fig, ax = plt.subplots(2, 2, figsize=(10, 6))

        def plot_hist(df, ax, color, label="all data"):
            df["target"] = (
                -np.abs(df["ionisation potential (eV)"].values - 5.5)
                - 1 * np.abs(df["fosc1"].values - 10)
                - 0.5 * np.abs(df["ES1"].values - 3)
            )
            df["ionisation potential (eV)"].hist(
                ax=ax[0, 0], bins=30, density=1, color=color, label=label
            )
            df["fosc1"].hist(ax=ax[0, 1], bins=30, density=1, color=color, label=label)
            df["ES1"].hist(ax=ax[1, 0], bins=30, density=1, color=color, label=label)
            df["target"].hist(ax=ax[1, 1], bins=30, density=1, color=color, label=label)
            # add vertical line showing target
            ax[0, 0].axvline(5.5, color="k", linestyle="--")
            ax[0, 1].axvline(10, color="k", linestyle="--")
            ax[1, 0].axvline(3, color="k", linestyle="--")
            ax[1, 1].axvline(0, color="k", linestyle="--")
            # add label to vertical line
            ax[0, 0].text(
                5.5,
                0.5,
                "target",
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                color="b",
            )
            ax[0, 1].text(
                10,
                0.2,
                "target",
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                color="b",
            )
            ax[1, 0].text(
                3,
                0.5,
                "target",
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                color="b",
            )
            ax[1, 1].text(
                0,
                0.2,
                "target",
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                color="b",
            )

        plot_hist(df=df_all, ax=ax, color="#21918c")
        # get a color list for the diffetent datasets
        color_list = ["#5ec962", "#f9c74f", "#f3722c", "#f94144", "#90be6d", "#577590"]
        for id, df in enumerate(df_list):
            plot_hist(df, ax, color=color_list[id], label=label_list[id])
        # add legend for the figure on top showing the different datasets
        ax[0, 0].set_xlabel("Ionisation Potential (eV)")
        ax[0, 1].set_xlabel("First Excited State Oscillator Strength")
        ax[1, 0].set_xlabel("Excited State Energy (eV)")
        ax[1, 0].set_xlim([1, 4])
        ax[0, 1].set_xlim([0, 10])
        ax[1, 1].set_xlabel("Combined Target Function")

        for ax in ax.flatten():
            ax.grid(False)
            ax.set_ylabel("Density")
            ax.set_yticks([])
            ax.legend()
        plt.tight_layout()

    def plot_histogram_fragment(
        self, column_name, df_list, df_total, number_of_fragments, label_list
    ):
        fig, axs = plt.subplots(3, 2, figsize=(6, 6), sharex="col", sharey="row")

        color_list = ["#5ec962", "#f9c74f", "#f3722c", "#f94144", "#90be6d", "#577590"]

        for i in range(number_of_fragments):
            df_total[f"{column_name}_{i}"].hist(
                ax=axs[i // 2, i % 2],
                bins=20,
                edgecolor="black",
                density=True,
                label="all",
                color="#21918c",
            )
            for id, df in enumerate(df_list):
                df[f"{column_name}_{i}"].hist(
                    ax=axs[i // 2, i % 2],
                    bins=20,
                    edgecolor="black",
                    density=True,
                    label=label_list[id],
                    color=color_list[id],
                )
            axs[i // 2, i % 2].set_xlabel(f"{column_name}_{i}")

        # set xlabel and y label for the last row
        for ax in axs.flatten():
            # ax.set_yscale('log')
            ax.grid(False)
            ax.set_ylabel("Density")
            ax.set_yticks([])
            ax.legend()
        plt.tight_layout()

    def get_all_possible_syntax(self):
        from itertools import combinations_with_replacement

        perm = product(
            list(range(self.number_of_fragments)), repeat=self.number_of_fragments
        )
        # Print the obtained permutations
        possible_syntax = []
        for i in list(perm):
            append = False

            # check that the first element is 0
            if i[0] == 0:
                # check that the element is not higher than its position
                for pos, id in enumerate(i):
                    if id > pos:
                        append = False
                        break
                    if id != pos and i[id] == id:
                        # print(id,i[id],pos,i[pos])
                        append = True
                    elif id == pos:
                        append = True
                    else:
                        append = False
                        break
            if append:
                possible_syntax.append(i)
        return possible_syntax

    def generate_interactive_condition_V2(self, df_total: pd.DataFrame):
        # function to generate an interactive prompt to select the condition
        # SP is the search space object
        # return the interactive widget
        def add_condition(columns: str, operation: str, value: str, fragment: int):
            # condition syntax should follow the following condition:
            # "'column'#operation#value" e.g. "'IP (eV)'#>=#6.5"
            condition = f"'{columns}'#{operation}#{value}"
            self.add_condition(condition, fragment)

        def remove_condition(columns: str, operation: str, value: str, fragment: int):
            # condition syntax should follow the following condition:
            # "'column'#operation#value" e.g. "'IP (eV)'#>=#6.5"
            condition = f"'{columns}'#{operation}#{value}"

            self.remove_condition(condition, fragment)

        def clear_condition(fragment: int):
            self.conditions_list[fragment] = []

        # Interactive widget for conditions selection
        columns_dropdown = widgets.Dropdown(
            options=self.df_precursors.select_dtypes(include=["int", "float"]).columns,
            description="Property of fragment:",
        )
        # Interactive widget for fragment selection
        fragment_dropdown = widgets.Dropdown(
            options=[(i) for i in range(self.number_of_fragments)],
            description="Fragment:",
        )
        # Interactive widget for operation selection
        operation_dropdown = widgets.Dropdown(
            options=[
                (">=", ">="),
                ("<=", "<="),
                ("==", "=="),
                ("!=", "!="),
                (">", ">"),
                ("<", "<"),
            ],
            description="Operation:",
        )
        # Interactive widget for value selection
        # add an input widget to enter the value
        value_dropdown = widgets.Text(description="Value:")
        # Interactive widget for add or remove condition
        add_remove_dropdown = widgets.Dropdown(
            options=[("Add", "add"), ("Remove", "remove")], description="Operation:"
        )
        # Interactive widger for adding the condition
        add_condition_button = widgets.Button(
            description="Add condition",
        )
        remove_condition_button = widgets.Button(
            description="remove condition",
        )
        clear_condition_button = widgets.Button(
            description="clear condition",
        )

        def on_click_add(b):
            add_condition(
                columns_dropdown.value,
                operation_dropdown.value,
                value_dropdown.value,
                fragment_dropdown.value,
            )
            self.get_space_size()
            number_of_elements_text.value = "{:.2e}".format(self.space_size)

            for i in range(self.number_of_fragments):
                display_conditions[i].options = self.conditions_list[i]

        def on_click_remove(b):
            remove_condition(
                columns_dropdown.value,
                operation_dropdown.value,
                value_dropdown.value,
                fragment_dropdown.value,
            )
            #self.redefine_search_space()
            self.get_space_size()
            number_of_elements_text.value = "{:.2e}".format(self.space_size)

            for i in range(self.number_of_fragments):
                display_conditions[i].options = self.conditions_list[i]

        def on_click_clear(b):
            clear_condition(
                fragment_dropdown.value,
            )

            self.get_space_size()
            number_of_elements_text.value = "{:.2e}".format(self.space_size)
            for i in range(self.number_of_fragments):
                display_conditions[i].options = self.conditions_list[i]

        clear_condition_button.on_click(on_click_clear)
        add_condition_button.on_click(on_click_add)
        remove_condition_button.on_click(on_click_remove)
        add_to_all_fragment_button = widgets.Button(
            description="Add to all fragments",
        )

        def on_click_add_to_all_fragment(b):
            for i in range(self.number_of_fragments):
                add_condition(
                    columns_dropdown.value,
                    operation_dropdown.value,
                    value_dropdown.value,
                    i,
                )
                self.get_space_size()
                number_of_elements_text.value = "{:.2e}".format(self.space_size)

                for i in range(self.number_of_fragments):
                    display_conditions[i].options = self.conditions_list[i]

        add_to_all_fragment_button.on_click(on_click_add_to_all_fragment)
        # Set up the layout of the widgets
        vbox_layout = Layout(display="flex", flex_flow="row", align_items="flex-start")
        Vb = VBox(
            [
                columns_dropdown,
                operation_dropdown,
                value_dropdown,
                fragment_dropdown,
                add_condition_button,
                remove_condition_button,
                clear_condition_button,
                add_to_all_fragment_button,
            ],
            layout=vbox_layout,
        )
        display(Vb)
        # put line for syntax chage and show some info about the searched space
        syntax_dropdown = widgets.Dropdown(
            options=self.get_all_possible_syntax(),
            description="Syntax:",
        )
        syntax_button = widgets.Button(
            description="Change syntax",
        )
        number_of_elements_text = widgets.Text(
            value="{:.2e}".format(self.space_size),
            description="Number of elements in search space:",
            disabled=True,
            style={"description_width": "initial"},
        )
        number_of_element_evaluated = widgets.Text(
            value=str(0),
            description="evaluated :",
            disabled=True,
            style={"description_width": "initial"},
        )
        df_total["normalised_target"] = (
            df_total["target"] - df_total["target"].min()
        ) / (df_total["target"].max() - df_total["target"].min())
        top5_percent_length = int(df_total.shape[0] * 0.05)
        min_target_5percent = df_total["target"].nlargest(n=top5_percent_length).min()
        top5_All_text = widgets.Text(
            value=str(min_target_5percent),
            description="min target to be anong the 5% highest:",
            disabled=True,
            style={"description_width": "initial"},
        )
        top5_current_text = widgets.Text(
            value=str(5),
            description="% among the top 5% in searched space :",
            disabled=True,
            style={"description_width": "initial"},
        )

        def on_click_syntax(b):
            self.syntax = syntax_dropdown.value
            self.get_space_size()
            number_of_elements_text.value = "{:.2e}".format(self.space_size)

            for i in range(self.number_of_fragments):
                display_conditions[i].options = self.conditions_list[i]
            #number_of_elements_text.value = "{:.2e}".format(self.space_size)

        syntax_button.on_click(on_click_syntax)
        Vb = VBox(
            [
                syntax_dropdown,
                syntax_button,
                number_of_elements_text,
                top5_All_text,
                top5_current_text,
                number_of_element_evaluated,
            ],
            layout=vbox_layout,
        )
        # Display the widget
        display(Vb)
        # add a button to add the condition
        display_conditions = []
        for i in range(self.number_of_fragments):
            display_conditions.append(widgets.SelectMultiple())
            display_conditions[i].options = self.conditions_list[i]
            display_conditions[i].description = f"Fragment {i}"
            display_conditions[i].layout.height = "100px"
            display_conditions[i].disabled = True
        # change display of list in to a table
        Vb = VBox(
            display_conditions,
            layout=vbox_layout,
        )
        display(Vb)
        # add a button to add the condition
        df_list = []
        label_list = []

        def add_to_hist_compare(b):
            self.redefine_search_space()

            number_of_elements_text.value = "{:.2e}".format(self.space_size)
            df_list.append(self.check_df_for_element_from_SP(df_to_check=df_total))
            label_list.append(label_text.value)
            # get the number of elements in the last df added to df_list that has target within 5% of the best target in df_total
            if df_list[-1].shape[0] > 0:
                top5_current_text.value = str(
                    df_list[-1][df_list[-1]["target"] > min_target_5percent].shape[0]
                    / df_list[-1].shape[0]
                    * 100
                )
            else:
                top5_current_text.value = 'no data'
            number_of_element_evaluated.value = "{:.2e}".format(df_list[-1].shape[0])
            print(len(df_list))
            widge_plot.kwargs["df_list"] = df_list
            widge_plot.kwargs["label_list"] = label_list
            widge_plot.update()
            hist_widget_plot.kwargs["df_list"] = df_list
            hist_widget_plot.kwargs["label_list"] = label_list

            hist_widget_plot.update()
            # save the search space properties in a table
            search_space_properties.append(
                {
                    "number of elements": self.space_size,
                    "syntax": self.syntax,
                    "conditions": self.conditions_list,
                    "Elements in top 5%": top5_current_text.value,
                    "number of elements evaluated": number_of_element_evaluated.value,
                }
            )

        def save_data(b):
            df_search_space_properties = pd.DataFrame.from_dict(search_space_properties)
            # add date and time inof into save dataframe
            date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            df_search_space_properties.to_pickle(
                f"Data/search_space_properties_{date_time}.pkl"
            )
            # save the figure

        add_to_hist_compare_button = widgets.Button(
            description="Add to histogram",
        )

        add_to_hist_compare_button.on_click(add_to_hist_compare)
        label_text = widgets.Text(
            value="",
            description="Label:",
            disabled=False,
        )
        save_data_button = widgets.Button(
            description="Save data",
        )
        save_data_button.on_click(save_data)

        vb = VBox(
            [label_text, add_to_hist_compare_button, save_data_button],
            layout=vbox_layout,
        )
        display(vb)

        widge_plot = interactive(
            self.plot_hist_compare,
            df_all=widgets.fixed(df_total),
            df_list=widgets.fixed(df_list),
            label_list=widgets.fixed(label_list),
        )
        # Interactive widget for column selection
        columns_dropdown_2 = widgets.Dropdown(
            options=[
                x.replace("_0", "")
                for x in df_total.select_dtypes(include=["int", "float"]).columns
                if "_0" in x
            ],
            description="Value:",
        )
        hist_widget_plot = interactive(
            self.plot_histogram_fragment,
            column_name=columns_dropdown_2,
            df_list=widgets.fixed(df_list),
            df_total=widgets.fixed(df_total),
            number_of_fragments=widgets.fixed(self.number_of_fragments),
            label_list=widgets.fixed(label_list),
        )
        vb = VBox(
            [widge_plot, hist_widget_plot],
            layout=vbox_layout,
        )
        display(vb)
        # save search space properties in table after each addistion to hist compare
        # save the search space properties in a table
        search_space_properties = [
            {
                "number of elements": self.space_size,
                "syntax": self.syntax,
                "conditions": self.conditions_list,
                "Elements in top 5%": 0,
                "number of elements evaluated": 0,

            },
        ]
