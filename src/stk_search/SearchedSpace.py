"""Define a class to represent the searched space.

Here i mainly introduce a function to plot the histogram of the searched space
and compare it with the histogram of the whole dataset
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import Layout, VBox, interactive, widgets

from stk_search.SearchSpace import SearchSpace


class SearchedSpace(SearchSpace):
    def plot_hist_compare(self, df_all, df_list, label_list,properties_to_plot=None):
        if properties_to_plot is None:
            properties_to_plot = []
        fig, ax = plt.subplots(1, len(properties_to_plot), figsize=(15, 10))
        ax = ax.flatten()
        def plot_hist(df, ax, color, label="all data") -> None:
            for axis_num,property in enumerate(properties_to_plot):
                df[property].hist(
                    ax=ax[axis_num],
                    bins=30,
                    density=1,
                    color=color,
                    label=label,
                    alpha=0.5,
                )
                ax[axis_num].set_xlabel(f"{property}")

        plot_hist(df=df_all, ax=ax, color="#21918c")
        # get a color list for the diffetent datasets
        color_list = [
            "#5ec962",
            "#f9c74f",
            "#f3722c",
            "#f94144",
            "#90be6d",
            "#577590",
            "#f8961e",
            "#e63946",
            "#a8dadc",
            "#457b9d",
        ]
        for id, df in enumerate(df_list):
            plot_hist(df, ax, color=color_list[id], label=label_list[id])
        for _ax in ax.flatten():
            # ax.set_yscale('log')
            _ax.grid(False)
            _ax.set_ylabel("Density")
            _ax.set_yticks([])
        # add legend for the figure on top showing the different datasets
        plt.tight_layout()
        return fig, ax

    def plot_histogram_fragment(
        self, column_name, df_list, df_total, number_of_fragments, label_list
    ):
        fig, axs = plt.subplots(
            3, 2, figsize=(12, 6), sharex="col", sharey="row"
        )

        color_list = [
            "#5ec962",
            "#f9c74f",
            "#f3722c",
            "#f94144",
            "#90be6d",
            "#577590",
            "#f8961e",
            "#e63946",
            "#a8dadc",
            "#457b9d",
        ]

        for i in range(number_of_fragments):
            range_min = df_total[f"{column_name}_{i}"].min()
            range_max = df_total[f"{column_name}_{i}"].max()

            df_total[f"{column_name}_{i}"].hist(
                ax=axs[i // 2, i % 2],
                bins=20,
                edgecolor="black",
                density=True,
                label="all",
                color="#21918c",
                range=(range_min, range_max),
            )
            for id, df in enumerate(df_list):
                df[f"{column_name}_{i}"].hist(
                    ax=axs[i // 2, i % 2],
                    bins=20,
                    edgecolor="black",
                    density=True,
                    label=label_list[id],
                    color=color_list[id],
                    range=(range_min, range_max),
                    alpha=0.5,
                )
            axs[i // 2, i % 2].set_xlabel(f"{column_name}_{i}")

        # set xlabel and y label for the last row
        # put the lengend on top of the figure

        for ax in axs.flatten():
            ax.grid(False)
            ax.set_ylabel("Density")
            ax.set_yticks([])
        plt.tight_layout()
        return fig, axs

    def get_all_possible_syntax(self):
        perm = product(
            list(range(self.number_of_fragments)),
            repeat=self.number_of_fragments,
        )
        # Print the obtained permutations
        possible_syntax = []
        for i in list(perm):
            append = False

            # check that the first element is 0
            if i[0] == 0:
                # check that the element is not higher than its position
                for pos, _id in enumerate(i):
                    if _id > pos:
                        append = False
                        break
                    if (_id!= pos and i[_id] == _id) or i[_id] == pos:
                        append = True
                    else:
                        append = False
                        break
            if append:
                possible_syntax.append(i)
        return possible_syntax

    def generate_interactive_condition_V2(self, df_total: pd.DataFrame,
                                          properties_to_plot=None):
        # function to generate an interactive prompt to select the condition
        # SP is the search space object
        # return the interactive widget
        if properties_to_plot is None:
            properties_to_plot = []
        def add_condition(
            columns: str, operation: str, value: str, fragment: int
        ) -> None:
            # condition syntax should follow the following condition:
            # "'column'#operation#value" e.g. "'IP (eV)'#>=#6.5"
            condition = f"'{columns}'#{operation}#{value}"
            self.add_condition(condition, fragment)

        def remove_condition(
            columns: str, operation: str, value: str, fragment: int
        ) -> None:
            # condition syntax should follow the following condition:
            # "'column'#operation#value" e.g. "'IP (eV)'#>=#6.5"
            condition = f"'{columns}'#{operation}#{value}"

            self.remove_condition(condition, fragment)

        def clear_condition(fragment: int) -> None:
            self.conditions_list[fragment] = []

        # Interactive widget for conditions selection
        columns_dropdown = widgets.Dropdown(
            options=self.df_precursors.select_dtypes(
                include=[np.number]
            ).columns,
            description="Property of fragment:",
        )
        # Interactive widget for fragment selection
        fragment_dropdown = widgets.Dropdown(
            options=list(range(self.number_of_fragments)),
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
        widgets.Dropdown(
            options=[("Add", "add"), ("Remove", "remove")],
            description="Operation:",
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

        def on_click_add(b) -> None:
            add_condition(
                columns_dropdown.value,
                operation_dropdown.value,
                value_dropdown.value,
                fragment_dropdown.value,
            )
            self.get_space_size()
            number_of_elements_text.value = f"{self.space_size:.2e}"

            for i in range(self.number_of_fragments):
                display_conditions[i].options = self.conditions_list[i]

        def on_click_remove(b) -> None:
            remove_condition(
                columns_dropdown.value,
                operation_dropdown.value,
                value_dropdown.value,
                fragment_dropdown.value,
            )
            # self.redefine_search_space()
            self.get_space_size()
            number_of_elements_text.value = f"{self.space_size:.2e}"

            for i in range(self.number_of_fragments):
                display_conditions[i].options = self.conditions_list[i]

        def on_click_clear(b) -> None:
            clear_condition(
                fragment_dropdown.value,
            )

            self.get_space_size()
            number_of_elements_text.value = f"{self.space_size:.2e}"
            for i in range(self.number_of_fragments):
                display_conditions[i].options = self.conditions_list[i]

        clear_condition_button.on_click(on_click_clear)
        add_condition_button.on_click(on_click_add)
        remove_condition_button.on_click(on_click_remove)
        add_to_all_fragment_button = widgets.Button(
            description="Add to all fragments",
        )

        def on_click_add_to_all_fragment(b) -> None:
            for i in range(self.number_of_fragments):
                add_condition(
                    columns_dropdown.value,
                    operation_dropdown.value,
                    value_dropdown.value,
                    i,
                )
                self.get_space_size()
                number_of_elements_text.value = f"{self.space_size:.2e}"

                for i in range(self.number_of_fragments):
                    display_conditions[i].options = self.conditions_list[i]

        add_to_all_fragment_button.on_click(on_click_add_to_all_fragment)
        # Set up the layout of the widgets
        vbox_layout = Layout(
            display="flex", flex_flow="row", align_items="flex-start"
        )
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
            value=f"{self.space_size:.2e}",
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
        min_target_5percent = (
            df_total["target"].nlargest(n=top5_percent_length).min()
        )
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

        def on_click_syntax(b) -> None:
            self.syntax = syntax_dropdown.value
            self.get_space_size()
            number_of_elements_text.value = f"{self.space_size:.2e}"

            for i in range(self.number_of_fragments):
                display_conditions[i].options = self.conditions_list[i]
            # number_of_elements_text.value = "{:.2e}".format(self.space_size)

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
        # save search space properties in table after each addition to hist compare
        # save the search space properties in a table
        search_space_properties = [
            {
                "number of elements": self.space_size,
                "syntax": self.syntax,
                "conditions": [[] for i in range(self.number_of_fragments)],
                "Elements in top 5%": 0,
                "number of elements evaluated": 0,
            },
        ]

        def add_to_hist_compare(b) -> None:
            # self.redefine_search_space()
            self.list_fragment = self.generate_list_fragment(
                self.generation_type
            )
            self.get_space_size()
            number_of_elements_text.value = f"{self.space_size:.2e}"
            df_list.append(
                self.check_df_for_element_from_sp(df_to_check=df_total)
            )
            label_list.append(label_text.value)
            # get the number of elements in the last df added to df_list that has target within 5% of the best target in df_total
            if df_list[-1].shape[0] > 0:
                top5_current_text.value = str(
                    df_list[-1][
                        df_list[-1]["target"] > min_target_5percent
                    ].shape[0]
                    / df_list[-1].shape[0]
                    * 100
                )
            else:
                top5_current_text.value = "no data"
            number_of_element_evaluated.value = f"{df_list[-1].shape[0]:.2e}"
            widge_plot.kwargs["df_list"] = df_list
            widge_plot.kwargs["label_list"] = label_list
            widge_plot.update()
            hist_widget_plot.kwargs["df_list"] = df_list
            hist_widget_plot.kwargs["label_list"] = label_list

            hist_widget_plot.update()
            # save the search space properties in a table
            conditions = [x.copy() for x in self.conditions_list]
            search_space_properties.append(
                {
                    "number of elements": self.space_size,
                    "syntax": self.syntax,
                    "conditions": conditions,
                    "Elements in top 5%": top5_current_text.value,
                    "number of elements evaluated": number_of_element_evaluated.value,
                }
            )

        def save_data(b) -> None:
            from pathlib import Path
            path_to_save = "search_space/search_space_properties.pkl"
            Path("search_space").mkdir(parents=True, exist_ok=True)
            if Path(path_to_save).exists():
                df_search_space_properties = pd.read_pickle(path_to_save)
                df_search_space_properties_2 = pd.DataFrame.from_dict(
                    search_space_properties
                )
                df_search_space_properties = pd.concat(
                    [
                        df_search_space_properties,
                        df_search_space_properties_2[1:],
                    ]
                )
                df_search_space_properties = df_search_space_properties.reset_index(drop=True)
            else:
                df_search_space_properties = pd.DataFrame.from_dict(
                    search_space_properties
                )
            df_search_space_properties.to_pickle(
                "search_space/search_space_properties.pkl"
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
            properties_to_plot = widgets.fixed(properties_to_plot),
        )
        # Interactive widget for column selection
        columns_dropdown_2 = widgets.Dropdown(
            options=self.df_precursors.select_dtypes(
                include=[np.number]
            ).columns,
            description="Property of fragment:",
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
