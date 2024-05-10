import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import Layout, VBox, interact, widgets


class Search_Space:
    """
    class that contains the chemical space to search over
    it is defined by the number of fragments and the syntax of the fragment forming the oligomer
    """

    def __init__(
        self,
        number_of_fragments: int,
        df: pd.DataFrame,
        features_frag: list,
        generation_type: str = "manual",
    ):
        self.number_of_fragments = number_of_fragments
        self.df_precursors = df
        self.features_frag = features_frag
        self.syntax = [0, 1, 2, 3, 4, 5]
        self.conditions_list = [[] for i in range(self.number_of_fragments)]
        self.generation_type = generation_type
        self.update()

    def update(self):
        self.list_fragment = self.generate_list_fragment(
            self.generation_type
        )  # list of list of index of the fragment ( need to be the same length as the number of fragments)
        self.get_space_size()

    def generate_list_fragment(self, generation_type="random"):
        # generate the list of list of index of the fragment ( need to be the same length as the number of fragments)
        # generation type can either be random or most_diverse or manual or conditional

        list_fragment = []
        if generation_type == "random":
            for i in range(self.number_of_fragments):
                list_fragment.append(
                    np.random.choice(
                        self.df_precursors.index.to_list(), 8, replace=False
                    )
                )
        elif generation_type == "most_diverse":
            list_fragment = generate_list_most_diverse_fragment(
                number_fragment=8, df=self.df_precursors
            )
        elif generation_type == "conditional":
            for i in range(self.number_of_fragments):
                df_filtered = self.df_precursors.copy()
                for condition in self.conditions_list[i]:
                    condition_exp = condition.split("#")
                    expression = (
                        "df_filtered["
                        + condition_exp[0]
                        + "]"
                        + condition_exp[1]
                        + condition_exp[2]
                    )
                    df_filtered = df_filtered[eval(expression)]
                list_fragment.append(df_filtered.index.to_list())
        elif generation_type == "manual":
            list_fragment = [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
                [11, 12],
            ]
        return list_fragment

    def get_space_size(self):
        x = 1

        [
            x := x * len(y)
            for y in [
                self.list_fragment[z]
                for z in set(self.syntax[: self.number_of_fragments])
            ]
        ]
        self.space_size = x
        return x

    def generate_syntax(self):
        # generate the syntax of the oligomer
        syntax = []
        for i in range(self.number_of_fragments):
            syntax.append([f"frag_{i}_{x}" for x in self.features_frag])
        return syntax

    def generate_dataframe_with_search_space(self):
        id_list_not_to_merge = []
        for i in set(self.syntax):
            if i == 0:
                df_multi = self.df_precursors.loc[list(self.list_fragment[0])][
                    self.features_frag
                ]
            else:
                df_multi = df_multi.merge(
                    self.df_precursors.loc[list(self.list_fragment[i])][
                        self.features_frag
                    ],
                    how="cross",
                    suffixes=("", "_" + str(i)),
                )
            id_list_not_to_merge.append(i)
        df_multi = df_multi.rename(
            columns={
                c: c + "_0"
                for c in df_multi.columns
                if c in self.features_frag
            }
        )

        for pos, id in enumerate(self.syntax):
            if pos in id_list_not_to_merge:
                continue
            else:
                df_multi = df_multi.merge(
                    self.df_precursors.loc[list(self.list_fragment[id])][
                        self.features_frag
                    ],
                    left_on=f"InChIKey_{id}",
                    right_on="InChIKey",
                    suffixes=("", "_" + str(pos)),
                )
                df_multi = df_multi.rename(
                    columns={
                        c: c + f"_{pos}"
                        for c in df_multi.columns
                        if c in self.features_frag
                    }
                )

        # print(f"shape of the dataframe {df_multi.shape}")
        return df_multi

    def random_generation_df(self, num_element):
        import random

        id_list_not_to_merge = []
        max_fragment = int(num_element ** (1 / len(set(self.syntax)))) + 1
        df_list = [None] * self.number_of_fragments
        for i in set(self.syntax):

            if i == 0:
                df_list[i] = self.df_precursors.loc[
                    list(random.sample(self.list_fragment[0], max_fragment))
                ][self.features_frag]
                df_multi = df_list[i]

            else:
                df_list[i] = self.df_precursors.loc[
                    list(random.sample(self.list_fragment[i], max_fragment))
                ][self.features_frag]
                df_multi = df_multi.merge(
                    df_list[i],
                    how="cross",
                    suffixes=("", "_" + str(i)),
                )
            id_list_not_to_merge.append(i)
        df_multi = df_multi.rename(
            columns={
                c: c + "_0"
                for c in df_multi.columns
                if c in self.features_frag
            }
        )
        for pos, id in enumerate(self.syntax):
            if pos in id_list_not_to_merge:
                continue
            else:
                df_multi = df_multi.merge(
                    df_list[id],
                    left_on=f"InChIKey_{id}",
                    right_on="InChIKey",
                    suffixes=("", "_" + str(pos)),
                )
                df_multi = df_multi.rename(
                    columns={
                        c: c + f"_{pos}"
                        for c in df_multi.columns
                        if c in self.features_frag
                    }
                )

        df_multi = df_multi.sample(num_element)
        # print(f"shape of the dataframe {df_multi.shape}")

        return df_multi

    def add_condition(self, condition: str, fragment: int):
        # condition syntax should follow the following condition:
        # "'column'#operation#value" e.g. "'IP (eV)'#>=#6.5"
        self.conditions_list[fragment].append(condition)

    def remove_condition(self, condition: str, fragment: int):
        # condition syntax should follow the following condition:
        # "'column'#operation#value" e.g. "'IP (eV)'#>=#6.5"
        self.conditions_list[fragment].remove(condition)

    def check_df_for_element_from_SP(self, df_to_check: pd.DataFrame):
        # check if the condition is respected by the search space
        # show that each fragment respect the condition
        # return the list of the InChIKey of the fragment that respect the condition
        df_mult_filtered = df_to_check.copy()
        for i in range(self.number_of_fragments):
            df_precursor_filter = self.df_precursors.copy()
            for condition in self.conditions_list[i]:
                condition_exp = condition.split("#")
                expression = (
                    "df_precursor_filter["
                    + condition_exp[0]
                    + "]"
                    + condition_exp[1]
                    + condition_exp[2]
                )
                df_precursor_filter = df_precursor_filter[eval(expression)]

            df_mult_filtered = df_mult_filtered[
                df_mult_filtered[f"InChIKey_{i}"].isin(
                    df_precursor_filter["InChIKey"]
                )
            ]
        # check if the fragment in the dataframe respect the syntax in the syntax list
        for pos, id in enumerate(self.syntax):
            if id == pos:
                pass
            else:
                df_mult_filtered = df_mult_filtered[
                    df_mult_filtered[f"InChIKey_{id}"]
                    == df_mult_filtered[f"InChIKey_{pos}"]
                ]

        return df_mult_filtered

    def redefine_search_space(self):
        # redifine the search space based on the conditions
        # and return the new search space as a dataframe
        self.list_fragment = self.generate_list_fragment(
            self.generation_type
        )  # list of list of index of the fragment ( need to be the same length as the number of fragments)
        self.get_space_size()
        if self.space_size < 1e6:
            return self.generate_dataframe_with_search_space()
        elif self.space_size < 1e8:
            print("space too big but will take element randomly")
            df_search_space = self.generate_dataframe_with_search_space()
            # reduce the size of the search space by taking random elements
            return df_search_space.sample(n=int(1e6), replace=False)
        else:
            print("space way too big")

    def plot_histogram_precursor(self):
        def plot_histogram(column_name):
            plt.figure(figsize=(6, 4))
            plt.hist(df[column_name], bins=20, edgecolor="black")
            plt.xlabel(column_name)
            plt.ylabel("Frequency")
            plt.title(f"Histogram of {column_name}")
            plt.show()

        df = self.df_precursors
        # Interactive widget for column selection
        columns_dropdown = widgets.Dropdown(
            options=df.select_dtypes(include=["int", "float"]).columns,
            description="Value:",
        )
        # Set up the layout of the widgets

        vbox_layout = Layout(
            display="flex", flex_flow="row", align_items="flex-start"
        )
        # Display the widget
        interact(
            plot_histogram, column_name=columns_dropdown, layout=vbox_layout
        )
        plt.show()

    def add_TSNE_to_df_precuros(self):
        # plot t_SNE of the search space
        from rdkit.Chem import rdFingerprintGenerator
        from sklearn import manifold

        mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048
        )
        # bit vectors:
        df = self.df_precursors
        fp_list = [
            mfpgen.GetFingerprintAsNumPy(mol) for mol in df["mol_opt_2"]
        ]
        TSNE = manifold.TSNE(n_components=2, random_state=0)
        self.precursor_TSNE_X_2d = TSNE.fit_transform(np.array(fp_list))

    def plot_chem_space(self):
        # plot X_2d as a scatter plot
        # check if precursor_TSNE_X_2d in the class
        if not hasattr(self, "precursor_TSNE_X_2d"):
            self.add_TSNE_to_df_precuros()
        plt.scatter(
            self.precursor_TSNE_X_2d[:, 0],
            self.precursor_TSNE_X_2d[:, 1],
            c="0.8",
        )
        # get color list for the different fragment
        # get marker list for the different fragment
        marker_list = ["o", "s", "^", "v", "<", ">"]
        color_list = [
            "#21918c",
            "#5ec962",
            "#f2c14e",
            "#f78154",
            "#f24e4e",
            "#f24e4e",
        ]
        for id, index_of_precursor_chosen in enumerate(self.list_fragment):
            plt.scatter(
                self.precursor_TSNE_X_2d[index_of_precursor_chosen, 0],
                self.precursor_TSNE_X_2d[index_of_precursor_chosen, 1],
                c=color_list[id],
                marker=marker_list[id],
                label=f"fragment{id}",
                s=50,
            )
        # plt.xlabel('T_SNE 1')
        # plt.ylabel('T_SNE 2')
        plt.legend()
        # put the legent on top of the plot and make it into 6 columns
        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncol=3,
            mode="expand",
            borderaxespad=0.0,
        )
        # plt.title('Fragement space',fontsize=20)
        # make sure that the title does not overlap with the legend
        plt.subplots_adjust(top=0.8)
        # plt.colorbar(label='IP (eV)')
        plt.grid(False)
        plt.axis("off")
        plt.show()

    def toJSON(self):
        return json.dumps(
            self, default=lambda o: o.__dict__, sort_keys=True, indent=4
        )

    def save_space(self, filename: str):
        # save the space in a jason file
        self.conditions_list.to_json(filename)
        # save the conditions ist in a jason file
        self.list_fragment.to_json(filename)
        # save the dataframe of the precursors
        self.df_precursors.to_json(filename)
        # save the other parameters
        self.number_of_fragments.to_json(filename)
        self.features_frag.to_json(filename)
        self.syntax.to_json(filename)

    def generate_interactive_condition(self):
        # function to generate an interactive prompt to select the condition
        # SP is the search space object
        # return the interactive widget
        def add_condition(
            columns: str, operation: str, value: str, fragment: int
        ):
            # condition syntax should follow the following condition:
            # "'column'#operation#value" e.g. "'IP (eV)'#>=#6.5"
            condition = f"'{columns}'#{operation}#{value}"
            self.add_condition(condition, fragment)

        def remove_condition(
            columns: str, operation: str, value: str, fragment: int
        ):
            # condition syntax should follow the following condition:
            # "'column'#operation#value" e.g. "'IP (eV)'#>=#6.5"
            condition = f"'{columns}'#{operation}#{value}"

            self.remove_condition(condition, fragment)

        # Interactive widget for conditions selection
        columns_dropdown = widgets.Dropdown(
            options=self.df_precursors.select_dtypes(
                include=[np.number]
            ).columns,
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

        def on_click_add(b):
            add_condition(
                columns_dropdown.value,
                operation_dropdown.value,
                value_dropdown.value,
                fragment_dropdown.value,
            )
            for i in range(self.number_of_fragments):
                display_conditions[i].options = self.conditions_list[i]

        def on_click_remove(b):
            remove_condition(
                columns_dropdown.value,
                operation_dropdown.value,
                value_dropdown.value,
                fragment_dropdown.value,
            )
            for i in range(self.number_of_fragments):
                display_conditions[i].options = self.conditions_list[i]

        add_condition_button.on_click(on_click_add)
        remove_condition_button.on_click(on_click_remove)

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
            ],
            layout=vbox_layout,
        )
        # Display the widget
        # add a button to add the condition
        display(Vb)
        display_conditions = []
        for i in range(self.number_of_fragments):
            display_conditions.append(widgets.SelectMultiple())
            display_conditions[i].options = self.conditions_list[i]
            display_conditions[i].description = f"Fragment {i}"
            display_conditions[i].layout.height = "200px"
            display_conditions[i].disabled = True
        # change display of list in to a table
        Vb = VBox(
            [display_conditions[i] for i in range(self.number_of_fragments)],
            layout=vbox_layout,
        )
        display(Vb)
        # interact(add_condition, columns=columns_dropdown, operation=operation_dropdown, value=value_dropdown, fragment=fragment_dropdown,add_condition=add_condition_button, layout=vbox_layout)
        # interact(remove_condition, condition=f"'{columns_dropdown.value}'#{operation_dropdown.value}#{value_dropdown.value}", fragment=fragment_dropdown, layout=vbox_layout)
        # interact(plot_histogram, column_name=columns_dropdown, layout=vbox_layout)
        plt.show()
        self.plot_histogram_precursor()

    def generate_list_most_diverse_fragment(
        number_fragment: int, df: pd.DataFrame
    ):
        print(len(df))
        fpgen = AllChem.GetRDKitFPGenerator()
        fps = []
        for x in df["mol_opt_2"].values:
            try:
                fps.append(fpgen.GetFingerprint(x))
            except:
                print(x)
                break
        low_IP = df["TSNE_1d"].min()
        sep = (df["TSNE_1d"].max() - low_IP) / number_fragment
        fps_list_index = []
        for i in range(number_fragment):
            df_test = df[df["TSNE_1d"] > low_IP + sep * i]
            df_test = df_test[df_test["TSNE_1d"] < low_IP + sep * (i + 1)]
            list_of_fragement = df_test.index.to_list()
            fps_list_index.append(
                np.random.choice(
                    list_of_fragement,
                    min(15, len(list_of_fragement)),
                    replace=False,
                )
            )
        return fps_list_index
