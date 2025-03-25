"""class SearchSpace.

class that contains the chemical space to search over
it is defined by the number of fragments and the syntax of the fragment forming the oligomer
it also contains the conditions that need to be respected by the building blocks.

Attributes
----------
number_of_fragments : int
    number of fragments in the oligomer
df_precursors : pd.DataFrame
    dataframe containing the building blocks inchikeys and features
generation_type : str
    type of generation of the search space
syntax : list
    list of the syntax of the oligomer
conditions_list : list
    list of the conditions that need to be respected by the building blocks


"""
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import Layout, interact, widgets


class SearchSpace:
    """class that contains the chemical space to search over
    it is defined by the number of fragments and the syntax of the fragment forming the oligomer
    it also contains the conditions that need to be respected by the building blocks.

    Attributes
    ----------
    number_of_fragments : int
        number of fragments in the oligomer
    df_precursors : pd.DataFrame
        dataframe containing the building blocks inchikeys and features
    generation_type : str
        type of generation of the search space
    syntax : list
        list of the syntax of the oligomer
    conditions_list : list
        list of the conditions that need to be respected by the building blocks


    """  # noqa: D205

    def __init__(
        self,
        number_of_fragments: int,
        df: pd.DataFrame,
        # features_frag: list,
        generation_type: str = "conditional",
    ):
        """Initialize the search space.

        Parameters
        ----------
        number_of_fragments : int
            number of fragments in the oligomer
        df : pd.DataFrame
            dataframe containing the building blocks inchikeys and features
        features_frag : list
            list of the features of the building blocks
        generation_type : str
            type of generation of the search space

        """
        self.number_of_fragments = number_of_fragments
        self.df_precursors = df
        self.syntax = [0, 1, 2, 3, 4, 5]
        self.conditions_list = [[] for i in range(self.number_of_fragments)]
        self.generation_type = generation_type
        self.space_size = 0
        self.update()

    def add_condition(self, condition: str, fragment: int):
        """Add a condition to the condition list.
        
        condition syntax should follow the following condition:
        "'column'#operation#value" e.g. "'IP (eV)'#>=#6.5".

        Parameters
        ----------
        condition : str
            condition to add
        fragment : int
            fragment position to which the condition is added

        """
        self.conditions_list[fragment].append(condition)

    def remove_condition(self, condition: str, fragment: int):
        """Remove a condition from the condition list.

        condition syntax should follow the following condition:
        "'column'#operation#value" e.g. "'IP (eV)'#>=#6.5".

        Parameters
        ----------
        condition : str
            condition to remove
        fragment : int
            fragment position from which the condition is removed

        """
        self.conditions_list[fragment].remove(condition)

    def check_df_for_element_from_sp(self, df_to_check: pd.DataFrame):
        """Check if the dataframe respect the conditions of the search space.
        
        Parameters
        ----------
        df_to_check : pd.DataFrame
            dataframe to check

        Returns
        -------
        pd.DataFrame
            dataframe containing the elements that respect the conditions

        """
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
        for pos, pos_id in enumerate(self.syntax):
            if pos_id == pos:
                pass
            else:
                df_mult_filtered = df_mult_filtered[
                    df_mult_filtered[f"InChIKey_{pos_id}"]
                    == df_mult_filtered[f"InChIKey_{pos}"]
                ]
        return df_mult_filtered

    def update(self):
        """Update the search space based on the conditions changes the list of fragment and recomputes the space size.

        runs the following functions:
        - generate_list_fragment:
            list of list of index of the fragment ( need to be the same length as the number of fragments)
        - get_space_size    

        """
        self.list_fragment = (
            self.generate_list_fragment()
        )
        self.get_space_size()

    def generate_list_fragment(self):
        """Generate the list of list of index of the fragment.

        generate the list of list of index of the fragment ( need to be the same length as the number of fragments)
        generation type can either be random or most_diverse or manual or conditional
        
        Returns
        -------
        list
            list of list of index of the fragment

        """
        list_fragment = []
        if self.generation_type == "conditional":
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
        elif self.generation_type == "manual":
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
        """Compute the size of the search space.

        Returns
        -------
        int
            size of the search space

        """
        x = 1
        for z in set(self.syntax[: self.number_of_fragments]):
            x = x * len( self.list_fragment[z])
        self.space_size = x
        return x

    def plot_histogram_precursor(self):
        """Plot the histogram of the precursors."""
        
        def plot_histogram(column_name) -> None:
            plt.figure(figsize=(6, 4))
            plt.hist(df_to_plot[column_name], bins=20, edgecolor="black")
            plt.xlabel(column_name)
            plt.ylabel("Frequency")
            plt.title(f"Histogram of {column_name}")
            plt.show()

        df_to_plot = self.df_precursors
        # Interactive widget for column selection
        columns_dropdown = widgets.Dropdown(
            options=df_to_plot.select_dtypes(include=["int", "float"]).columns,
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

    def random_generation_df(self, num_elements):
        """
        Generate a random DataFrame of molecular fragments based on the syntax.

        This function randomly selects molecular fragments from the precursor DataFrame
        (`self.df_precursors`) and combines them according to the specified syntax (`self.syntax`).
        The resulting DataFrame contains a random subset of molecules with the desired structure.

        Args:
        ----
        num_elements : int
            The number of elements (rows) to generate in the resulting DataFrame.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing randomly generated molecular fragments
            combined according to the syntax.

        Notes:
        -----
        - The function ensures that the fragments are combined based on the syntax rules.
        - The `features_frag` attribute specifies the features to include in the resulting DataFrame.
        """
        import random

        # List to track fragment IDs that should not be merged
        excluded_ids = []

        # Calculate the maximum number of fragments to sample for each syntax group
        max_fragments = int(num_elements ** (1 / len(set(self.syntax)))) + 1
        # Initialize a list to store DataFrames for each fragment group
        fragment_dfs = [None] * self.number_of_fragments

        # Generate DataFrames for each unique syntax group
        for group_id in set(self.syntax):
            if group_id == 0:
                # Select random fragments for the first group
                fragment_dfs[group_id] = self.df_precursors.loc[
                    list(random.sample(self.list_fragment[0], max_fragments))
                ]
                combined_df = fragment_dfs[group_id]
            else:
                # Select random fragments for other groups and merge with the combined DataFrame
                fragment_dfs[group_id] = self.df_precursors.loc[
                    list(random.sample(self.list_fragment[group_id], max_fragments))
                ]
                combined_df = combined_df.merge(
                    fragment_dfs[group_id],
                    how="cross",
                    suffixes=("", f"_{group_id}"),
                )
            excluded_ids.append(group_id)


        # Merge fragments based on the syntax
        for position, group_id in enumerate(self.syntax):
            if position in excluded_ids:
                continue
            else:
                combined_df = combined_df.merge(
                    fragment_dfs[group_id],
                    left_on=f"InChIKey_{group_id}",
                    right_on="InChIKey",
                    suffixes=("", f"_{position}"),
                )

        # Randomly sample the desired number of elements
        combined_df = combined_df.sample(num_elements)
        combined_df.reset_index(drop=True, inplace=True)
        combined_df = combined_df[[col for col in combined_df.columns if "InChIKey" in col]]
        return combined_df