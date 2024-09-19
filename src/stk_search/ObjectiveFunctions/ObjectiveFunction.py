"""Module containing the classes for the objective functions.

Here we have the base class ObjectiveFunction and the LookUpTable class.
The base class is used to define the structure of the objective functions.

The LookUpTable class is used to evaluate the fitness of the elements by looking up the fitness in a database.

"""


import numpy as np
import stk


def get_inchi_key(molecule):
    """Get the InChIKey of the molecule.

    Args:
    ----
        molecule: stk.BuildingBlock
        the molecule

    Returns:
    -------
        str
        the InChIKey of the molecule

    """
    return stk.InchiKey().get_key(molecule)


class ObjectiveFunction:
    """Base class for objective functions.

    The objective function is the function that will be used to evaluate the fitness of the molecules in the search.

    Functions
    ---------
    evaluate_element(element, multi_fidelity=False)
        Evaluates the fitness of the element
        takes as an input a list of building blocks and returns the fitness of the element

    """

    def __init__(self):
        """Initialise the objective function."""
        self.multi_fidelity = False  # default value

    def evaluate_element(self, element):
        """Evaluate the fitness of the element.

        takes as an input a list of building blocks and returns the fitness of the element.

        Args:
        ----
            element: list
            list of building blocks


        Returns:
        -------
            float
            the fitness of the element
            str
            the identifier of the element

        """
        for x in element:
            if isinstance(x, (int, np.float64)):
                return float(x), "test"
        return None


class LookUpTable(ObjectiveFunction):
    """Class for look up table objective functions.

    The look up table objective function is used to evaluate the fitness of the elements by looking up the fitness in a database.

    """

    def __init__(self, df_look_up, fragment_size, target_name="target", aim=0):
        """Initialise the look up table objective function.

        Args:
        ----
            df_look_up: pd.DataFrame
            the dataframe containing the look up table
            the dataframe should contain the InChIKeys of the fragments in the form of 'InChIKey_0', 'InChIKey_1', etc.
            and the target column
            and the InChIKeys of the molecule

            fragment_size: int
            the size of the fragments

            target_name: str
            the name of the target column

            aim: int or float
            the aim of the fitness function
            if the aim is an int, the fitness function will be the negative absolute difference between the target and the aim

        """
        super().__init__()
        self.df_look_up = df_look_up
        self.fragment_size = fragment_size
        self.target_name = target_name
        self.aim = aim
        self.check_database()

    def check_database(self):
        """Check the database."""
        if self.df_look_up is None:
            msg = "No database found"
            raise ValueError(msg)
        if "InChIKey" not in self.df_look_up.columns:
            msg = "No InChIKey column found"
            raise ValueError(msg)
        if self.target_name not in self.df_look_up.columns:
            msg = "No target column found"
            raise ValueError(msg)
        if any(
            f"InChIKey_{i}" not in self.df_look_up.columns
            for i in range(self.fragment_size)
        ):
            msg = "No fragment columns found or not enough fragment columns"
            raise ValueError(msg)

    def evaluate_element(self, element):
        """Evaluate the fitness of the element.

        Takes as an input a list of building blocks and returns the fitness of the element.

        Args:
        ----
            element: list
            list of building blocks
            s

        Returns:
        -------
            float
            the fitness of the element
            str
            the identifier of the element in the form of an InChIKey

        """
        columns = [f"InChIKey_{i}" for i in range(self.fragment_size)]
        if self.multi_fidelity:
            columns.append("fidelity")
        results = element.merge(
            self.df_look_up,
            on=columns,
            how="left",
        )

        results = results.drop_duplicates(
            subset=[f"InChIKey_{i}" for i in range(self.fragment_size)],
        )
        if results[self.target_name].isna().any():
            msg = "missing data"
            raise ValueError(msg)
        if isinstance(self.aim, (int, float)):
            target = -np.abs(results[self.target_name][0] - self.aim)
        else:
            target = results[self.target_name][0]
        return target, results["InChIKey"][0]

