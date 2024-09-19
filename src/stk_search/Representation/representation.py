"""Base class for the molecular Representation of the molecules to be used for eather the BO or the surrogate EA ."""

import numpy as np
import torch


class Representation:
    """Base clasee to Generate the Representation of the constructed molecules."""

    def __init__(self):
        """Initialise the class."""
        self.name = "default"

    def generate_repr(self, building_blocks):
        """Generate the Representation of the elements.

        Args:
        ----
            building_blocks (pd.dataframe): table of building blocks nmaed with their InChIKey
        Returns:
            torch.tensor: Representation of the constructed molecule

        """
        init_rpr = []
        building_blocks = building_blocks.copy()
        num_frag = building_blocks.shape[1]
        for i in range(num_frag):
            building_blocks["InChIKey"] = building_blocks[
                f"InChIKey_{i}"
            ].astype(str)
            if len(init_rpr) == 0:
                init_rpr = np.zeros_like(
                    building_blocks[f"InChIKey_{i}"].values
                )
            else:
                init_rpr = np.concatenate(
                    [
                        init_rpr,
                        np.zeros_like(building_blocks[f"InChIKey_{i}"].values),
                    ],
                    axis=1,
                )
        return torch.tensor(
            np.array(init_rpr.astype(float)), dtype=torch.float32
        )
