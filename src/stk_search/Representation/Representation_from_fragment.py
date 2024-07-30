import numpy as np
import pandas as pd
import torch


class Representation_from_fragment:
    """This class is used to generate the representation of the elements."""

    def __init__(self, df_precursors, frag_properties):
        """Initialise the class.

        Args:
        ----
            df_precursors (pd.dataframe): table of building blocks nmaed with their InChIKey
            frag_properties (set): set of properties to be used for the representation

        """
        self.df_precursors = df_precursors
        self.frag_properties = frag_properties
        self.name = "Representation_from_fragment"

    def generate_repr(self, building_blocks):
        """Generate the representation of the elements.

        Args:
        ----
            building_blocks (pd.dataframe): table of building blocks named with their InChIKey
        Returns:
            torch.tensor: representation of the constructed molecule

        """
        frag_properties = self.frag_properties
        frag_properties = frag_properties.union(["InChIKey"])
        elements_curr = building_blocks.copy()
        num_frag = elements_curr.shape[1]
        init_rpr = []
        for i in range(num_frag):
            elements_curr["InChIKey"]=elements_curr[f"InChIKey_{i}"].astype(str)
            df_eval = pd.merge(elements_curr,self.df_precursors[frag_properties], on="InChIKey", how="left", suffixes=("", f"_{i}"))
            if len(init_rpr)==0:
                init_rpr = df_eval[df_eval.columns[num_frag+1:]].values
            else:
                init_rpr = np.concatenate([init_rpr,df_eval[df_eval.columns[num_frag+1:]].values],axis=1)

        return torch.tensor(np.array(init_rpr.astype(float)), dtype=torch.float32)
