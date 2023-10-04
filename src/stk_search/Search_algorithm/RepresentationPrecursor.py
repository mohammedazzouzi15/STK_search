import numpy as np
import torch


class RepresentationPrecursor:
    def __init__(self, df_precursors, frag_properties):
        self.df_precursors = df_precursors
        self.frag_properties = frag_properties

    def generate_repr(self, elements):
        init_rpr = []
        for elem in elements:
            init_rpr.append(
                np.concatenate(
                    np.concatenate(
                        [
                            self.df_precursors[
                                self.df_precursors["InChIKey"] == x
                            ][self.frag_properties].values
                            for x in elem
                        ]
                    )
                )
            )
        X_explored_BO = torch.tensor(np.array(init_rpr), dtype=torch.float32)
        X_explored_BO = (X_explored_BO - X_explored_BO.mean(axis=0)) / (
            X_explored_BO.std(axis=0)
        )
        X_explored_BO = torch.nan_to_num(X_explored_BO)
        return X_explored_BO
