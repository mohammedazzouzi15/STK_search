import numpy as np
import torch
import pandas as pd

class RepresentationMorganFP:
    def __init__(self):

    def generate_repr(self):
        # If path exists, load the data
        if self.df_path:
            df_MP = pd.read_pickle('data/output/Full_dataset/df_total_subset_16_11_23_MFP.csv')
            print('Morgan FP dataframe found')
            # Get the Morgan FP data from 'morgan_fp' column
            X_explored_BO = torch.tensor(np.array(df_MP['Morgan_fingerprints'].values.tolist()), dtype=torch.float32)
            print('Morgan FP data loaded')
        else:
            raise ValueError("Path to Morgan FP data not found, generate the fingerprint data first using Morgan_fp_generation.py")

        return X_explored_BO
