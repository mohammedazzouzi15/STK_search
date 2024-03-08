import numpy as np
import torch
import pandas as pd

class RepresentationPrecursor:
    def __init__(self, df_precursors, frag_properties):
        self.df_precursors = df_precursors
        self.frag_properties = frag_properties

    def generate_repr(self, elements):
        init_rpr = []
        frag_properties = self.frag_properties
        frag_properties = frag_properties.union(['InChIKey'])
        elements_curr = elements.copy()
        num_frag = elements_curr.shape[1]
        init_rpr = []
        for i in range(num_frag):
            elements_curr['InChIKey']=elements_curr[f'InChIKey_{i}'].astype(str)
            df_eval = pd.merge(elements_curr,self.df_precursors[frag_properties], on='InChIKey', how='left', suffixes=('', f'_{i}'))
            if len(init_rpr)==0:
                init_rpr = df_eval[df_eval.columns[num_frag+1:]].values
            else:
                init_rpr = np.concatenate([init_rpr,df_eval[df_eval.columns[num_frag+1:]].values],axis=1)   

        X_explored_BO = torch.tensor(np.array(init_rpr.astype(float)), dtype=torch.float32)
        return X_explored_BO
