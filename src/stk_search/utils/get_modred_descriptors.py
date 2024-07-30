import os

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from stk_search.utils import tanimoto_similarity_utils


def main(df_total_path,save_path,number=0):
    os.makedirs(save_path, exist_ok=True)
    df_total = pd.read_csv(df_total_path)
    df_total["mol"] = df_total["InChIKey"].apply(tanimoto_similarity_utils.get_mol_from_df_single)
    calc = Calculator(descriptors, ignore_3D=True)
    descriptors_df = calc.pandas(df_total["mol"])
    frag_properties = descriptors_df.select_dtypes(include=[np.number]).columns
    descriptors_df = descriptors_df[frag_properties].copy()
    descriptors_df["InChIKey"] = df_total["InChIKey"]
    descriptors_df = descriptors_df.dropna()
    descriptors_df = descriptors_df.reset_index(drop=True)
    descriptors_df.to_csv(save_path+f"descriptors_{number}.csv", index=False)
    #pca = PCA(n_components=2)
    #frag_properties = descriptors_df.select_dtypes(include=[np.number]).columns
    #pca_transformed = pca.fit_transform(descriptors_df[frag_properties])
    #pca_df = pd.DataFrame(pca_transformed, columns=["PC1", "PC2"])
    #pca_df["InChIKey"] = descriptors_df["InChIKey"]
    #pca_df.to_csv(save_path+f"pca{number}.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_total_path", type=str, help="Path to the total dataframe")
    parser.add_argument("--save_path", type=str, help="Path to save the descriptors and pca dataframe")
    parser.add_argument("--number", type=int, help="Number to append to the file name", default=0)
    args = parser.parse_args()
    main(args.df_total_path,args.save_path,args.number)
