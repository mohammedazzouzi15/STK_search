import pandas as pd
import numpy as np

def test_random():
    top_mol_count = 300
    number_of_iteration = 450
    df = pd.read_csv("/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/30K_benchmark_150524.csv")
    print("max value in the dataset", df["target"].max())
    #df.dropna(subset=["target"], inplace=True)
    #df.drop_duplicates(subset=["InChIKey"], inplace=True)
    print(df.shape)
    max_list = []
    min_value = df["target"].sort_values().iloc[-top_mol_count]
    top_count_found_list = []
    print(min_value)
    for i in range(500):
        df_sample = df.sample(number_of_iteration)
        max_list.append(df_sample["target"].max())
        number_of_top_candidates_in_sample = df_sample[df_sample["target"] > min_value].shape[0]
        top_count_found_list.append(number_of_top_candidates_in_sample)
    print("Results")
    print("Max value")
    print(np.mean(max_list))
    print(np.min(max_list))
    print(np.max(max_list))
    print("Number of top candidates")
    print(np.mean(top_count_found_list))
    print(np.min(top_count_found_list))
    print(np.max(top_count_found_list)) 


    return df

if __name__ == "__main__":
    test_random()