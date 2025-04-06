import pickle

import numpy as np
import pandas as pd
import swifter  # noqa: F401
from stk_search.Search_algorithm import (
    Search_algorithm,
)


def initialisation(number_of_initial_elements = 50):
    number_of_initial_elements = 50
    EA = Search_algorithm.evolution_algorithm()
    EA.number_of_parents = 10
    EA.number_of_random = 2
    EA.num_added_random = 0
    EA.selection_method_mutation = "top"
    EA.selection_method_cross = "top"
    df = pd.read_csv(
        "/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/30K_benchmark_150524.csv"
    )
    df = df.dropna(subset=["target"])
    df = df.reset_index(drop=True)
    SearchSpace_loc = "data_example/data_benchmark/SearchSpace_6_frag_full.pkl"
    print("max value in the dataset", df["target"].max())
    df_precursor = pd.read_pickle(
        "data_example/data_benchmark/df_properties.pkl"
    )
    df_precursor = df_precursor.reset_index(drop=True)
    print("size of df_precursor", df_precursor.shape)
    sp = pickle.load(open(SearchSpace_loc, "rb"))
    sp.df_precursors = df_precursor
    sp.update()
    df = df.dropna(subset=["target"])
    df = sp.check_df_for_element_from_sp(df_to_check=df)
    df= df.sort_values("target", ascending=False)
    print(df.shape)
    df_search = df.sample(50)
    df_search_top = df_search.sort_values("target", ascending=False).head(3)
    print("max_initial", df_search["target"].max())
    return EA, df, df_search, df_search_top, sp


def check_mutation(EA, df, df_search, df_search_top, sp):
    columns = [f"InChIKey_{i}" for i in range(6)]
    
    elements_to_mutate = EA.run_selection_method(
        EA.selection_method_mutation,
        df_search["target"].to_list(),
        df_search[columns],
    )
    df_elements_to_mutate = pd.DataFrame(
        elements_to_mutate,
        columns=[
            f"InChIKey_{x}" for x in range(elements_to_mutate.shape[1])
        ],  # check this for generalization
    )
    df_elements_to_mutate = sp.check_df_for_element_from_sp(
        df_to_check=df_elements_to_mutate
    )
    print("shape of df_elements_to_mutate", df_elements_to_mutate.shape)

    results_to_mutate = df_elements_to_mutate.merge(
        df_search_top,
        on=columns,
        how="left",
    )

    results_to_mutate.dropna(subset=["target"], inplace=True)
    print("shape of results_to_mutate", results_to_mutate.shape)


def test_generation(EA, df, df_search, df_search_top, sp):
    elements = EA.Generate_element_to_evaluate(
        df_search["target"].to_list(),
        df_search[[f"InChIKey_{i}" for i in range(6)]],
        sp,
    )
    df_elements = pd.DataFrame(
        elements,
        columns=[
            f"InChIKey_{x}" for x in range(elements.shape[1])
        ],  # check this for generalization
    )
    # print("shape of df_elements", df_elements.shape)

    df_elements = sp.check_df_for_element_from_sp(df_to_check=df_elements)
    columns = [f"InChIKey_{i}" for i in range(6)]
    results = df_elements.merge(
        df,
        on=columns,
        how="left",
    )
    results.dropna(subset=["target"], inplace=True)
    print("shape of df_elements", df_elements.drop_duplicates().shape)
    print("shape of df_search_top", df_search_top.shape)
    results_top = df_elements.merge(
        df_search_top,
        on=columns,
        how="left",
    )
    results_top = results_top.dropna(subset=["target"])
    df_search_top.to_csv(
        "df_search_top.csv", index=False
    )
    results_top.to_csv(
        "testEA.csv", index=False
    )
    print("shape of results_top", results_top.drop_duplicates().shape)
    print("max of results_top", results_top["target"].max())
    if results_top["target"].max() > results["target"].max():
        raise ValueError(
            "max value in the elements is greater than the max value in the initial space"
        )
    print("shape of results", results.drop_duplicates().shape)
    print("max_elements_to_choose from:", results["target"].max())
    if results["target"].max() < df_search["target"].max():
        raise ValueError(
            "max value in the elements is less than the max value in the initial space"
        )


def test_generate_element_to_choose(EA, df, df_search, sp,population_size = 20):
    df_search.reset_index(drop=True, inplace=True)
    columns = [f"InChIKey_{i}" for i in range(6)]
    df_elements, df_search_1 = EA.generate_df_elements_to_choose_from(
        df_search[columns],
        df_search["target"].to_list(),
        sp,
        benchmark=True,
        df_total=df,
    )
    #print("shape of df_elements", df_elements.shape)
    results = df_elements.merge(
        df,
        on=columns,
        how="left",
    )
    merged = results.merge(df_search[columns], on=columns, how="left", indicator=True)
    filtered_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    #print("shape of filtered_df", filtered_df.shape)
    if filtered_df.shape[0] < population_size:
        #print("size of df lower than the population size")
        random_element = df.sample(population_size)
    else:
        random_element = filtered_df.sample(population_size)
    # results.dropna(subset=["target"], inplace=True)
    #print("shape of results", results.shape)
    #print("max_elements_to_choose from:", results["target"].max())
    df_without_elements = df[~df["InChIKey"].isin(df_search["InChIKey"])]
    probability = 0
    #probability = check_probability_of_getting_a_better_element(results, df_without_elements)
    return probability ,random_element


def check_probability_of_getting_a_better_element(df_to_check, df_reference):
    """Calculate the probability of randomly selecting a better element
    from `df_to_check` compared to `df_reference`.

    Parameters
    ----------
    df_to_check : pd.DataFrame
        The DataFrame containing the elements to check.
        Must have a "target" column.

    df_reference : pd.DataFrame
        The reference DataFrame to compare against.
        Must have a "target" column.

    Returns
    -------
    float
        The probability of selecting a better element from `df_to_check`.

    """
    # Ensure both DataFrames have the "target" column
    if (
        "target" not in df_to_check.columns
        or "target" not in df_reference.columns
    ):
        raise ValueError("Both DataFrames must have a 'target' column.")

    # Get the target values from both DataFrames
    targets_to_check = df_to_check["target"].values
    targets_reference = df_reference["target"].values

    # Count the number of times an element in `df_to_check` is better
    better_count = 0
    total_comparisons = 0
    num_choices = 10
    for _ in range(10):  # Perform Monte Carlo sampling
        # Randomly select `num_choices` elements from `df_to_check`
        chosen_targets = np.random.choice(
            targets_to_check, size=num_choices, replace=False
        )
        chosen_targets_reference = np.random.choice(
            targets_reference, size=num_choices, replace=False
        )
        # Compare the best of the chosen elements to all elements in `df_reference`
        best_chosen = max(chosen_targets)
        best_chosen_reference = max(chosen_targets_reference)
        if best_chosen > best_chosen_reference:
            better_count += 1
        total_comparisons += 1


    # Calculate the probability
    probability = (
        better_count / total_comparisons if total_comparisons > 0 else 0
    )
    #print(f"Probability of getting a better element: {probability:.4f}")
    return probability


def test_random(number_of_iteration = 100):
    top_mol_count = 300
    df = pd.read_csv("/media/mohammed/Work/STK_search/Example_notebooks/data_example/data_benchmark/30K_benchmark_150524.csv")
    print("max value in the dataset", df["target"].max())
    df.dropna(subset=["target"], inplace=True)
    df.drop_duplicates(subset=["InChIKey"], inplace=True)
    #print(df.shape)
    max_list = []
    min_value = df["target"].sort_values().iloc[-top_mol_count]
    top_count_found_list = []
    #print(min_value)
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
    #print("Number of top candidates")
    #print(np.mean(top_count_found_list))
    #print(np.min(top_count_found_list))
    #print(np.max(top_count_found_list)) 


    return np.mean(max_list)


def main():
    number_of_initial_elements = 50
    number_of_iteration = 400
    EA, df, df_search, df_search_top, sp = initialisation()
    check_mutation(EA, df, df_search, df_search_top, sp)
    test_generation(EA, df, df_search, df_search_top, sp)
    return
    probability_list = []
    population_size = 1
    for i in range(0, number_of_iteration, population_size):
        probability,random_element = test_generate_element_to_choose(EA, df, df_search, sp,population_size = population_size)
        
        df_search = pd.concat([df_search, random_element])
        probability_list.append(probability)
    print("mean probability", np.mean(probability_list))
    print("min probability", np.min(probability_list))
    print("shape df_search", df_search["target"].shape)   
    print("max df_search", df_search["target"].max()) 

    
    mean_random = test_random(number_of_iteration)
    print("mean_random", mean_random, "max_EA", df_search["target"].max())

    


if __name__ == "__main__":
    main()
