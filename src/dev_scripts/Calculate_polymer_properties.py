import pandas as pd
from stk_search.ObjectiveFunctions.IpEs1Fosc import IpEs1Fosc



def main(df_path,oligomer_size):
    #read_dataframme from path
    df_total = pd.read_csv(df_path)
    #load the data and the precursors dataset
    df_elements = df_total[[f"InChIKey_{i}" for i in range(oligomer_size)]]
    calculator = IpEs1Fosc(oligomer_size =oligomer_size)
    for i in range(df_elements.shape[0]):
        try:
            fitness_function, Inchikey = calculator.evaluate_element(df_elements.iloc[[i]])
            print(f"Fitness function: {fitness_function} Inchikey: {Inchikey}")
        except Exception as e:
            print(f"Error in {df_elements.iloc[i]}")
            print(e)
            continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate the properties of the polymer")
    parser.add_argument("--df_path", type=str, help="Path to the dataframe")
    parser.add_argument("--oligomer_size", type=int, help="Size of the oligomer")
    args = parser.parse_args()
    main(args.df_path,args.oligomer_size)
