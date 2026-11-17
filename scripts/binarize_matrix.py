import numpy as np
import pandas as pd
"""This script binarize influenece matrix, with threshold PERCENTILE """
INPUT_CSV_PATH = "../../influence_matrix.csv"
OUTPUT_BINARY_CSV_PATH = "../influence/binary_influence_matrix.csv"
PERCENTILE = 95


def create_binary_matrix(input_path: str, output_path: str, percentile: float):
    df = pd.read_csv(input_path, index_col=0)
    raw_matrix = df.values

    non_zero_values = raw_matrix[raw_matrix > 1e-10] # I assume that values <1e-10 are so small to consider them as 0 
    threshold_value = np.percentile(non_zero_values, percentile)

    binary_matrix = (raw_matrix > threshold_value).astype(int)

    np.fill_diagonal(binary_matrix, 0)

    total_possible_edges = binary_matrix.shape[0] * (binary_matrix.shape[0] - 1)
    active_edges = binary_matrix.sum()

    print(f"Chosen percentile:              {percentile}%")
    print(f"Computed threshold:         {threshold_value:.6e} ({threshold_value:.6f})")
    print(f"Num of agents:              {binary_matrix.shape[0]}")
    print(f"Active interactions (influence = 1):      {active_edges} of {total_possible_edges} possible")
    print("================================")

    binary_df = pd.DataFrame(binary_matrix, index=df.index, columns=df.columns)
    binary_df.to_csv(output_path)
    print(f"Saved at: {output_path}")


if __name__ == "__main__":
    create_binary_matrix(INPUT_CSV_PATH, OUTPUT_BINARY_CSV_PATH, PERCENTILE)