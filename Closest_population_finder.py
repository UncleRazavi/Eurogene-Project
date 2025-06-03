import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def get_closest_populations(sample_df, ref_df):
    results = {}
    for sample_name, sample_coords in sample_df.iterrows():
        distances = {}
        for ref_name, ref_coords in ref_df.iterrows():
            distance = np.linalg.norm(sample_coords - ref_coords)
            distances[ref_name] = distance
        results[sample_name] = sorted(distances.items(), key=lambda x: x[1])
    return results

def plot_top_matches(sample_name, top_matches):
    labels = [x[0] for x in top_matches]
    values = [x[1] for x in top_matches]

    plt.barh(labels[::-1], values[::-1])
    plt.title(f"Top 5 Closest Populations to {sample_name}")
    plt.xlabel("Euclidean Distance")
    plt.tight_layout()
    plt.show()

def main(sample_path, reference_path):
    if not os.path.exists(sample_path) or not os.path.exists(reference_path):
        print("Error: One or both input files not found.")
        return

    # Load the data
    sample_df = pd.read_csv(sample_path, index_col=0)
    ref_df = pd.read_csv(reference_path, index_col=0)

    # Compute distances
    closest = get_closest_populations(sample_df, ref_df)

    # Display and plot results for first sample
    sample_name = list(closest.keys())[0]
    top_matches = closest[sample_name][:5]

    print(f"\nTop 5 closest populations to {sample_name}:")
    for name, dist in top_matches:
        print(f"{name}: {dist:.4f}")

    plot_top_matches(sample_name, top_matches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find genetically closest populations.")
    parser.add_argument("--sample", required=True, help="Path to your sample CSV file")
    parser.add_argument("--reference", required=True, help="Path to reference population CSV file")

    args = parser.parse_args()
    main(args.sample, args.reference)
