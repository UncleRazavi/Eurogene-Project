import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import os

# List of ancient populations to use as sources
TARGET_SOURCES = {
    "Turkey_N", "Russia_Samara_EBA_Yamnaya", "Iran_Wezmeh_N.SG", "Israel_Natufian",
    "China_AmurRiver_N", "Georgia_Kotias.SG", "Russia_Karelia_HG", "Russia_Baikal_EN",
    "Morocco_Iberomaurusian"
}

def load_data(target_path, ancient_path):

    target_data = pd.read_csv(target_path, index_col=0)
    ancient_data = pd.read_csv(ancient_path, index_col=0)

    ancient_data['Population'] = ancient_data.index.to_series().apply(lambda x: x.split(':')[0])
    ancient_data = ancient_data[ancient_data['Population'].isin(TARGET_SOURCES)].copy()
    numeric_columns = ancient_data.select_dtypes(include=[np.number]).columns
    ancient_numeric = ancient_data[numeric_columns]
    ancient_averaged = ancient_data.groupby(ancient_data['Population']).mean()
    return target_data, ancient_averaged

def fit_nnls(target_vector, sources_matrix):
    coeffs, _ = nnls(sources_matrix.T, target_vector)
    coeffs /= coeffs.sum()
    return coeffs

def plot_pie_chart(name, populations, coeffs):
    plt.figure(figsize=(7, 7))
    wedges, texts, autotexts = plt.pie(
        coeffs,
        labels=None, 
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 10}
    )

    plt.legend(wedges, populations, title="Sources", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    plt.axis('equal')
    plt.title(f"Ancestry of {name}")
    plt.tight_layout()
    plt.show()

def main(target_path, ancient_path):
    target_data, ancient_averaged = load_data(target_path, ancient_path)
    populations = list(ancient_averaged.index)
    sources_matrix = ancient_averaged.values

    print("Using populations:", populations)
    print("=" * 40)

    for sample_name, pcs in target_data.iterrows():
        coefficients = fit_nnls(pcs.values, sources_matrix)
        print(f"{sample_name}:")
        for pop, coef in zip(populations, coefficients):
            print(f"  {pop:20} -> {coef:.2%}")
        print("-" * 40)

        plot_pie_chart(sample_name, populations, coefficients)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G25 Ancestry Deconvolution using NNLS")
    parser.add_argument('--target', required=True, help="Path to target dataset (.txt)")
    parser.add_argument('--ancient', required=True, help="Path to ancient dataset (.txt)")
    args = parser.parse_args()

    # Check if files exist
    for filepath in [args.target, args.ancient]:
        if not os.path.isfile(filepath):
            print(f"[Error] File not found: {filepath}")
            exit(1)

    main(args.target, args.ancient)
