import argparse
import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls

from g25_utils import PC_COLUMNS, fit_distance, load_g25_csv, write_results


# --------------------------------------------------
# Default ancient source panel
# --------------------------------------------------
TARGET_SOURCES = {
    "Turkey_N",
    "Russia_Samara_EBA_Yamnaya",
    "Iran_Wezmeh_N.SG",
    "Israel_Natufian",
    "China_AmurRiver_N",
    "Georgia_Kotias.SG",
    "Russia_Karelia_HG",
    "Russia_Baikal_EN",
    "Morocco_Iberomaurusian"
}


# --------------------------------------------------
def read_sources(source_path):
    if not source_path:
        return TARGET_SOURCES

    with open(source_path, "r", encoding="utf-8") as f:
        sources = {
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        }

    if not sources:
        raise ValueError("No valid sources found.")

    return sources


# --------------------------------------------------
def load_data(target_path, ancient_path, sources):

    target_data = load_g25_csv(target_path, "Target")

    # ensure PC order
    target_data = target_data[PC_COLUMNS]

    ancient = pd.read_csv(ancient_path, index_col=0)

    ancient["Population"] = ancient.index.to_series().apply(
        lambda x: x.split(":")[0]
    )

    ancient = ancient[ancient["Population"].isin(sources)].copy()

    missing = [c for c in PC_COLUMNS if c not in ancient.columns]
    if missing:
        raise ValueError(f"Missing PC columns in ancient dataset: {missing}")

    ancient[PC_COLUMNS] = ancient[PC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    ancient = ancient.dropna(subset=PC_COLUMNS)

    grouped = ancient.groupby("Population")[PC_COLUMNS].mean()

    if grouped.empty:
        raise ValueError("No matching ancient populations found.")

    return target_data, grouped


# --------------------------------------------------
def fit_nnls(target_vec, source_matrix):
    """
    Solve mixture using NNLS
    source_matrix shape: (sources, PCs)
    target_vec shape: (PCs)
    """

    A = source_matrix.T
    b = target_vec

    coeffs, _ = nnls(A, b)

    total = coeffs.sum()

    if total <= 0:
        raise ValueError("NNLS returned zero solution")

    return coeffs / total


# --------------------------------------------------
def run_ancestry(target_path, ancient_path, source_path=None):
    """
    Import-safe function
    Returns dict for first sample
    """

    sources = read_sources(source_path)
    target_data, ancient = load_data(target_path, ancient_path, sources)

    pops = list(ancient.index)
    matrix = ancient.values

    results = {}

    for name, row in target_data.iterrows():

        coeffs = fit_nnls(row.values, matrix)

        results[name] = {
            pop: float(w)
            for pop, w in zip(pops, coeffs)
        }

    return results


# --------------------------------------------------
def ensure_dir(path):
    directory = os.path.dirname(path) if path else ""
    if directory:
        os.makedirs(directory, exist_ok=True)


# --------------------------------------------------
def write_outputs(df, output_csv, output_json):

    if output_csv:
        ensure_dir(output_csv)

    if output_json:
        ensure_dir(output_json)

    write_results(df, output_csv, output_json)


# --------------------------------------------------
def plot_ancestry(df, output_path):
    if not output_path:
        return

    ensure_dir(output_path)

    plot_df = df[df["proportion"] > 0].copy()
    plot_df = plot_df.sort_values("proportion", ascending=False)

    labels = plot_df["source"]
    values = plot_df["percent"]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels[::-1], values[::-1], color="#4c9f70")
    ax.set_xlabel("Estimated ancestry (%)")
    ax.set_title(f"NNLS Ancestry Decomposition: {plot_df['sample'].iloc[0]}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"PNG plot saved to {output_path}")


# --------------------------------------------------
def main(target_path, ancient_path, source_path, output_png, output_csv, output_json):

    sources = read_sources(source_path)
    target_data, ancient = load_data(target_path, ancient_path, sources)

    pops = list(ancient.index)
    matrix = ancient.values

    rows = []

    for name, row in target_data.iterrows():

        coeffs = fit_nnls(row.values, matrix)

        modeled = coeffs @ matrix
        dist = fit_distance(row.values, modeled)

        print(f"\n{name}")

        for pop, coef in zip(pops, coeffs):

            print(f"{pop:25} {coef:.2%}")

            rows.append({
                "sample": name,
                "source": pop,
                "proportion": float(coef),
                "percent": float(coef * 100),
                "fit_distance": float(dist)
            })

        print(f"Fit distance: {dist:.6f}")

    df = pd.DataFrame(rows)

    plot_ancestry(df, output_png)
    write_outputs(df, output_csv, output_json)


# --------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="G25 Ancestry Decomposition using NNLS"
    )

    parser.add_argument("--target", required=True)
    parser.add_argument("--ancient", required=True)
    parser.add_argument("--sources", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)

    args = parser.parse_args()

    main(
        args.target,
        args.ancient,
        args.sources,
        args.output,
        args.output_csv,
        args.output_json
    )
