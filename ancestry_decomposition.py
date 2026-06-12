import argparse
import pandas as pd
import numpy as np
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
    ancient = pd.read_csv(ancient_path, index_col=0)

    ancient["Population"] = ancient.index.to_series().apply(
        lambda x: x.split(":")[0]
    )

    ancient = ancient[ancient["Population"].isin(sources)].copy()

    missing = [c for c in PC_COLUMNS if c not in ancient.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    ancient[PC_COLUMNS] = ancient[PC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    ancient = ancient.dropna()

    grouped = ancient.groupby("Population")[PC_COLUMNS].mean()

    return target_data, grouped


# --------------------------------------------------
def fit_nnls(target_vec, source_matrix):
    coeffs, _ = nnls(source_matrix.T, target_vec)

    total = coeffs.sum()
    if total <= 0:
        raise ValueError("NNLS returned zero solution")

    return coeffs / total


# --------------------------------------------------
def run_ancestry(target_path, ancient_path, source_path=None):
    """
    IMPORT-SAFE FUNCTION FOR ATLAS MODULE
    Returns: {population: proportion}
    """

    sources = read_sources(source_path)
    target_data, ancient = load_data(target_path, ancient_path, sources)

    pops = list(ancient.index)
    matrix = ancient.values

    for _, row in target_data.iterrows():
        coeffs = fit_nnls(row.values, matrix)

        return {
            pop: float(w)
            for pop, w in zip(pops, coeffs)
        }

    return {}


# --------------------------------------------------
def write_outputs(df, output_csv, output_json):
    write_results(df, output_csv, output_json)


# --------------------------------------------------
def main(target_path, ancient_path, source_path, output_csv, output_json):
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
                "fit_distance": dist
            })

        print(f"Fit distance: {dist:.4f}")

    df = pd.DataFrame(rows)
    write_outputs(df, output_csv, output_json)


# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G25 Ancestry Decomposition using NNLS"
    )

    parser.add_argument("--target", required=True)
    parser.add_argument("--ancient", required=True)
    parser.add_argument("--sources", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)

    args = parser.parse_args()

    main(
        args.target,
        args.ancient,
        args.sources,
        args.output_csv,
        args.output_json
    )