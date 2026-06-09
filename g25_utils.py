import os

import numpy as np
import pandas as pd


PC_COLUMNS = [f"PC{i}" for i in range(1, 26)]


def load_g25_csv(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} file not found: {path}")

    df = pd.read_csv(path, index_col=0)
    missing = [col for col in PC_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")

    pc_df = df[PC_COLUMNS].copy()
    pc_df = pc_df.apply(pd.to_numeric, errors="coerce")

    if pc_df.isnull().any().any():
        bad_rows = pc_df[pc_df.isnull().any(axis=1)].index.tolist()
        preview = ", ".join(str(row) for row in bad_rows[:5])
        raise ValueError(f"{label} has non-numeric or missing PC values in: {preview}")

    if pc_df.empty:
        raise ValueError(f"{label} has no samples.")

    return pc_df


def write_results(results_df, output_csv=None, output_json=None):
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"CSV results saved to {output_csv}")

    if output_json:
        results_df.to_json(output_json, orient="records", indent=2)
        print(f"JSON results saved to {output_json}")


def fit_distance(target_vector, modeled_vector):
    return float(np.linalg.norm(target_vector - modeled_vector))
