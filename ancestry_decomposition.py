import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls

from g25_utils import PC_COLUMNS, fit_distance, load_g25_csv, write_results

# List of ancient populations to use as sources
TARGET_SOURCES = {
    "Turkey_N", "Russia_Samara_EBA_Yamnaya", "Iran_Wezmeh_N.SG", "Israel_Natufian",
    "China_AmurRiver_N", "Georgia_Kotias.SG", "Russia_Karelia_HG", "Russia_Baikal_EN",
    "Morocco_Iberomaurusian"
}
def read_sources(source_path):
    if not source_path:
        return TARGET_SOURCES

    with open(source_path, "r", encoding="utf-8") as source_file:
        sources = {
            line.strip()
            for line in source_file
            if line.strip() and not line.strip().startswith("#")
        }

    if not sources:
        raise ValueError(f"No sources found in {source_path}")
    return sources

def load_data(target_path, ancient_path, target_sources):
    target_data = load_g25_csv(target_path, "Target")
    ancient_data = pd.read_csv(ancient_path, index_col=0)

    ancient_data['Population'] = ancient_data.index.to_series().apply(lambda x: x.split(':')[0])
    ancient_data = ancient_data[ancient_data['Population'].isin(target_sources)].copy()
    if ancient_data.empty:
        raise ValueError("No matching ancient source populations found.")

    missing_cols = [col for col in PC_COLUMNS if col not in ancient_data.columns]
    if missing_cols:
        raise ValueError(f"Ancient data is missing required columns: {', '.join(missing_cols)}")

    ancient_data[PC_COLUMNS] = ancient_data[PC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if ancient_data[PC_COLUMNS].isnull().any().any():
        raise ValueError("Ancient data has non-numeric or missing PC values in selected sources.")

    ancient_averaged = ancient_data.groupby('Population')[PC_COLUMNS].mean()
    missing_sources = sorted(set(target_sources) - set(ancient_averaged.index))
    if missing_sources:
        print("[Warning] Sources not found:", ", ".join(missing_sources))

    return target_data, ancient_averaged

def fit_nnls(target_vector, sources_matrix):
    coeffs, _ = nnls(sources_matrix.T, target_vector)
    total = coeffs.sum()
    if total <= 0:
        raise ValueError("NNLS returned all-zero coefficients; cannot normalize ancestry proportions.")
    coeffs /= total
    return coeffs

def plot_pie_chart(name, populations, coeffs, output_path=None):

    plt.figure(figsize=(9, 9))
    
    non_zero_indices = np.where(coeffs > 1e-4)[0] 
    filtered_coeffs = coeffs[non_zero_indices]
    filtered_populations = [populations[i] for i in non_zero_indices]

    colors = plt.cm.Paired(np.linspace(0, 1, len(filtered_populations))) 
    
    wedges, texts, autotexts = plt.pie(
        filtered_coeffs,
        labels=None,  
        autopct='%1.1f%%', 
        startangle=140,
        colors=colors,
        textprops={'fontsize': 10}
    )

    displayed_percentages = set()
    for i, autotext in enumerate(autotexts):
        percent_str = autotext.get_text()
        if percent_str: 
            try:
            
                percent_value = float(percent_str.strip('%'))
                if percent_value in displayed_percentages:
                    autotext.set_text('') 
                else:
                    displayed_percentages.add(percent_value)
            except ValueError:
                pass
            
    legend_labels = []
    for i, pop in enumerate(filtered_populations):
        percent = filtered_coeffs[i] * 100
        legend_labels.append(f"{pop} ({percent:.1f}%)")

    plt.legend(wedges, legend_labels, title="Sources", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    plt.axis('equal')
    plt.title(f"Ancestry of {name}")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def safe_plot_path(output_path, sample_name, sample_count):
    if not output_path:
        return None
    if sample_count == 1:
        return output_path

    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(sample_name))
    root, ext = output_path.rsplit(".", 1) if "." in output_path else (output_path, "png")
    return f"{root}_{safe_name}.{ext}"

def main(target_path, ancient_path, source_path=None, output=None, output_csv=None, output_json=None):
    target_sources = read_sources(source_path)
    target_data, ancient_averaged = load_data(target_path, ancient_path, target_sources)
    populations = list(ancient_averaged.index)
    sources_matrix = ancient_averaged.values
    result_rows = []

    print("Using populations:", populations)
    print("=" * 40)

    for sample_name, pcs in target_data.iterrows():
        coefficients = fit_nnls(pcs.values, sources_matrix)
        modeled_vector = coefficients @ sources_matrix
        residual_distance = fit_distance(pcs.values, modeled_vector)
        print(f"{sample_name}:")
        for pop, coef in zip(populations, coefficients):
            print(f"  {pop:20} -> {coef:.2%}")
            result_rows.append({
                "sample": sample_name,
                "source": pop,
                "proportion": float(coef),
                "percent": float(coef * 100),
                "fit_distance": residual_distance,
            })
        print(f"  {'Fit distance':20} -> {residual_distance:.4f}")
        print("-" * 40)

        plot_path = safe_plot_path(output, sample_name, len(target_data))
        if output or len(target_data) == 1:
            plot_pie_chart(sample_name, populations, coefficients, plot_path)

    results_df = pd.DataFrame(result_rows)
    write_results(results_df, output_csv, output_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G25 Ancestry Deconvolution using NNLS")
    parser.add_argument('--target', required=True, help="Path to target dataset (.csv)")
    parser.add_argument('--ancient', required=True, help="Path to ancient dataset (.csv)")
    parser.add_argument('--sources', help="Optional text file with one source population per line")
    parser.add_argument('--output', help="Path to save the pie chart image")
    parser.add_argument('--output_csv', help="Path to save ancestry results as CSV")
    parser.add_argument('--output_json', help="Path to save ancestry results as JSON")
    args = parser.parse_args()

    try:
        main(args.target, args.ancient, args.sources, args.output, args.output_csv, args.output_json)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Error] {exc}")
