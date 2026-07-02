import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns
from scipy.spatial.distance import squareform, pdist 

from g25_utils import load_g25_csv, write_results

def ensure_parent_dir(path):
    directory = os.path.dirname(path) if path else ""
    if directory:
        os.makedirs(directory, exist_ok=True)

def get_closest_populations(sample_df, ref_df):
    results = {}
    for sample_name, sample_coords in sample_df.iterrows():
        diff = ref_df.subtract(sample_coords, axis="columns")
        distances = np.linalg.norm(diff.values, axis=1)
        matches = list(zip(ref_df.index, distances))
        results[sample_name] = sorted(matches, key=lambda x: x[1])
    return results

def plot_top_matches(sample_name, top_matches, output_path=None):
    labels = [x[0] for x in top_matches]
    values = [x[1] for x in top_matches]

    plt.barh(labels[::-1], values[::-1])
    plt.title(f"Top {len(top_matches)} Closest Populations to {sample_name}")
    plt.xlabel("Euclidean Distance")
    plt.tight_layout()
    
    if output_path:
        ensure_parent_dir(output_path)
        plt.savefig(output_path)
        print(f" Plot saved to {output_path}")
    else:
        plt.show()

def flatten_results(closest, top_n):
    rows = []
    for sample_name, matches in closest.items():
        for rank, (name, dist) in enumerate(matches[:top_n], start=1):
            rows.append({
                "sample": sample_name,
                "rank": rank,
                "match": name,
                "distance": float(dist),
            })
    return pd.DataFrame(rows)

def plot_clustermap(sample_name, sample_vector, top_matches_df, output_path=None):
    print(f" Generating spacious clustermap for {sample_name}...")
    
    combined_df = pd.concat(
        [sample_vector.to_frame(name=sample_name).T, top_matches_df]
)      
    distance_matrix = pd.DataFrame(
        squareform(pdist(combined_df, metric='euclidean')),
        columns=combined_df.index,
        index=combined_df.index
    )
    
    num_items = len(combined_df)
    fig_size = max(10, num_items * 0.5) 
    
    g = sns.clustermap(
        distance_matrix,
        cmap="viridis_r",  
        annot=True,      
        fmt=".2f",
        figsize=(fig_size, fig_size),       
        annot_kws = { "size" :8 },             
        tree_kws={"linewidths": 1.5}       
    )
    
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    
    g.fig.suptitle(f"Clustermap for {sample_name} and Top {len(top_matches_df)} Populations", y=1.02)
    
    if output_path:
        clustermap_output_path = output_path.replace('.png', f'_{sample_name}_clustermap.png')
        ensure_parent_dir(clustermap_output_path)
        g.savefig(clustermap_output_path, bbox_inches='tight', dpi=300) 
        print(f" Clustermap generously saved to {clustermap_output_path}")
    else:
        plt.show()


def main(sample_path, reference_path, top_n=5, output=None, output_csv=None, output_json=None, plot_type="bar"):
    sample_df = load_g25_csv(sample_path, "Sample")
    ref_df = load_g25_csv(reference_path, "Reference")
    closest = get_closest_populations(sample_df, ref_df)
    results_df = flatten_results(closest, top_n)

    for sample_name, matches in closest.items():
        top_matches = matches[:top_n]
        print(f"\nTop {top_n} closest populations to {sample_name}:")
        for rank, (name, dist) in enumerate(top_matches, start=1):
            print(f"{rank:>2}. {name}: {dist:.4f}")

        plot_path = None
        if output:
            if len(closest) == 1:
                plot_path = output
            else:
                safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(sample_name))
                root, ext = output.rsplit(".", 1) if "." in output else (output, "png")
                plot_path = f"{root}_{safe_name}.{ext}"
        
        if plot_type == "bar":
            plot_top_matches(sample_name, top_matches, plot_path)
        elif plot_type == "clustermap":
            sample_vector = sample_df.loc[sample_name]
            top_match_names = [x[0] for x in top_matches]
            top_matches_df = ref_df.loc[top_match_names] 
            plot_clustermap(sample_name, sample_vector, top_matches_df, plot_path)

    write_results(results_df, output_csv, output_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find genetically closest populations.")
    parser.add_argument("--sample", required=True, help="Path to your sample CSV file")
    parser.add_argument("--reference", required=True, help="Path to reference population CSV file")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top matches to display (default: 5)")
    parser.add_argument("--output", type=str, help="Path to save the plot image (optional)")
    parser.add_argument("--output_csv", type=str, help="Path to save tabular results as CSV")
    parser.add_argument("--output_json", type=str, help="Path to save tabular results as JSON")
    parser.add_argument("--plot_type", type=str, default="bar", choices=["bar", "clustermap"], help="نوع نمودار خروجی: bar یا clustermap")

    args = parser.parse_args()
    try:
        main(args.sample, args.reference, args.top_n, args.output, args.output_csv, args.output_json, args.plot_type)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Error] {exc}")
