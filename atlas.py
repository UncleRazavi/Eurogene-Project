import argparse
import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from g25_utils import PC_COLUMNS
from ancestry_decomposition import run_ancestry


# --------------------------------------------------
# Geographic anchors
# --------------------------------------------------
GEO_MAP = {
    "Turkey_N": (39.0, 35.0),
    "Russia_Samara_EBA_Yamnaya": (53.2, 50.2),
    "Iran_Wezmeh_N.SG": (34.5, 47.0),
    "Israel_Natufian": (31.8, 35.2),
    "Georgia_Kotias.SG": (42.0, 44.0),
    "Russia_Karelia_HG": (61.5, 33.0),
    "China_AmurRiver_N": (50.3, 127.5),
    "Russia_Baikal_EN": (53.0, 108.0),
    "Morocco_Iberomaurusian": (34.0, -6.0),
}


# --------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, index_col=0)

    missing = [c for c in PC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df[PC_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()


# --------------------------------------------------
def fit_pca(df):
    pca = PCA(n_components=3)
    pca.fit(df.values)
    return pca


# --------------------------------------------------
def project(pca, df):
    coords = pca.transform(df.values)
    out = pd.DataFrame(coords, columns=["PC1", "PC2", "PC3"])
    out["Label"] = df.index
    return out


# --------------------------------------------------
def cluster(df, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    df = df.copy()
    df["Cluster"] = km.fit_predict(df[["PC1", "PC2", "PC3"]]).astype(str)
    return df


# --------------------------------------------------
def ancestry_to_geo(weights):
    lat, lon = 0.0, 0.0
    total = 0.0

    for pop, w in weights.items():
        if pop in GEO_MAP:
            la, lo = GEO_MAP[pop]
            lat += la * w
            lon += lo * w
            total += w

    if total == 0:
        return None

    return lat / total, lon / total


# --------------------------------------------------
def compute_distances(df, target_name):
    t = df[df["Label"] == target_name]
    tx, ty, tz = t[["PC1", "PC2", "PC3"]].values[0]

    df = df.copy()
    df["Distance"] = np.sqrt(
        (df["PC1"] - tx) ** 2 +
        (df["PC2"] - ty) ** 2 +
        (df["PC3"] - tz) ** 2
    )

    return df.sort_values("Distance")


# --------------------------------------------------
def build_plot(df, target_name, geo_point):

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scattergeo"}]],
        column_widths=[0.45, 0.55],
        subplot_titles=("3D PCA Genetic Space", "Eurasian Atlas Map")
    )

    target = df[df["Label"] == target_name]
    others = df[df["Label"] != target_name]

    # ---------------- PCA SPACE ----------------
    if "Cluster" in others.columns:
        for c in sorted(others["Cluster"].unique()):
            d = others[others["Cluster"] == c]

            fig.add_trace(go.Scatter3d(
                x=d["PC1"],
                y=d["PC2"],
                z=d["PC3"],
                mode="markers",
                marker=dict(size=3),
                name=f"Cluster {c}"
            ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=target["PC1"],
        y=target["PC2"],
        z=target["PC3"],
        mode="markers+text",
        marker=dict(size=10, symbol="diamond"),
        text=[target_name],
        name="Target"
    ), row=1, col=1)

    # ---------------- MAP ----------------
    for pop, (lat, lon) in GEO_MAP.items():
        fig.add_trace(go.Scattergeo(
            lat=[lat],
            lon=[lon],
            mode="markers+text",
            marker=dict(size=8, color="blue"),
            text=[pop],
            textposition="top center",
            name=pop
        ), row=1, col=2)

    # centroid
    if geo_point:
        fig.add_trace(go.Scattergeo(
            lat=[geo_point[0]],
            lon=[geo_point[1]],
            mode="markers+text",
            marker=dict(size=14, color="green"),
            text=["Ancestry Centroid"],
            name="Centroid"
        ), row=1, col=2)

    # ---------------- MAP STYLE ----------------
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        showland=True,
        landcolor="rgb(240,240,240)",
        coastlinecolor="black"
    )

    fig.update_layout(
        height=1200,
        width=2200,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
        title="Eurogene Atlas — PCA + Clustering + Geographic Projection"
    )

    return fig


# --------------------------------------------------
def main(modern, target, ancient, output_html, clusters):

    modern_df = load_data(modern)
    target_df = load_data(target)

    pca = fit_pca(modern_df)

    modern_pca = project(pca, modern_df)
    target_pca = project(pca, target_df)

    modern_pca = cluster(modern_pca, clusters)

    df = pd.concat([modern_pca, target_pca], ignore_index=True)
    df = compute_distances(df, target_df.index[0])

    ancestry = run_ancestry(target, ancient)
    geo = ancestry_to_geo(ancestry)

    fig = build_plot(df, target_df.index[0], geo)

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)

    print("\n Ancestry breakdown:")
    for k, v in ancestry.items():
        print(f"{k:30} {v:.2%}")

    print("\n Centroid:", geo)


# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--modern", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--ancient", required=True)
    parser.add_argument("--output_html", default="Results/atlas.html")
    parser.add_argument("--clusters", type=int, default=6)

    args = parser.parse_args()

    main(
        args.modern,
        args.target,
        args.ancient,
        args.output_html,
        args.clusters
    )