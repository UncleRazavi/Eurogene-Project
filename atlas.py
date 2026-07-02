
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from ancestry_decomposition import run_ancestry
from g25_utils import PC_COLUMNS, fit_distance, load_g25_csv


# ============================================================================
# Population metadata
# ============================================================================

POPULATION_INFO = {

    # ===========================
    # Paleolithic
    # ===========================

    "Russia_MA1": {
        "era": "Paleolithic",
        "coord": (54.9, 82.9),
    },

    "Russia_AfontovaGora3": {
        "era": "Paleolithic",
        "coord": (56.0, 92.8),
    },

    "Italy_Villabruna": {
        "era": "Paleolithic",
        "coord": (46.2, 11.9),
    },

    "Spain_ElMiron": {
        "era": "Paleolithic",
        "coord": (43.3, -3.6),
    },

    "Switzerland_Bichon": {
        "era": "Paleolithic",
        "coord": (46.9, 6.7),
    },

    # ===========================
    # Mesolithic
    # ===========================

    "Russia_Karelia_HG": {
        "era": "Mesolithic",
        "coord": (61.5, 33.0),
    },

    "Georgia_Kotias.SG": {
        "era": "Mesolithic",
        "coord": (42.0, 44.0),
    },

    "Russia_Baikal_EN": {
        "era": "Mesolithic",
        "coord": (53.0, 108.0),
    },

    # ===========================
    # Early Neolithic
    # ===========================

    "Turkey_N": {
        "era": "Early Neolithic",
        "coord": (39.0, 35.0),
    },

    "Iran_Wezmeh_N.SG": {
        "era": "Early Neolithic",
        "coord": (34.5, 47.0),
    },

    "Israel_Natufian": {
        "era": "Early Neolithic",
        "coord": (31.8, 35.2),
    },

    "China_AmurRiver_N": {
        "era": "Early Neolithic",
        "coord": (50.3, 127.5),
    },

    "Morocco_Iberomaurusian": {
        "era": "Early Neolithic",
        "coord": (34.0, -6.0),
    },

    # ===========================
    # Early Bronze Age
    # ===========================

    "Russia_Samara_EBA_Yamnaya": {
        "era": "Early Bronze",
        "coord": (53.2, 50.2),
    },
}


# ============================================================================
# Data loading
# ============================================================================

def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load and validate PCA coordinate data.
    """

    df = pd.read_csv(path, index_col=0)

    missing = [c for c in PC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PC columns: {missing}")

    pc_df = df[PC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    if pc_df.isnull().values.any():
        raise ValueError("Non-numeric or missing values detected.")

    return pc_df


# ============================================================================
# PCA projection
# ============================================================================

def fit_pca(df: pd.DataFrame, n_components: int = 3) -> PCA:
    """
    Fit PCA model.
    """

    pca = PCA(n_components=n_components)
    pca.fit(df.values)

    return pca


def project(pca: PCA, df: pd.DataFrame) -> pd.DataFrame:
    """
    Project samples into PCA space.
    """

    coords = pca.transform(df.values)

    out = pd.DataFrame(
        coords,
        columns=["PC1", "PC2", "PC3"],
        index=df.index,
    )

    out["Label"] = out.index

    return out


# ============================================================================
# Clustering
# ============================================================================

def cluster(df: pd.DataFrame, k: int = 5) -> Tuple[pd.DataFrame, float]:
    """
    Apply KMeans clustering and compute silhouette score.
    """

    n_samples = len(df)
    if n_samples < 2:
        raise ValueError("At least two reference samples are required for clustering.")

    k = min(k, n_samples - 1)
    if k < 2:
        raise ValueError("At least two clusters are required for silhouette scoring.")

    km = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=42,
    )

    labels = km.fit_predict(df[["PC1", "PC2", "PC3"]])

    df = df.copy()
    df["Cluster"] = labels.astype(str)

    sil = silhouette_score(
        df[["PC1", "PC2", "PC3"]],
        labels
    )

    return df, sil


# ============================================================================
# Distance analysis
# ============================================================================

def compute_distances(
    df: pd.DataFrame,
    target_name: str,
) -> pd.DataFrame:
    """
    Compute Euclidean distances in PCA space.
    """

    if target_name not in df.index:
        raise ValueError(f"{target_name} not found.")

    target = df.loc[target_name, ["PC1", "PC2", "PC3"]].values

    distances = []

    for idx, row in df.iterrows():

        vec = row[["PC1", "PC2", "PC3"]].values

        dist = fit_distance(target, vec)

        distances.append(dist)

    out = df.copy()
    out["Distance"] = distances

    return out.sort_values("Distance")


# ============================================================================
# Geographic projection
# ============================================================================

def ancestry_to_geo(
    weights: Dict[str, float]
) -> Optional[Tuple[float, float]]:
    """
    Convert ancestry-like weights into a heuristic geographic centroid.

    This is a visualization heuristic only.
    """

    lat_sum = 0.0
    lon_sum = 0.0
    total = 0.0

    for pop, weight in weights.items():

        info = POPULATION_INFO.get(pop)

        if not info:
            continue

        lat, lon = info["coord"]

        lat_sum += lat * weight
        lon_sum += lon * weight
        total += weight

    if total == 0:
        return None

    return lat_sum / total, lon_sum / total


# ============================================================================
# Visualization
# ============================================================================

def build_plot(
    df: pd.DataFrame,
    pca: PCA,
    target_name: str,
    geo_point: Optional[Tuple[float, float]] = None,
    ancestry_weights: Optional[Dict[str, float]] = None,
    nearest: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Build interactive PCA + geographic atlas plot.
    """

    var = pca.explained_variance_ratio_ * 100
    plot_df = df.drop(index=target_name, errors="ignore")
    if nearest is None:
        nearest = compute_distances(df, target_name).drop(index=target_name, errors="ignore").head(12)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "scattergeo"}],
            [None, {"type": "xy"}],
        ],
        column_widths=[0.68, 0.32],
        row_heights=[0.58, 0.42],
        subplot_titles=(
            "3D PCA Genetic Space",
            "Ancestry Source Map",
            "Closest Reference Populations",
        ),
    )

    # ---------------------------------------------------------------------
    # PCA scatter
    # ---------------------------------------------------------------------

    fig.add_trace(

        go.Scatter3d(
            x=plot_df["PC1"],
            y=plot_df["PC2"],
            z=plot_df["PC3"],
            mode="markers",
            text=plot_df["Label"],
            marker=dict(
                size=4,
                color=plot_df["Cluster"].astype(int),
                colorscale="Turbo",
                opacity=0.72,
            ),
            name="Reference samples",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "PC1: %{x:.3f}<br>"
                "PC2: %{y:.3f}<br>"
                "PC3: %{z:.3f}<br>"
                "<extra></extra>"
            ),
        ),

        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=nearest["PC1"],
            y=nearest["PC2"],
            z=nearest["PC3"],
            mode="markers+text",
            text=nearest["Label"],
            textposition="top center",
            marker=dict(
                size=6,
                color="white",
                line=dict(width=1, color="black"),
            ),
            name="Closest references",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Distance: %{customdata:.4f}<br>"
                "PC1: %{x:.3f}<br>"
                "PC2: %{y:.3f}<br>"
                "PC3: %{z:.3f}<br>"
                "<extra></extra>"
            ),
            customdata=nearest["Distance"],
        ),
        row=1,
        col=1,
    )
    # Highlight target
    target = df.loc[target_name]

    fig.add_trace(

        go.Scatter3d(
            x=[target["PC1"]],
            y=[target["PC2"]],
            z=[target["PC3"]],
            mode="markers+text",
            text=[target_name],
            marker=dict(
                size=12,
                color="red",
                symbol="diamond",
            ),
            name="Target",
        ),

        row=1,
        col=1,
    )

    # ---------------------------------------------------------------------
    # Geographic projection
    # ---------------------------------------------------------------------

    if geo_point is not None:
        if ancestry_weights:
            source_rows = []
            for pop, weight in ancestry_weights.items():
                info = POPULATION_INFO.get(pop)
                if info and weight > 0:
                    lat, lon = info["coord"]
                    source_rows.append((pop, weight, lat, lon, info["era"]))

            if source_rows:
                source_df = pd.DataFrame(
                    source_rows,
                    columns=["Population", "Weight", "Lat", "Lon", "Era"],
                ).sort_values("Weight", ascending=False)

                fig.add_trace(
                    go.Scattergeo(
                        lon=source_df["Lon"],
                        lat=source_df["Lat"],
                        mode="markers+text",
                        text=source_df["Population"],
                        textposition="top center",
                        marker=dict(
                            size=(source_df["Weight"] * 45 + 7),
                            color=source_df["Weight"],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Weight", len=0.35),
                            line=dict(width=1, color="white"),
                        ),
                        name="Ancestry sources",
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Weight: %{customdata[0]:.1%}<br>"
                            "Era: %{customdata[1]}<br>"
                            "<extra></extra>"
                        ),
                        customdata=source_df[["Weight", "Era"]],
                    ),
                    row=1,
                    col=2,
                )
        fig.add_trace(

            go.Scattergeo(
                lon=[geo_point[1]],
                lat=[geo_point[0]],
                mode="markers+text",
                text=[target_name],
                marker=dict(
                    size=14,
                    color="crimson",
                    symbol="star",
                ),
                name="Projected centroid",
            ),

            row=1,
            col=2,
        )

    fig.add_trace(
        go.Bar(
            x=nearest["Distance"][::-1],
            y=nearest["Label"][::-1],
            orientation="h",
            marker=dict(color="#55c2ff"),
            name="Closest distance",
            hovertemplate="<b>%{y}</b><br>Distance: %{x:.4f}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # ---------------------------------------------------------------------
    # Layout polish
    # ---------------------------------------------------------------------

    fig.update_layout(

        template="plotly_dark",

        height=920,
        width=1500,

        title=dict(
            text=(
                f"<b>Eurasian Genetic Atlas</b><br>"
                f"<sup>"
                f"PC1={var[0]:.2f}% | "
                f"PC2={var[1]:.2f}% | "
                f"PC3={var[2]:.2f}% variance explained"
                f"</sup>"
            ),
            x=0.5,
        ),

        scene=dict(
            xaxis_title=f"PC1 ({var[0]:.2f}%)",
            yaxis_title=f"PC2 ({var[1]:.2f}%)",
            zaxis_title=f"PC3 ({var[2]:.2f}%)",
            bgcolor="rgb(10,10,10)",
        ),

        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(40,40,40)",
            showcountries=True,
            countrycolor="gray",
            bgcolor="rgb(15,15,15)",
            lataxis=dict(range=[15, 70]),
            lonaxis=dict(range=[-15, 135]),
        ),

        xaxis=dict(title="PCA-space distance"),
        yaxis=dict(title=""),

        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
        ),

        annotations=[
            dict(
                text=(
                    "Exploratory PCA-space visualization. "
                    "Geographic projection is heuristic and "
                    "should not be interpreted as direct ancestry inference."
                ),
                x=0.5,
                y=-0.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=11, color="lightgray"),
            )
        ]
    )

    return fig


def ensure_parent_dir(path: str | Path) -> None:
    parent = Path(path).parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Main atlas pipeline
# ============================================================================

def run_atlas(
    target_path: str,
    reference_path: str,
    ancient_path: Optional[str] = None,
    source_path: Optional[str] = None,
    k_clusters: int = 5,
    output_html: Optional[str] = None,
):
    """
    Full atlas pipeline.
    """

    ref_df = load_data(reference_path)
    target_df = load_g25_csv(target_path, "Target")

    pca = fit_pca(ref_df)

    projected_ref = project(pca, ref_df)

    projected_ref, silhouette = cluster(projected_ref, k_clusters)

    projected_target = project(pca, target_df)
    projected_target["Cluster"] = "-1"

    target_name = projected_target.index[0]
    projected = pd.concat([projected_ref, projected_target], axis=0)

    geo_point = None
    ancestry_weights = None

    if ancient_path:

        ancestry = run_ancestry(
            target_path,
            ancient_path,
            source_path,
        )

        if target_name in ancestry:
            ancestry_weights = ancestry[target_name]
            geo_point = ancestry_to_geo(ancestry_weights)

    fig = build_plot(
        projected,
        pca,
        target_name,
        geo_point,
        ancestry_weights=ancestry_weights,
        nearest=compute_distances(projected, target_name).drop(index=target_name, errors="ignore").head(12),
    )

    if output_html:
        ensure_parent_dir(output_html)
        fig.write_html(output_html, include_plotlyjs="cdn")
        print(f"[+] Atlas saved to: {output_html}")

    print(f"[+] Silhouette score: {silhouette:.3f}")

    return fig, projected


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive Eurasian PCA Atlas"
    )

    parser.add_argument("--target", required=True)
    parser.add_argument("--reference", "--modern", dest="reference", required=True)

    parser.add_argument("--ancient")
    parser.add_argument("--sources")

    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        "--output_html",
        dest="output",
        default="atlas.html",
    )

    args = parser.parse_args()

    run_atlas(
        target_path=args.target,
        reference_path=args.reference,
        ancient_path=args.ancient,
        source_path=args.sources,
        k_clusters=args.clusters,
        output_html=args.output,
    )

