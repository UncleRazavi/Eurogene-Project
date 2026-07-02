# Eurogenes G25 Genetic Analysis

This project analyzes Eurogenes G25-style coordinates with PCA visualization,
closest-population matching, and NNLS ancestry decomposition.

## Project Layout

- `Data/g25_data.csv` - modern reference G25 coordinates
- `Data/my_sample.csv` - target sample coordinates
- `Data/Global25_PCA_scaled (Ancient Individuals).csv` - ancient source data
- `Results/` - generated HTML, screenshots, PNG plots, CSV files, and JSON files

## Generated Results

The current bundled sample is `Mixed_Turkmen25_Tatar75`.

Generated files:

- `Results/atlas.html` - interactive Plotly atlas
- `Results/atlas_screenshot.png` - screenshot of the atlas
- `Results/closest.png` - closest-population bar chart
- `Results/closest_Mixed_Turkmen25_Tatar75_clustermap.png` - closest-population clustermap
- `Results/closest.csv` - closest-population table
- `Results/closest.json` - closest-population JSON
- `Results/ancestry_decomposition.png` - NNLS ancestry plot
- `Results/ancestry.csv` - NNLS ancestry table
- `Results/ancestry.json` - NNLS ancestry JSON

## Run The Atlas

```bash
python atlas.py --modern Data/g25_data.csv --target Data/my_sample.csv --ancient "Data/Global25_PCA_scaled (Ancient Individuals).csv" --output_html Results/atlas.html --clusters 6
```

The atlas includes:

- 3D PCA reference cloud
- highlighted target sample
- labeled nearest reference populations
- ancestry source map
- closest-reference distance chart

## Closest Population Finder

```bash
python Closest_population_finder.py --sample Data/my_sample.csv --reference Data/g25_data.csv --top_n 10 --output Results/closest.png --output_csv Results/closest.csv --output_json Results/closest.json
```

Optional clustermap:

```bash
python Closest_population_finder.py --sample Data/my_sample.csv --reference Data/g25_data.csv --top_n 10 --output Results/closest.png --plot_type clustermap
```

## Ancestry Decomposition

```bash
python ancestry_decomposition.py --target Data/my_sample.csv --ancient "Data/Global25_PCA_scaled (Ancient Individuals).csv" --output Results/ancestry_decomposition.png --output_csv Results/ancestry.csv --output_json Results/ancestry.json
```

Default ancient sources:

- Turkey_N
- Russia_Samara_EBA_Yamnaya
- Iran_Wezmeh_N.SG
- Israel_Natufian
- China_AmurRiver_N
- Georgia_Kotias.SG
- Russia_Karelia_HG
- Russia_Baikal_EN
- Morocco_Iberomaurusian

## Input Format

Input CSV files must include `PC1` through `PC25`:

```csv
,PC1,PC2,PC3,...,PC25
SampleName,val1,val2,val3,...,val25
```

## Notes

PCA is trained on modern reference populations. Ancient populations are used for
NNLS ancestry inference and the heuristic geographic centroid. These outputs are
for exploratory population genetics visualization and should not be interpreted
as direct ancestry proof.
