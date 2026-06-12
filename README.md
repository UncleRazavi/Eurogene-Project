# Eurogenes G25 Genetic Analysis

This project analyzes genetic data from the **Eurogenes G25 dataset** using dimensionality reduction, clustering, and ancestry modeling techniques.

It includes:

- PCA-based population structure visualization (2D/3D)
- KMeans clustering of genetic populations
- Interactive geographic “genetic atlas”
- NNLS-based ancestry decomposition
- Closest population matching via Euclidean distance
- Combined visualization of genetic + geographic structure

---

#  Genetic Population Matcher

The `closest_population_finder.py` script finds genetically closest reference populations to a sample using Euclidean distance across 25-dimensional G25 coordinates.

## Features:
- Computes genetic distance to reference populations
- Finds closest matches for each sample
- Validates PC1–PC25 input format
- Outputs results in CSV / JSON format
- Optional visualization (bar plot of closest populations)

---

## Input Format

## `my_sample.csv`

## A CSV file containing one or more samples with 25-dimensional coordinates:

```csv
,PC1,PC2,PC3,...,PC25
SampleName,val1,val2,val3,...,val25
``` 
## Example 
```csv
,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,PC11,PC12,PC13,PC14,PC15,PC16,PC17,PC18,PC19,PC20,PC21,PC22,PC23,PC24,PC25
Iranian_Persian_Shiraz:SHII20,0.094473,0.105615,-0.07203,-0.027132,-0.046162,-0.006136,0.001175,-0.009692,-0.038246,-0.017312,0.00341,-0.004646,0,-0.004679,0.004614,0.013259,-0.010691,0.002534,0.004399,-0.004877,0.008984,-0.001855,-0.002095,-0.005061,0.00479
```

```bash 
python Closest_population_finder.py --sample my_sample.csv --reference g25_data.csv --top_n 10 --output Results/closest.png --output_csv Results/closest.csv 
--output_json Results/closest.json
```

# Ancestry Decomposition (NNLS Model)

# Estimates ancestral composition using non-negative least squares regression.

Each individual is modeled as a weighted mixture of ancient populations.

- Features
- Ancient population-based modeling
- NNLS optimization
- Ancestry proportions estimation
- Pie chart visualization
- Fit quality (error distance)
- Custom source panels

## Ancient Sources
- Turkey_N
- Russia_Samara_EBA_Yamnaya
- Iran_Wezmeh_N.SG
- Israel_Natufian
- China_AmurRiver_N
- Georgia_Kotias.SG
- Russia_Karelia_HG
- Russia_Baikal_EN
- Morocco_Iberomaurusian

## Run
```bash
python ancestry_decomposition.py \
  --target my_sample.csv \
  --ancient "Global25_PCA_scaled (Ancient Individuals).csv" \
  --output Results/ancestry.png \
  --output_csv Results/ancestry.csv \
  --output_json Results/ancestry.json
  ```


## Custom Source Panel
```bash
python ancestry_decomposition.py \
  --target my_sample.csv \
  --ancient "Global25_PCA_scaled (Ancient Individuals).csv" \
  --sources sources.txt
```
Each line:

- Turkey_N
- Israel_Natufian
- Iran_Wezmeh_N.SG

# Interactive Genetic Atlas

## The atlas.py module provides a full interactive visualization system combining:

- PCA (trained on modern populations)
- KMeans clustering
- Geographic projection of populations
- NNLS ancestry centroid mapping
- Interactive Plotly HTML output
# Features
- 3D PCA genetic space
- Clustered modern populations
- Eurasian geographic map
- Ancestry centroid projection
- Fully interactive HTML output

```bash
python atlas.py --modern Data/g25_data.csv --target Data/my_sample.csv --ancient "Global25_PCA_scaled (Ancient Individuals).csv" --output_html Results/atlas.html --clusters 6
  ```

# Notes
PCA is trained only on modern populations
Ancient populations are used only for ancestry inference
Outputs are fully reproducible and script-based
Designed for exploratory population genetics research
## Scientific Interpretation
PCA captures major genetic variation axes
NNLS estimates admixture proportions
Geographic projection approximates ancestral centers
Clustering reveals structure in modern populations
## Future Improvements
Migration flow animation (Bronze Age expansions)
Temporal PCA (Neolithic → Iron Age)
Heatmap-based ancestry density maps
Streamlit web dashboard
Ancient DNA time slider
