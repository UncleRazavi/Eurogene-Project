# Eurogenes G25 Genetic Analysis

This project analyzes genetic data from the **Eurogenes G25** dataset using various data science and machine learning techniques. It includes:

- Dimensionality reduction 
- Clustering of populations and individuals
- Visualization and interpretation of population structure
- Closest population matching via Euclidean distance


# Genetic Population Matcher

The `closest_population_finder.py` script finds the genetically closest reference populations to a user-provided sample by computing **Euclidean distances** between 25-dimensional coordinate vectors (e.g., from PCA).

###  Features:
- Computes genetic distances from reference populations
- Prints top 5 closest matches
- Visualizes the result with a horizontal bar chart

---

##  Input Format

###  `my_sample.csv`
A CSV file with a **single** sample and 25-dimensional coordinates:

```csv
,PC1,PC2,PC3,...,PC25
SampleName,val1,val2,val3,...,val25
Example
,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,PC11,PC12,PC13,PC14,PC15,PC16,PC17,PC18,PC19,PC20,PC21,PC22,PC23,PC24,PC25
Iranian_Persian_Shiraz:SHII20,0.094473,0.105615,-0.07203,-0.027132,-0.046162,-0.006136,0.001175,-0.009692,-0.038246,-0.017312,0.00341,-0.004646,0,-0.004679,0.004614,0.013259,-0.010691,0.002534,0.004399,-0.004877,0.008984,-0.001855,-0.002095,-0.005061,0.00479
