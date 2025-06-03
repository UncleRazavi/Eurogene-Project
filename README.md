# Eurogenes G25 Genetic Analysis

This project analyzes genetic data from the Eurogenes G25 dataset using various data science and machine learning techniques. It includes clustering, dimensionality reduction among populations and individuals.
##  Project Structure

- **Eurogenes Project.ipynb**: Main Jupyter Notebook containing all analysis steps, visualizations, and code.

# Genetic Population Matcher

This Python script finds the closest reference populations to a given genetic sample based on Euclidean distance between coordinates (e.g., PCA components or other genetic dimensions).

##  Files

- `find_closest_pops.py`: Main script to compute and visualize the closest populations.
- `my_sample.csv`: CSV file containing a single sample with 25-dimensional coordinates.
- `reference_populations.csv`: CSV file containing multiple reference populations with 25-dimensional coordinates.
- `README.md`: Documentation for usage.

##  Input Format

### `my_sample.csv`

A CSV file with the following format (header + one row sample):

```csv
,PC1,PC2,...,PC25
SampleName,val1,val2,...,val25
