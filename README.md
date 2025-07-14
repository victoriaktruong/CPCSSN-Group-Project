# CPCSSN-Group-Project

This repository contains all the data wrangling and analyses scripts for **examining the association between laboratory biomarkers and bipolar disorder diagnosis**. 


EMBED PROJECT DIAGRAM INTO README

---

## Repository Structure

### `functions/`
Contains all scripts defining functions that are used in the Jupyter Notebooks, which includes data wrangling, preprocessing, exploratory data analysis (EDA), and modeling. Scripts are listed in order of use:

1. `wrangling.py`: Contains functions for loading, cleaning, filtering, and restructuring datasets.
2. `missing_labs.py`: Contains functions for loading a csv dataset, identifying columns with lab markers, printing and visualizing a missingness summary of the lab marker columns, imputing missing lab values using mean or median (depending on skewness of data), and saving a complete imputed dataset to csv.
3. `eda.py`: Contains functions for simple exploratory data analysis, including summarizing dataset structure and content, calculating descriptive statistics, and creating visualizations (histograms, correlation matrices, boxplots, and scatter plots) to analyze distributions. 
4. `eda_model.py`: Contains functions for mapping and encoding variables, plotting individual trajectories over time for a list of variables, and histograms and boxplots of trajectories.

### `notebooks/`
Contains the Jupyter Notebooks used to process and analyze the data.

`main.ipynb`: This is the primary notebook that performs the full pipeline: loading data, wrangling, preprocessing, EDA.

`analysis.ipynb`: This is the secondary notebook that performs the analytical procedures, including exploratory data analysis (EDA) to examine trends in laboratory markers over time, as well as statistical modeling.

### `supplementary/`
Contains the supplementary notebooks that explore or test additional wrangling, preprocessing, and modeling scenarios that are not in the `main.ipynb`. Notebooks are listed in order:
1. `data_wrangling.ipynb`: Focuses on cohort selection and lab filtering for bipolar disorder patients.
2. `missing_labs.ipnyb`: Focuses on visualizing missingness to confirm that the correct lab columns are being selected and imputed appropriately. Also contains a function that only selects lab columns that have less than 50% missing data (ultimately was not used in main.ipnyb). 
3. `eda_modelling.ipynb`: Focuses on analyzing the trends of laboratory markers, testing their association with bipolar disorder, and applying statistical modeling to validate these associations.
---

## How to Use
1. Install Python 3.11

