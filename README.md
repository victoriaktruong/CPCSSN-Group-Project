# CPCSSN-Group-Project

This repository contains all the data wrangling and analyses scripts for **examining the association between laboratory biomarkers and bipolar disorder diagnosis**. 


EMBED PROJECT DIAGRAM INTO README

---

## Repository Structure

### `functions/`
Contains all scripts defining functions that are used in the Jupyter Notebooks, which includes data wrangling, preprocessing, exploratory data analysis (EDA), and modeling. Scripts are listed in order of use:

1. `wrangling.py`: Contains functions for loading, cleaning, filtering, and restructuring datasets.
2. `missing_labs.py`:
3. `eda.py`:
4. `eda_model.py`: Contains functions for mapping and encoding variables, Plot individual trajectories over time for a list of variables, and histgrams and boxplots of trajectories plots.

### `notebooks/`
Contains the Jupyter Notebooks used to process and analyze the data.

`main.ipynb`: This is the primary notebook that performs the full pipeline: loading data, wrangling, preprocessing, EDA.
`Analysis.ipynb`: This is the secondary notebook that performs the analytical procedures, including exploratory data analysis (EDA) to examine trends in laboratory markers over time, as well as statistical modeling.

#### `supplementary/`
Contains the supplementary notebooks that explore or test additional wrangling, preprocessing, and modeling scenarios that are not in the `main.ipynb`. Notebooks are listed in order:
1. `data-wrangling.ipynb`: Focuses on cohort selection and lab filtering for bipolar disorder patients.
2. 
3. 
4. `EDA_MODELLING.ipynb`: Focuses on the workflow for analyzing the trends of laboratory markers, testing their association with bipolar disorder, and applying statistical modeling to validate these associations.
---

## How to Use
1. Install Python 3.11

