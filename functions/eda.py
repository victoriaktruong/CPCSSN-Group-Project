#!/usr/bin/env python
# coding: utf-8

# In[11]:


#load packages
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


# In[27]:


#reused
def load_data(file_path):
    """Load dataset from CSV."""
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


# In[28]:


#reused
def get_lab_columns(df, meta_cols, predefined_labs=None):
    """
    Identify lab columns in a dataset.

    Parameters:
    - df: DataFrame
    - meta_cols: List of metadata columns to exclude
    - predefined_labs: Optional list of lab columns (if provided, use directly)

    Returns:
    - List of lab columns
    """
    if predefined_labs:
        # Use explicitly provided lab columns
        lab_cols = predefined_labs
        print("Using predefined lab columns.")
    else:
        # Auto-detect: all columns not in meta_cols
        lab_cols = [col for col in df.columns if col not in meta_cols]
        print("Auto-detected lab columns:", lab_cols)

    return lab_cols


# In[29]:


def summarize_dataframes(datasets):
    """
    Print concise summary of DataFrame structure and content.

    Parameters:
    - datasets

    Returns:
    - summary
    """
    for name, df in datasets.items():
        print(f"{name} summary")
        print(df.info())
        print()


# In[30]:


def summarize_numeric_statistics(datasets, numeric_columns):
    """
    Summarize basic statistics for numeric columns.

    Parameters:
    - datasets (dict): Dictionary of DataFrames.
    - numeric_columns (dict): Dictionary with names as keys and lists of numeric columns as values.

    Returns:
    -  A dictionary with names as keys and summary statistics as values
    """
    summaries = {}
    for name, df in datasets.items():
        if name in numeric_columns:
            summaries[name] = df[numeric_columns[name]].describe()
            print(f"\n{name} statistics:")
            print(summaries[name])
    return summaries


# In[31]:


def plot_histograms(df, numeric_cols):
    """
    Plot histograms for numeric columns.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - numeric_cols (list): Numeric column names.

     Returns:
    - Histogram(s)
    """
    n_cols = min(len(numeric_cols), 4)
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        df[col].dropna().hist(bins=10)
        plt.title(col.replace('_', ' '))
    plt.tight_layout()
    plt.show()


# In[32]:


def plot_correlation_matrix(df, numeric_cols):
    """
    Plot correlation matrix for numeric columns.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - numeric_cols (list): Numeric column names.

    Returns:
    - corr_matrix (pd.DataFrame): Correlation matrix.
    """
    corr_matrix = df[numeric_cols].corr()
    print("\nCorrelation matrix:")
    print(corr_matrix)
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    return corr_matrix


# In[33]:


def plot_boxplots(df, numeric_cols):
    """
    Plot boxplots for numeric columns.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - numeric_cols (list): Numeric column names.

    Returns:
    - boxplot(s)
    """
    n_cols = min(len(numeric_cols), 4)
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        plt.boxplot(df[col].dropna())
        plt.title(col.replace('_', ' '))
    plt.tight_layout()
    plt.show()


# In[34]:


def plot_scatter_pairs(df, col_pairs):
    """
    Plot scatter plots for column pairs.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - col_pairs (list): List of (x_col, y_col, title) tuples.

    Returns: 
    - scatterplot(s)
    """
    plt.figure(figsize=(10, 4))
    for i, (x_col, y_col, title) in enumerate(col_pairs, 1):
        plt.subplot(1, 2, i)
        plt.scatter(df[x_col], df[y_col])
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel(y_col.replace('_', ' '))
        plt.title(title)
    plt.tight_layout()
    plt.show()


# In[35]:


def check_unique_values(df, col):
    """
    Print unique values in a column.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - col (str): Column name.

    Returns:
    - unique_values 
    """
    unique_values = df[col].unique()
    print(f"Unique values in {col}: {unique_values}")
    return unique_values


# In[36]:


#import the cleaned dataset after missing value analysis 
os.chdir(r"C:\Users\selen\OneDrive\Desktop\BD_data")

# Load and preprocess datasets
file_paths = {
    'two_vars': 'cleaned_dataset_two_vars.csv',
    'all_vars': 'cleaned_dataset_all_vars.csv'
}
datasets = {name: load_data(path) for name, path in file_paths.items()}

expected_shapes = {
    'two_vars': (1389, 8),
    'all_vars': (1389, 13)
}
columns_to_keep = {
    'two_vars': ['Patient_ID', 'PerformedDate', 'first_any_dx_date', 'DiagnosisCode_calc', 
                 'DiagnosisText_calc', 'Lab_Timing', 'TOTAL CHOLESTEROL', 'FASTING GLUCOSE'],
    'all_vars': ['Patient_ID', 'PerformedDate', 'first_any_dx_date', 'DiagnosisCode_calc', 
                 'DiagnosisText_calc', 'Lab_Timing', 'TOTAL CHOLESTEROL', 'FASTING GLUCOSE', 
                 'HBA1C', 'HDL', 'LDL', 'INR', 'GLUCOSE TOLERANCE']
}
cleaning_steps = [
    ('PerformedDate', 'datetime'),
    ('first_any_dx_date', 'datetime'),
    ('DiagnosisText_calc', 'uppercase'),
    ('DiagnosisText_calc', 'strip')
]

df_two_vars = datasets['two_vars']
df_all_vars = datasets['all_vars']

# Get concise summary of the structure and content of DataFrames
summarize_dataframes(datasets)

# Calculate summary statistics for lab markers
lab_cols_two = ['TOTAL CHOLESTEROL', 'FASTING GLUCOSE']
lab_cols_all = ['TOTAL CHOLESTEROL', 'FASTING GLUCOSE', 'HBA1C', 'HDL', 'LDL', 'INR', 'GLUCOSE TOLERANCE']
numeric_columns = {
    'two_vars': lab_cols_two,
    'all_vars': lab_cols_all
}
summaries = summarize_numeric_statistics(datasets, numeric_columns)

# Plot histograms for lab markers (all_vars dataset, default 10 bins)
print("Histogram with 10 bins")
plot_histograms(df_all_vars, lab_cols_all)

# Plot correlation matrix for all lab columns
corr_matrix = plot_correlation_matrix(df_all_vars, lab_cols_all)

# Correlation matrix excluding GLUCOSE TOLERANCE
lab_cols_reduced = ['TOTAL CHOLESTEROL', 'FASTING GLUCOSE', 'HBA1C', 'HDL', 'LDL', 'INR']
corr_matrix_reduced = plot_correlation_matrix(df_all_vars, lab_cols_reduced)

# Check why GLUCOSE TOLERANCE has NaN correlations
check_unique_values(df_all_vars, 'GLUCOSE TOLERANCE')

# Plot boxplots to analyze outliers
plot_boxplots(df_all_vars, lab_cols_reduced)

# Plot scatter pairs for high-correlation combinations
plot_scatter_pairs(df_all_vars, [
    ('TOTAL CHOLESTEROL', 'LDL', 'Total Cholesterol vs LDL'),
    ('FASTING GLUCOSE', 'HBA1C', 'Fasting Glucose vs HBA1C')
])


# In[ ]:





# In[ ]:




