#!/usr/bin/env python
# coding: utf-8

# In[27]:


#load packages
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


# In[28]:


#Reused function 
def load_csv(file_paths, delimiter=','):
    """
    Load multiple CSV files from a dictionary of dataset names and file paths

    Parameters:
    - file_paths: dictionary of dataset names and file paths
    - delimiter: separator used in the CSV files

    Returns:
    - datasets: dictionary of loaded DataFrames with dataset names as keys
    """
    datasets = {}
    for name, path in file_paths.items():
        datasets[name] = pd.read_csv(path, sep=delimiter, on_bad_lines='skip', low_memory=False)
    return datasets


# In[29]:


#Reused function
def select_columns(datasets, columns_to_keep):
    """
    Select specific columns for each dataset based on a provided dictionary

    Parameters:
    - datasets: dictionary of DataFrames with dataset names as keys
    - columns_to_keep: dictionary mapping dataset names to a list of columns to keep

    Returns:
    - datasets: dictionary of DataFrames with only the selected columns
    """
    for name, columns in columns_to_keep.items():
        datasets[name] = datasets[name][columns]
    return datasets


# In[30]:


def preprocess_data(df, cleaning_steps):
    """
    Apply multiple cleaning steps to a DataFrame

    Parameters:
    - df: DataFrame to clean
    - cleaning_steps: list of tuples (column_name, method)
        - Method:
            - uppercase: converts strings to uppercase
            - strip: removes leading or trailing spaces
            - dropna: drops rows where column is missing
            - datetime: converts to datetime format

    Returns:
    - Cleaned DataFrame
    """
    for col, method in cleaning_steps:
        if method == 'uppercase':
            df[col] = df[col].fillna('').str.upper()
        elif method == 'strip':
            df[col] = df[col].astype(str).str.strip()
        elif method == 'dropna':
            df = df.dropna(subset=[col])
        elif method == 'datetime':
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


# In[31]:


def summarize_dataframes(datasets):
    """
    Print concise summary of DataFrame structure and content.

    Parameters:
    - datasets: dictionary of DataFrames with dataset names as keys

    Returns:
    - None (prints summaries)
    """
    for name, df in datasets.items():
        print(f"{name} summary")
        print(df.info())
        print()


# In[32]:


def summarize_lab_statistics(datasets, lab_columns):
    """
    Calculate and print summary statistics for lab markers.

    Parameters:
    - datasets: dictionary of DataFrames
    - lab_columns: dictionary mapping dataset names to lists of lab columns

    Returns:
    - summaries: dictionary of summary statistics DataFrames
    """
    summaries = {}
    for name, df in datasets.items():
        if name in lab_columns:
            summaries[name] = df[lab_columns[name]].describe()
            print(f"{name}:", summaries[name])
    return summaries


# In[33]:


def plot_lab_histograms(df, lab_cols, bins=10, figsize=(10, 6)):
    """
    Plot histograms for lab markers to check distributions.

    Parameters:
    - df: DataFrame containing lab data
    - lab_cols: list of lab column names
    - bins: number of bins for histograms (default: 10)
    - figsize: tuple of figure size (width, height)

    Returns:
    - None (displays plots)
    """
    plt.figure(figsize=figsize)  
    for i, col in enumerate(lab_cols, 1): 
        plt.subplot(2, 4, i)
        plt.hist(df[col], bins=bins)  
        plt.title(col.replace('_', ' ')) 
    plt.tight_layout()
    plt.show()


# In[34]:


def plot_correlation_matrix(df, lab_cols, figsize=(8, 6)):
    """
    Calculate and plot correlation matrix for lab markers.

    Parameters:
    - df: DataFrame containing lab data
    - lab_cols: list of lab column names
    - figsize: tuple of figure size (width, height)

    Returns:
    - corr_matrix: correlation matrix DataFrame
    """
    corr_matrix = df[lab_cols].corr()
    print(corr_matrix)

    plt.figure(figsize=figsize)
    plt.imshow(corr_matrix, cmap='coolwarm')
    plt.colorbar()

    # Set ticks and labels
    plt.xticks(range(len(lab_cols)), lab_cols, rotation=45, ha='right')
    plt.yticks(range(len(lab_cols)), lab_cols)

    # Add correlation values
    for i in range(len(lab_cols)):
        for j in range(len(lab_cols)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center')

    plt.title('Correlation Matrix of Lab Markers')
    plt.tight_layout()
    plt.show()

    return corr_matrix


# In[35]:


def plot_boxplots(df, lab_cols, figsize=(10, 8)):
    """
    Plot boxplots for lab markers to identify outliers.

    Parameters:
    - df: DataFrame containing lab data
    - lab_cols: list of lab column names
    - figsize: tuple of figure size (width, height)

    Returns:
    - None (displays plots)
    """
    plt.figure(figsize=figsize)
    for i, col in enumerate(lab_cols, 1):
        plt.subplot(2, 3, i)
        plt.boxplot(df[col])
        plt.title(col.replace('_', ' '))
    plt.tight_layout()
    plt.show()


# In[36]:


def plot_scatter_pairs(df, col_pairs, figsize=(10, 4)):
    """
    Plot scatter plots for specified column pairs.

    Parameters:
    - df: DataFrame containing the data
    - col_pairs: list of tuples (x_col, y_col, title)
    - figsize: tuple of figure size (width, height)

    Returns:
    - None (displays plots)
    """
    plt.figure(figsize=figsize)
    for i, (x_col, y_col, title) in enumerate(col_pairs, 1):
        plt.subplot(1, 2, i)
        plt.scatter(df[x_col], df[y_col])
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel(y_col.replace('_', ' '))
        plt.title(title)
    plt.tight_layout()
    plt.show()


# In[37]:


def check_unique_values(df, col):
    """
    Check and print unique values in a specified column.

    Parameters:
    - df: DataFrame
    - col: column name to check

    Returns:
    - unique_values: array of unique values
    """
    unique_values = df[col].unique()
    print(f"Unique values in {col}:", unique_values)
    return unique_values


# In[38]:


def summarize_diagnosis_counts(df, diag_col='DiagnosisText_calc'):
    """
    Summarize counts of diagnosis text.

    Parameters:
    - df: DataFrame containing diagnosis data
    - diag_col: column name for diagnosis text

    Returns:
    - counts: Series of diagnosis counts
    """
    counts = df[diag_col].value_counts()
    print("\nDiagnosis Text counts:\n", counts)
    return counts


# In[39]:


#import the cleaned dataset after missing value analysis 
os.chdir(r"C:\Users\selen\OneDrive\Desktop\BD_data")

# Load and preprocess datasets
file_paths = {
    'two_vars': 'cleaned_dataset_two_vars.csv',
    'all_vars': 'cleaned_dataset_all_vars.csv'
}
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
datasets = load_and_verify_datasets(file_paths, expected_shapes)
datasets = select_columns(datasets, columns_to_keep)
for name, df in datasets.items():
    datasets[name] = preprocess_data(df, cleaning_steps)
df_two_vars = datasets['two_vars']
df_all_vars = datasets['all_vars']

# Check the number of rows and columns for the two csv files to make sure they match with Uththami's
print("df_two_vars:", df_two_vars.shape, df_two_vars.columns.tolist())
print("\ndf_all_vars:", df_all_vars.shape, df_all_vars.columns.tolist())

#Are there any missing values remaining?
print("Missing values in two-vars dataset:", df_two_vars.isnull().sum().sum())
print("Missing values in all-vars dataset:", df_all_vars.isnull().sum().sum())

#get concise summary of the structure and content of two dataframes
print("two_vars summary")
print(df_two_vars.info())
print("\nall_vars summary")
print(df_all_vars.info())

#the output is as we would expect based on our previous analyses

#Calculate summary statistics for lab markers using describe()
lab_cols_two = ['TOTAL CHOLESTEROL', 'FASTING GLUCOSE']
lab_cols_all = ['TOTAL CHOLESTEROL', 'FASTING GLUCOSE', 'HBA1C', 'HDL', 'LDL', 'INR', 'GLUCOSE TOLERANCE']
print("two_vars:", df_two_vars[lab_cols_two].describe())
print("all_vars:", df_all_vars[lab_cols_all].describe())

# Plot histograms for lab markers to check distributions
#Histogram with default 10 bins
print("Histogram with 10 bins")
plot_lab_histograms(df_all_vars, lab_cols_all, bins=10)

#Histogram with 30 bins to emphasize the concentration around mean
print("\nHistogram with 30 bins")
plot_lab_histograms(df_all_vars, lab_cols_all, bins=30)

#Check correlation matrix 
corr_matrix = plot_correlation_matrix(df_all_vars, lab_cols_all)

#To avoid NaN results, exclude GLUCOSE TOLERANCE 
lab_cols = ['TOTAL CHOLESTEROL', 'FASTING GLUCOSE', 'HBA1C', 'HDL', 'LDL', 'INR']
corr_matrix_reduced = plot_correlation_matrix(df_all_vars, lab_cols)

#Why is glucose tolerance NaN? 
#There was just 1 observation = 4 in the dataset and GLUCOSE TOLERANCE was imputed with mean
#Therefore maybe we should drop the variable? 
check_unique_values(df_all_vars, 'GLUCOSE TOLERANCE')

#Based on summary statistics and the histograms, we might need to analyze outliers 
plot_boxplots(df_all_vars, lab_cols)

#From the correlation matrix, we know that LDL-TOTAL CHOLESTEROL and FASTING GLUCOSE-HBA1C are the highest correlation combinations
plot_scatter_pairs(df_all_vars, [
    ('TOTAL CHOLESTEROL', 'LDL', 'Total Cholesterol vs LDL'),
    ('FASTING GLUCOSE', 'HBA1C', 'Fasting Glucose vs HBA1C')
])

# Showing counts of diagnosis 
summarize_diagnosis_counts(df_all_vars)


# In[ ]:




