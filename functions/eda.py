#load packages
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

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
