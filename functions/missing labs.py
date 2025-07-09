import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

def load_data(file_path):
    """Load dataset from CSV."""
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df

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


def summarize_missingness(df, lab_cols):
    """Print and visualize missingness summary."""
    missing_counts = df[lab_cols].isnull().sum().sort_values(ascending=False)
    missing_frac = df[lab_cols].isnull().mean().sort_values(ascending=False)
    
    print("\nMissing values per lab variable:")
    print(missing_counts)
    
    # Bar plot of missing counts
    missing_counts.plot.bar()
    plt.title("Missing Value Count per Lab Marker")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()
    
    # Missingness matrix
    msno.matrix(df[lab_cols])
    plt.show()
    
    return missing_frac

def plot_distributions(df, lab_cols):
    """Plot distributions of lab markers to inform imputation strategy."""
    n_cols = 3
    n_rows = -(-len(lab_cols) // n_cols)  # Ceiling division
    plt.figure(figsize=(15, 4 * n_rows))

    for i, col in enumerate(lab_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True)
        plt.title(f"{col} distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def impute_labs(
    df, meta_cols, lab_cols, skew_threshold=1.0, 
    missing_threshold=0.5, save_prefix="cleaned_labs", drop_missing=True
):
    """
    Impute missing lab values and save cleaned dataset.
    
    Parameters:
    - df: DataFrame
    - meta_cols: list of metadata columns
    - lab_cols: list of lab marker columns
    - skew_threshold: abs(skew) > threshold uses median, else mean
    - missing_threshold: drop lab vars with > threshold missingness (if drop_missing=True)
    - save_prefix: prefix for saved CSV file
    - drop_missing: if False, keeps all lab variables regardless of missingness
    
    Returns:
    - cleaned DataFrame, list of dropped columns
    """
    clean_df = df[meta_cols + lab_cols].copy()

    # Decide which lab columns to impute
    missing_frac = clean_df[lab_cols].isnull().mean()
    if drop_missing:
        keep_cols = missing_frac[missing_frac <= missing_threshold].index.tolist()
        dropped_cols = missing_frac[missing_frac > missing_threshold].index.tolist()
        print(f"Dropping columns with >{missing_threshold*100:.0f}% missing: {dropped_cols}")
    else:
        keep_cols = lab_cols
        dropped_cols = []
        print("Keeping all lab columns regardless of missingness.")

    # Impute missing values
    for col in keep_cols:
        skew = clean_df[col].dropna().skew()
        if abs(skew) > skew_threshold:
            impute_value = clean_df[col].median()
            method = "median"
        else:
            impute_value = clean_df[col].mean()
            method = "mean"

        print(f"Imputing {col} using {method} (skewness={skew:.2f})")
        clean_df[col] = clean_df[col].fillna(impute_value)

    # Save cleaned dataset
    output_path = f"{save_prefix}.csv"
    clean_df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path}")

    return clean_df, dropped_cols


