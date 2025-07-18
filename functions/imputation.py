import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import missingno as msno

#function to load csv file
def load_data(file_path):
    """Load dataset from CSV."""
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df

#function to code sex variable as numeric
def encode_sex(df, column="Sex"):
    """Encode 'Sex' as numeric: Female=0, Male=1."""
    df[column] = df[column].map({"Female": 0, "Male": 1})
    print("Encoded Sex: Female=0, Male=1")
    return df

#function to drop lab columns with more than 60% missing data
def drop_missing_cols(df, meta_cols, threshold=0.6):
    """
    Drop lab markers with more than 60% missingness.
    Parameters:
    - df: DataFrame
    - meta_cols: List of metadata columns (excluded from missingness check)
    - threshold: Proportion of missingness allowed (default=0.6 for 60%)
    """
    #identify lab columns (anything that's not a metadata column)
    lab_cols = [col for col in df.columns if col not in meta_cols]
    
    #calculate missingness percentage for each lab marker
    missing_frac = df[lab_cols].isnull().mean()
    
    #identify columns to drop
    columns_to_drop = missing_frac[missing_frac > threshold].index.tolist()
    
    #drop columns
    df.drop(columns=columns_to_drop, inplace=True)
    
    print(f"Dropped columns with >{threshold*100:.0f}% missingness: {columns_to_drop}")
    return df

#function to scale auxiliary variables like sex and age (give them more weight in the knn than other lab markers)
def scale_aux_vars(df, scale_factors):
    """
    Scale auxiliary variables by specified factors.
    Example scale_factors: {"Age": 2, "Sex": 2}
    """
    for col, factor in scale_factors.items():
        if col in df.columns:
            df[col] *= factor
            print(f"Scaled {col} by {factor}")
    return df


#function to apply knn imputation
def knn_imp(df, knn_cols, n_neighbors=5, weights='uniform', exclude_cols=None):
    """
    Apply KNN imputation to specified columns.
    
    Parameters:
    - df: DataFrame
    - knn_cols: List of columns to include in KNN (predictors and targets)
    - n_neighbors: Number of neighbors (default=5)
    - weights: Weight function ('uniform' or 'distance')
    - exclude_cols: Columns to use as predictors but not to be imputed (default=None)
    """
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputed = imputer.fit_transform(df[knn_cols])
    imputed_df = pd.DataFrame(imputed, columns=knn_cols, index=df.index)
    
    #keep original values for exclude_cols (sex and age)
    if exclude_cols:
        for col in exclude_cols:
            imputed_df[col] = df[col]
        print(f"Excluded columns from imputation: {exclude_cols}")
    
    #update df with imputed values
    df[knn_cols] = imputed_df
    print(f"Applied KNN imputation (k={n_neighbors}, weights={weights})")
    return df

#function to scale auxiliary variables back down to normal
def unscale_aux_vars(df, scale_factors):
    """
    Reverse scaling of auxiliary variables before saving.
    
    Parameters:
    df: DataFrame
    Example scale factors: {"Age": 2, "Sex": 2}
    
    Returns:
    - df with variables scaled back down
    """
    for col, factor in scale_factors.items():
        if col in df.columns:
            df[col] = df[col] / factor
            print(f"Unscaled {col} by dividing by {factor}")
    return df



#function to save imputed dataset as csv
def save_csv(df, filename):
    """Save cleaned DataFrame to CSV."""
    df.to_csv(filename, index=False)
    print(f"Saved cleaned dataset to {filename}")




