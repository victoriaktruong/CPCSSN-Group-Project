import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Map one variable to categories and encode as ordered categorical or numeric.
def map_encode(
    data,
    code='code',
    mapping_dict=None,
    ordered_levels=None,
    new_label='new_label',
    new_factor='new_factor',
    new_num='new_num'
):
    data = data.copy()

    data[new_label] = data[code].astype(str).map(mapping_dict)

    data[new_factor] = pd.Categorical(
        data[new_label],
        categories=ordered_levels,
        ordered=True
    )

    data[new_num] = data[new_factor].cat.codes

    return data

import seaborn as sns
import matplotlib.pyplot as plt

# Plot by Group
import seaborn as sns
import matplotlib.pyplot as plt

def plot_by_group(
    df, 
    y_vars, 
    x='BD_Code', 
    hue=None, 
    plot_type='box', 
    palette='Set2',
    figsize=(8, 5),
    bins=20,
    kde=False
):
    """
    Flexible grouped plotting function with support for histogram.

    Parameters:
    - df: pandas DataFrame
    - y_vars: list of str, variables to plot on y-axis
    - x: str, grouping variable (default: 'BD_Code')
    - hue: str or None, optional second grouping variable (e.g., 'Sex')
    - plot_type: str, one of 'box', 'violin', 'bar', or 'hist'
    - palette: str, seaborn color palette
    - figsize: tuple, figure size
    - bins: int, number of bins for histograms
    - kde: bool, whether to show KDE in histogram
    """

    for y in y_vars:
        plt.figure(figsize=figsize)

        if plot_type == 'box':
            sns.boxplot(data=df, x=x, y=y, hue=hue, palette=palette)
            plt.title(f'{y} by {x} (Boxplot)')
        elif plot_type == 'violin':
            sns.violinplot(data=df, x=x, y=y, hue=hue, palette=palette, inner='box')
            plt.title(f'{y} by {x} (Violinplot)')
        elif plot_type == 'bar':
            sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette, errorbar='ci')
            plt.title(f'{y} by {x} (Barplot with CI)')
        elif plot_type == 'hist':
            sns.histplot(data=df, x=y, hue=hue or x, palette=palette, bins=bins, kde=kde, multiple='stack')
            plt.title(f'Distribution of {y} by {hue or x} (Histogram)')
        else:
            raise ValueError("plot_type must be one of 'box', 'violin', 'bar', or 'hist'")

        plt.tight_layout()
        plt.show()



# Kruskal-Wallis test for each variable across groups
from scipy.stats import kruskal

def kruskal_test(
    data,
    variables,
    group_col,
    min_group_size=3,
    verbose=True,
    return_effect_size=False,
    export_path=None
):

    results = []

    for var in variables:
        # Drop missing values in both variable and group column
        df_var = data.dropna(subset=[var, group_col])
        group_counts = df_var.groupby(group_col)[var].count()

        # Keep only groups with at least min_group_size
        valid_groups = group_counts[group_counts >= min_group_size].index.tolist()

        filtered_groups = []
        for g in valid_groups:
            values = df_var[df_var[group_col] == g][var].values
            if len(np.unique(values)) > 1: 
                filtered_groups.append(g)

        if len(filtered_groups) < 2:
            if verbose:
                print(f"{var}: Not enough valid groups for comparison (n_groups < 2).")
            results.append({
                'variable': var,
                'n_groups': len(filtered_groups),
                'H_stat': np.nan,
                'p_value': np.nan
            })
            continue

        try:
            groups = [df_var[df_var[group_col] == g][var].values for g in filtered_groups]
            stat, p = kruskal(*groups)
            N = sum(len(g) for g in groups)
            eta_sq = stat / (N - 1) if return_effect_size else None

            if verbose:
                print(f'{var}: H={stat:.3f}, p={p:.4f}' + (f", η²={eta_sq:.4f}" if return_effect_size else ""))

            results.append({
                'variable': var,
                'n_groups': len(filtered_groups),
                'H_stat': stat,
                'p_value': p
            })
        except ValueError as e:
            if verbose:
                print(f"{var}: Test failed - {e}")
            results.append({
                'variable': var,
                'n_groups': len(filtered_groups),
                'H_stat': np.nan,
                'p_value': np.nan
            })

    result_df = pd.DataFrame(results)

    if export_path:
        result_df.to_csv(export_path, index=False)
        if verbose:
            print(f"Results saved to {export_path}")

    return result_df
