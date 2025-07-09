import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Map diagnosis codes to broader state categories and encode as ordered categorical and numeric.
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

# Plot individual trajectories over time for a list of variables.
def plot_trends(
    data,
    variables,
    date='Date',
    hue='ID',
    xlabel='Date',
    ylabel='Value'
):
    data[date] = pd.to_datetime(data[date])

    n = len(variables)
    rows = (n + 1) // 2
    plt.figure(figsize=(15, 5 * rows))

    for i, var in enumerate(variables, 1):
        plt.subplot(rows, 2, i)
        sns.lineplot(data=data, x=date, y=var, hue=hue, alpha=0.3, legend=None)
        plt.title(var)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()

# Plot trends of lab variables over grouped axis (e.g., VisitOrder), grouped by hue (e.g., State).
def plot_grouped_trends(
    data,
    variables,
    group_x='group_x',
    group_hue='group_hue',
    xlabel='x',
    ylabel='y',
    title_prefix='',
    dropna=True
):
    n = len(variables)
    rows = (n + 1) // 2
    plt.figure(figsize=(15, 5 * rows))

    for i, var in enumerate(variables, 1):
        plt.subplot(rows, 2, i)

        summary = data.groupby([group_hue, group_x])[var].mean().reset_index()

        if dropna:
            summary = summary.dropna(subset=[group_hue, var])

        if summary.empty:
            print(f"[Skip] No data for {var}")
            continue

        sns.lineplot(data=summary, x=group_x, y=var, hue=group_hue, marker='o')
        plt.title(f'{title_prefix}{var} Trend by {group_hue}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(title=group_hue, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

# Plot histograms of lab variables grouped by a group
def plot_grouped_histograms(data, variables, group='group', 
                        bins='auto', element='step', stat='density',
                        common_norm=False, figsize=(15, 20), layout=(4, 2)):
    n_rows, n_cols = layout
    total_plots = len(variables)
    
    plt.figure(figsize=figsize)
    for i, var in enumerate(variables, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(data=data, x=var, hue=group,
                     bins=bins, element=element, stat=stat,
                     common_norm=common_norm)
        plt.title(f'Histogram of {var} by {group}')
        plt.xlabel(var)
        plt.ylabel(stat.capitalize())
    plt.tight_layout()
    plt.show()

# Plot boxplots of lab variables grouped by a hue
def plot_grouped_boxplots(data, variables, group='group', 
                          figsize=(15, 20), layout=(4, 2), orient='v'):
    n_rows, n_cols = layout
    total_plots = len(variables)

    plt.figure(figsize=figsize)
    for i, var in enumerate(variables, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(data=data, x=group if orient == 'v' else var,
                    y=var if orient == 'v' else group)
        plt.title(f'Boxplot of {var} by {group}')
        plt.xlabel(group if orient == 'v' else var)
        plt.ylabel(var if orient == 'v' else group)
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
