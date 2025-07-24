#!/usr/bin/env python3
# coding: utf-8

# This script contains modular, reusable functions for exploratory data analysis (EDA).
# It was developed using key references:
# 1. Mukhiya & Ahmed (2020) – EDA workflows and summary techniques.
# 2. Oluleye (2023) – Visualization recipes and modular analysis patterns.
# 3. VanderPlas (2022) – Data manipulation and plotting with Python libraries.

# Generative AI tools (e.g., ChatGPT) assisted by:
# - Translating concepts into code tailored to specific needs.
# - Debugging and improving code readability and modularity.
# - Providing quick answers to technical questions.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from typing import List, Dict, Tuple, Optional, Union
from functools import wraps
import os
from datetime import datetime

def validate_columns(df: pd.DataFrame, cols: List[str]) -> None:
    """Validate if columns exist in DataFrame."""
    missing = [col for col in cols if col and col not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found. Available: {list(df.columns)}")

def sample_rows(df: pd.DataFrame, nrows: int, replace: bool = False) -> Optional[pd.DataFrame]:
    """Select a random sample of rows from a DataFrame."""
    try:
        return df.sample(n=nrows, replace=replace, random_state=42)
    except Exception:
        return None

def summarize_dataframes(df: pd.DataFrame, name: str = "dataset") -> None:
    """Print simplified DataFrame summary including shape, columns, and missing values."""
    try:
        print(f"\n=== {name} Summary ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nData Types:\n" + df.dtypes.to_string())
        print("\nMissing Values:\n" + df.isnull().sum().to_string())
        print(f"Total Missing Values: {df.isnull().sum().sum()}")
        print(f"Percentage Missing: {df.isnull().mean().mean() * 100:.2f}%")
    except Exception:
        pass

def plot_distribution(distribution_type: str = 'normal', size: int = 10000, figsize: Tuple[int, int] = (6, 4), **kwargs) -> None:
    """Plot a probability distribution (uniform, normal, exponential, binomial)."""
    try:
        if distribution_type not in ['uniform', 'normal', 'exponential', 'binomial']:
            raise ValueError(f"Unsupported distribution: {distribution_type}")
        plt.figure(figsize=figsize)
        if distribution_type == 'uniform':
            data = stats.uniform.rvs(size=size, loc=kwargs.get('loc', 20), scale=kwargs.get('scale', 25))
            sns.histplot(data, bins=100, kde=True, stat='density')
        elif distribution_type == 'normal':
            data = stats.norm.rvs(size=size, loc=kwargs.get('loc', 20), scale=kwargs.get('scale', 30))
            sns.histplot(data, bins=100, kde=True, stat='density')
        elif distribution_type == 'exponential':
            data = stats.expon.rvs(size=size, scale=kwargs.get('scale', 1), loc=kwargs.get('loc', 0))
            sns.histplot(data, bins=100, kde=True, stat='density')
        elif distribution_type == 'binomial':
            n = kwargs.get('n', 10)
            data = stats.binom.rvs(n=n, p=kwargs.get('p', 0.8), size=size)
            sns.histplot(data, bins=range(n+2), kde=False, stat='probability')
        plt.title(f"{distribution_type.capitalize()} Distribution")
        plt.xlabel("Value")
        plt.ylabel("Density" if distribution_type != 'binomial' else "Probability")
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def plot_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = "Line Chart",
                   xlabel: str = "X", ylabel: str = "Y", figsize: Tuple[int, int] = (6, 4),
                   max_points: int = 1000) -> None:
    """Plot a line chart for two continuous variables."""
    try:
        validate_columns(df, [x_col, y_col])
        if len(df) > max_points:
            df = sample_rows(df, max_points)
            if df is None:
                return
        plt.figure(figsize=figsize)
        plt.plot(df[x_col], df[y_col], linewidth=1.5, marker='o', markersize=4)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def plot_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = "Bar Chart",
                  figsize: Tuple[int, int] = (6, 4), max_points: int = 1000) -> None:
    """Plot a bar chart with value annotations."""
    try:
        validate_columns(df, [x_col, y_col])
        if len(df) > max_points:
            df = sample_rows(df, max_points)
            if df is None:
                return
        plt.figure(figsize=figsize)
        plot = sns.barplot(x=x_col, y=y_col, data=df)
        for rect in plot.patches:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height + 0.5, f'{height:.0f}',
                     ha='center', va='bottom', fontsize=8)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def _jitter(values: Union[pd.Series, np.ndarray], jitter: float = 0.5) -> np.ndarray:
    """Add random noise to values for better visualization in scatter plots."""
    try:
        return values + np.random.uniform(-jitter, jitter, len(values))
    except Exception:
        return values

def plot_scatter_pairs(df: pd.DataFrame, col_pairs: List[Tuple[str, str, str]],
                      hue_col: Optional[str] = None, jitter: float = 0.0, alpha: float = 0.6,
                      log_x: bool = False, log_y: bool = False, figsize: Tuple[int, int] = (10, 3),
                      max_points: int = 1000) -> None:
    """Plot scatter plots for column pairs with regression lines."""
    try:
        cols = [col for pair in col_pairs for col in pair[:2]] + ([hue_col] if hue_col else [])
        validate_columns(df, cols)
        if len(df) > max_points:
            df = sample_rows(df, max_points)
            if df is None:
                return
        plt.figure(figsize=figsize)
        for i, (x_col, y_col, title) in enumerate(col_pairs, 1):
            plt.subplot(1, min(len(col_pairs), 3), i)
            x_data = df[x_col] if not log_x else np.log(df[x_col] + 1)
            y_data = df[y_col] if not log_y else np.log(df[y_col] + 1)
            if jitter > 0:
                x_data = _jitter(x_data, jitter)
                y_data = _jitter(y_data, jitter)
            sns.scatterplot(x=x_data, y=y_data, hue=df[hue_col] if hue_col else None, alpha=alpha)
            sns.regplot(x=x_data, y=y_data, scatter=False)
            plt.xlabel(x_col.replace('_', ' ').title() + (' (log)' if log_x else ''))
            plt.ylabel(y_col.replace('_', ' ').title() + (' (log)' if log_y else ''))
            plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def plot_bubble_chart(df: pd.DataFrame, x_col: str, y_col: str, size_col: str,
                     color_col: Optional[str] = None, title: str = "Bubble Chart",
                     jitter: float = 0.0, alpha: float = 0.6, log_x: bool = False, log_y: bool = False,
                     figsize: Tuple[int, int] = (6, 4), max_points: int = 1000) -> None:
    """Plot a bubble chart with size and color encoding."""
    try:
        validate_columns(df, [x_col, y_col, size_col] + ([color_col] if color_col else []))
        if len(df) > max_points:
            df = sample_rows(df, max_points)
            if df is None:
                return
        plt.figure(figsize=figsize)
        x_data = df[x_col] if not log_x else np.log(df[x_col] + 1)
        y_data = df[y_col] if not log_y else np.log(df[y_col] + 1)
        if jitter > 0:
            x_data = _jitter(x_data, jitter)
            y_data = _jitter(y_data, jitter)
        size_range = df[size_col].max() - df[size_col].min()
        size = 50 * (df[size_col] - df[size_col].min()) / (size_range if size_range != 0 else 1)
        sns.scatterplot(x=x_data, y=y_data, size=size, hue=df[color_col] if color_col else None, alpha=alpha)
        plt.xlabel(x_col.replace('_', ' ').title() + (' (log)' if log_x else ''))
        plt.ylabel(y_col.replace('_', ' ').title() + (' (log)' if log_y else ''))
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def plot_hexbin(df: pd.DataFrame, x_col: str, y_col: str, title: str = "Hexbin Plot",
                gridsize: int = 20, figsize: Tuple[int, int] = (6, 4), max_points: int = 1000) -> None:
    """Plot a hexbin density plot."""
    try:
        validate_columns(df, [x_col, y_col])
        if len(df) > max_points:
            df = sample_rows(df, max_points)
            if df is None:
                return
        plt.figure(figsize=figsize)
        plt.hexbin(df[x_col], df[y_col], gridsize=gridsize, cmap='Greys')
        plt.colorbar(label='Count')
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def plot_percentiles(df: pd.DataFrame, x_col: str, y_col: str, bins: int = 10, percentiles: List[int] = [25, 50, 75],
                    title: str = "Percentile Plot", figsize: Tuple[int, int] = (6, 4), max_points: int = 1000) -> None:
    """Plot percentiles of y_col across bins of x_col."""
    try:
        validate_columns(df, [x_col, y_col])
        if len(df) > max_points:
            df = sample_rows(df, max_points)
            if df is None:
                return
        df_clean = df[[x_col, y_col]].dropna()
        bin_edges = np.linspace(df_clean[x_col].min(), df_clean[x_col].max(), bins + 1)
        indices = np.digitize(df_clean[x_col], bin_edges)
        groups = df_clean.groupby(indices)
        x_means = [group[x_col].mean() if not group.empty else np.nan for _, group in groups]
        plt.figure(figsize=figsize)
        for percent in percentiles:
            y_values = [np.percentile(group[y_col], percent) if not group.empty else np.nan for _, group in groups]
            plt.plot(x_means, y_values, label=f'{percent}th Percentile')
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def plot_pie_chart(df: pd.DataFrame, category_col: str, value_col: str, title: str = "Pie Chart",
                  figsize: Tuple[int, int] = (6, 4)) -> None:
    """Plot a pie chart for categorical data."""
    try:
        validate_columns(df, [category_col, value_col])
        plt.figure(figsize=figsize)
        plt.pie(df[value_col], labels=df[category_col], autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(title)
        plt.show()
        plt.close()
    except Exception:
        pass

def plot_stacked_area(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str = "Stacked Area Plot",
                      figsize: Tuple[int, int] = (6, 4), max_points: int = 1000) -> None:
    """Plot a stacked area chart for multiple numeric variables."""
    try:
        validate_columns(df, [x_col] + y_cols)
        if len(df) > max_points:
            df = sample_rows(df, max_points)
            if df is None:
                return
        plt.figure(figsize=figsize)
        plt.stackplot(df[x_col], *[df[col] for col in y_cols], labels=y_cols, alpha=0.7)
        plt.legend(loc='upper left')
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def plot_table_chart(df: pd.DataFrame, row_labels: List[str], col_labels: List[str], data: List[List[float]],
                    title: str = "Table Chart", figsize: Tuple[int, int] = (8, 6)) -> None:
    """Plot a table chart combining bar chart and table."""
    try:
        if len(data) != len(row_labels) or any(len(row) != len(col_labels) for row in data):
            raise ValueError("Data dimensions must match row_labels and col_labels")
        plt.figure(figsize=figsize)
        colors = plt.cm.Greys(np.linspace(0.2, 0.7, len(row_labels)))
        index = np.arange(len(col_labels)) + 0.3
        bar_width = 0.7
        y_offset = np.zeros(len(col_labels))
        cell_text = []
        for row in range(len(data)):
            plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
            y_offset += data[row]
            cell_text.append([f'{x:.1f}' for x in y_offset])
        plt.table(cellText=cell_text, rowLabels=row_labels, rowColours=colors,
                  colLabels=col_labels, loc='bottom')
        plt.ylabel("Units")
        plt.xticks([])
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def plot_polar_chart(labels: List[str], values1: List[float], values2: Optional[List[float]] = None,
                     title: str = "Polar Chart", legend_labels: Tuple[str, str] = ("Series 1", "Series 2"),
                     figsize: Tuple[int, int] = (6, 4)) -> None:
    """Plot a polar chart for comparing two sets of values."""
    try:
        if not all(isinstance(v, (int, float)) for v in values1) or (values2 and not all(isinstance(v, (int, float)) for v in values2)):
            raise ValueError("Values must be numeric")
        if len(labels) != len(values1) or (values2 and len(labels) != len(values2)):
            raise ValueError("Labels and values must have the same length")
        values1 = values1 + [values1[0]]
        if values2:
            values2 = values2 + [values2[0]]
        theta = np.linspace(0, 2 * np.pi, len(labels) + 1)
        plt.figure(figsize=figsize)
        ax = plt.subplot(polar=True)
        ax.plot(theta, values1, label=legend_labels[0])
        ax.fill(theta, values1, alpha=0.2)
        if values2:
            ax.plot(theta, values2, label=legend_labels[1])
        ax.set_thetagrids(np.degrees(theta[:-1]), labels)
        plt.legend(loc='upper right')
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def plot_histograms(df: pd.DataFrame, numeric_cols: List[str], bins: int = 30, kde: bool = True,
                   show_normal: bool = False, color: Optional[str] = 'gray',
                   figsize: Tuple[int, int] = (10, 3), max_points: int = 1000) -> None:
    """Plot histograms for numeric columns with optional normal distribution overlay and custom color."""
    try:
        validate_columns(df, numeric_cols)
        if len(df) > max_points:
            df = sample_rows(df, max_points)
            if df is None:
                return
        n_cols = min(len(numeric_cols), 4)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        plt.figure(figsize=(figsize[0], figsize[1] * n_rows))
        for i, col in enumerate(numeric_cols, 1):
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                plt.subplot(n_rows, n_cols, i)
                sns.histplot(df[col].dropna(), bins=bins, kde=kde, stat='density',
                            color=color, edgecolor='black')
                if show_normal:
                    mu = df[col].mean()
                    sigma = df[col].std()
                    x = np.linspace(df[col].min(), df[col].max(), 100)
                    plt.plot(x, stats.norm.pdf(x, mu, sigma), '--', label='Normal Dist',
                            color='black')
                    plt.axvline(x=mu, linewidth=1.5, color='black', label='Mean')
                plt.title(col.replace('_', ' ').title())
                plt.xlabel(col)
                plt.ylabel('Density')
                plt.legend()
            else:
                continue
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def plot_lollipop_chart(df: pd.DataFrame, group_col: str, value_col: str, title: str = "Lollipop Chart",
                       xlabel: str = "Group", ylabel: str = "Value", figsize: Tuple[int, int] = (6, 4),
                       max_points: int = 1000) -> None:
    """Plot a lollipop chart to display rankings of mean values by group."""
    try:
        validate_columns(df, [group_col, value_col])
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            raise ValueError(f"Column {value_col} must be numeric")
        if len(df) > max_points:
            df = sample_rows(df, max_points)
            if df is None:
                return
        processed_df = df[[group_col, value_col]].groupby(group_col).mean().reset_index().sort_values(value_col)
        processed_df[group_col] = processed_df[group_col].astype(str)
        plt.figure(figsize=figsize)
        plt.vlines(x=range(len(processed_df)), ymin=0, ymax=processed_df[value_col], color='black', alpha=0.7, linewidth=1.5)
        plt.scatter(range(len(processed_df)), processed_df[value_col], s=50, color='black', alpha=0.7)
        for i, val in enumerate(processed_df[value_col]):
            plt.text(i, val + 0.5, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(range(len(processed_df)), processed_df[group_col].str.title(), rotation=45, ha='right')
        plt.ylim(0, processed_df[value_col].max() * 1.2)
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def plot_violin_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = "Violin Plot",
                     xlabel: str = "Category", ylabel: str = "Value", figsize: Tuple[int, int] = (6, 4),
                     max_points: int = 1000) -> None:
    """Plot a violin chart to show distribution of a numeric variable across categories."""
    try:
        validate_columns(df, [x_col, y_col])
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            raise ValueError(f"Column {y_col} must be numeric")
        if len(df) > max_points:
            df = sample_rows(df, max_points)
            if df is None:
                return
        plt.figure(figsize=figsize)
        sns.violinplot(x=x_col, y=y_col, data=df)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception:
        pass

def compute_correlation(df: pd.DataFrame, x_col: str, y_col: str, corr_type: str = 'pearson',
                       log_x: bool = False, log_y: bool = False) -> Optional[float]:
    """Compute correlation coefficient (Pearson or Spearman)."""
    try:
        validate_columns(df, [x_col, y_col])
        x_data = df[x_col] if not log_x else np.log(df[x_col] + 1)
        y_data = df[y_col] if not log_y else np.log(df[y_col] + 1)
        df_clean = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
        if df_clean.empty:
            raise ValueError("No valid data after dropping NaNs")
        if corr_type == 'pearson':
            corr = df_clean['x'].corr(df_clean['y'], method='pearson')
        elif corr_type == 'spearman':
            corr = df_clean['x'].corr(df_clean['y'], method='spearman')
        else:
            raise ValueError(f"Unsupported correlation type: {corr_type}")
        print(f"{corr_type.capitalize()} correlation between {x_col} and {y_col}: {corr:.4f}")
        return corr
    except Exception:
        return None

def compute_descriptive_stats(df: pd.DataFrame, numeric_cols: List[str]) -> Optional[Dict[str, Dict]]:
    """Compute comprehensive descriptive statistics for numeric columns."""
    try:
        stats_dict = {}
        for col in numeric_cols:
            if col in df.columns:
                try:
                    data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if data.empty:
                        continue
                    Q1 = np.percentile(data, 25)
                    Q3 = np.percentile(data, 75)
                    stats_dict[col] = {
                        'count': data.count(),
                        'mean': data.mean(),
                        'median': data.median(),
                        'mode': data.mode()[0] if not data.mode().empty else np.nan,
                        'std': data.std(),
                        'variance': data.var(),
                        'skewness': data.skew(),
                        'kurtosis': data.kurt(),
                        'min': data.min(),
                        'max': data.max(),
                        'percentile_25': Q1,
                        'percentile_50': np.percentile(data, 50),
                        'percentile_75': Q3,
                        'iqr': Q3 - Q1,
                        'missing_count': df[col].isnull().sum(),
                        'missing_percentage': df[col].isnull().mean() * 100
                    }
                    print(f"\n=== Descriptive Statistics for {col} ===")
                    for stat, value in stats_dict[col].items():
                        print(f"{stat}: {value:.4f}")
                except Exception:
                    continue
            else:
                continue
        return stats_dict
    except Exception:
        return None

def perform_groupby_analysis(df: pd.DataFrame, group_cols: List[str], agg_cols: List[str],
                           agg_funcs: Dict[str, Union[str, List[str]]]) -> Optional[pd.DataFrame]:
    """Perform groupby analysis with specified aggregation functions."""
    try:
        validate_columns(df, group_cols + agg_cols)
        grouped = df.groupby(group_cols)[agg_cols].agg(agg_funcs).reset_index()
        print(f"\n=== Groupby Analysis by {', '.join(group_cols)} ===")
        print(grouped)
        return grouped
    except Exception:
        return None

def create_crosstab(df: pd.DataFrame, index: Union[str, List[str]], columns: Union[str, List[str]],
                   values: Optional[str] = None, aggfunc: Optional[str] = None,
                   normalize: bool = False, margins: bool = False, margins_name: str = "Total",
                   figsize: Tuple[int, int] = (6, 5)) -> Optional[pd.DataFrame]:
    """Create a cross-tabulation table with optional aggregation and normalization."""
    try:
        if isinstance(index, str):
            index = [index]
        if isinstance(columns, str):
            columns = [columns]
        validate_columns(df, index + columns + ([values] if values else []))
        crosstab = pd.crosstab(index=[df[col] for col in index],
                              columns=[df[col] for col in columns],
                              values=df[values] if values else None,
                              aggfunc=aggfunc,
                              normalize=normalize if not values else False,
                              margins=margins,
                              margins_name=margins_name)
        print(f"\n=== Cross-Tabulation ===")
        print(crosstab)
        plt.figure(figsize=figsize)
        sns.heatmap(crosstab, annot=True, cmap='Greys', fmt='.2f' if values else 'd')
        plt.title("Cross-Tabulation Heatmap")
        plt.tight_layout()
        plt.show()
        plt.close()
        return crosstab
    except Exception:
        return None

def recommend_chart_type(data_types: Dict[str, str], purpose: str) -> List[str]:
    """Recommend chart types based on data types and analysis purpose."""
    try:
        continuous_cols = [col for col, dtype in data_types.items() if dtype == 'continuous']
        categorical_cols = [col for col, dtype in data_types.items() if dtype == 'categorical']
        recommendations = []
        if purpose == 'correlation' and len(continuous_cols) >= 2:
            recommendations.extend(['Scatter Plot', 'Bubble Plot', 'Hexbin Plot', 'Percentile Plot', 'Correlation Matrix'])
        if purpose == 'distribution' and continuous_cols:
            recommendations.extend(['Histogram', 'Density Plot', 'Box Plot'])
        if purpose == 'composition' and categorical_cols and continuous_cols:
            recommendations.extend(['Pie Chart', 'Bar Chart', 'Table Chart'])
        if purpose == 'change' and continuous_cols:
            recommendations.extend(['Line Chart', 'Stacked Area Plot'])
        if purpose == 'ranking' and categorical_cols and continuous_cols:
            recommendations.extend(['Lollipop Chart', 'Bar Chart'])
        if purpose == 'groups' and categorical_cols:
            recommendations.extend(['Bar Chart', 'Cross-Tabulation', 'Table Chart'])
        if not recommendations:
            return ['Bar Chart', 'Scatter Plot']
        return recommendations
    except Exception:
        return ['Bar Chart', 'Scatter Plot']

def perform_correlation_analysis(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str],
                                target_col: Optional[str] = None, pairs: Optional[List[Tuple[str, str]]] = None,
                                figsize: Tuple[int, int] = (8, 6), max_points: int = 1000) -> Optional[Dict]:
    """Perform correlation analysis: univariate, bivariate, and multivariate."""
    try:
        if len(df) < 2:
            return None
        if len(df) > max_points:
            df = sample_rows(df, max_points)
            if df is None:
                return None
        results = {}
        valid_numeric_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        valid_categorical_cols = [col for col in categorical_cols if col in df.columns]

        # Univariate Analysis: Histograms and Box Plots
        print("\n=== Univariate Analysis ===")
        for col in valid_numeric_cols:
            plt.figure(figsize=figsize)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()
            plt.close()

            plt.figure(figsize=figsize)
            sns.boxplot(x=col, data=df)
            plt.title(f"Box Plot of {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()
            plt.close()

        # Bivariate Analysis: Scatter Plots and Box Plots
        print("\n=== Bivariate Analysis ===")
        if pairs:
            for col1, col2 in pairs:
                if col1 in df.columns and col2 in df.columns and pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    plt.figure(figsize=figsize)
                    sns.scatterplot(x=df[col1], y=df[col2])
                    plt.title(f"Scatter Plot: {col1} vs {col2}")
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                    corr, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                    results[(col1, col2)] = {'correlation': corr, 'p_value': p_value}
                    print(f"Correlation between {col1} and {col2}:")
                    print(f"Pearson correlation: {corr:.4f}")
                    print(f"P-value: {p_value:.4e}")

        # Bivariate Analysis: Numeric vs Categorical (Box Plots)
        for cat_col in valid_categorical_cols:
            if len(df[cat_col].unique()) > 0:
                for num_col in valid_numeric_cols:
                    plt.figure(figsize=figsize)
                    sns.boxplot(x=cat_col, y=num_col, data=df)
                    plt.title(f"{num_col} by {cat_col}")
                    plt.xlabel(cat_col)
                    plt.ylabel(num_col)
                    plt.tight_layout()
                    plt.show()
                    plt.close()

        # Multivariate Analysis: Pair Plot and Correlation Heatmap
        print("\n=== Multivariate Analysis ===")
        if len(valid_numeric_cols) >= 2:
            sns.pairplot(df, vars=valid_numeric_cols, hue=target_col if target_col in df.columns else None, kind='reg')
            plt.suptitle("Pair Plot of Numeric Variables", y=1.02)
            plt.show()
            plt.close()

            corr_matrix = df[valid_numeric_cols].corr(method='pearson')
            results['correlation_matrix'] = corr_matrix
            print("\nPearson Correlation Matrix:")
            print(corr_matrix)

            plt.figure(figsize=(figsize[0] + 2, figsize[1] + 2))
            sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns, 
                        annot=True, cmap='Greys', vmin=-1, vmax=1)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.show()
            plt.close()
        return results
    except Exception:
        return None

def perform_levene_test(df: pd.DataFrame, numeric_cols: List[str], group_col: str, group1: str, group2: str,
                       alpha: float = 0.05) -> Optional[Dict[str, Dict]]:
    """Perform Levene's test for variance equality between two groups for specified numeric columns."""
    try:
        validate_columns(df, numeric_cols + [group_col])
        results = {}
        
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # Extract data for each group and drop NaN values
            group1_data = df[df[group_col] == group1][col].dropna()
            group2_data = df[df[group_col] == group2][col].dropna()
            
            # Initialize result dictionary for this column
            results[col] = {
                'statistic': None,
                'p_value': None,
                'interpretation': None,
                'sample_size_group1': len(group1_data),
                'sample_size_group2': len(group2_data)
            }
            
            # Check if there is sufficient data
            if len(group1_data) > 1 and len(group2_data) > 1:
                # Perform Levene's test
                stat, p_value = stats.levene(group1_data, group2_data)
                results[col]['statistic'] = stat
                results[col]['p_value'] = p_value
                
                # Print results
                print(f"\n=== Levene's Test for {col} ===")
                print(f"Statistic: {stat:.4f}")
                print(f"P-value: {p_value:.4e}")
                
                # Interpret results
                if p_value < alpha:
                    interpretation = (
                        f"Reject null hypothesis: Variances are unequal for {col} "
                        f"between {group1} and {group2}"
                    )
                    print(interpretation)
                else:
                    interpretation = (
                        f"Fail to reject null hypothesis: Variances are equal for {col} "
                        f"between {group1} and {group2}"
                    )
                    print(interpretation)
                results[col]['interpretation'] = interpretation
            else:
                interpretation = (
                    f"Skipping Levene's test for {col}: Insufficient data "
                    f"(Sample size {group1}: {len(group1_data)}, {group2}: {len(group2_data)})"
                )
                print(interpretation)
                results[col]['interpretation'] = interpretation
                
        return results
    except Exception:
        return None