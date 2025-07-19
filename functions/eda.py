#load packages
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def summarize_dataframes(df, name="dataset"):
    """
    Print concise summary of DataFrame structure and content.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - name (str): Name of the dataset for display.

    Returns:
    - None
    """
    try:
        print(f"\n{name} summary:")
        print(df.info())
        print()
    except Exception as e:
        print(f"Error in summarize_dataframes: {e}")

def summarize_numeric_statistics(df, numeric_columns, name="dataset"):
    """
    Summarize basic statistics for numeric columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - numeric_columns (list): List of numeric column names.
    - name (str): Name of the dataset for display.

    Returns:
    - pd.DataFrame: Summary statistics
    """
    try:
        missing_cols = [col for col in numeric_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame")
        non_numeric_cols = [col for col in numeric_columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric_cols:
            print(f"Warning: Columns {non_numeric_cols} are not numeric and will be skipped")
            numeric_columns = [col for col in numeric_columns if col not in non_numeric_cols]
        
        if not numeric_columns:
            raise ValueError("No valid numeric columns to summarize")
        
        summaries = df[numeric_columns].describe()
        print(f"\n{name} statistics:")
        print(summaries)
        return summaries
    except Exception as e:
        print(f"Error in summarize_numeric_statistics: {e}")
        return None

def plot_line_chart(df, x_col, y_col, title="Line Chart", xlabel="X", ylabel="Y"):
    """
    Plot a line chart for two continuous variables.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - x_col (str): X-axis column (e.g., Date).
    - y_col (str): Y-axis column (e.g., Price).
    - title (str): Plot title.
    - xlabel (str): X-axis label.
    - ylabel (str): Y-axis label.

    Returns:
    - None
    """
    try:
        plt.figure(figsize=(14, 10))
        plt.plot(df[x_col], df[y_col])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_line_chart: {e}")

def plot_histograms(df, numeric_cols, bins=10):
    """
    Plot histograms for numeric columns.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - numeric_cols (list): Numeric column names.
    - bins (int): Number of bins for histograms.

    Returns:
    - None
    """
    try:
        n_cols = min(len(numeric_cols), 4)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(numeric_cols, 1):
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                plt.subplot(n_rows, n_cols, i)
                n, bins, patches = plt.hist(df[col].dropna(), bins=bins, density=1)
                plt.title(col.replace('_', ' '))
                
                # Add normal distribution
                mu = df[col].mean()
                sigma = df[col].std()
                y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((bins - mu) / sigma)**2))
                plt.plot(bins, y, '--', label='Normal Dist')
                plt.axvline(x=mu, linewidth=3, color='g', label='Mean')
                plt.legend()
            else:
                print(f"Warning: Skipping {col} (not found or not numeric)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_histograms: {e}")

def plot_boxplots(df, numeric_cols):
    """
    Plot boxplots for numeric columns.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - numeric_cols (list): Numeric column names.

    Returns:
    - None
    """
    try:
        n_cols = min(len(numeric_cols), 4)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(numeric_cols, 1):
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                plt.subplot(n_rows, n_cols, i)
                plt.boxplot(df[col].dropna())
                plt.title(col.replace('_', ' '))
            else:
                print(f"Warning: Skipping {col} (not found or not numeric)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_boxplots: {e}")

def plot_correlation_matrix(df, numeric_cols):
    """
    Plot correlation matrix for numeric columns.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - numeric_cols (list): Numeric column names.

    Returns:
    - pd.DataFrame: Correlation matrix
    """
    try:
        valid_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if not valid_cols:
            raise ValueError("No valid numeric columns for correlation matrix")
        
        corr_matrix = df[valid_cols].corr()
        print("\nCorrelation matrix:")
        print(corr_matrix)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        return corr_matrix
    except Exception as e:
        print(f"Error in plot_correlation_matrix: {e}")
        return None

def plot_scatter_pairs(df, col_pairs):
    """
    Plot scatter plots for column pairs.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - col_pairs (list): List of (x_col, y_col, title) tuples.

    Returns:
    - None
    """
    try:
        plt.figure(figsize=(10, 4))
        for i, (x_col, y_col, title) in enumerate(col_pairs, 1):
            if x_col in df.columns and y_col in df.columns:
                plt.subplot(1, min(len(col_pairs), 2), i)
                plt.scatter(df[x_col], df[y_col], alpha=0.5)
                plt.xlabel(x_col.replace('_', ' '))
                plt.ylabel(y_col.replace('_', ' '))
                plt.title(title)
            else:
                print(f"Warning: Skipping pair ({x_col}, {y_col}) (columns not found)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_scatter_pairs: {e}")

def plot_bubble_chart(df, x_col, y_col, size_col, color_col=None, title="Bubble Chart"):
    """
    Plot a bubble chart for two continuous variables with size and optional color.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - x_col (str): X-axis column.
    - y_col (str): Y-axis column.
    - size_col (str): Column for bubble size.
    - color_col (str, optional): Column for bubble color.
    - title (str): Plot title.

    Returns:
    - None
    """
    try:
        plt.figure(figsize=(8, 6))
        if color_col:
            plt.scatter(df[x_col], df[y_col], s=50*df[size_col], c=df[color_col], alpha=0.3)
        else:
            plt.scatter(df[x_col], df[y_col], s=50*df[size_col], alpha=0.3)
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel(y_col.replace('_', ' '))
        plt.title(title)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_bubble_chart: {e}")

def plot_bar_chart(df, x_col, y_col, title="Bar Chart"):
    """
    Plot a bar chart for categorical vs. numeric variables.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - x_col (str): Categorical column for x-axis.
    - y_col (str): Numeric column for y-axis.
    - title (str): Plot title.

    Returns:
    - None
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(df[x_col], df[y_col])
        plt.xticks(rotation=20)
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel(y_col.replace('_', ' '))
        plt.title(title)
        for i, v in enumerate(df[y_col]):
            plt.text(i, v + 0.5, str(int(v)), ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_bar_chart: {e}")

def plot_pie_chart(df, category_col, value_col, title="Pie Chart"):
    """
    Plot a pie chart for categorical data.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - category_col (str): Categorical column.
    - value_col (str): Numeric column for proportions.
    - title (str): Plot title.

    Returns:
    - None
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.pie(df[value_col], labels=df[category_col], autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(title)
        plt.show()
    except Exception as e:
        print(f"Error in plot_pie_chart: {e}")

def plot_stacked_area(df, x_col, y_cols, title="Stacked Area Plot"):
    """
    Plot a stacked area chart for multiple numeric variables over a continuous variable.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - x_col (str): X-axis column (e.g., Date).
    - y_cols (list): List of numeric columns to stack.
    - title (str): Plot title.

    Returns:
    - None
    """
    try:
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(y_cols):
            plt.plot([], [], color=f'C{i}', label=col)
        plt.stackplot(df[x_col], *[df[col] for col in y_cols], colors=[f'C{i}' for i in range(len(y_cols))])
        plt.legend()
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' '))
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_stacked_area: {e}")

def plot_table_chart(df, categories, values, title="Table Chart"):
    """
    Plot a table chart with bars and a table.

    Parameters:
    - df (pd.DataFrame): DataFrame with categories and values.
    - categories (list): List of category names.
    - values (list): List of lists with numeric values.
    - title (str): Plot title.

    Returns:
    - None
    """
    try:
        colors = plt.cm.OrRd(np.linspace(0, 0.7, len(categories)))
        index = np.arange(len(values[0])) + 0.3
        bar_width = 0.7
        y_offset = np.zeros(len(values[0]))
        fig, ax = plt.subplots(figsize=(10, 6))
        cell_text = []
        for row in range(len(values)):
            plt.bar(index, values[row], bar_width, bottom=y_offset, color=colors[row])
            y_offset = y_offset + values[row]
            cell_text.append(['%1.1f' % x for x in y_offset])
            i = 0
            for rect in plt.bar(index, values[row], bar_width, bottom=y_offset):
                ax.text(rect.get_x() + rect.get_width()/2, y_offset[i], '%d' % int(y_offset[i]), ha='center', va='bottom')
                i += 1
        the_table = plt.table(cellText=cell_text, rowLabels=categories, colLabels=[str(i) for i in range(len(values[0]))], loc='bottom')
        plt.ylabel("Values")
        plt.xticks([])
        plt.title(title)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_table_chart: {e}")

def plot_polar_chart(df, theta_col, value_cols, labels, title="Polar Chart"):
    """
    Plot a polar chart for multiple variables.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - theta_col (str): Column for angles (e.g., categories).
    - value_cols (list): List of columns for values.
    - labels (list): Labels for legend.
    - title (str): Plot title.

    Returns:
    - None
    """
    try:
        theta = np.linspace(0, 2 * np.pi, len(df[theta_col]))
        plt.figure(figsize=(10, 6))
        plt.subplot(polar=True)
        (lines, labels_theta) = plt.thetagrids(range(0, 360, int(360/len(df[theta_col]))), df[theta_col])
        for i, col in enumerate(value_cols):
            plt.plot(theta, df[col], label=labels[i])
            plt.fill(theta, df[col], alpha=0.2)
        plt.legend(labels=labels, loc=1)
        plt.title(title)
        plt.show()
    except Exception as e:
        print(f"Error in plot_polar_chart: {e}")

def check_unique_values(df, col):
    """
    Print unique values in a column.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - col (str): Column name.

    Returns:
    - np.ndarray: Unique values
    """
    try:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
        unique_values = df[col].unique()
        print(f"Unique values in {col}: {unique_values}")
        return unique_values
    except Exception as e:
        print(f"Error in check_unique_values: {e}")
        return None


def plot_lollipop_chart(df, group_col, value_col, title="Lollipop Chart", xlabel="Group", ylabel="Value", figsize=(16, 10), color='firebrick', alpha=0.7):
    """
    Plot a lollipop chart to display rankings of mean values by group.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - group_col (str): Categorical column to group by.
    - value_col (str): Numeric column to compute mean.
    - title (str): Plot title.
    - xlabel (str): X-axis label.
    - ylabel (str): Y-axis label.
    - figsize (tuple): Figure size.
    - color (str): Color for lines and markers.
    - alpha (float): Transparency for lines and markers.

    Returns:
    - None
    """
    try:
        # Validate inputs
        if group_col not in df.columns:
            raise ValueError(f"Column {group_col} not found in DataFrame")
        if value_col not in df.columns:
            raise ValueError(f"Column {value_col} not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            raise ValueError(f"Column {value_col} must be numeric")
        if len(df[group_col].unique()) < 1:
            print(f"Warning: Insufficient unique values in {group_col} for lollipop chart")
            return

        # Group by group_col and compute mean of value_col
        processed_df = df[[group_col, value_col]].groupby(group_col).mean().reset_index()
        processed_df.sort_values(value_col, inplace=True)

        # Plot lollipop chart
        fig, ax = plt.subplots(figsize=figsize, dpi=80)
        ax.vlines(x=processed_df.index, ymin=0, ymax=processed_df[value_col], color=color, alpha=alpha, linewidth=2)
        ax.scatter(x=processed_df.index, y=processed_df[value_col], s=75, color=color, alpha=alpha)
        
        # Annotate values
        for row in processed_df.itertuples():
            ax.text(row.Index, getattr(row, value_col) + 0.5, s=f"{getattr(row, value_col):.2f}", 
                    horizontalalignment='center', verticalalignment='bottom', fontsize=14)
        
        # Customize plot
        ax.set_title(title, fontdict={'size': 22})
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(processed_df.index)
        ax.set_xticklabels(processed_df[group_col].str.upper(), rotation=65, fontdict={'horizontalalignment': 'right', 'size': 12})
        ax.set_ylim(0, processed_df[value_col].max() * 1.2)  # Adjust y-limit for visibility
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_lollipop_chart: {e}")


def compute_descriptive_stats(df, numeric_cols):
    """
    Compute and print descriptive statistics for numeric columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - numeric_cols (list): List of numeric column names.

    Returns:
    - dict: Dictionary with statistics for each column.
    """
    stats_dict = {}
    try:
        for col in numeric_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                stats_dict[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'mode': df[col].mode()[0] if not df[col].mode().empty else np.nan,
                    'std': df[col].std(),
                    'variance': df[col].var(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurt(),
                    'percentile_25': np.percentile(df[col].dropna(), 25),
                    'percentile_50': np.percentile(df[col].dropna(), 50),
                    'percentile_75': np.percentile(df[col].dropna(), 75),
                    'IQR': np.percentile(df[col].dropna(), 75) - np.percentile(df[col].dropna(), 25)
                }
                print(f"\nDescriptive statistics for {col}:")
                for stat, value in stats_dict[col].items():
                    print(f"{stat}: {value:.4f}")
            else:
                print(f"Warning: Skipping {col} (not found or not numeric)")
        return stats_dict
    except Exception as e:
        print(f"Error in compute_descriptive_stats: {e}")
        return {}


# Titanic-style correlation analysis function
def perform_titanic_style_correlation_analysis(df, numeric_cols, categorical_cols, target_col=None, pairs=None):
    """
    Perform Titanic-style correlation analysis: univariate, bivariate, and multivariate.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - numeric_cols (list): Numeric column names.
    - categorical_cols (list): Categorical column names.
    - target_col (str): Target column for survival-like analysis (e.g., 'Comorbidity').
    - pairs (list): List of tuples (col1, col2) for specific pair correlations.

    Returns:
    - dict: Dictionary with analysis results.
    """
    try:
        if len(df) < 2:
            print(f"Warning: Insufficient data points ({len(df)}) for reliable correlation analysis")
            return {}
        
        results = {}
        valid_numeric_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        valid_categorical_cols = [col for col in categorical_cols if col in df.columns]

        # Univariate Analysis: Histograms and Box Plots
        print("\n=== Univariate Analysis ===")
        for col in valid_numeric_cols:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.show()

            plt.figure(figsize=(8, 6))
            sns.boxplot(x=col, data=df)
            plt.title(f"Box Plot of {col}")
            plt.xlabel(col)
            plt.show()

        # Bivariate Analysis: Scatter Plots and Box Plots
        print("\n=== Bivariate Analysis ===")
        if pairs:
            for col1, col2 in pairs:
                if col1 in df.columns and col2 in df.columns and pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    plt.figure(figsize=(8, 6))
                    plt.scatter(df[col1], df[col2])
                    plt.title(f"Scatter Plot: {col1} vs {col2}")
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.show()

                    corr, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                    results[(col1, col2)] = {'correlation': corr, 'p_value': p_value}
                    print(f"Correlation between {col1} and {col2}:")
                    print(f"Pearson correlation: {corr:.4f}")
                    print(f"P-value: {p_value:.4e}")

        # Bivariate Analysis: Numeric vs Categorical (Box Plots)
        for cat_col in valid_categorical_cols:
            if len(df[cat_col].unique()) > 0:
                for num_col in valid_numeric_cols:
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(x=cat_col, y=num_col, data=df)
                    plt.title(f"{num_col} by {cat_col}")
                    plt.xlabel(cat_col)
                    plt.ylabel(num_col)
                    plt.show()

        # Multivariate Analysis: Pair Plot and Correlation Heatmap
        print("\n=== Multivariate Analysis ===")
        if len(valid_numeric_cols) >= 2:
            sns.pairplot(df, vars=valid_numeric_cols, hue=target_col if target_col in df.columns else None, kind='reg')
            plt.suptitle("Pair Plot of Numeric Variables", y=1.02)
            plt.show()

            corr_matrix = df[valid_numeric_cols].corr(method='pearson')
            results['correlation_matrix'] = corr_matrix
            print("\nPearson Correlation Matrix:")
            print(corr_matrix)

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns, 
                        annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Error in perform_titanic_style_correlation_analysis: {e}")
        
  