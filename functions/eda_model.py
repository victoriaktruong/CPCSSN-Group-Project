import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# only this function uses in main file
# encode variables to factor and orders
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

# Several kinds of Plots supporting for grouped data
def plot_by_group(
    df, 
    y_vars, 
    x='', 
    hue=None, 
    plot_type='box', 
    palette='Set2',
    figsize=(8, 5),
    bins=20,
    kde=False
):

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

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
