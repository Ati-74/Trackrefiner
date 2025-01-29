import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_frequency(array, feature_related_col_name):
    """
    Calculate the frequency of unique values in a given array and return the result as a DataFrame.

    This function computes the frequency of each unique value in the input array and returns
    the results in a pandas DataFrame with columns for the unique values and their respective frequencies.

    :param numpy.ndarray array:
        A 1D array of values for which the frequency distribution is to be calculated.
    :param str feature_related_col_name:
        The name to use for the column representing unique values in the output DataFrame.

    :returns:
        pandas.DataFrame: A DataFrame with the following columns:
        - '<feature_related_col_name>': The unique values from the array.
        - 'Frequency': The count of each unique value.
    """

    # Get unique values and their counts
    unique_values, counts = np.unique(array, return_counts=True)

    # Create a DataFrame
    df = pd.DataFrame({
        feature_related_col_name: unique_values,
        'Frequency': counts
    })

    return df


def plot_frequency_distribution(x_positions, grouped_sel_feature, bin_labels, feature, feature_name, y_axis_value,
                                output_path):
    """
    Plot the frequency distribution of a selected feature and save the output.

    :param numpy.ndarray x_positions:
        An array of x-axis positions for the grouped frequency bars.
    :param pandas.DataFrame grouped_sel_feature:
        A DataFrame containing grouped frequency data to plot.
    :param list bin_labels:
        Labels for the x-axis bins.
    :param str feature:
        The name of the feature being plotted.
    :param str feature_name:
        A descriptive label for the feature (used for x-axis labeling).
    :param str y_axis_value:
        A descriptive label for y-axis labeling.
    :param str output_path:
        The directory path where the plot will be saved.
    """

    # Plot configuration
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for cp_omnipose_abs_error and tr_abs_error
    ax.bar(x_positions, grouped_sel_feature["Frequency"], label="Trackrefiner", color="blue")

    # Add labels and title
    ax.set_xlabel(feature_name, fontsize=16, fontweight='bold', labelpad=15)
    ax.set_ylabel(y_axis_value, fontsize=16, fontweight='bold', labelpad=15)
    ax.set_title('')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bin_labels, rotation=90)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{output_path}/plots/{feature}.jpg', dpi=600)
    plt.clf()
    plt.close()


def draw_feature_distribution(df, features_dict, label_col, interval_time, doubling_time, output_path):
    """
    Generate and plot the distribution of selected features from a DataFrame.

    This function iterates through a dictionary of features, calculates their frequency distribution,
    assigns them to bins, and visualizes the distribution with bar plots.

    :param pandas.DataFrame df:
        The input DataFrame containing feature data to analyze.
    :param dict features_dict:
        A dictionary mapping feature column names to descriptive labels and type of analysis for plotting.
    :param str label_col:
        Name of the column for object labels.
    :param float interval_time:
        Time interval to scale the 'LifeHistory' feature.
    :param float doubling_time:
        Threshold for filtering 'LifeHistory' values.
    :param str output_path:
        The directory path where the plots will be saved.

    :returns:
        None. The plots are saved as JPG files in the specified output directory.
    """

    os.makedirs(f'{output_path}/plots/', exist_ok=True)

    for feature in features_dict.items():

        feature_related_col_name = feature[0]
        feature_name = feature[1][0]
        y_axis_value = feature[1][1]
        analysis_type = feature[1][2]
        if analysis_type == 'lifehistory_based':

            selected_rows = df.drop_duplicates(subset='id')[[feature_related_col_name]].copy()
            selected_rows = selected_rows.loc[~ selected_rows[feature_related_col_name].isna()]

            if feature_related_col_name == 'LifeHistory':

                selected_rows['LifeHistory'] *= interval_time

                # filter
                selected_rows = selected_rows.loc[selected_rows['LifeHistory'] >= doubling_time]

                sel_feature_frequency = calculate_frequency(selected_rows[feature_related_col_name].values,
                                                            feature_related_col_name)
            else:
                sel_feature_frequency = calculate_frequency(selected_rows[feature_related_col_name].values,
                                                            feature_related_col_name)
        elif analysis_type == 'family_based':

            selected_rows = df.drop_duplicates(subset=label_col)[[feature_related_col_name]].copy()
            selected_rows = selected_rows.loc[~ selected_rows[feature_related_col_name].isna()]
            sel_feature_frequency = calculate_frequency(selected_rows[feature_related_col_name].values,
                                                        feature_related_col_name)

        else:

            selected_rows = df.loc[~ df[feature_related_col_name].isna()]
            sel_feature_frequency = calculate_frequency(selected_rows[feature_related_col_name].values,
                                                        feature_related_col_name)

        # Calculate bin edges using numpy.histogram_bin_edges with 'doane'
        bin_edges = np.histogram_bin_edges(sel_feature_frequency[feature_related_col_name], bins='doane')

        # Check if the min or max value falls outside the bin edges
        extend_lower = False
        if sel_feature_frequency[feature_related_col_name].min() < bin_edges[0]:
            # Extend lower
            bin_edges = np.insert(bin_edges, 0, sel_feature_frequency[feature_related_col_name].min() - 1)
            extend_lower = True

        extend_upper = False
        if sel_feature_frequency[feature_related_col_name].max() > bin_edges[-1]:
            # Extend upper
            bin_edges = np.append(bin_edges, sel_feature_frequency[feature_related_col_name].max() + 1)
            extend_upper = True

        # Assign each LifeHistory value to a bin
        sel_feature_frequency["bin"] = pd.cut(sel_feature_frequency[feature_related_col_name], bins=bin_edges,
                                              labels=False, include_lowest=True)

        grouped_sel_feature = sel_feature_frequency.groupby("bin").sum()

        # Ensure all bins are represented
        all_bins = pd.Series(0, index=range(len(bin_edges) - 1))  # Create all possible bins
        grouped_sel_feature = grouped_sel_feature.reindex(all_bins.index, fill_value=0)

        # Create x positions for grouped bars
        x_positions = np.arange(len(grouped_sel_feature.index))

        # Create labels for the bins
        bin_labels = []
        if feature_related_col_name in ['LifeHistory', 'Division_Family_Count']:
            for i in range(len(bin_edges) - 1):
                if i == 0:
                    if extend_lower:
                        bin_labels.append(f"<{bin_edges[1]:.0f}")  # Label for the new lower bin
                    else:
                        bin_labels.append(f"[{bin_edges[i]:.0f},{bin_edges[i + 1]:.0f}]")
                elif i == len(bin_edges) - 2:
                    if extend_upper:
                        bin_labels.append(f">{bin_edges[-2]:.0f}")  # Label for the new upper bin
                    else:
                        bin_labels.append(f"({bin_edges[i]:.0f},{bin_edges[i + 1]:.0f}]")
                else:
                    bin_labels.append(f"({bin_edges[i]:.0f},{bin_edges[i + 1]:.0f}]")
        elif feature_related_col_name in ['Elongation_Rate', 'Velocity']:
            for i in range(len(bin_edges) - 1):
                if i == 0:
                    if extend_lower:
                        bin_labels.append(f"<{bin_edges[1]:.4f}")  # Label for the new lower bin
                    else:
                        bin_labels.append(f"[{bin_edges[i]:.4f},{bin_edges[i + 1]:.4f}]")
                elif i == len(bin_edges) - 2:
                    if extend_upper:
                        bin_labels.append(f">{bin_edges[-2]:.4f}")  # Label for the new upper bin
                    else:
                        bin_labels.append(f"({bin_edges[i]:.4f},{bin_edges[i + 1]:.4f}]")
                else:
                    bin_labels.append(f"({bin_edges[i]:.4f},{bin_edges[i + 1]:.4f}]")
        else:

            for i in range(len(bin_edges) - 1):
                if i == 0:
                    if extend_lower:
                        bin_labels.append(f"<{bin_edges[1]:.2f}")  # Label for the new lower bin
                    else:
                        bin_labels.append(f"[{bin_edges[i]:.2f},{bin_edges[i + 1]:.2f}]")
                elif i == len(bin_edges) - 2:
                    if extend_upper:
                        bin_labels.append(f">{bin_edges[-2]:.2f}")  # Label for the new upper bin
                    else:
                        bin_labels.append(f"({bin_edges[i]:.2f},{bin_edges[i + 1]:.2f}]")
                else:
                    bin_labels.append(f"({bin_edges[i]:.2f},{bin_edges[i + 1]:.2f}]")

        plot_frequency_distribution(x_positions, grouped_sel_feature, bin_labels, feature_related_col_name,
                                    feature_name, y_axis_value, output_path)
