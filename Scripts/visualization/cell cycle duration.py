import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calc_freq(array, postfix):

    """
    Calculate the frequency of unique values in an array and return a DataFrame.

    :param np.ndarray array:
        Array of numerical values for which frequencies are calculated.
    :param str postfix:
        A string to append to the frequency column name for identification.

    :return:
        - **df** (*pd.DataFrame*): DataFrame containing unique values and their corresponding frequencies.
    """

    # Get unique values and their counts
    unique_values, counts = np.unique(array, return_counts=True)

    # Create a DataFrame
    df = pd.DataFrame({
        'LifeHistory': unique_values,
        'Frequency' + postfix: counts
    })

    return df


def plot_cp_omnipose_tr():

    """
    Generate and save a bar plot comparing absolute errors from CellProfiler (CP) and TrackRefiner (TR).

    This function creates a grouped bar chart comparing absolute tracking errors from CP and TR across
    different bins of LifeHistory values. The bars are color-coded, and the plot is saved as a high-resolution image.

    The dataset name determines whether a legend is displayed.

    :global dataset_name:
        Name of the dataset being processed, used for labeling and file naming.
    :global grouped_cp_tr:
        DataFrame containing absolute errors from CP and TR, grouped by bins.
    :global x, width, bin_labels, output_path:
        Variables used for plotting (bin positions, bar width, bin labels, and output path).

    :return:
        - Saves the generated plot to the specified output directory.
    """

    # Plot configuration
    #
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for cp_omnipose_abs_error and tr_abs_error
    ax.bar(x - width / 2, grouped_cp_tr["cp_omnipose_abs_error"], width, label="CellProfiler", color="blue")
    ax.bar(x + width / 2, grouped_cp_tr["tr_abs_error"], width, label="TrackRefiner", color="orange")

    # Add labels and title
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=0, fontsize=20)  # Set font size to 12
    plt.yticks(fontsize=20)

    # Add legend
    if dataset_name == 'clara 6A':
        ax.legend(title="", fontsize=16, title_fontsize=18)

    # Adjust x-axis limits to reduce space before the first bin and after the last bin
    ax.set_xlim(-0.5, len(x) - 0.5)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{output_path}/{dataset_name}_cp_omnipose_tr.png', dpi=600)
    plt.clf()
    plt.close()


def plot_gt():

    """
    Generate and save a bar plot representing ground truth (GT) frequency distribution.

    This function creates a bar chart displaying the frequency distribution of LifeHistory values
    in the ground truth dataset. The plot is saved as a high-resolution image.

    :global dataset_name:
        Name of the dataset being processed, used for labeling and file naming.
    :global grouped_gt:
        DataFrame containing frequency counts of LifeHistory values in the ground truth.
    :global x, width, output_path:
        Variables used for plotting (bin positions, bar width, and output path).

    :return:
        - Saves the generated plot to the specified output directory.
    """

    # Plot configuration
    # figsize=(10, 6)
    fig, ax = plt.subplots()

    # Plot bars for cp_omnipose_abs_error and tr_abs_error
    ax.bar(x, grouped_gt['FrequencyGT'], width, label="GT", color="black")

    # Add labels and title
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title('')
    ax.set_xticks(x)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.yticks(fontsize=16)

    # Adjust x-axis limits to reduce space before the first bin and after the last bin
    ax.set_xlim(-0.3, len(x) - 0.7)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{output_path}/{dataset_name}_gt.png', dpi=600)
    plt.clf()
    plt.close()


if __name__ == '__main__':

    dict_pref = {'unconstrained_1': {'Interval time (min)': 2, 'doubling time (min)': 20},
                 'unconstrained_2': {'Interval time (min)': 2, 'doubling time (min)': 20},
                 'unconstrained_3': {'Interval time (min)': 2, 'doubling time (min)': 20},
                 'unconstrained_4': {'Interval time (min)': 10, 'doubling time (min)': 20},
                 'constrained_1': {'Interval time (min)': 3, 'doubling time (min)': 20},
                 'constrained_2': {'Interval time (min)': 5, 'doubling time (min)': 20},
                 'constrained_3': {'Interval time (min)': 2, 'doubling time (min)': 45},
                 'constrained_4': {'Interval time (min)': 2, 'doubling time (min)': 45}}

    gt_path = '../../09.GroundTruth/'
    cp_postprocessing_path = '../../05.CP_Omnipose_post_processing/'
    tr_output_path = '../../07.runTrackrefiner/'
    output_path = '../../10.Visualization/cell cycle duration/'

    for i, dataset_name in enumerate(dict_pref.keys()):

        print(dataset_name)

        # min
        interval_time = dict_pref[dataset_name]['Interval time (min)']

        # min
        life_history_lowe_bound = 20

        gt_life_history_df = \
            pd.read_csv(f'{gt_path}/ground_truth_{dataset_name}/post_processing/GT_LifeHistory_based_Analysis.csv')

        cp_omnipose_life_history_df = \
            pd.read_csv(f'{cp_postprocessing_path}/CP_Omnipose_post_processing_{dataset_name}/'
                        f'CP_Omnipose_LifeHistory_based_Analysis.csv')

        tr_life_history_df = \
            pd.read_csv(f'{tr_output_path}/trackrefiner_{dataset_name}/Trackrefiner.FilterObjects_Average_analysis.csv')
        tr_life_history_df = tr_life_history_df.drop_duplicates(subset=['id'], keep='last')

        gt_life_history_df['LifeHistory'] *= interval_time
        cp_omnipose_life_history_df['LifeHistory'] *= interval_time
        tr_life_history_df['LifeHistory'] *= interval_time

        # filter
        gt_life_history_df = gt_life_history_df.loc[gt_life_history_df['LifeHistory'] >= life_history_lowe_bound]
        cp_omnipose_life_history_df = \
            cp_omnipose_life_history_df.loc[cp_omnipose_life_history_df['LifeHistory'] >= life_history_lowe_bound]
        tr_life_history_df = tr_life_history_df.loc[tr_life_history_df['LifeHistory'] >= life_history_lowe_bound]

        gt_frequency = calc_freq(gt_life_history_df['LifeHistory'].values, 'GT')
        cp_omnipose_frequency = calc_freq(cp_omnipose_life_history_df['LifeHistory'].values, 'CP_Omnipose')
        tr_frequency = calc_freq(tr_life_history_df['LifeHistory'].values, 'Trackrefiner')

        merged_df = gt_frequency.merge(cp_omnipose_frequency, on='LifeHistory', how='outer')

        merged_df = merged_df.merge(tr_frequency, on='LifeHistory', how='outer')
        merged_df = merged_df.fillna(0)

        gt_frequency = merged_df[['LifeHistory', 'FrequencyGT']]

        merged_df['cp_omnipose_abs_error'] = np.abs(merged_df['FrequencyGT'] - merged_df['FrequencyCP_Omnipose'])
        merged_df['tr_abs_error'] = np.abs(merged_df['FrequencyGT'] - merged_df['FrequencyTrackrefiner'])

        # print(merged_df)

        merged_df = merged_df[['LifeHistory', 'cp_omnipose_abs_error', 'tr_abs_error']]

        # Calculate bin edges using numpy.histogram_bin_edges with 'doane'
        bin_edges = np.histogram_bin_edges(merged_df["LifeHistory"], bins='doane')

        # Assign each LifeHistory value to a bin
        merged_df["bin"] = pd.cut(merged_df["LifeHistory"], bins=bin_edges, labels=False, include_lowest=True)
        gt_frequency['bin'] = pd.cut(gt_frequency["LifeHistory"], bins=bin_edges, labels=False, include_lowest=True)

        # Expected bins based on bin_edges
        expected_bins = range(len(bin_edges) - 1)
        # Assigned bins
        assigned_bins = merged_df["bin"].unique()

        # Find missing bins
        missing_bins = set(expected_bins) - set(assigned_bins)

        # Group by bins and calculate sum values for each bin
        grouped_cp_tr = merged_df.groupby("bin").sum()

        grouped_cp_tr = grouped_cp_tr.reindex(expected_bins, fill_value=0)

        grouped_gt = gt_frequency.groupby("bin").sum()
        grouped_gt = grouped_gt.reindex(expected_bins, fill_value=0)

        # Create x positions for grouped bars
        x = np.arange(len(grouped_cp_tr.index))  # X positions
        width = 0.4  # Bar width

        # Get bin ranges for x-axis labels
        bin_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i + 1]:.0f}" for i in range(len(bin_edges) - 1)]

        plot_cp_omnipose_tr()
        plot_gt()
