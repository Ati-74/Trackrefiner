import pandas as pd
import os
from ExperimentalDataProcessing import bacteria_analysis


def lineage_based_analysis(df):
    """
    Conducts lineage-based analysis on the input DataFrame.

    @params df DataFrame DataFrame containing bacteria data.

    Returns:
    - results DataFrame Results of the analysis containing the number of divisions for each label.
    """

    # Get unique labels present in the DataFrame.
    uniq_label = list(set(df["label"].values))
    result_dict = {"label": [], "NumberOfDivision": []}

    # Iterate over each unique label.
    for label in uniq_label:
        # Filter the DataFrame for rows corresponding to the current label.
        df_current_label = df.loc[df["label"] == label]

        # Find rows where division occurred.
        division_df = df_current_label.loc[df_current_label["divideFlag"] == True]

        # Add the number of divisions and the label to the result dictionary.
        if division_df.shape[0] > 1:
            result_dict["NumberOfDivision"].append(division_df.shape[0])
        else:
            # If no division occurred, append 0.
            result_dict["NumberOfDivision"].append(0)
        result_dict["label"].append(label)

    # Convert the results dictionary to a DataFrame.
    results = pd.DataFrame.from_dict(result_dict, orient="index").transpose()

    return results


def find_num_cells_in_each_time_step(df):
    """
    Finds the number of cells present at each time step in the input DataFrame.

    @param df DataFrame DataFrame containing bacteria data.

    Returns:
    - results DataFrame Results of the analysis containing the number of cells for each time step.
    """

    # Get unique time steps present in the DataFrame.
    uniq_time_steps = list(set(df["TimeStep"].values))

    result_dict = {"TimeStep": [], "NumberOfCells": []}

    # Iterate over each unique time step.
    for timestep in uniq_time_steps:
        # Filter the DataFrame for rows corresponding to the current time step.
        df_current_timestep = df.loc[df["TimeStep"] == timestep]

        # Add the time step and the number of cells present to the result dictionary.
        result_dict["TimeStep"].append(timestep)
        result_dict["NumberOfCells"].append(df_current_timestep.shape[0])

    # Convert the results dictionary to a DataFrame.
    results = pd.DataFrame.from_dict(result_dict, orient="index").transpose()

    return results


def process_data(input_file, interval_time, output_directory, prefix_name):
    """
    Processes the data from the input_file and saves the results to various CSV files in the output_directory.

    @param input_file str Path to the input CSV file containing bacteria data (output of Cellprofiler or GroundTruth).
    @param interval_time float Interval time between frames.
    @param output_directory str Directory where the processed results will be saved.
    @param prefix_name str prefix for output file names

    """

    # Read the data from the CSV file into a DataFrame.
    data_frame = pd.read_csv(input_file)

    # Filter out rows where AreaShape_MajorAxisLength is zero and reset index.
    data_frame = data_frame.loc[data_frame["AreaShape_MajorAxisLength"] != 0].reset_index(drop=True)
    data_frame = data_frame.reset_index(drop=True)

    # Process the tracking data using the bacteria_analysis function.
    # Returns a modified DataFrame and a life-history based analysis DataFrame.
    df, life_history_based_analysis = bacteria_analysis(data_frame, interval_time, um_per_pixel)

    # Perform lineage-based analysis and count the number of cells in each time step.
    lineage_based_analysis_results = lineage_based_analysis(df)
    num_cells_in_each_time_step_results = find_num_cells_in_each_time_step(df)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define paths for saving the results as CSV files.
    path_life_history = output_directory + prefix_name + "_LifeHistory_based_Analysis"
    path_lineage = output_directory + prefix_name + "_lineage_based_analysis"
    path_bacteria_based_features = output_directory + prefix_name + "_bacteria_feature_analysis"
    path_num_cells_in_each_time_step = output_directory + prefix_name + "_Num_cells_in_each_timeStep"

    # Save the results to CSV files in the specified output_directory.
    life_history_based_analysis.to_csv(path_life_history + ".csv", index=False)
    lineage_based_analysis_results.to_csv(path_lineage + ".csv", index=False)
    df.to_csv(path_bacteria_based_features + ".csv", index=False)
    num_cells_in_each_time_step_results.to_csv(path_num_cells_in_each_time_step + ".csv", index=False)


if __name__ == "__main__":

    # interval time unit: minute
    datasets_dict = {'unconstrained_1': {'interval_time': 2}, 'unconstrained_2': {'interval_time': 2},
                     'unconstrained_3': {'interval_time': 2}, 'unconstrained_4': {'interval_time': 10},
                     'constrained_1': {'interval_time': 3}, 'constrained_3': {'interval_time': 2},
                     'constrained_4': {'interval_time': 2}, 'constrained_2': {'interval_time': 5}}

    um_per_pixel = 0.144

    cp_omnipose_output_dir = '../../04.run_CellProfiler_Omnipose/'
    gt_output_dir = '../../09.GroundTruth/'

    cp_omnipose_post_processing_put_path = '../../05.CP_Omnipose_post_processing/'
    gt_post_processing_put_path = '../../09.GroundTruth/'

    for dataset_name in datasets_dict.keys():

        print(dataset_name)
        print('CP-Omnipose Post-Processing')

        prefix_name_value = 'CP_Omnipose'

        cp_omnipose_output_path = f'{cp_omnipose_output_dir}/cellProfiler_omnipose_{dataset_name}/FilterObjects.csv'

        out_path = f'{cp_omnipose_post_processing_put_path}/CP_Omnipose_post_processing_{dataset_name}/'

        interval_time_dataset = datasets_dict[dataset_name]['interval_time']

        # Process the data
        process_data(cp_omnipose_output_path, interval_time_dataset, out_path, prefix_name_value)

        print('Ground Truth Post-Processing')

        prefix_name_value = 'GT'

        cp_omnipose_output_path = f'{gt_output_dir}/ground_truth_{dataset_name}/{dataset_name}.GT.csv'

        out_path = f'{gt_output_dir}/ground_truth_{dataset_name}/post_processing/'

        interval_time_dataset = datasets_dict[dataset_name]['interval_time']

        # Process the data from the current path using the previously defined process_data function.
        process_data(cp_omnipose_output_path, interval_time_dataset, out_path, prefix_name_value)
