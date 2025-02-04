import numpy as np
import pandas as pd
from Trackrefiner.core.correction.action.helper import extract_bacteria_features
from Trackrefiner.core.correction.action.featuresCalculation.calcElongationRate import calculate_elongation_rate
from Trackrefiner.core.correction.action.featuresCalculation.calcVelocity import calc_average_velocity
from Trackrefiner.core.correction.action.featuresCalculation.fluorescenceIntensity import determine_final_cell_type


def process_bacterial_life_and_family(processed_cp_out_df, interval_time, elongation_rate_method, assigning_cell_type,
                                      cell_type_array, label_col, center_coord_cols):
    """
    Calculates additional features related to the life history of bacteria from a processed
    CellProfiler output DataFrame.


    :param pandas.DataFrame processed_cp_out_df:
        Processed CellProfiler output DataFrame containing measured bacterial features.

    :param float interval_time:
        Time interval between consecutive images (in minutes).

    :param str elongation_rate_method:
        Method for calculating the Elongation rate. Options:
        - 'Average': Computes the average elongation rate.
        - 'Linear Regression': Estimates the elongation rate using linear regression.

    :param bool assigning_cell_type:
        If True, assigns cell types to objects based on intensity thresholds.

    :param numpy.ndarray cell_type_array:
        A boolean array indicating the channel in which the intensity exceeds the threshold for each bacterium.

    :param str label_col:
        Column name in the CellProfiler DataFrame that indicates the label of bacteria during tracking.

    :param dict center_coord_cols:
        A dictionary with keys 'x' and 'y', specifying the column names for center coordinates.

    :returns:
        processed_cp_out_df (pandas.DataFrame):
            A DataFrame including new features related to the life history and family of bacteria.
        data_frame_with_selected_col (pandas.DataFrame):
            A subset of `processed_cp_out_df` that includes specific columns to save in numpy output files.


    """

    processed_cp_out_df["time"] = pd.to_numeric(processed_cp_out_df["ImageNumber"]) * interval_time
    processed_cp_out_df["radius"] = processed_cp_out_df['AreaShape_MinorAxisLength'] / 2
    processed_cp_out_df['Average_Length'] = np.nan
    processed_cp_out_df['Average_Length'] = \
        processed_cp_out_df.groupby('id')['AreaShape_MajorAxisLength'].transform('mean')

    columns_to_initialize = ["Elongation_Rate", "startVol", "targetVol", "pos", "dir", "ends",
                             "strainRate", "strainRate_rolling", 'Velocity', 'Instant_Elongation_Rate',
                             'Instant_Velocity', 'Division_Family_Count']

    for column in columns_to_initialize:
        processed_cp_out_df[column] = ''

    # useful for calculation of rolling average
    window_size = 5  # time steps

    bacteria_id = processed_cp_out_df['id'].unique()

    for bacterium_id in bacteria_id:

        bacterium_life_history = processed_cp_out_df.loc[processed_cp_out_df['id'] == bacterium_id]
        elongation_rate = calculate_elongation_rate(bacterium_life_history, interval_time, elongation_rate_method)
        velocity = calc_average_velocity(bacterium_life_history, center_coord_cols, interval_time)

        # length of bacteria when they are born
        birth_length = bacterium_life_history["AreaShape_MajorAxisLength"].values[0]
        division_length = bacterium_life_history["AreaShape_MajorAxisLength"].values[-1]

        strain_rate_list = []
        old_length = birth_length
        old_position = np.sqrt(bacterium_life_history[center_coord_cols['x']].values[0] ** 2 +
                               bacterium_life_history[center_coord_cols['y']].values[0] ** 2)

        processed_cp_out_df.loc[bacterium_life_history.index, "Elongation_Rate"] = elongation_rate
        processed_cp_out_df.loc[bacterium_life_history.index, "Velocity"] = velocity
        processed_cp_out_df.loc[bacterium_life_history.index, "startVol"] = birth_length
        processed_cp_out_df.loc[bacterium_life_history.index, "targetVol"] = division_length

        for idx, bacterium in bacterium_life_history.iterrows():

            # https://github.com/cellmodeller/CellModeller/blob/master/CellModeller/Biophysics/BacterialModels/CLBacterium.py#L674
            strain_rate = (bacterium["AreaShape_MajorAxisLength"] - old_length) / old_length
            processed_cp_out_df.at[idx, "strainRate"] = strain_rate

            # rolling average
            strain_rate_list.append(strain_rate)
            if len(strain_rate_list) > window_size:
                # ignore first element
                strain_rate_rolling = np.mean(strain_rate_list[1:])
            else:
                strain_rate_rolling = np.mean(strain_rate_list)

            if bacterium['age'] > 1:
                instantaneous_elongation_rate = (
                    round((bacterium["AreaShape_MajorAxisLength"] - old_length) / interval_time, 3))

                # instantaneous velocity
                pos2 = np.sqrt(bacterium[center_coord_cols['x']] ** 2 + bacterium[center_coord_cols['y']] ** 2)
                instantaneous_velocity = round((pos2 - old_position) / interval_time, 3)

            else:
                instantaneous_elongation_rate = np.nan
                instantaneous_velocity = np.nan

            old_length = bacterium["AreaShape_MajorAxisLength"]
            old_position = np.sqrt(bacterium[center_coord_cols['x']] ** 2 + bacterium[center_coord_cols['y']] ** 2)

            processed_cp_out_df.at[idx, "strainRate_rolling"] = strain_rate_rolling
            processed_cp_out_df.at[idx, 'Instant_Elongation_Rate'] = instantaneous_elongation_rate
            processed_cp_out_df.at[idx, 'Instant_Velocity'] = instantaneous_velocity

            bacterium_features = extract_bacteria_features(bacterium, center_coord_cols)

            bacterium_center_position = [bacterium_features['center_x'], bacterium_features['center_y']]
            processed_cp_out_df.at[idx, "pos"] = bacterium_center_position

            processed_cp_out_df.at[idx, "dir"] = [np.cos(bacterium["AreaShape_Orientation"]),
                                                  np.sin(bacterium["AreaShape_Orientation"])]

            # find end points
            end_points = [[bacterium['Endpoint1_X'], bacterium['Endpoint1_Y']],
                          [bacterium['Endpoint2_X'], bacterium['Endpoint2_Y']]]

            processed_cp_out_df.at[idx, "ends"] = end_points

    processed_cp_out_df['Average_Instant_Velocity'] = \
        processed_cp_out_df.groupby('id')['Instant_Velocity'].transform('mean')

    if assigning_cell_type:
        # determine final cell type of each bacterium
        processed_cp_out_df = determine_final_cell_type(processed_cp_out_df, cell_type_array)

    # Iterate over each unique label.
    for label in processed_cp_out_df[label_col].unique():
        # Filter the DataFrame for rows corresponding to the current label.
        df_current_label = processed_cp_out_df.loc[processed_cp_out_df[label_col] == label]

        # Find rows where division occurred.
        mothers_df = df_current_label.loc[~ df_current_label["Max_Daughter_Mother_Length_Ratio"].isna()]

        processed_cp_out_df.loc[df_current_label.index, 'Division_Family_Count'] = mothers_df.shape[0]

    # rename some columns
    processed_cp_out_df = processed_cp_out_df.rename(columns={'age': 'cellAge'})

    data_frame_with_selected_col = \
        processed_cp_out_df.rename(columns={'ImageNumber': 'stepNum', 'AreaShape_MajorAxisLength': 'length',
                                            label_col: 'label', 'age': 'cellAge', 'Elongation_Rate': 'growthRate'})
    if 'cellType' in data_frame_with_selected_col.columns:

        data_frame_with_selected_col = data_frame_with_selected_col[
            ['stepNum', 'ObjectNumber', 'id', 'label', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol',
             'targetVol', 'parent_id', 'pos', 'time', 'radius', 'length', 'ends', 'dir', 'cellType', 'strainRate',
             'strainRate_rolling']]
    else:
        data_frame_with_selected_col = data_frame_with_selected_col[
            ['stepNum', 'ObjectNumber', 'id', 'label', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol',
             'targetVol', 'parent_id', 'pos', 'time', 'radius', 'length', 'ends', 'dir', 'strainRate',
             'strainRate_rolling']]

    return processed_cp_out_df, data_frame_with_selected_col
