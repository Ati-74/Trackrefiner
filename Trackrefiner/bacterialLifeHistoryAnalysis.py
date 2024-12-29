import numpy as np
from Trackrefiner.correction.action.helper import calculate_bac_endpoints, extract_bacteria_features
from Trackrefiner.correction.action.featuresCalculation.calcGrowthRate import calculate_growth_rate
from Trackrefiner.correction.action.featuresCalculation.fluorescenceIntensity import determine_final_cell_type


def process_bacterial_life_and_family(processed_cp_out_df, interval_time, growth_rate_method, assigning_cell_type,
                                      cell_type_array, label_col, center_coord_cols):

    """
    Calculates additional features related to the life history of bacteria from a processed
    CellProfiler output DataFrame.


    :param pandas.DataFrame processed_cp_out_df:
        Processed CellProfiler output DataFrame containing measured bacterial features.

    :param float interval_time:
        Time interval between consecutive images (in minutes).

    :param str growth_rate_method:
        Method for calculating the growth rate. Options:
        - 'Average': Computes the average growth rate.
        - 'Linear Regression': Estimates the growth rate using linear regression.

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

    columns_to_initialize = ["growthRate", "startVol", "targetVol", "pos", "time", "radius", "dir", "ends",
                             "strainRate", "strainRate_rolling"]
    for column in columns_to_initialize:
        processed_cp_out_df[column] = ''

    # useful for calculation of rolling average
    window_size = 5  # time steps

    bacteria_id = processed_cp_out_df['id'].unique()

    for bacterium_id in bacteria_id:

        bacterium_life_history = processed_cp_out_df.loc[processed_cp_out_df['id'] == bacterium_id]
        elongation_rate = calculate_growth_rate(bacterium_life_history, interval_time, growth_rate_method)

        # length of bacteria when they are born
        birth_length = bacterium_life_history.iloc[0]["AreaShape_MajorAxisLength"]
        division_length = bacterium_life_history.iloc[-1]["AreaShape_MajorAxisLength"]

        strain_rate_list = []
        old_length = birth_length
        for idx, bacterium in bacterium_life_history.iterrows():
            processed_cp_out_df.at[idx, "growthRate"] = elongation_rate
            processed_cp_out_df.at[idx, "startVol"] = birth_length
            processed_cp_out_df.at[idx, "targetVol"] = division_length
            # https://github.com/cellmodeller/CellModeller/blob/master/CellModeller/Biophysics/BacterialModels/CLBacterium.py#L674
            strain_rate = (bacterium["AreaShape_MajorAxisLength"] - old_length) / old_length
            processed_cp_out_df.at[idx, "strainRate"] = strain_rate
            old_length = processed_cp_out_df.iloc[idx]["AreaShape_MajorAxisLength"]

            # rolling average
            strain_rate_list.append(strain_rate)
            if len(strain_rate_list) > window_size:
                # ignore first element
                strain_rate_rolling = np.mean(strain_rate_list[1:])
            else:
                strain_rate_rolling = np.mean(strain_rate_list)
            processed_cp_out_df.at[idx, "strainRate_rolling"] = strain_rate_rolling

            bacterium_features = extract_bacteria_features(bacterium, center_coord_cols)

            bacterium_center_position = [bacterium_features['center_x'], bacterium_features['center_y']]
            processed_cp_out_df.at[idx, "pos"] = bacterium_center_position

            processed_cp_out_df.at[idx, "time"] = bacterium["ImageNumber"] * interval_time

            processed_cp_out_df.at[idx, "radius"] = bacterium_features['radius']

            processed_cp_out_df.at[idx, "dir"] = [np.cos(bacterium["AreaShape_Orientation"]),
                                                  np.sin(bacterium["AreaShape_Orientation"])]

            # find end points
            end_points = calculate_bac_endpoints(bacterium_center_position, bacterium_features['major'],
                                                 bacterium_features['orientation'])

            processed_cp_out_df.at[idx, "ends"] = end_points

    if assigning_cell_type:
        # determine final cell type of each bacterium
        processed_cp_out_df = determine_final_cell_type(processed_cp_out_df, cell_type_array)

    # rename some columns
    processed_cp_out_df.rename(columns={'ImageNumber': 'stepNum', 'AreaShape_MajorAxisLength': 'length',
                                        label_col: 'label', 'age': 'cellAge'}, inplace=True)
    if assigning_cell_type:
        data_frame_with_selected_col = processed_cp_out_df[
            ['stepNum', 'ObjectNumber', 'id', 'label', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol',
             'targetVol', 'parent_id', 'pos', 'time', 'radius', 'length', 'ends', 'dir', 'cellType', 'strainRate',
             'strainRate_rolling']]
    else:
        data_frame_with_selected_col = processed_cp_out_df[
            ['stepNum', 'ObjectNumber', 'id', 'label', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol',
             'targetVol', 'parent_id', 'pos', 'time', 'radius', 'length', 'ends', 'dir', 'strainRate',
             'strainRate_rolling']]

    return processed_cp_out_df, data_frame_with_selected_col
