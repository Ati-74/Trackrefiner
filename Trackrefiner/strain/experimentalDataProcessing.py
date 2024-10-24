import numpy as np
from Trackrefiner.strain.correction.action.helperFunctions import find_vertex, bacteria_features
from Trackrefiner.strain.correction.action.calcGrowthRate import calculate_growth_rate
from Trackrefiner.strain.correction.action.fluorescenceIntensity import final_cell_type


def bacteria_analysis_func(data_frame, interval_time, growth_rate_method, assigning_cell_type, cell_type_array,
                           label_col, center_coordinate_columns):
    """
    goal: assign
    """

    # same Bacteria features
    data_frame["cellAge"] = ''
    data_frame["growthRate"] = ''
    data_frame["startVol"] = ''
    data_frame["targetVol"] = ''

    # single cell Features
    data_frame["pos"] = ''
    data_frame["time"] = ''
    data_frame["radius"] = ''
    data_frame["dir"] = ''
    data_frame["ends"] = ''
    data_frame["strainRate"] = ''
    data_frame["strainRate_rolling"] = ''

    # useful for calculation of rolling average
    window_size = 5  # time steps

    bacteria_id = data_frame['id'].unique()

    for bacterium_id in bacteria_id:
        # print("Calculating new features for bacterium id: " + str(bacterium_id))

        bacterium_life_history = data_frame.loc[data_frame['id'] == bacterium_id]
        elongation_rate = calculate_growth_rate(bacterium_life_history, interval_time, growth_rate_method)

        # length of bacteria when they are born
        birth_length = bacterium_life_history.iloc[0]["AreaShape_MajorAxisLength"]
        division_length = bacterium_life_history.iloc[-1]["AreaShape_MajorAxisLength"]

        cell_age = 1

        strain_rate_list = []
        old_length = birth_length
        for idx, bacterium in bacterium_life_history.iterrows():
            data_frame.at[idx, "cellAge"] = cell_age
            data_frame.at[idx, "growthRate"] = elongation_rate
            data_frame.at[idx, "startVol"] = birth_length
            data_frame.at[idx, "targetVol"] = division_length
            # https://github.com/cellmodeller/CellModeller/blob/master/CellModeller/Biophysics/BacterialModels/CLBacterium.py#L674
            strain_rate = (bacterium["AreaShape_MajorAxisLength"] - old_length) / old_length
            data_frame.at[idx, "strainRate"] = strain_rate
            old_length = data_frame.iloc[idx]["AreaShape_MajorAxisLength"]

            # rolling average
            strain_rate_list.append(strain_rate)
            if len(strain_rate_list) > window_size:
                # ignore first element
                strain_rate_rolling = np.mean(strain_rate_list[1:])
            else:
                strain_rate_rolling = np.mean(strain_rate_list)
            data_frame.at[idx, "strainRate_rolling"] = strain_rate_rolling

            bacterium_features = bacteria_features(bacterium, center_coordinate_columns)

            bacterium_center_position = [bacterium_features['center_x'], bacterium_features['center_y']]
            data_frame.at[idx, "pos"] = bacterium_center_position

            data_frame.at[idx, "time"] = bacterium["ImageNumber"] * interval_time

            data_frame.at[idx, "radius"] = bacterium_features['radius']

            data_frame.at[idx, "dir"] = [np.cos(bacterium["AreaShape_Orientation"]),
                                         np.sin(bacterium["AreaShape_Orientation"])]

            # find end points
            end_points = find_vertex(bacterium_center_position, bacterium_features['major'],
                                     bacterium_features['orientation'])

            data_frame.at[idx, "ends"] = end_points

            cell_age += 1

    if assigning_cell_type:
        # determine final cell type of each bacterium
        data_frame = final_cell_type(data_frame, cell_type_array)

    # rename some columns
    data_frame.rename(columns={'ImageNumber': 'stepNum', 'AreaShape_MajorAxisLength': 'length',
                               label_col: 'label'}, inplace=True)
    if assigning_cell_type:
        data_frame_with_selected_col = data_frame[
            ['stepNum', 'ObjectNumber', 'id', 'label', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol',
             'targetVol', 'parent_id', 'pos', 'time', 'radius', 'length', 'ends', 'dir', 'cellType', 'strainRate',
             'strainRate_rolling']]
    else:
        data_frame_with_selected_col = data_frame[
            ['stepNum', 'ObjectNumber', 'id', 'label', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol',
             'targetVol', 'parent_id', 'pos', 'time', 'radius', 'length', 'ends', 'dir', 'strainRate',
             'strainRate_rolling']]

    return data_frame, data_frame_with_selected_col
