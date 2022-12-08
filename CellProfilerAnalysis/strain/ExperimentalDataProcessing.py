import numpy as np
from CellProfilerAnalysis.strain.correction.action.processing import find_vertex, bacteria_features
from CellProfilerAnalysis.strain.correction.action.CalcGrowthRate import calculate_growth_rate
from CellProfilerAnalysis.strain.correction.action.FluorescenceIntensity import final_cell_type


def bacteria_analysis_func(data_frame, interval_time, growth_rate_method):
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

            major, radius, orientation, center_x, center_y = bacteria_features(bacterium)
            bacterium_center_position = [center_x, center_y]
            data_frame.at[idx, "pos"] = bacterium_center_position

            data_frame.at[idx, "time"] = bacterium["ImageNumber"] * interval_time

            data_frame.at[idx, "radius"] = radius

            data_frame.at[idx, "dir"] = [np.cos(bacterium["AreaShape_Orientation"]),
                                         np.sin(bacterium["AreaShape_Orientation"])]

            # find end points
            end_points = find_vertex(bacterium_center_position, major, orientation)

            data_frame.at[idx, "ends"] = end_points

            cell_age += 1

    # determine final cell type of each bacterium
    data_frame = final_cell_type(data_frame)

    # rename some columns
    data_frame.rename(columns={'ImageNumber': 'stepNum', 'AreaShape_MajorAxisLength': 'length',
                               'TrackObjects_Label_50': 'label'}, inplace=True)
    data_frame = data_frame[
        ['stepNum', 'id', 'label', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol', 'targetVol',
         'parent_id', 'pos', 'time', 'radius', 'length', 'ends', 'dir', 'cellType', 'strainRate', 'strainRate_rolling']]

    return data_frame
