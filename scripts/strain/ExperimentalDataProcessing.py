import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def lineage_bacteria_after_this_time_step(data_frame, bacteria):
    data_frame_of_lineage = data_frame.loc[(data_frame["TrackObjects_Label_50"] == bacteria["TrackObjects_Label_50"]) &
                                           (data_frame["ImageNumber"] >= bacteria["ImageNumber"])]
    return data_frame_of_lineage


def division_occurrence(data_frame_of_lineage, bacteria, bac_id):
    parent_time_step_of_cell = bacteria["ImageNumber"]
    parent_index_of_cell = bacteria["ObjectNumber"]

    division_occ = False
    life_history_index = []

    bacteria_index = bac_id
    life_history_index.append(bacteria_index)
    last_time_step = data_frame_of_lineage["ImageNumber"].iloc[-1]

    while (division_occ is False) and (last_time_step != parent_time_step_of_cell):
        relative_bacteria_in_next_timestep = data_frame_of_lineage.loc[
            (data_frame_of_lineage["TrackObjects_ParentImageNumber_50"] == parent_time_step_of_cell) &
            (data_frame_of_lineage["TrackObjects_ParentObjectNumber_50"] == parent_index_of_cell)]

        number_of_relative_bacteria = relative_bacteria_in_next_timestep.shape[0]

        if number_of_relative_bacteria == 1:
            parent_index_of_cell = relative_bacteria_in_next_timestep.iloc[0]["ObjectNumber"]
            parent_time_step_of_cell = relative_bacteria_in_next_timestep.iloc[0]["ImageNumber"]
            bacteria_index = relative_bacteria_in_next_timestep.index.tolist()[0]
            life_history_index.append(bacteria_index)
        elif number_of_relative_bacteria == 2:
            division_occ = True
            last_timestep_before_division = parent_time_step_of_cell

        elif number_of_relative_bacteria == 0:  # interrupt
            last_timestep_before_division = parent_time_step_of_cell
            break

    if division_occ is False and last_time_step == parent_time_step_of_cell:
        last_timestep_before_division = data_frame_of_lineage["ImageNumber"].iloc[-1]

    division_status = {"division_occ": division_occ, "last_timestep_before_division": last_timestep_before_division,
                       "lifeHistoryIndex": life_history_index}

    return division_status


def find_life_history(data_frame_of_lineage, life_history_index):
    df_life_history = data_frame_of_lineage.loc[life_history_index]

    return df_life_history


def calculate_average_growth_rate(division_length, birth_length, t):
    elongation_rate = round((math.log(division_length) - math.log(birth_length)) / t, 3)

    return elongation_rate


def calculate_linear_regression_growth_rate(time, length):
    linear_regressor = LinearRegression()  # create object for the class
    # perform linear regression
    linear_regressor.fit(np.array(time).reshape(-1, 1), np.log(np.array(length).reshape(-1, 1)))
    elongation_rate = round(linear_regressor.coef_[0][0], 3)

    return elongation_rate


def calculate_growth_rate(df_life_history, interval_time, growth_rate_method):
    life_history_length = df_life_history.shape[0]

    # calculation of new feature
    division_length = df_life_history.iloc[[-1]]["AreaShape_MajorAxisLength"].values[0]
    # length of bacteria when they are born
    birth_length = df_life_history.iloc[[0]]["AreaShape_MajorAxisLength"].values[0]
    # this condition checks the life history of bacteria
    # If the bacterium exists only one time step: NaN will be reported.
    if life_history_length > 1:
        if growth_rate_method == "Average":
            t = life_history_length * interval_time
            elongation_rate = calculate_average_growth_rate(division_length, birth_length, t)
        if growth_rate_method == "Linear Regression":
            # linear regression
            time = df_life_history["ImageNumber"].values * interval_time
            length = df_life_history["AreaShape_MajorAxisLength"].values
            elongation_rate = calculate_linear_regression_growth_rate(time, length)
    else:
        elongation_rate = "NaN"  # shows: bacterium is present for only one timestep.

    return elongation_rate


def find_same_id_bacteria(data_frame, id_of_bacteria):
    data_frame_of_same_bacteria = data_frame.loc[data_frame["id"] == id_of_bacteria]
    same_bacteria_index = data_frame_of_same_bacteria.index.tolist()

    return same_bacteria_index


def completing_parent_id(data_frame):
    bacteria_id = 1
    for index, row in data_frame.iterrows():
        if (data_frame.iloc[index]["lineage"] != '') and (data_frame.iloc[index]["id"] >= bacteria_id):
            id_of_bacteria = data_frame.iloc[index]["id"]
            same_bacteria_index = find_same_id_bacteria(data_frame, id_of_bacteria)
            for idx in same_bacteria_index:
                data_frame.at[idx, "lineage"] = data_frame.iloc[index]["lineage"]
            bacteria_id = data_frame.iloc[index]["id"] + 1

    return data_frame


def assign_cell_type(fluorescence_df):
    """
    this function has been written by Aaron Yip (https://github.com/cheekolegend)

    Assigns cell type based on the fluorescence intensity of a cell.
    cell_type number is ascending based on alphabetical order, followed by types based on channel overlap.
        
    @param fluorescence_df  DataFrame   Fluorescence columns for a specific cell
    (i.e. a single row of CPdata['Intensity_MeanIntensity_...'])
    
    return cell_type        int         cell_type identifier; cell_type = 0 means no fluorescence   
    
    TODO: allow user to select the fluorescence intensity threshold for selection
    """

    # Initialize storage variables
    cell_type = 0

    n_channels = len(fluorescence_df)

    # TODO: could do this in a loop to make it work for arbitrary n_channels; would need to specify a vector for the
    #  threshold value to pass.
    if n_channels == 1:
        if fluorescence_df[0] > 0.1:
            cell_type = 1

    if n_channels == 2:
        channel_threshold_passed = [False, False]
        if fluorescence_df[0] > 0.1:
            channel_threshold_passed[0] = True
        if fluorescence_df[1] > 0.1:
            channel_threshold_passed[1] = True

        # Identify cell type
        if channel_threshold_passed == [True, False]:
            cell_type = 1
        elif channel_threshold_passed == [False, True]:
            cell_type = 2
        elif channel_threshold_passed == [True, True]:
            cell_type = 3

    return cell_type


# elongation rate: tracking each bacterium and calculate its elongation rate based on formula:
# (ln(l(last))-ln(l(first))/age
# some odd types in calculating elongation rate:
# 1) life history =1 ---> I store elongation rate as NaN
# Furthermore:
# I find the first time step that the bacterium is born and the last time step that the bacterium exists,
# also the length of bacteria in first time step of its life history and also the length of bacteria
# in the last time step of its life history.

def bacteria_analysis_func(data_frame, interval_time, growth_rate_method):
    # same Bacteria features
    data_frame["id"] = ''
    data_frame["divideFlag"] = False
    data_frame["cellAge"] = ''
    data_frame["growthRate"] = ''
    data_frame["LifeHistory"] = ''
    data_frame["startVol"] = ''
    data_frame["targetVol"] = ''
    data_frame["lineage"] = ''

    # single cell Features
    data_frame["pos"] = ''
    data_frame["time"] = ''
    data_frame["radius"] = ''

    # this part has been written by Aaron Yip (https://github.com/cheekolegend)
    '''
        start
    '''
    data_frame["cellType"] = ''

    # fluorescence intensity columns
    fluorescence_intensities = [col for col in data_frame if col.startswith('Intensity_MeanIntensity')]
    '''
        end
    '''

    id_of_bacteria = 1
    for index, row in data_frame.iterrows():
        if not data_frame.iloc[index]["growthRate"]:

            data_frame_of_lineage = lineage_bacteria_after_this_time_step(data_frame, row)

            division_status = division_occurrence(data_frame_of_lineage, row, index)

            df_life_history = find_life_history(data_frame_of_lineage, division_status["lifeHistoryIndex"])
            elongation_rate = calculate_growth_rate(df_life_history, interval_time, growth_rate_method)

            life_history_index = df_life_history.index.tolist()
            life_history_length = df_life_history.shape[0]
            division_length = df_life_history.iloc[[-1]]["AreaShape_MajorAxisLength"].values[0]
            # length of bacteria when they are born
            birth_length = df_life_history.iloc[[0]]["AreaShape_MajorAxisLength"].values[0]
            cell_age = 1

            for idx in life_history_index:
                data_frame.at[idx, "id"] = id_of_bacteria
                data_frame.at[idx, "cellAge"] = cell_age
                data_frame.at[idx, "growthRate"] = elongation_rate
                data_frame.at[idx, "LifeHistory"] = life_history_length
                data_frame.at[idx, "startVol"] = birth_length
                data_frame.at[idx, "targetVol"] = division_length
                cell_age += 1
            id_of_bacteria += 1

            if division_status["division_occ"]:
                last_time_step_of_bacteria = life_history_index[-1]
                data_frame.at[last_time_step_of_bacteria, "divideFlag"] = True

                # daughters
                division_time = division_status["last_timestep_before_division"] + 1
                data_frame_of_daughters = data_frame_of_lineage.loc[(data_frame_of_lineage["ImageNumber"]
                                                                     == division_time)]
                daughter_index_list = data_frame_of_daughters.index.tolist()

                parent_id = id_of_bacteria

                for daughter_idx in daughter_index_list:
                    data_frame.at[daughter_idx, "lineage"] = parent_id

        # SingleCellFeatures
        # this part has been modified by Aaron Yip (https://github.com/cheekolegend)
        '''
            start
        '''
        try:
            data_frame.at[index, "pos"] = [row["Location_Center_X"], row["Location_Center_Y"]]
        except:
            data_frame.at[index, "pos"] = [row["AreaShape_Center_X"], row["AreaShape_Center_Y"]]

        '''
            end
        '''
        data_frame.at[index, "time"] = row["ImageNumber"] * interval_time

        # this line has been written by Aaron Yip (https://github.com/cheekolegend)
        data_frame.at[index, "radius"] = row["AreaShape_MinorAxisLength"] / 2
        
        # modification of bacterium orientation
        # -(angle + 90) * np.pi / 180
        data_frame.at[index, "AreaShape_Orientation"] = \
            (-1 * data_frame.iloc[index]["AreaShape_Orientation"] + 90) * np.pi / 180

        # For fluorescence readings; maybe this would be cleaner if placed in different function?
        if fluorescence_intensities:
            data_frame.at[index, "cellType"] = assign_cell_type(row[fluorescence_intensities])

    data_frame = completing_parent_id(data_frame)
    # rename some columns
    final_data_frame = data_frame.rename(
        columns={'AreaShape_MajorAxisLength': 'length', 'AreaShape_Orientation': 'orientation'})
    final_data_frame = final_data_frame[
        ['ImageNumber', 'id', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol', 'targetVol', 'lineage',
         'pos', 'time', 'radius', 'length', 'orientation', 'cellType']]
    return final_data_frame
