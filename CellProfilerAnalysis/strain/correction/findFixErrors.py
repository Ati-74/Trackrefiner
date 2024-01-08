import numpy as np
import pandas as pd
import time

from CellProfilerAnalysis.strain.correction.action.helperFunctions import bacteria_life_history, remove_rows, \
    convert_to_um, \
    find_vertex, angle_convert_to_radian
from CellProfilerAnalysis.strain.correction.action.fluorescenceIntensity import assign_cell_type
from CellProfilerAnalysis.strain.correction.nanLabel import modify_nan_labels
from CellProfilerAnalysis.strain.correction.withoutParentError import correction_without_parent
from CellProfilerAnalysis.strain.correction.badDaughters import detect_remove_bad_daughters_to_mother_link
from CellProfilerAnalysis.strain.correction.redundantParentLink import detect_and_remove_redundant_parent_link
from CellProfilerAnalysis.strain.correction.action.helperFunctions import calculate_orientation_angle
from CellProfilerAnalysis.strain.correction.noiseRemover import noise_remover


def calculate_slope_intercept(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def assign_feature_find_errors(dataframe, intensity_threshold, check_cell_type):
    """
    goal: assign new features like: `id`, `divideFlag`, `daughters_index`, `bad_division_flag`,
    `unexpected_end`, `division_time`, `transition`, `LifeHistory`, `parent_id` to bacteria and find errors

    @param dataframe dataframe bacteria features value
    @param intensity_threshold float min intensity value of channel
    """
    dataframe['checked'] = False

    dataframe['endppoint1_X'] = ''
    dataframe['endppoint1_Y'] = ''
    dataframe['endppoint2_X'] = ''
    dataframe['endppoint2_Y'] = ''

    dataframe['noise_bac'] = False
    dataframe["id"] = ''
    dataframe["divideFlag"] = False
    dataframe['daughters_index'] = ''
    dataframe['bad_division_flag'] = False
    dataframe['unexpected_end'] = False
    dataframe['division_time'] = 0
    dataframe['transition'] = False
    dataframe["LifeHistory"] = ''
    dataframe["parent_id"] = ''
    dataframe["daughter_orientation"] = ''
    dataframe["daughter_length_to_mother"] = ''
    dataframe["daughter_distance_to_mother"] = ''
    dataframe["bac_length_to_back"] = ''
    dataframe["bac_length_to_back_orientation_changes"] = ''
    dataframe["max_daughter_len_to_mother"] = ''
    dataframe["bacteria_movement"] = ''
    dataframe["bacteria_slope"] = ''

    last_time_step = sorted(dataframe['ImageNumber'].unique())[-1]
    # bacterium id
    bacterium_id = 1
    transition = False

    # columns name
    parent_image_number_col = [col for col in dataframe.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in dataframe.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    for row_index, row in dataframe.iterrows():
        if not dataframe.iloc[row_index]["checked"]:
            bacterium_status = bacteria_life_history(dataframe, row, row_index, last_time_step)
            parent_img_num = row[parent_image_number_col]
            parent_obj_num = row[parent_object_number_col]
            if parent_img_num != 0 and parent_obj_num != 0:
                parent = dataframe.loc[(dataframe["ImageNumber"] == parent_img_num) & (dataframe["ObjectNumber"] ==
                                                                                       parent_obj_num)]
                parent_id = parent['id'].values.tolist()[0]
                transition = False
            else:
                parent_id = 0
                if row['ImageNumber'] > 1:
                    transition = True

            for list_indx, indx in enumerate(bacterium_status['lifeHistoryIndex']):
                dataframe.at[indx, 'checked'] = True

                current_bacterium_endpoints = find_vertex(
                    [dataframe.iloc[indx]["AreaShape_Center_X"], dataframe.iloc[indx]["AreaShape_Center_Y"]],
                    dataframe.iloc[indx]["AreaShape_MajorAxisLength"],
                    dataframe.iloc[indx]["AreaShape_Orientation"])

                # Convert points to vectors
                # Calculate slopes and intercepts
                slope1, intercept1 = calculate_slope_intercept(current_bacterium_endpoints[0],
                                                               current_bacterium_endpoints[1])

                dataframe.at[indx, 'endppoint1_X'] = current_bacterium_endpoints[0][0]
                dataframe.at[indx, 'endppoint1_Y'] = current_bacterium_endpoints[0][1]
                dataframe.at[indx, 'endppoint2_X'] = current_bacterium_endpoints[1][0]
                dataframe.at[indx, 'endppoint2_Y'] = current_bacterium_endpoints[1][1]
                dataframe.at[indx, 'bacteria_slope'] = slope1

                dataframe.at[indx, 'id'] = bacterium_id
                if list_indx > 0:
                    dataframe.at[indx, "bac_length_to_back"] = \
                        dataframe.iloc[indx]["AreaShape_MajorAxisLength"] / \
                        dataframe.iloc[bacterium_status['lifeHistoryIndex'][list_indx - 1]]["AreaShape_MajorAxisLength"]

                    prev_bacterium_endpoints = find_vertex(
                        [dataframe.iloc[bacterium_status['lifeHistoryIndex'][list_indx - 1]]["AreaShape_Center_X"],
                         dataframe.iloc[bacterium_status['lifeHistoryIndex'][list_indx - 1]]["AreaShape_Center_Y"]],
                        dataframe.iloc[bacterium_status['lifeHistoryIndex'][list_indx - 1]][
                            "AreaShape_MajorAxisLength"],
                        dataframe.iloc[bacterium_status['lifeHistoryIndex'][list_indx - 1]]["AreaShape_Orientation"])

                    center_movement = \
                        np.sqrt((dataframe.iloc[indx]["AreaShape_Center_X"] -
                                 dataframe.iloc[bacterium_status['lifeHistoryIndex'][list_indx - 1]][
                                     "AreaShape_Center_X"]) ** 2 +
                                (dataframe.iloc[indx]["AreaShape_Center_Y"] -
                                 dataframe.iloc[bacterium_status['lifeHistoryIndex'][list_indx - 1]][
                                     "AreaShape_Center_Y"]) ** 2)

                    endpoint1_movement = \
                        np.sqrt((current_bacterium_endpoints[0][0] - prev_bacterium_endpoints[0][0]) ** 2 +
                                (current_bacterium_endpoints[0][1] - prev_bacterium_endpoints[0][1]) ** 2)

                    endpoint1_endpoint2_movement = \
                        np.sqrt((current_bacterium_endpoints[0][0] - prev_bacterium_endpoints[1][0]) ** 2 +
                                (current_bacterium_endpoints[0][1] - prev_bacterium_endpoints[1][1]) ** 2)

                    endpoint2_movement = \
                        np.sqrt((current_bacterium_endpoints[1][0] - prev_bacterium_endpoints[1][0]) ** 2 +
                                (current_bacterium_endpoints[1][1] - prev_bacterium_endpoints[1][1]) ** 2)

                    endpoint2_endpoint1_movement = \
                        np.sqrt((current_bacterium_endpoints[1][0] - prev_bacterium_endpoints[0][0]) ** 2 +
                                (current_bacterium_endpoints[1][1] - prev_bacterium_endpoints[0][1]) ** 2)

                    dataframe.at[indx, "bacteria_movement"] = min(center_movement, endpoint1_movement,
                                                                  endpoint2_movement, endpoint1_endpoint2_movement,
                                                                  endpoint2_endpoint1_movement)

                    slope2, intercept2 = calculate_slope_intercept(prev_bacterium_endpoints[0],
                                                                   prev_bacterium_endpoints[1])

                    # Calculate orientation angle
                    orientation_angle = calculate_orientation_angle(slope1, slope2)

                    dataframe.at[indx, "bac_length_to_back_orientation_changes"] = orientation_angle

                dataframe.at[indx, 'LifeHistory'] = bacterium_status['life_history']
                if bacterium_status['division_occ']:
                    dataframe.at[indx, 'divideFlag'] = bacterium_status['division_occ']
                    dataframe.at[indx, 'daughters_index'] = bacterium_status['daughters_index']
                    dataframe.at[indx, 'division_time'] = bacterium_status['division_time']
                    dataframe.at[indx, 'divideFlag'] = bacterium_status['division_occ']
                    dataframe.at[indx, 'bad_division_flag'] = bacterium_status['bad_division_occ']
                    dataframe.at[indx, 'division_time'] = bacterium_status['division_time']

                dataframe.at[indx, 'parent_id'] = parent_id

            last_bacterium_in_life_history = bacterium_status['lifeHistoryIndex'][-1]
            dataframe.at[last_bacterium_in_life_history, 'unexpected_end'] = bacterium_status['unexpected_end']

            dataframe.at[last_bacterium_in_life_history, "daughter_length_to_mother"] = \
                bacterium_status['daughter_len'] / \
                dataframe.iloc[last_bacterium_in_life_history]["AreaShape_MajorAxisLength"]

            dataframe.at[last_bacterium_in_life_history, "max_daughter_len_to_mother"] = \
                bacterium_status["max_daughter_len"] / \
                dataframe.iloc[last_bacterium_in_life_history]["AreaShape_MajorAxisLength"]

            dataframe.at[row_index, 'transition'] = transition

            if bacterium_status['division_occ'] and not bacterium_status['bad_division_occ']:
                daughters_df = dataframe.iloc[bacterium_status['daughters_index']]

                daughter1_endpoints = find_vertex([daughters_df["AreaShape_Center_X"].values.tolist()[0],
                                                   daughters_df["AreaShape_Center_Y"].values.tolist()[0]],
                                                  daughters_df["AreaShape_MajorAxisLength"].values.tolist()[0],
                                                  daughters_df["AreaShape_Orientation"].values.tolist()[0])

                daughter2_endpoints = find_vertex([daughters_df["AreaShape_Center_X"].values.tolist()[1],
                                                   daughters_df["AreaShape_Center_Y"].values.tolist()[1]],
                                                  daughters_df["AreaShape_MajorAxisLength"].values.tolist()[1],
                                                  daughters_df["AreaShape_Orientation"].values.tolist()[1])

                slope_daughter1, intercept_daughter1 = calculate_slope_intercept(daughter1_endpoints[0],
                                                                                 daughter1_endpoints[1])

                slope_daughter2, intercept_daughter2 = calculate_slope_intercept(daughter2_endpoints[0],
                                                                                 daughter2_endpoints[1])

                # Calculate orientation angle
                daughters_orientation_angle = calculate_orientation_angle(slope_daughter1, slope_daughter2)
                dataframe.at[last_bacterium_in_life_history, "daughter_orientation"] = daughters_orientation_angle

            bacterium_id += 1

    # assign cell type
    if check_cell_type:
        dataframe = assign_cell_type(dataframe, intensity_threshold)
    dataframe.drop(labels='checked', axis=1, inplace=True)
    return dataframe


def data_cleaning(raw_df):
    """
    goal:   1. remove related rows to bacteria with zero MajorAxisLength
            2. Correct the labels of bacteria whose labels are nan.

    @param raw_df dataframe bacteria features value
    """

    # find bacteria with zero length
    zero_length_bac = raw_df.loc[raw_df["AreaShape_MajorAxisLength"] == 0]
    # remove related rows to bacteria with zero MajorAxisLength
    raw_df = raw_df.loc[raw_df["AreaShape_MajorAxisLength"] != 0]

    # columns name
    parent_image_number_col = [col for col in raw_df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in raw_df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    # The parent image number and parent object number of the daughters (or same bacteria) of the zero-length mother
    # should now be set to zero
    for indx, row in zero_length_bac.iterrows():
        parent_img_number = row['ImageNumber']
        parent_obj_number = row['ObjectNumber']

        daughters = raw_df.loc[(raw_df[parent_image_number_col] == parent_img_number) &
                               (raw_df[parent_object_number_col] == parent_obj_number)]

        for daughter_indx, daughter in daughters.iterrows():
            raw_df.at[daughter_indx, parent_image_number_col] = 0
            raw_df.at[daughter_indx, parent_object_number_col] = 0

    raw_df = raw_df.reset_index(drop=True)
    modified_df = modify_nan_labels(raw_df)

    return modified_df


def data_modification(dataframe, intensity_threshold, check_cell_type):
    # 1. remove related rows to bacteria with zero MajorAxisLength
    # 2. Correct the labels of bacteria whose labels are nan.
    dataframe = data_cleaning(dataframe)
    dataframe = assign_feature_find_errors(dataframe, intensity_threshold, check_cell_type)

    return dataframe


def data_conversion(dataframe, um_per_pixel=0.144):
    dataframe = convert_to_um(dataframe, um_per_pixel)
    dataframe = angle_convert_to_radian(dataframe)

    return dataframe


def find_fix_errors(dataframe, sorted_npy_files_list, neighbors_df, number_of_gap=0, um_per_pixel=0.144,
                    intensity_threshold=0.1, check_cell_type=True, interval_time=1, min_life_history_of_bacteria=20):
    logs_list = []

    dataframe = data_conversion(dataframe, um_per_pixel)
    dataframe = data_modification(dataframe, intensity_threshold, check_cell_type)

    data_preparation_time = time.time()
    data_preparation_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data_preparation_time))
    data_preparation_log = "At " + data_preparation_time_str + ',data preparation was completed.'
    print(data_preparation_log)
    logs_list.append(data_preparation_log)

    start_tracking_errors_correction_time = time.time()
    start_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                              time.localtime(start_tracking_errors_correction_time))

    start_tracking_errors_correction_log = "At " + start_tracking_errors_correction_time_str + \
                                           ", the correction of the tracking errors commenced."
    print(start_tracking_errors_correction_log)
    logs_list.append(start_tracking_errors_correction_log)

    # dataframe = merged_bacteria(dataframe, check_cell_type)

    # remove noise objects
    dataframe, noise_objects_log_list = noise_remover(dataframe)

    logs_list.extend(noise_objects_log_list)

    # more than two daughters
    dataframe = detect_remove_bad_daughters_to_mother_link(dataframe, sorted_npy_files_list, um_per_pixel)

    # redundant links
    dataframe = detect_and_remove_redundant_parent_link(dataframe, sorted_npy_files_list, um_per_pixel)

    # try to assign new link
    df, assign_new_link_log = correction_without_parent(dataframe, sorted_npy_files_list, neighbors_df, number_of_gap,
                                                        check_cell_type, interval_time, um_per_pixel,
                                                        min_life_history_of_bacteria)
    logs_list.extend(assign_new_link_log)

    # remove incorrect bacteria
    df = remove_rows(df, 'noise_bac', False)

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))

    end_tracking_errors_correction_log = "At " + end_tracking_errors_correction_time_str + \
                                         ", the corrections to the tracking errors were completed."

    print(end_tracking_errors_correction_log)
    logs_list.append(end_tracking_errors_correction_log)

    return df, logs_list
