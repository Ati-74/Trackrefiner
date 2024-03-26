import numpy as np
import pandas as pd
import time
from CellProfilerAnalysis.strain.correction.action.helperFunctions import bacteria_life_history, remove_rows, \
    convert_to_um, find_vertex, angle_convert_to_radian, calculate_slope_intercept, \
    adding_features_to_each_timestep_except_first, adding_features_only_for_last_time_step, \
    adding_features_related_to_division, calc_distance_between_daughters
from CellProfilerAnalysis.strain.correction.action.fluorescenceIntensity import assign_cell_type
from CellProfilerAnalysis.strain.correction.nanLabel import modify_nan_labels
from CellProfilerAnalysis.strain.correction.withoutParentError import correction_without_parent
from CellProfilerAnalysis.strain.correction.badDaughters import detect_remove_bad_daughters_to_mother_link
from CellProfilerAnalysis.strain.correction.redundantParentLink import detect_and_remove_redundant_parent_link
from CellProfilerAnalysis.strain.correction.noiseRemover import noise_remover
from CellProfilerAnalysis.strain.correction.neighborChecking import neighbor_checking
from CellProfilerAnalysis.strain.correction.incorrectSameLink import incorrect_same_link
from CellProfilerAnalysis.strain.correction.unExpectedEnd import unexpected_end_bacteria
from CellProfilerAnalysis.strain.correction.action.multiRegionsCorrection import multi_region_correction
from CellProfilerAnalysis.strain.correction.action.finalMatching import final_matching


def assign_feature_find_errors(dataframe, intensity_threshold, check_cell_type, neighbor_df):
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
    dataframe["difference_neighbors"] = 0
    dataframe["parent_id"] = ''
    dataframe["daughter_orientation"] = ''
    dataframe["daughter_length_to_mother"] = ''
    dataframe["daughters_distance"] = ''
    dataframe["bac_length_to_back"] = ''
    dataframe["next_to_first_bac_length_ratio"] = ''
    dataframe["mother_last_to_first_bac_length_ratio"] = ''
    dataframe["bac_length_to_back_orientation_changes"] = ''
    dataframe["max_daughter_len_to_mother"] = ''
    dataframe["bacteria_movement"] = ''
    dataframe["bacteria_slope"] = ''
    dataframe["direction_of_motion"] = 0
    dataframe["angle_between_neighbor_motion_bac_motion"] = 0

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

            dataframe.at[row_index, 'transition'] = transition

            for list_indx, indx in enumerate(bacterium_status['lifeHistoryIndex']):
                dataframe.at[indx, 'checked'] = True
                
                try:
                    center_x = dataframe.iloc[indx]["AreaShape_Center_X"]
                    center_y = dataframe.iloc[indx]["AreaShape_Center_Y"]
                except:
                    center_x = dataframe.iloc[indx]["Location_Center_X"]
                    center_y = dataframe.iloc[indx]["Location_Center_Y"]
                
                current_bacterium_endpoints = find_vertex(
                    [center_x, center_y],
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

                dataframe.at[indx, 'LifeHistory'] = bacterium_status['life_history']

                if list_indx > 0:
                    dataframe = adding_features_to_each_timestep_except_first(dataframe, list_indx, indx,
                                                                              bacterium_status, neighbor_df)

                if bacterium_status['division_occ']:
                    dataframe = adding_features_related_to_division(dataframe, indx, bacterium_status)

                    if indx == bacterium_status['lifeHistoryIndex'][-1] and not bacterium_status["bad_division_occ"]:
                        dataframe.at[indx, 'daughters_distance'] = \
                            calc_distance_between_daughters(dataframe, bacterium_status["daughters_index"][0],
                                                            bacterium_status["daughters_index"][1])

                    if list_indx > 0 and indx != bacterium_status['lifeHistoryIndex'][-1]:
                        dataframe.at[indx, "next_to_first_bac_length_ratio"] = \
                            dataframe.iloc[indx]["AreaShape_MajorAxisLength"] / \
                            dataframe.iloc[bacterium_status['lifeHistoryIndex'][0]]["AreaShape_MajorAxisLength"]

                    elif list_indx > 0:
                        dataframe.at[indx, "mother_last_to_first_bac_length_ratio"] = \
                            dataframe.iloc[bacterium_status['lifeHistoryIndex'][-1]][
                                "AreaShape_MajorAxisLength"] / \
                            dataframe.iloc[bacterium_status['lifeHistoryIndex'][0]]["AreaShape_MajorAxisLength"]
                else:
                    if list_indx > 0:
                        dataframe.at[indx, "next_to_first_bac_length_ratio"] = \
                            dataframe.iloc[indx]["AreaShape_MajorAxisLength"] / \
                            dataframe.iloc[bacterium_status['lifeHistoryIndex'][0]]["AreaShape_MajorAxisLength"]

                dataframe.at[indx, 'parent_id'] = parent_id

            last_bacterium_in_life_history = bacterium_status['lifeHistoryIndex'][-1]
            dataframe = adding_features_only_for_last_time_step(dataframe, last_bacterium_in_life_history,
                                                                bacterium_status)

            bacterium_id += 1

    # assign cell type
    if check_cell_type:
        dataframe = assign_cell_type(dataframe, intensity_threshold)

    # dataframe.drop(labels='checked', axis=1, inplace=True)
    return dataframe


def data_cleaning(raw_df):
    """
    goal:   Correct the labels of bacteria whose labels are nan.

    @param raw_df dataframe bacteria features value
    """

    modified_df = modify_nan_labels(raw_df)

    return modified_df


def data_modification(dataframe, intensity_threshold, check_cell_type, neighbors_df):
    # 1. remove related rows to bacteria with zero MajorAxisLength
    # 2. Correct the labels of bacteria whose labels are nan.
    dataframe = data_cleaning(dataframe)
    dataframe = assign_feature_find_errors(dataframe, intensity_threshold, check_cell_type, neighbors_df)

    return dataframe


def data_conversion(dataframe, um_per_pixel=0.144):
    dataframe['color_mask'] = ''
    dataframe = convert_to_um(dataframe, um_per_pixel)
    dataframe = angle_convert_to_radian(dataframe)

    return dataframe


def find_fix_errors(dataframe, sorted_npy_files_list, neighbors_df, number_of_gap=0, um_per_pixel=0.144,
                    intensity_threshold=0.1, check_cell_type=True, interval_time=1, min_life_history_of_bacteria=20):

    logs_list = []
    logs_df = pd.DataFrame(columns=dataframe.columns)

    dataframe = data_conversion(dataframe, um_per_pixel)

    # correction of multi regions
    dataframe, neighbors_df = multi_region_correction(dataframe, sorted_npy_files_list, neighbors_df, um_per_pixel)

    dataframe = data_modification(dataframe, intensity_threshold, check_cell_type, neighbors_df)

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

    raw_data_frame = dataframe.copy()

    # dataframe = merged_bacteria(dataframe, check_cell_type)
    # remove noise objects
    dataframe, neighbors_df, noise_objects_log_list, logs_df = noise_remover(dataframe, neighbors_df, logs_df)

    logs_list.extend(noise_objects_log_list)

    # check neighbors
    dataframe = neighbor_checking(dataframe, neighbors_df)

    # more than two daughters
    dataframe, logs_df = detect_remove_bad_daughters_to_mother_link(dataframe, neighbors_df, sorted_npy_files_list,
                                                                    logs_df)

    # redundant links
    dataframe, logs_df = detect_and_remove_redundant_parent_link(dataframe, neighbors_df, sorted_npy_files_list, logs_df)

    # try to assign new link
    df, assign_new_link_log, logs_df = correction_without_parent(dataframe, neighbors_df, sorted_npy_files_list,
                                                                 number_of_gap, check_cell_type, interval_time,
                                                                 min_life_history_of_bacteria, logs_df)
                                                                 
    logs_list.extend(assign_new_link_log)

    df, logs_df = incorrect_same_link(df, neighbors_df, sorted_npy_files_list, min_life_history_of_bacteria,
                                      interval_time, logs_df)

    df, logs_df = unexpected_end_bacteria(df, neighbors_df, sorted_npy_files_list, min_life_history_of_bacteria,
                                          interval_time, logs_df)

    df, logs_df = final_matching(df, neighbors_df, min_life_history_of_bacteria, interval_time, sorted_npy_files_list,
                                 logs_df)

    df = remove_rows(df, 'noise_bac', False)

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))

    end_tracking_errors_correction_log = "At " + end_tracking_errors_correction_time_str + \
                                         ", the corrections to the tracking errors were completed."

    print(end_tracking_errors_correction_log)
    logs_list.append(end_tracking_errors_correction_log)

    logs_df = logs_df.sort_values(by=['ImageNumber', 'ObjectNumber'])
    logs_df = logs_df.drop_duplicates(subset=['ImageNumber', 'ObjectNumber'], keep='last')
    logs_df.rename(columns={'ImageNumber': 'stepNum', 'AreaShape_MajorAxisLength': 'length'}, inplace=True)
    logs_df = logs_df[["stepNum", "ObjectNumber", "length",	"noise_bac", "unexpected_end",	"transition"]]

    return df, logs_list, logs_df, neighbors_df
