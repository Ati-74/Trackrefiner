import numpy as np
import pandas as pd

from CellProfilerAnalysis.strain.correction.action.helperFunctions import distance_normalization, \
    calculate_trajectory_direction, calc_neighbors_dir_motion
from CellProfilerAnalysis.strain.correction.action.compareBacteria import (make_initial_distance_matrix,
                                                                           receive_new_link_cost, optimize_assignment)
from CellProfilerAnalysis.strain.correction.action.findOutlier import find_daughter_len_to_mother_ratio_outliers, \
    find_bac_len_to_bac_ratio_boundary
from CellProfilerAnalysis.strain.correction.action.bacteriaModification import remove_redundant_link


def detect_and_remove_redundant_parent_link(dataframe, neighbor_df, sorted_npy_files_list, logs_df):
    try:
        dataframe['AreaShape_Center_X']
        center_str = 'AreaShape_'
    except:
        center_str = 'Location_'      

    num_redundant_links = None

    while num_redundant_links != 0:

        dataframe_for_neighbors = dataframe.loc[dataframe['noise_bac'] == False]
        neighbor_changes = dataframe_for_neighbors['difference_neighbors'].values.tolist()
        neighbor_changes = [v for v in neighbor_changes if v != '']
        max_neighbor_changes = max(neighbor_changes)

        # note: it's only for division (mother - daughters relation)
        # check daughter length (sum daughters length or max daughter length) to mother length
        daughters_to_mother_ratio_list_outliers = find_daughter_len_to_mother_ratio_outliers(dataframe)

        bacteria_with_redundant_parent_link_error = \
            dataframe.loc[(dataframe["daughter_length_to_mother"].isin(
                daughters_to_mother_ratio_list_outliers['daughter_length_to_mother'])) |
                          (dataframe["max_daughter_len_to_mother"].isin(
                              daughters_to_mother_ratio_list_outliers['max_daughter_len_to_mother']))
                          ]

        num_redundant_links = bacteria_with_redundant_parent_link_error.shape[0]

        # print("number of redundant parent link error: ")
        # print(bacteria_with_redundant_parent_link_error.shape[0])
        # print("more information: ")
        # print(bacteria_with_redundant_parent_link_error)

        bac_len_to_bac_ratio_boundary = find_bac_len_to_bac_ratio_boundary(dataframe)

        for parent_indx, parent_with_redundant_link in bacteria_with_redundant_parent_link_error.iterrows():

            bacteria_in_current_time_step = dataframe.loc[
                dataframe['ImageNumber'] == parent_with_redundant_link['ImageNumber']]

            daughters_at_first_time_step_of_life_history = \
                dataframe.loc[(dataframe['parent_id'] == dataframe.iloc[parent_indx]['id']) &
                              (dataframe['ImageNumber'] == parent_with_redundant_link['ImageNumber'] + 1)]

            bac_len_to_bac_ratio_df = daughters_at_first_time_step_of_life_history['AreaShape_MajorAxisLength'] / \
                                      parent_with_redundant_link['AreaShape_MajorAxisLength']

            candidate_daughters_at_first_time_step_of_life_history = \
                daughters_at_first_time_step_of_life_history[bac_len_to_bac_ratio_df >
                                                             bac_len_to_bac_ratio_boundary['avg'] -
                                                             1.96 * bac_len_to_bac_ratio_boundary['std']]

            incorrect_daughters_at_first_time_step_of_life_history = \
                daughters_at_first_time_step_of_life_history[bac_len_to_bac_ratio_df <
                                                             bac_len_to_bac_ratio_boundary['avg'] -
                                                             1.96 * bac_len_to_bac_ratio_boundary['std']]

            if candidate_daughters_at_first_time_step_of_life_history.shape[0] == 0:
                candidate_daughters_at_first_time_step_of_life_history = \
                    incorrect_daughters_at_first_time_step_of_life_history

            if candidate_daughters_at_first_time_step_of_life_history.shape[0] > 1:

                bacteria_in_next_time_step = dataframe.loc[dataframe['ImageNumber'] ==
                                                           parent_with_redundant_link['ImageNumber'] + 1]

                # check the cost of daughters to mother
                overlap_df, distance_df = make_initial_distance_matrix(sorted_npy_files_list, bacteria_in_current_time_step,
                                                                       parent_with_redundant_link.to_frame().transpose(),
                                                                       bacteria_in_next_time_step,
                                                                       daughters_at_first_time_step_of_life_history)

                normalized_distance_df = distance_normalization(dataframe, distance_df)

                cost_df = np.sqrt((1 - overlap_df) ** 2 + normalized_distance_df ** 2)

                neighbors_bacteria_obj_nums = \
                    neighbor_df.loc[
                        (neighbor_df['First Image Number'] == parent_with_redundant_link['ImageNumber']) &
                        (neighbor_df['First Object Number'] ==
                         parent_with_redundant_link['ObjectNumber'])]['Second Object Number'].values.tolist()

                neighbors_bacteria_info = \
                    dataframe.loc[(dataframe['ImageNumber'] == parent_with_redundant_link['ImageNumber']) &
                                  (dataframe['ObjectNumber'].isin(neighbors_bacteria_obj_nums))]

                parent_life_history = dataframe.loc[dataframe['id'] == parent_with_redundant_link['id']]
                parent_increase_length_ratio = [v for v in parent_life_history['bac_length_to_back'] if v != '']

                if len(parent_increase_length_ratio) > 0:
                    parent_increase_length_ratio_avg = np.average(parent_increase_length_ratio)
                else:
                    parent_increase_length_ratio_avg = 1

                neighbors_dir_motion = calc_neighbors_dir_motion(dataframe, parent_with_redundant_link, neighbor_df)

                daughter_neighbor_changes = (
                    max(candidate_daughters_at_first_time_step_of_life_history['difference_neighbors'].values.tolist()))

                if daughter_neighbor_changes != 0:
                    max_neighbor_changes = daughter_neighbor_changes

                for daughter_bac_ndx, daughter_bac in candidate_daughters_at_first_time_step_of_life_history.iterrows():
                    length_ratio = daughter_bac["AreaShape_MajorAxisLength"] / \
                                   parent_with_redundant_link["AreaShape_MajorAxisLength"]

                    direction_of_motion = \
                        calculate_trajectory_direction(
                            np.array([dataframe.iloc[parent_indx][center_str + "Center_X"],
                                      dataframe.iloc[parent_indx][center_str + "Center_Y"]]),
                            np.array([dataframe.iloc[daughter_bac_ndx][center_str + "Center_X"],
                                      dataframe.iloc[daughter_bac_ndx][center_str + "Center_Y"]]))

                    if str(neighbors_dir_motion[0]) != 'nan':
                        diff_direction_motion = np.sqrt(np.power(direction_of_motion[0] - neighbors_dir_motion[0], 2) +
                                                        np.power(direction_of_motion[1] - neighbors_dir_motion[1], 2))
                    else:
                        diff_direction_motion = 0

                    if parent_life_history.shape[0] > 1:
                        if length_ratio >= 1 or length_ratio >= bac_len_to_bac_ratio_boundary['avg'] + 1.96 * \
                            bac_len_to_bac_ratio_boundary['std']:
                            cost_df.at[parent_indx, daughter_bac_ndx] = \
                                np.sqrt(np.power(cost_df.at[parent_indx, daughter_bac_ndx], 2)+ \
                                        np.power(diff_direction_motion, 2) + \
                                        np.power(daughter_bac['difference_neighbors'] / max_neighbor_changes, 2))

                        else:
                            cost_df.at[parent_indx, daughter_bac_ndx] = \
                                np.sqrt(np.power(cost_df.at[parent_indx, daughter_bac_ndx], 2) + \
                                        np.power(diff_direction_motion, 2) + \
                                        np.power(parent_increase_length_ratio_avg - length_ratio, 2) + \
                                        np.power(daughter_bac['difference_neighbors'] / max_neighbor_changes, 2))

                    else:
                        if length_ratio >= 1 or length_ratio >= bac_len_to_bac_ratio_boundary['avg'] + 1.96 * \
                                bac_len_to_bac_ratio_boundary['std']:
                            cost_df.at[parent_indx, daughter_bac_ndx] = \
                                np.sqrt(np.power(cost_df.at[parent_indx, daughter_bac_ndx], 2) + \
                                        np.power(diff_direction_motion, 2) +
                                        np.power(daughter_bac['difference_neighbors'] / max_neighbor_changes, 2)
                                        )
                        else:
                            cost_df.at[parent_indx, daughter_bac_ndx] = \
                                np.sqrt(np.power(cost_df.at[parent_indx, daughter_bac_ndx], 2) + \
                                        np.power(diff_direction_motion, 2) + \
                                        np.power(1 - length_ratio, 2) +
                                        np.power(daughter_bac['difference_neighbors'] / max_neighbor_changes, 2)
                                        )

                new_link_cost_df = receive_new_link_cost(dataframe, neighbor_df, sorted_npy_files_list, neighbors_bacteria_info,
                                                         bacteria_in_current_time_step,
                                                         candidate_daughters_at_first_time_step_of_life_history,
                                                         bacteria_in_next_time_step)

                final_cost = pd.concat([cost_df, new_link_cost_df])
                result_df = optimize_assignment(final_cost)

                if parent_indx in result_df['without parent index'].values.tolist():
                    correct_daughter = result_df.loc[result_df['without parent index'] == parent_indx]['Candida bacteria index in previous time step'].values.tolist()[0]
                else:
                    correct_daughter = cost_df.min().idxmin()

                wrong_daughter_index = [v for v in cost_df.columns.values.tolist() if v != correct_daughter][0]

                wrong_daughter_life_history = dataframe.loc[
                    dataframe['id'] == dataframe.iloc[wrong_daughter_index]['id']]

            else:
                wrong_daughter_life_history = dataframe.loc[dataframe['id'] ==
                                                            incorrect_daughters_at_first_time_step_of_life_history[
                                                                'id'].values.tolist()[0]]

            dataframe = remove_redundant_link(dataframe, wrong_daughter_life_history, neighbor_df)
            logs_df = pd.concat([logs_df, wrong_daughter_life_history.iloc[0].to_frame().transpose()],
                                ignore_index=True)
    return dataframe, logs_df
