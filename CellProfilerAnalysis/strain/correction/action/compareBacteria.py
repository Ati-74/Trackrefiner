from CellProfilerAnalysis.strain.correction.action.findOutlier import find_sum_daughter_len_to_mother_ratio_boundary, \
    find_max_daughter_len_to_mother_ratio_boundary, find_bac_len_to_bac_ratio_boundary
from CellProfilerAnalysis.strain.correction.action.fluorescenceIntensity import check_fluorescent_intensity
from CellProfilerAnalysis.strain.correction.action.helperFunctions import (find_vertex, calculate_slope_intercept,
                                                                           calculate_orientation_angle,
                                                                           calc_normalized_angle_between_motion,
                                                                           calculate_trajectory_direction,
                                                                           find_neighbors_info, calc_distance_matrix,
                                                                           distance_normalization)
from CellProfilerAnalysis.strain.correction.neighborChecking import check_num_neighbors
from CellProfilerAnalysis.strain.correction.action.costFinder import (adding_new_terms_to_cost_matrix,
                                                                      make_initial_distance_matrix)
from CellProfilerAnalysis.strain.correction.action.helperFunctions import calc_neighbors_dir_motion
from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np


def optimize_assignment(df):
    # Convert DataFrame to NumPy array
    cost_matrix = df.values

    # Applying the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Retrieve the original row indices and column names
    original_row_indices = df.index[row_ind]
    original_col_names = df.columns[col_ind]

    # Retrieve the costs for the selected elements
    selected_costs = cost_matrix[row_ind, col_ind]

    # Create a DataFrame for the results
    result_df = pd.DataFrame({
        'without parent index': original_row_indices,
        'Candida bacteria index in previous time step': original_col_names,
        'Cost': selected_costs
    })
    result_df['Total Minimized Cost'] = selected_costs.sum()

    return result_df


def compare_daughters_bacteria(daughter1, daughter2):
    endpoint12_distance_df = calc_distance_matrix(daughter1, daughter2, 'endppoint1_X', 'endppoint1_Y',
                                                  'endppoint2_X', 'endppoint2_Y')
    endpoint21_distance_df = calc_distance_matrix(daughter1, daughter2, 'endppoint2_X', 'endppoint2_Y',
                                                  'endppoint1_X', 'endppoint1_Y')

    # Concatenate the DataFrames
    combined_df = pd.concat([endpoint12_distance_df, endpoint21_distance_df])

    # Group by index and find the min value for each cell
    distance_df = combined_df.groupby(level=0).min()

    # Convert all elements to float, coercing errors to NaN
    distance_df = distance_df.applymap(lambda x: pd.to_numeric(x, errors='coerce'))

    return distance_df


def optimization_transition_cost(df, masks_dict, all_bac_in_without_source_time_step_df, without_source_bacteria,
                                 all_bacteria_in_prev_time_step, check_cellType, neighbors_df,
                                 min_life_history_of_bacteria):

    # note: check_fluorescent_intensity(transition_bac, candidate_parent_bacterium)

    neighbor_changes = df['difference_neighbors'].values.tolist()
    neighbor_changes = [v for v in neighbor_changes if v != '']
    max_neighbor_changes = max(neighbor_changes) + 1

    bac_len_to_bac_ratio_boundary = find_bac_len_to_bac_ratio_boundary(df)
    max_daughter_len_to_mother_ratio_boundary = find_max_daughter_len_to_mother_ratio_boundary(df)
    sum_daughter_len_to_mother_ratio_boundary = find_sum_daughter_len_to_mother_ratio_boundary(df)
    redundant_link_dict = {}

    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    all_bac_in_prev_time_step_without_noise_bac = \
        all_bacteria_in_prev_time_step.loc[all_bacteria_in_prev_time_step['noise_bac'] == False]

    overlap_df, distance_df = make_initial_distance_matrix(masks_dict, all_bac_in_without_source_time_step_df,
                                                           without_source_bacteria,
                                                           all_bacteria_in_prev_time_step,
                                                           all_bac_in_prev_time_step_without_noise_bac)

    normalized_distance_df = distance_normalization(df, distance_df)

    cost_df = np.sqrt(np.power(1 - overlap_df, 2) + np.power(normalized_distance_df, 2))

    # Replace 0s with NaN
    overlap_df_replaced = overlap_df.replace(0, np.nan)
    max_overlap_bac_indx = overlap_df_replaced.idxmax(axis=1).unique().tolist()
    min_distance_bac_ndx = distance_df.idxmin(axis=1).unique().tolist()

    nearest_bac_ndx = max_overlap_bac_indx
    nearest_bac_ndx.extend(min_distance_bac_ndx)

    unique_nearest_bac_ndx = [v for v in set(nearest_bac_ndx) if str(v) != 'nane']

    nearest_bac_next_time_step_df = df.loc[df.index.isin(unique_nearest_bac_ndx)]

    neighbors_bac_to_nearest_bac = neighbors_df.loc[(neighbors_df['First Image Number'] ==
                                                     nearest_bac_next_time_step_df['ImageNumber'].values.tolist()[0]) &
                                                    (neighbors_df['First Object Number'].isin(
                                                        nearest_bac_next_time_step_df['ObjectNumber'].values.tolist()))]

    if neighbors_bac_to_nearest_bac.shape[0] > 0:
        bac_under_invest_prev_time_step = df.loc[(df.index.isin(unique_nearest_bac_ndx)) |
                                                 ((df['ObjectNumber'].isin(
                                                     neighbors_bac_to_nearest_bac['Second Object Number'])) &
                                                  (df['ImageNumber'] ==
                                                   neighbors_bac_to_nearest_bac['Second Image Number'].values.tolist()[
                                                       0]))]
    else:
        bac_under_invest_prev_time_step = df.loc[df.index.isin(unique_nearest_bac_ndx)]

    receiver_of_bac_under_invest_link = \
        df.loc[(df[parent_object_number_col].isin(bac_under_invest_prev_time_step['ObjectNumber'])) &
               (df[parent_image_number_col] == bac_under_invest_prev_time_step['ImageNumber'].values.tolist()[0])]

    cost_df = cost_df[bac_under_invest_prev_time_step.index.values.tolist()]

    # now check the cost of maintaining the link
    maintenance_cost_df = calc_maintenance_cost(df, masks_dict, all_bacteria_in_prev_time_step,
                                                bac_under_invest_prev_time_step,
                                                all_bac_in_without_source_time_step_df, neighbors_df,
                                                receiver_of_bac_under_invest_link, F=True)

    # prev_maintenance_cost_df = calc_maintenance_cost(raw_df, masks_dict, all_bacteria_in_prev_time_step,
    #                                                 all_bacteria_in_prev_time_step,
    #                                                 all_bac_in_without_source_time_step_df,
    #                                                 neighbors_df, all_bac_in_without_source_time_step_df, F=True)

    for without_source_link_bac_ndx, without_source_link_bac in without_source_bacteria.iterrows():
        for candidate_source_bac_ndx in cost_df.columns:

            candidate_source_bac = df.loc[candidate_source_bac_ndx]

            cost_df, redundant_link_dict = \
                adding_new_terms_to_cost_matrix(df, masks_dict, cost_df, maintenance_cost_df,
                                                candidate_source_bac_ndx,
                                                candidate_source_bac,  without_source_link_bac_ndx,
                                                without_source_link_bac, neighbors_df, redundant_link_dict,
                                                max_neighbor_changes, bac_len_to_bac_ratio_boundary,
                                                sum_daughter_len_to_mother_ratio_boundary,
                                                max_daughter_len_to_mother_ratio_boundary,
                                                min_life_history_of_bacteria, all_bacteria_in_prev_time_step,
                                                all_bac_in_without_source_time_step_df)

    # Run the optimization
    result_df = optimize_assignment(cost_df)

    return result_df, redundant_link_dict


def division_detection_cost(df, masks_dict, source_incorrect_same_link, all_bac_in_source_time_step,
                            min_life_history_of_bacteria_time_step, target_incorrect_same_link,
                            all_bac_in_target_time_step, neighbors_bacteria_info, neighbors_indx_dict):

    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    max_daughter_len_to_mother_ratio_boundary = find_max_daughter_len_to_mother_ratio_boundary(df)
    sum_daughter_len_to_mother_ratio_boundary = find_sum_daughter_len_to_mother_ratio_boundary(df)

    overlap_df, distance_df = make_initial_distance_matrix(masks_dict, all_bac_in_source_time_step,
                                                           source_incorrect_same_link, all_bac_in_target_time_step,
                                                           neighbors_bacteria_info, daughter_flag=True)

    normalized_distance_df = distance_normalization(df, distance_df)

    cost_df = np.sqrt(np.power(1 - overlap_df, 2) + np.power(normalized_distance_df, 2))

    for source_bac_ndx, source_bac_cost in cost_df.iterrows():

        source_bac = df.iloc[source_bac_ndx]

        if source_bac['LifeHistory'] < min_life_history_of_bacteria_time_step:
            # it means this bacterium can not have a division
            cost_df.at[source_bac_ndx, :] = 999
        else:
            target_bac = target_incorrect_same_link.loc[target_incorrect_same_link[parent_object_number_col] ==
                                                        source_bac['ObjectNumber']]
            for col in cost_df.columns:
                if col not in neighbors_indx_dict[target_bac.index.values.tolist()[0]]:
                    cost_df.at[source_bac_ndx, col] = 999
                else:
                    max_daughter_len_to_mother = df.iloc[col]['AreaShape_MajorAxisLength'] / \
                                                 source_bac['AreaShape_MajorAxisLength']

                    sum_daughters_len_to_mother = \
                        (df.iloc[col]['AreaShape_MajorAxisLength'] +
                         target_bac['AreaShape_MajorAxisLength'].values.tolist()[0]) / \
                        source_bac['AreaShape_MajorAxisLength']

                    dist_daughter1_daughter2 = compare_daughters_bacteria(df.iloc[col].to_frame().transpose(),
                                                                          target_bac)

                    cost_df.at[source_bac_ndx, col] = np.sqrt(np.power(cost_df.loc[source_bac_ndx][col], 2) +
                                                              np.power(dist_daughter1_daughter2.values.tolist()[0], 2))

                    if max_daughter_len_to_mother >= 1 or max_daughter_len_to_mother > \
                            (max_daughter_len_to_mother_ratio_boundary['avg'] + 1.96 *
                             max_daughter_len_to_mother_ratio_boundary['std']):

                        cost_df.at[source_bac_ndx, col] = 999

                    elif sum_daughters_len_to_mother < sum_daughter_len_to_mother_ratio_boundary['avg'] - 1.96 * \
                            sum_daughter_len_to_mother_ratio_boundary['std'] or sum_daughters_len_to_mother > \
                            sum_daughter_len_to_mother_ratio_boundary['avg'] + 1.96 * \
                            sum_daughter_len_to_mother_ratio_boundary['std']:

                        cost_df.at[source_bac_ndx, col] = 999

    # Run the optimization
    result_df = optimize_assignment(cost_df)

    return result_df


def final_division_detection_cost(df, masks_dict, source_incorrect_same_link, all_bac_in_source_time_step,
                            min_life_history_of_bacteria_time_step, target_incorrect_same_link,
                            all_bac_in_target_time_step, neighbors_bacteria_info, neighbors_indx_dict):

    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    overlap_df, distance_df = make_initial_distance_matrix(masks_dict, all_bac_in_source_time_step,
                                                           source_incorrect_same_link,
                                                           all_bac_in_target_time_step, neighbors_bacteria_info,
                                                           daughter_flag=True)

    normalized_distance_df = distance_normalization(df, distance_df)

    cost_df = np.sqrt(np.power(1 - overlap_df, 2) + np.power(normalized_distance_df, 2))

    for source_bac_ndx, source_bac_cost in cost_df.iterrows():

        source_bac = df.iloc[source_bac_ndx]

        if source_bac['LifeHistory'] < min_life_history_of_bacteria_time_step:
            # it means this bacterium can not have a division
            cost_df.at[source_bac_ndx, :] = 999
        else:
            target_bac = target_incorrect_same_link.loc[target_incorrect_same_link[parent_object_number_col] ==
                                                        source_bac['ObjectNumber']]
            for col in cost_df.columns:
                if col not in neighbors_indx_dict[target_bac.index.values.tolist()[0]]:
                    cost_df.at[source_bac_ndx, col] = 999
                else:
                    max_daughter_len_to_mother = df.iloc[col]['AreaShape_MajorAxisLength'] / \
                                                 source_bac['AreaShape_MajorAxisLength']

                    dist_daughter1_daughter2 = compare_daughters_bacteria(df.iloc[col].to_frame().transpose(),
                                                                          target_bac)

                    cost_df.at[source_bac_ndx, col] = np.sqrt(np.power(cost_df.loc[source_bac_ndx][col], 2) +
                                                              np.power(dist_daughter1_daughter2.values.tolist()[0], 2))

                    if max_daughter_len_to_mother >= 1:
                        cost_df.at[source_bac_ndx, col] = 999

    # Run the optimization
    print(cost_df)
    print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
    result_df = optimize_assignment(cost_df)

    return result_df


def receive_new_link_cost(df, neighbors_df, masks_dict, neighbors_bacteria_info, all_bac_in_source_time_step,
                          target_incorrect_link, all_bac_in_target_time_step):

    df_for_max_neighbors = df.loc[df['noise_bac'] == False]
    neighbor_changes = df_for_max_neighbors['difference_neighbors'].values.tolist()
    neighbor_changes = [v for v in neighbor_changes if v != '']
    max_neighbor_changes = max(neighbor_changes) + 1

    bac_len_to_bac_ratio_boundary = find_bac_len_to_bac_ratio_boundary(df)

    overlap_df, distance_df = make_initial_distance_matrix(masks_dict, all_bac_in_source_time_step,
                                                           neighbors_bacteria_info, all_bac_in_target_time_step,
                                                           target_incorrect_link)

    normalized_distance_df = distance_normalization(df, distance_df)

    cost_df = np.sqrt(np.power(1 - overlap_df, 2) + np.power(normalized_distance_df, 2))

    for neighbors_bac_ndx, neighbors_bac_cost in cost_df.iterrows():

        neighbors_bac = df.iloc[neighbors_bac_ndx]

        neighbor_next_time_step = \
            all_bac_in_target_time_step.loc[all_bac_in_target_time_step['id'] == neighbors_bac['id']]

        neighbor_daughters_next_time_step = \
            all_bac_in_target_time_step.loc[all_bac_in_target_time_step['parent_id'] == neighbors_bac['id']]

        for col in cost_df.columns:
            if neighbor_daughters_next_time_step.shape[0] > 0 or neighbor_next_time_step.shape[0] > 0:
                cost_df.at[neighbors_bac_ndx, col] = 999
            else:
                bac_to_bac_length_ratio = df.iloc[col]['AreaShape_MajorAxisLength'] / \
                                          neighbors_bac['AreaShape_MajorAxisLength']

                neighbors_changes = check_num_neighbors(df, neighbors_df, df.iloc[col], neighbors_bac)

                cost_df.at[neighbors_bac_ndx, col] = np.sqrt(np.power(cost_df.loc[neighbors_bac_ndx][col], 2) +
                                                             np.power(neighbors_changes / max_neighbor_changes, 2))

                if bac_to_bac_length_ratio < bac_len_to_bac_ratio_boundary['avg'] - 1.96 * \
                        bac_len_to_bac_ratio_boundary['std']:
                    cost_df.at[neighbors_bac_ndx, col] = 999

    cost_df = cost_df[~(cost_df == 999).all(axis=1)]

    return cost_df


def adding_new_link_cost(df, neighbors_df, masks_dict, source_incorrect_same_link, all_bac_in_source_time_step,
                         target_incorrect_same_link,  all_bac_in_target_time_step, neighbors_bacteria_info,
                         neighbors_indx_dict):
    try:
        df['AreaShape_Center_X']
        center_str = 'AreaShape_'
    except:
        center_str = 'Location_'       

    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    neighbor_changes = df['difference_neighbors'].values.tolist()
    neighbor_changes = [v for v in neighbor_changes if v != '']
    max_neighbor_changes = max(neighbor_changes) + 1

    bac_len_to_bac_ratio_boundary = find_bac_len_to_bac_ratio_boundary(df)

    overlap_df, distance_df = make_initial_distance_matrix(masks_dict, all_bac_in_source_time_step,
                                                           source_incorrect_same_link,
                                                           all_bac_in_target_time_step, neighbors_bacteria_info)

    normalized_distance_df = distance_normalization(df, distance_df)

    cost_df = np.sqrt(np.power(1 - overlap_df, 2) + np.power(normalized_distance_df, 2))

    for source_bac_ndx, source_bac_cost in cost_df.iterrows():

        source_bac = df.iloc[source_bac_ndx]

        target_bac = target_incorrect_same_link.loc[target_incorrect_same_link[parent_object_number_col] ==
                                                    source_bac['ObjectNumber']]
        for col in cost_df.columns:
            if col not in neighbors_indx_dict[target_bac.index.values.tolist()[0]]:
                cost_df.at[source_bac_ndx, col] = 999
            else:
                bac_to_bac_length_ratio = df.iloc[col]['AreaShape_MajorAxisLength'] / \
                                          source_bac['AreaShape_MajorAxisLength']

                next_bac_endpoints = find_vertex([df.iloc[col][center_str + "Center_X"],
                                                  df.iloc[col][center_str + "Center_Y"]],
                                                 df.iloc[col]["AreaShape_MajorAxisLength"],
                                                 df.iloc[col]["AreaShape_Orientation"])

                source_bac_endpoints = find_vertex([source_bac[center_str + "Center_X"],
                                                    source_bac[center_str + "Center_Y"]],
                                                   source_bac["AreaShape_MajorAxisLength"],
                                                   source_bac["AreaShape_Orientation"])

                slope_next_bac, intercept_next_bac = calculate_slope_intercept(next_bac_endpoints[0],
                                                                               next_bac_endpoints[1])

                slope_source_bac, intercept_source_bac = calculate_slope_intercept(source_bac_endpoints[0],
                                                                                   source_bac_endpoints[1])

                angle_between_daughters = calculate_orientation_angle(slope_next_bac, slope_source_bac)

                neighbors_changes = check_num_neighbors(df, neighbors_df, df.iloc[col], source_bac)

                cost_df.at[source_bac_ndx, col] = np.sqrt(np.power(cost_df.loc[source_bac_ndx][col], 2) +
                                                          np.power(angle_between_daughters, 2) +
                                                          np.power(neighbors_changes / max_neighbor_changes, 2))

                if bac_to_bac_length_ratio < bac_len_to_bac_ratio_boundary['avg'] - 1.96 * \
                        bac_len_to_bac_ratio_boundary['std']:
                    cost_df.at[source_bac_ndx, col] = 999

    # Run the optimization
    result_df = optimize_assignment(cost_df)

    return result_df


def calc_maintenance_cost(df, masks_dict, all_bac_in_source_time_step, sel_source_bacteria_info,
                          all_bac_in_target_time_step, neighbor_df, sel_target_bacteria_info, F=False):
    try:
        df['AreaShape_Center_X']
        center_str = 'AreaShape_'
    except:
        center_str = 'Location_'       

    if sel_target_bacteria_info.shape[0] > 0 and sel_source_bacteria_info.shape[0] > 0:
        bac_len_to_bac_ratio_boundary = find_bac_len_to_bac_ratio_boundary(df)

        neighbors_overlap_df, neighbors_distance_df = \
            make_initial_distance_matrix(masks_dict, all_bac_in_source_time_step, sel_source_bacteria_info,
                                         all_bac_in_target_time_step, sel_target_bacteria_info, maintain=True)

        neighbor_changes = df['difference_neighbors'].values.tolist()
        neighbor_changes = [v for v in neighbor_changes if v != '']
        max_neighbor_changes = max(neighbor_changes) + 1

        normalized_distance_df = distance_normalization(df, neighbors_distance_df)

        maintenance_cost = np.sqrt(np.power(1 - neighbors_overlap_df, 2) + np.power(normalized_distance_df, 2))

        for row_indx in maintenance_cost.index:
            for col in maintenance_cost.columns:
                num_daughters = df.loc[(df['ImageNumber'] == sel_target_bacteria_info['ImageNumber'].values.tolist()[0]) &
                                       (df['parent_id'] == df.iloc[row_indx]['id'])]

                neighbors_dir_motion = calc_neighbors_dir_motion(df, df.iloc[row_indx], neighbor_df)
                direction_of_motion = \
                    calculate_trajectory_direction(
                        np.array([df.iloc[row_indx][center_str + "Center_X"],
                                  df.iloc[row_indx][center_str + "Center_Y"]]),
                        np.array([df.iloc[col][center_str + "Center_X"],
                                  df.iloc[col][center_str + "Center_Y"]]))

                if str(neighbors_dir_motion[0]) != 'nan':
                    angle_between_motion = calc_normalized_angle_between_motion(neighbors_dir_motion, direction_of_motion)
                else:
                    angle_between_motion = 0

                if num_daughters.shape[0] > 0:
                    maintenance_cost.at[row_indx, col] = np.sqrt(np.power(maintenance_cost.loc[row_indx][col], 2) +
                                                                 np.power(angle_between_motion, 2))
                else:
                    len_ratio = df.iloc[col]["AreaShape_MajorAxisLength"] / df.iloc[row_indx]["AreaShape_MajorAxisLength"]

                    source_life_history_before_target = \
                        df.loc[(df['ImageNumber'] < sel_target_bacteria_info['ImageNumber'].values.tolist()[0]) &
                               (df['id'] == df.iloc[row_indx]['id'])]

                    source_length_ratio = source_life_history_before_target['bac_length_to_back'].values.tolist()
                    source_length_ratio = [v for v in source_length_ratio if v != '']

                    if len(source_length_ratio) > 0:
                        source_length_ratio_avg = np.average(source_length_ratio)
                    else:
                        source_length_ratio_avg = 1

                    if maintenance_cost.loc[row_indx][col] != 999:
                        if len_ratio >= 1 or len_ratio >= bac_len_to_bac_ratio_boundary['avg'] + 1.96 * \
                                bac_len_to_bac_ratio_boundary['std']:
                            maintenance_cost.at[row_indx, col] = (
                                np.sqrt(np.power(maintenance_cost.loc[row_indx][col], 2) +
                                        np.power(df.iloc[col]["difference_neighbors"] /max_neighbor_changes, 2) +
                                        np.power(angle_between_motion, 2)))
                        else:
                            if source_life_history_before_target.shape[0] > 1:
                                maintenance_cost.at[row_indx, col] = (
                                    np.sqrt(np.power(maintenance_cost.loc[row_indx][col], 2) +
                                            np.power(source_length_ratio_avg - len_ratio, 2) +
                                            np.power(df.iloc[col]["difference_neighbors"] / max_neighbor_changes, 2) +
                                            np.power(angle_between_motion, 2)
                                            ))
                            else:
                                maintenance_cost.at[row_indx, col] = (
                                    np.sqrt(np.power(maintenance_cost.loc[row_indx][col], 2) +
                                            np.power(1 - len_ratio, 2) +
                                            np.power(df.iloc[col]["difference_neighbors"] / max_neighbor_changes, 2) +
                                            np.power(angle_between_motion, 2)
                                            ))
    else:
        maintenance_cost = pd.DataFrame(columns=[sel_target_bacteria_info.index],
                                           index=sel_source_bacteria_info.index, data=999)

    return maintenance_cost


def adding_new_link_to_unexpected(df, neighbors_df, masks_dict, unexpected_end_bac_in_current_time_step,
                                  all_bac_in_current_time_step, all_bac_in_next_time_step,
                                  min_life_history_of_bacteria_time_step):
    try:
        df['AreaShape_Center_X']
        center_str = 'AreaShape_'
    except:
        center_str = 'Location_'   

    neighbor_changes = df['difference_neighbors'].values.tolist()
    neighbor_changes = [v for v in neighbor_changes if v != '']
    max_neighbor_changes = max(neighbor_changes) + 1

    bac_len_to_bac_ratio_boundary = find_bac_len_to_bac_ratio_boundary(df)

    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    all_bac_in_next_time_step_without_noise_bac = \
        all_bac_in_next_time_step.loc[all_bac_in_next_time_step['noise_bac'] == False]

    overlap_df, distance_df = make_initial_distance_matrix(masks_dict, all_bac_in_current_time_step,
                                                           unexpected_end_bac_in_current_time_step,
                                                           all_bac_in_next_time_step,
                                                           all_bac_in_next_time_step_without_noise_bac)

    normalized_distance_df = distance_normalization(df, distance_df)

    cost_df = np.sqrt(np.power(1 - overlap_df, 2) + np.power(normalized_distance_df, 2))

    # Replace 0s with NaN
    overlap_df_replaced = overlap_df.replace(0, np.nan)
    max_overlap_bac_indx = overlap_df_replaced.idxmax(axis=1).unique().tolist()
    min_distance_bac_ndx = distance_df.idxmin(axis=1).unique().tolist()

    nearest_bac_ndx = max_overlap_bac_indx
    nearest_bac_ndx.extend(min_distance_bac_ndx)

    unique_nearest_bac_ndx = [v for v in set(nearest_bac_ndx) if str(v) != 'nane']

    nearest_bac_next_time_step_df = df.loc[df.index.isin(unique_nearest_bac_ndx)]
    # nearest_bac_next_time_step_df = df.loc[df.index.isin(overlap_df.columns.values.tolist())]

    neighbors_bac_to_nearest_bac = neighbors_df.loc[(neighbors_df['First Image Number'] ==
                                                     nearest_bac_next_time_step_df['ImageNumber'].values.tolist()[0]) &
                                                    (neighbors_df['First Object Number'].isin(
                                                        nearest_bac_next_time_step_df['ObjectNumber'].values.tolist()))]

    if neighbors_bac_to_nearest_bac.shape[0] > 0:
        bac_under_invest = df.loc[(df.index.isin(unique_nearest_bac_ndx)) |
                                  ((df['ObjectNumber'].isin(neighbors_bac_to_nearest_bac['Second Object Number'])) &
                                   (df['ImageNumber'] ==
                                    neighbors_bac_to_nearest_bac['Second Image Number'].values.tolist()[0]))]
    else:
        bac_under_invest = df.loc[df.index.isin(unique_nearest_bac_ndx)]

    source_of_bac_under_invest_link = \
        df.loc[(df['ObjectNumber'].isin(bac_under_invest[parent_object_number_col])) &
               (df['ImageNumber'] == bac_under_invest['ImageNumber'].values.tolist()[0] - 1)]

    cost_df = cost_df[bac_under_invest.index.values.tolist()]

    # now check the cost of maintaining the link
    maintenance_cost_df = calc_maintenance_cost(df, masks_dict, all_bac_in_current_time_step,
                                                    source_of_bac_under_invest_link,
                                                    all_bac_in_next_time_step, neighbors_df, bac_under_invest, F=True)

    candidate_new_bac_daughter_list_id = {}

    for unexpected_ndx, unexpected_bac in unexpected_end_bac_in_current_time_step.iterrows():
        for col in cost_df.columns:

            unexpected_bac_life_history = df.loc[df['id'] == unexpected_bac['id']]
            next_bac_life_history = df.loc[
                (df['id'] == df.iloc[col]["id"]) & (df['ImageNumber'] < df.iloc[col]["ImageNumber"])]
            unexpected_bac_bac_length_to_back = [v for v in unexpected_bac_life_history['bac_length_to_back'] if
                                                 v != '']

            if len(unexpected_bac_bac_length_to_back) > 0:
                unexpected_bac_bac_length_to_back_avg = np.average(unexpected_bac_bac_length_to_back)
            else:
                unexpected_bac_bac_length_to_back_avg = 1

            if next_bac_life_history.shape[0] == 0 or next_bac_life_history.shape[0] >= 1:

                length_ratio = df.iloc[col]['AreaShape_MajorAxisLength'] / \
                               df.iloc[unexpected_ndx]['AreaShape_MajorAxisLength']

                neighbors_dir_motion = calc_neighbors_dir_motion(df, unexpected_bac, neighbors_df)

                direction_of_motion = \
                    calculate_trajectory_direction(np.array([df.iloc[unexpected_ndx][center_str + "Center_X"],
                                                             df.iloc[unexpected_ndx][center_str + "Center_Y"]]),
                                                   np.array([df.iloc[col][center_str + "Center_X"],
                                                             df.iloc[col][center_str + "Center_Y"]]))

                if str(neighbors_dir_motion[0]) != 'nan':
                    angle_between_motion = calc_normalized_angle_between_motion(neighbors_dir_motion,
                                                                                direction_of_motion)
                else:
                    angle_between_motion = 0

                # check neighbors
                difference_neighbors, common_neighbors = check_num_neighbors(df, neighbors_df, df.iloc[unexpected_ndx],
                                                                             df.iloc[col], return_common_elements=True)

                if difference_neighbors > common_neighbors:
                    cost_df.at[unexpected_ndx, col] = 999

                elif length_ratio >= 1 or length_ratio >= \
                        bac_len_to_bac_ratio_boundary['avg'] - 1.96 * bac_len_to_bac_ratio_boundary['std']:

                    if length_ratio < 1:
                        cost_df.at[unexpected_ndx, col] = \
                            np.sqrt(np.power(cost_df.loc[unexpected_ndx][col], 2) +
                                    np.power(angle_between_motion, 2) +
                                    np.power(unexpected_bac_bac_length_to_back_avg - length_ratio, 2) +
                                    np.power(difference_neighbors / max_neighbor_changes, 2))

                    else:
                        cost_df.at[unexpected_ndx, col] = np.sqrt(np.power(cost_df.loc[unexpected_ndx][col], 2) +
                                                                  np.power(angle_between_motion, 2) +
                                                                  np.power(difference_neighbors / max_neighbor_changes,
                                                                           2))
                else:
                    candidate_daughter_ndx = neighboring_cost(df, neighbors_df, cost_df, maintenance_cost_df,
                                                              unexpected_bac, unexpected_ndx, df.iloc[col])

                    if len(candidate_daughter_ndx) > 0:
                        candidate_new_bac_daughter_list_id[col] = candidate_daughter_ndx
                        cost_df.at[unexpected_ndx, col] = np.sqrt(np.power(cost_df.loc[unexpected_ndx][col], 2) +
                                                                  np.power(angle_between_motion, 2) +
                                                                  np.power(difference_neighbors / max_neighbor_changes,
                                                                           2)
                                                                  )
                    else:
                        cost_df.at[unexpected_ndx, col] = 999

    cost_before_compare_with_maintenance_df = cost_df

    for bac_ndx, bac in bac_under_invest.iterrows():

        source_of_this_link = df.loc[(df['ObjectNumber'] == bac[parent_object_number_col]) &
                                     (df['ImageNumber'] == (bac['ImageNumber'] - 1))]

        if source_of_this_link.shape[0] > 0:
            if maintenance_cost_df.loc[source_of_this_link.index.values.tolist()[0]][bac_ndx] <= cost_df[bac_ndx].min():

                cost_df = cost_df.drop(columns=[bac_ndx])
                cost_before_compare_with_maintenance_df[bac_ndx] = 999

                if bac_ndx in candidate_new_bac_daughter_list_id.keys():
                    candidate_new_bac_daughter_list_id.pop(bac_ndx)

            else:
                cost_df[bac_ndx][cost_df[bac_ndx] >= \
                                 maintenance_cost_df.loc[source_of_this_link.index.values.tolist()[0]][bac_ndx]] = 999

    # Run the optimization
    result_df = optimize_assignment(cost_df)

    return result_df, candidate_new_bac_daughter_list_id, cost_before_compare_with_maintenance_df, maintenance_cost_df


def neighboring_cost(df, neighbors_df, cost_df, maintenance_cost_df, source_bac, source_bac_ndx, target_bac):
    target_bac_neighbors_df = find_neighbors_info(df, neighbors_df, target_bac)
    max_daughter_len_boundary = find_max_daughter_len_to_mother_ratio_boundary(df)
    sum_daughter_len_boundary = find_sum_daughter_len_to_mother_ratio_boundary(df)

    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    candidate_daughter_ndx = []

    for neighbor_bac_ndx, neighbor_bac in target_bac_neighbors_df.iterrows():
        length_ratio = neighbor_bac['AreaShape_MajorAxisLength'] / source_bac['AreaShape_MajorAxisLength']
        sum_length_ratio = (neighbor_bac['AreaShape_MajorAxisLength'] + target_bac['AreaShape_MajorAxisLength']) / \
                           source_bac['AreaShape_MajorAxisLength']

        if length_ratio >= 1 or length_ratio > max_daughter_len_boundary['avg'] + 1.96 * max_daughter_len_boundary[
            'std']:
            # it can not be a good candidate
            pass
        elif sum_length_ratio > sum_daughter_len_boundary['avg'] + 1.96 * sum_daughter_len_boundary['std'] or \
                sum_length_ratio < sum_daughter_len_boundary['avg'] - 1.96 * sum_daughter_len_boundary['std']:
            # it can not be a good candidate
            pass
        else:
            prev_link_to_neighbor = df.loc[(df['ImageNumber'] == neighbor_bac[parent_image_number_col]) &
                                           (df['ObjectNumber'] == neighbor_bac[parent_object_number_col])]

            if prev_link_to_neighbor.shape[0] > 0:
                if neighbor_bac_ndx in maintenance_cost_df.columns.values.tolist():
                    maintenance_cost = \
                        maintenance_cost_df.loc[prev_link_to_neighbor.index][neighbor_bac_ndx].values.tolist()[0]
                    new_link_cost = cost_df.loc[source_bac_ndx][neighbor_bac_ndx]
                    if new_link_cost < maintenance_cost:
                        candidate_daughter_ndx.append(neighbor_bac_ndx)
            else:
                candidate_daughter_ndx.append(neighbor_bac_ndx)

    return candidate_daughter_ndx
