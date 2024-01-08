from CellProfilerAnalysis.strain.correction.action.findOutlier import find_sum_daughter_len_to_mother_ratio_boundary, \
    find_max_daughter_len_to_mother_ratio_boundary, find_bac_len_to_bac_ratio_boundary, find_bac_movement_boundary
from CellProfilerAnalysis.strain.correction.action.findOverlap import find_overlap_object_to_next_frame
from CellProfilerAnalysis.strain.correction.action.helperFunctions import calc_distance_matrix, distance_normalization
from CellProfilerAnalysis.strain.correction.action.fluorescenceIntensity import check_fluorescent_intensity
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


def compare_bacteria(current_img_npy, current_time_step_df, target_bacteria, next_time_step_img_npy,
                     bacteria_in_next_time_step, candidate_bacteria, um_per_pixel):

    overlap_df = find_overlap_object_to_next_frame(current_img_npy, current_time_step_df, target_bacteria,
                                                   next_time_step_img_npy, bacteria_in_next_time_step,
                                                   candidate_bacteria, um_per_pixel)

    # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)
    try:
        center_distance_df = calc_distance_matrix(target_bacteria, candidate_bacteria,
                                                  'Location_Center_X', 'Location_Center_Y')
    except TypeError:
        center_distance_df = calc_distance_matrix(target_bacteria, candidate_bacteria,
                                                  'AreaShape_Center_X', 'AreaShape_Center_Y')

    endpoint1_distance_df = calc_distance_matrix(target_bacteria, candidate_bacteria, 'endppoint1_X', 'endppoint1_Y')
    endpoint2_distance_df = calc_distance_matrix(target_bacteria, candidate_bacteria, 'endppoint2_X', 'endppoint2_Y')

    endpoint12_distance_df = calc_distance_matrix(target_bacteria, candidate_bacteria, 'endppoint1_X', 'endppoint1_Y',
                                                  'endppoint2_X', 'endppoint2_Y')
    endpoint21_distance_df = calc_distance_matrix(target_bacteria, candidate_bacteria, 'endppoint2_X', 'endppoint2_Y',
                                                  'endppoint1_X', 'endppoint1_Y')

    # Concatenate the DataFrames
    combined_df = pd.concat([center_distance_df, endpoint2_distance_df, endpoint1_distance_df, endpoint12_distance_df,
                             endpoint21_distance_df])

    # Group by index and find the min value for each cell
    distance_df = combined_df.groupby(level=0).min()

    # Convert all elements to float, coercing errors to NaN
    distance_df = distance_df.applymap(lambda x: pd.to_numeric(x, errors='coerce'))

    return overlap_df, distance_df


def optimization_transition_cost(df, current_img_npy, current_time_step_df, target_bacteria, next_time_step_img_npy,
                                 bacteria_in_next_time_step, candidate_bacteria, um_per_pixel, check_cellType,
                                 neighbors_df):

    # note: check_fluorescent_intensity(transition_bac, candidate_parent_bacterium)

    overlap_df, distance_df = compare_bacteria(current_img_npy, current_time_step_df, target_bacteria,
                                               next_time_step_img_npy, bacteria_in_next_time_step, candidate_bacteria,
                                               um_per_pixel)

    normalized_distance_df = distance_normalization(df, distance_df)

    cost_df = np.sqrt(np.power(1 - overlap_df, 2) + np.power(normalized_distance_df, 2))

    max_daughter_len_to_mother_ratio_boundary = find_max_daughter_len_to_mother_ratio_boundary(df)
    sum_daughter_len_to_mother_ratio_boundary = find_sum_daughter_len_to_mother_ratio_boundary(df)
    bac_len_to_bac_ratio_boundary = find_bac_len_to_bac_ratio_boundary(df)
    bac_movement_boundary = find_bac_movement_boundary(df)

    candidate_bacteria_time_step = candidate_bacteria['ImageNumber'].values.tolist()[0]
    current_time_step = current_time_step_df['ImageNumber'].values.tolist()[0]

    for candidate_bac_indx, candidate_bacterium in candidate_bacteria.iterrows():

        if candidate_bacterium['continued_life_history_before_transition']:
            if candidate_bacteria_time_step == current_time_step - 1:
                # This means that a division may have occurred (maybe the transition bacterium is the second daughter)

                other_daughter = df.loc[(df['ImageNumber'] == current_time_step) &
                                        (df['id'] == candidate_bacterium['id'])]

                # check the neighbors of other daughter
                neighbors_of_other_daughter = \
                    neighbors_df.loc[(neighbors_df['First Image Number'] ==
                                      other_daughter['ImageNumber'].values.tolist()[0]) &
                                     (neighbors_df['First Object Number'] ==
                                      other_daughter['ObjectNumber'].values.tolist()[0])]

                neighbors_of_other_daughter_object_number = \
                    neighbors_of_other_daughter['Second Object Number'].values.tolist()

                neighbors_of_other_daughter = df.loc[(df['ImageNumber'] == current_time_step) &
                                                     (df['ObjectNumber'].isin(neighbors_of_other_daughter_object_number))]

                neighbors_of_other_daughter_indx = neighbors_of_other_daughter.index.values.tolist()

                cost_df.loc[~cost_df.index.isin(neighbors_of_other_daughter_indx), candidate_bac_indx] = 999

                if len(list(set(cost_df[candidate_bac_indx].values.tolist()))) > 1 or \
                        int(list(set(cost_df[candidate_bac_indx].values.tolist()))[0]) != 999:

                    bac_without_parent_len_to_mother_ratio = target_bacteria['AreaShape_MajorAxisLength'] / \
                                                             candidate_bacterium['AreaShape_MajorAxisLength']

                    sum_daughter_len_to_mother_ratio = (target_bacteria['AreaShape_MajorAxisLength'] +
                                                        other_daughter['AreaShape_MajorAxisLength'].values.tolist()[0]) / \
                                                       candidate_bacterium['AreaShape_MajorAxisLength']

                    inappropriate_target_bac_based_on_max_length = \
                        bac_without_parent_len_to_mother_ratio[(bac_without_parent_len_to_mother_ratio >= 1) |
                                                               (bac_without_parent_len_to_mother_ratio >=
                                                                max_daughter_len_to_mother_ratio_boundary['avg'] + 1.96 *
                                                                max_daughter_len_to_mother_ratio_boundary[
                                                                    'std'])].index.tolist()

                    inappropriate_target_bac_based_on_sum_length = \
                        sum_daughter_len_to_mother_ratio[(sum_daughter_len_to_mother_ratio <=
                                                          sum_daughter_len_to_mother_ratio_boundary['avg'] - 1.96 *
                                                          sum_daughter_len_to_mother_ratio_boundary['std']) |
                                                         (sum_daughter_len_to_mother_ratio >=
                                                          sum_daughter_len_to_mother_ratio_boundary['avg'] + 1.96 *
                                                          sum_daughter_len_to_mother_ratio_boundary['std'])].index.tolist()

                    inappropriate_target_back = inappropriate_target_bac_based_on_max_length
                    inappropriate_target_back.extend(inappropriate_target_bac_based_on_sum_length)
                    cost_df.loc[list(set(inappropriate_target_back)), candidate_bac_indx] = 999

            else:
                # This means that a division may have occurred
                # (the division occurred before transition bacterium, but segmentation errors occurred).
                # only check the distance
                cost_df.loc[distance_df[candidate_bac_indx] > bac_movement_boundary['max'], candidate_bac_indx] = 999

        else:
            if candidate_bacteria_time_step == current_time_step - 1:
                # same bacteria
                bac_length_to_bac = target_bacteria['AreaShape_MajorAxisLength'] / \
                                    candidate_bacterium['AreaShape_MajorAxisLength']

                inappropriate_target_bac_based_on_back_length_to_back = \
                    bac_length_to_bac[bac_length_to_bac <= bac_len_to_bac_ratio_boundary['avg'] - 1.96 *
                                      bac_len_to_bac_ratio_boundary['std']].index.tolist()

                cost_df.loc[inappropriate_target_bac_based_on_back_length_to_back, candidate_bac_indx] = 999

            # check the distance
            cost_df.loc[distance_df[candidate_bac_indx] > bac_movement_boundary['max'], candidate_bac_indx] = 999

    # Run the optimization
    result_df = optimize_assignment(cost_df)

    return result_df
