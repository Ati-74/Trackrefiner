from Trackrefiner.strain.correction.action.findOutlier import find_sum_daughter_len_to_mother_ratio_boundary, \
    find_max_daughter_len_to_mother_ratio_boundary, find_bac_len_to_bac_ratio_boundary, find_upper_bound, \
    find_lower_bound
from Trackrefiner.strain.correction.action.helperFunctions import calculate_orientation_angle_batch
from Trackrefiner.strain.correction.action.costFinder import (adding_new_terms_to_cost_matrix,
                                                              make_initial_distance_matrix)
from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
from Trackrefiner.strain.correction.action.Modeling.calculation.iouCalForML import iou_calc
from Trackrefiner.strain.correction.action.Modeling.calculation.calcDistanceForML import calc_distance
from Trackrefiner.strain.correction.action.Modeling.calculation.lengthRatio import check_len_ratio
from Trackrefiner.strain.correction.action.Modeling.calculation.calMotionAlignmentAngle import calc_MotionAlignmentAngle
from Trackrefiner.strain.correction.neighborChecking import neighbor_checking
from Trackrefiner.strain.correction.action.helperFunctions import calculate_trajectory_direction_angle_all


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


def feature_space_adding_new_link_to_unexpected_beginning(neighbors_df, unexpected_begging_bac,
                                                          all_bac_in_unexpected_beginning_bac_time_step_df,
                                                          all_bacteria_in_prev_time_step, center_coordinate_columns,
                                                          parent_object_number_col, color_array, coordinate_array):

    overlap_df, distance_df = make_initial_distance_matrix(all_bac_in_unexpected_beginning_bac_time_step_df,
                                                           unexpected_begging_bac,
                                                           all_bacteria_in_prev_time_step,
                                                           all_bacteria_in_prev_time_step,
                                                           center_coordinate_columns,
                                                           color_array=color_array,
                                                           coordinate_array=coordinate_array)

    # remove columns with all values equal to zero
    overlap_df_replaced = overlap_df[(overlap_df != 0).any(axis=1)]
    max_overlap_bac_idx = overlap_df_replaced.idxmax(axis=1)
    min_distance_bac_ndx = distance_df.idxmin(axis=1)

    df_max_overlap_bac_idx = max_overlap_bac_idx.reset_index()
    df_min_distance_bac_ndx = min_distance_bac_ndx.reset_index()

    df_max_overlap_bac_idx.columns = ['unexpected beginning bac idx', 'candidate bac idx']
    df_min_distance_bac_ndx.columns = ['unexpected beginning bac idx', 'candidate bac idx']

    df_primary_candidate_bac_ndx = pd.concat([df_max_overlap_bac_idx, df_min_distance_bac_ndx],
                                             ignore_index=True).drop_duplicates()

    df_unexpected_bac_indo_with_candidates = df_primary_candidate_bac_ndx.merge(unexpected_begging_bac,
                                                                                left_on='unexpected beginning bac idx',
                                                                                right_on='index', how='inner')

    df_primary_candidate_bac_ndx_info = df_unexpected_bac_indo_with_candidates.merge(all_bacteria_in_prev_time_step,
                                                                                     left_on='candidate bac idx',
                                                                                     right_on='index', how='inner',
                                                                                     suffixes=('', '_candidate'))

    df_primary_candidate_bac_ndx_info_neighbors = \
        df_primary_candidate_bac_ndx_info.merge(neighbors_df,
                                                left_on=['ImageNumber_candidate', 'ObjectNumber_candidate'],
                                                right_on=['First Image Number', 'First Object Number'], how='left')

    df_primary_candidate_bac_ndx_info_neighbors_info = \
        df_primary_candidate_bac_ndx_info_neighbors.merge(all_bacteria_in_prev_time_step,
                                                          left_on=['Second Image Number', 'Second Object Number'],
                                                          right_on=['ImageNumber', 'ObjectNumber'], how='left',
                                                          suffixes=('', '_candidate_neighbors'))

    raw_cols = all_bacteria_in_prev_time_step.columns.tolist()
    candidate_columns = [v + '_candidate' for v in raw_cols]
    neighbor_candidate_columns = [v + '_candidate_neighbors' for v in raw_cols]

    candidate_df_cols = raw_cols.copy()
    candidate_df_cols.extend(candidate_columns)

    neighbor_candidate_df_cols = raw_cols.copy()
    neighbor_candidate_df_cols.extend(neighbor_candidate_columns)

    rename_neighbor_candidate_df_cols_dict = {}

    for i, col in enumerate(neighbor_candidate_columns):
        rename_neighbor_candidate_df_cols_dict[col] = raw_cols[i] + '_candidate'

    # Create DataFrames for the two groups of columns to be combined
    df_unexpected_beginning_with_can_bac = df_primary_candidate_bac_ndx_info_neighbors_info[candidate_df_cols]

    df_unexpected_beginning_with_can_neighbor_bac = \
        df_primary_candidate_bac_ndx_info_neighbors_info[neighbor_candidate_df_cols].rename(
            columns=rename_neighbor_candidate_df_cols_dict)

    # Concatenate the DataFrames
    final_candidate_bac = pd.concat([df_unexpected_beginning_with_can_bac,
                                     df_unexpected_beginning_with_can_neighbor_bac]).sort_index(
        kind='merge').reset_index(drop=True)

    # we don't check unexpected end bacteria in this phase
    final_candidate_bac = final_candidate_bac.loc[~ final_candidate_bac['ImageNumber_candidate'].isna()]

    final_candidate_bac = final_candidate_bac.loc[final_candidate_bac['unexpected_end_candidate'] == False]

    final_candidate_bac[['index', 'index_candidate']] = final_candidate_bac[['index', 'index_candidate']].astype(int)

    final_candidate_bac = final_candidate_bac.drop_duplicates(subset=['index', 'index_candidate'],
                                                              keep='first').reset_index(drop=True)

    bac_under_invest = all_bacteria_in_prev_time_step.loc[final_candidate_bac['index_candidate'].unique()]

    return bac_under_invest, final_candidate_bac


def feature_space_adding_new_link_to_unexpected_end(neighbors_df, unexpected_end_bac,
                                                    all_bac_in_unexpected_end_bac_time_step_df,
                                                    all_bacteria_in_next_time_step_to_unexpected_end_bac,
                                                    center_coordinate_columns, parent_object_number_col,
                                                    color_array, coordinate_array):
    overlap_df, distance_df = make_initial_distance_matrix(all_bac_in_unexpected_end_bac_time_step_df,
                                                           unexpected_end_bac,
                                                           all_bacteria_in_next_time_step_to_unexpected_end_bac,
                                                           all_bacteria_in_next_time_step_to_unexpected_end_bac,
                                                           center_coordinate_columns,
                                                           color_array=color_array,
                                                           coordinate_array=coordinate_array)

    # remove columns with all values equal to zero
    overlap_df_replaced = overlap_df[(overlap_df != 0).any(axis=1)]
    max_overlap_bac_idx = overlap_df_replaced.idxmax(axis=1)
    min_distance_bac_ndx = distance_df.idxmin(axis=1)

    df_max_overlap_bac_idx = max_overlap_bac_idx.reset_index()
    df_min_distance_bac_ndx = min_distance_bac_ndx.reset_index()

    df_max_overlap_bac_idx.columns = ['unexpected end bac idx', 'candidate bac idx']
    df_min_distance_bac_ndx.columns = ['unexpected end bac idx', 'candidate bac idx']

    df_primary_candidate_bac_ndx = pd.concat([df_max_overlap_bac_idx, df_min_distance_bac_ndx],
                                             ignore_index=True).drop_duplicates()

    df_unexpected_bac_info_with_candidates = \
        df_primary_candidate_bac_ndx.merge(unexpected_end_bac,
                                           left_on='unexpected end bac idx',
                                           right_on='index', how='inner')

    df_primary_candidate_bac_ndx_info = \
        df_unexpected_bac_info_with_candidates.merge(all_bacteria_in_next_time_step_to_unexpected_end_bac,
                                                     left_on='candidate bac idx',
                                                     right_on='index', how='inner',
                                                     suffixes=('', '_candidate'))

    df_primary_candidate_bac_ndx_info_neighbors = \
        df_primary_candidate_bac_ndx_info.merge(neighbors_df,
                                                left_on=['ImageNumber_candidate', 'ObjectNumber_candidate'],
                                                right_on=['First Image Number', 'First Object Number'], how='left')

    df_primary_candidate_bac_ndx_info_neighbors_info = \
        df_primary_candidate_bac_ndx_info_neighbors.merge(all_bacteria_in_next_time_step_to_unexpected_end_bac,
                                                          left_on=['Second Image Number', 'Second Object Number'],
                                                          right_on=['ImageNumber', 'ObjectNumber'], how='left',
                                                          suffixes=('', '_candidate_neighbors'))

    raw_cols = all_bacteria_in_next_time_step_to_unexpected_end_bac.columns.tolist()
    candidate_columns = [v + '_candidate' for v in raw_cols]
    neighbor_candidate_columns = [v + '_candidate_neighbors' for v in raw_cols]

    candidate_df_cols = raw_cols.copy()
    candidate_df_cols.extend(candidate_columns)

    neighbor_candidate_df_cols = raw_cols.copy()
    neighbor_candidate_df_cols.extend(neighbor_candidate_columns)

    rename_neighbor_candidate_df_cols_dict = {}

    for i, col in enumerate(neighbor_candidate_columns):
        rename_neighbor_candidate_df_cols_dict[col] = raw_cols[i] + '_candidate'

    # Create DataFrames for the two groups of columns to be combined
    df_unexpected_beginning_with_can_bac = df_primary_candidate_bac_ndx_info_neighbors_info[candidate_df_cols]

    df_unexpected_beginning_with_can_neighbor_bac = \
        df_primary_candidate_bac_ndx_info_neighbors_info[neighbor_candidate_df_cols].rename(
            columns=rename_neighbor_candidate_df_cols_dict)

    # Concatenate the DataFrames
    final_candidate_bac = pd.concat([df_unexpected_beginning_with_can_bac,
                                     df_unexpected_beginning_with_can_neighbor_bac
                                     ]).sort_index(kind='merge').reset_index(drop=True)

    # remove Nas because we used left join for neighbors
    final_candidate_bac = final_candidate_bac.dropna(subset=['index', 'index_candidate'])

    final_candidate_bac[['index', 'index_candidate']] = final_candidate_bac[['index', 'index_candidate']].astype(int)

    final_candidate_bac = final_candidate_bac.drop_duplicates(subset=['index', 'index_candidate'],
                                                              keep='first').reset_index(drop=True)

    bac_under_invest = all_bacteria_in_next_time_step_to_unexpected_end_bac.loc[
        final_candidate_bac['index_candidate'].unique()]

    return bac_under_invest, final_candidate_bac


def optimization_unexpected_beginning_cost(df, all_bac_in_unexpected_begging_bac_time_step_df,
                                           unexpected_begging_bacteria, all_bacteria_in_prev_time_step,
                                           check_cell_type, neighbors_df, neighbor_list_array,
                                           min_life_history_of_bacteria,
                                           parent_image_number_col, parent_object_number_col, center_coordinate_columns,
                                           comparing_divided_non_divided_model, non_divided_bac_model,
                                           divided_bac_model, color_array, coordinate_array):
    # note: check_fluorescent_intensity(unexpected_beginning_bac, candidate_parent_bacterium)

    max_daughter_len_to_mother_ratio_boundary = find_max_daughter_len_to_mother_ratio_boundary(df)
    sum_daughter_len_to_mother_ratio_boundary = find_sum_daughter_len_to_mother_ratio_boundary(df)

    redundant_link_dict_division = {}
    division_cost_dict = {}

    bac_under_invest_prev_time_step, final_candidate_bac = \
        feature_space_adding_new_link_to_unexpected_beginning(neighbors_df, unexpected_begging_bacteria,
                                                              all_bac_in_unexpected_begging_bac_time_step_df,
                                                              all_bacteria_in_prev_time_step, center_coordinate_columns,
                                                              parent_object_number_col,
                                                              color_array=color_array,
                                                              coordinate_array=coordinate_array)

    if final_candidate_bac.shape[0] > 0:

        receiver_of_bac_under_invest_link = \
            all_bac_in_unexpected_begging_bac_time_step_df.loc[
                all_bac_in_unexpected_begging_bac_time_step_df[parent_object_number_col].isin(
                    bac_under_invest_prev_time_step['ObjectNumber'])]

        # now check the cost of maintaining the link
        maintenance_cost_df = calc_maintenance_cost(df, all_bacteria_in_prev_time_step,
                                                    bac_under_invest_prev_time_step,
                                                    all_bac_in_unexpected_begging_bac_time_step_df,
                                                    neighbors_df, receiver_of_bac_under_invest_link,
                                                    center_coordinate_columns, parent_image_number_col,
                                                    parent_object_number_col,
                                                    comparing_divided_non_divided_model,
                                                    coordinate_array=coordinate_array)

        same_link_cost_df = \
            same_link_cost(df, neighbors_df, neighbor_list_array, final_candidate_bac.copy(), center_coordinate_columns,
                           '_candidate', '', parent_image_number_col, parent_object_number_col,
                           non_divided_bac_model, comparing_divided_non_divided_model, maintenance_cost_df,
                           maintenance_to_be_check='source', coordinate_array=coordinate_array)

        division_cost_df = daughter_cost(df, neighbors_df, neighbor_list_array, final_candidate_bac.copy(),
                                         center_coordinate_columns, '_candidate', '',
                                         parent_image_number_col, parent_object_number_col, divided_bac_model,
                                         comparing_divided_non_divided_model, min_life_history_of_bacteria,
                                         maintenance_cost_df, maintenance_to_be_check='source',
                                         coordinate_array=coordinate_array)

        if division_cost_df.shape[0] > 0:

            # now check division condition

            source_bac_next_time_step_dict = {}
            source_bac_daughters_dict = {}

            for source_bac_idx in division_cost_df.index.values:
                source_bac = all_bacteria_in_prev_time_step.loc[source_bac_idx]

                # Did the bacteria survive or divide?
                source_bac_next_time_step = \
                    all_bac_in_unexpected_begging_bac_time_step_df.loc[
                        all_bac_in_unexpected_begging_bac_time_step_df['id'] == source_bac['id']]

                source_bac_daughters = \
                    all_bac_in_unexpected_begging_bac_time_step_df.loc[
                        all_bac_in_unexpected_begging_bac_time_step_df['parent_id'] == source_bac['id']].copy()

                source_bac_next_time_step_dict[source_bac_idx] = source_bac_next_time_step
                source_bac_daughters_dict[source_bac_idx] = source_bac_daughters

            for unexpected_beginning_bac_idx in division_cost_df.columns.values:

                redundant_link_dict_division[unexpected_beginning_bac_idx] = {}
                division_cost_dict[unexpected_beginning_bac_idx] = {}

                unexpected_beginning_bac = \
                    all_bac_in_unexpected_begging_bac_time_step_df.loc[unexpected_beginning_bac_idx]

                # unexpected beginning = ub
                candidate_source_bac_ndx_for_this_ub_bac = \
                    division_cost_df[~ division_cost_df[unexpected_beginning_bac_idx].isna()].index.values

                for i, candidate_source_bac_ndx in enumerate(candidate_source_bac_ndx_for_this_ub_bac):
                    candidate_source_bac = all_bacteria_in_prev_time_step.loc[candidate_source_bac_ndx]

                    division_cost_dict, redundant_link_dict_division = \
                        adding_new_terms_to_cost_matrix(division_cost_df, maintenance_cost_df, division_cost_dict,
                                                        candidate_source_bac_ndx, candidate_source_bac,
                                                        unexpected_beginning_bac_idx, unexpected_beginning_bac,
                                                        redundant_link_dict_division,
                                                        sum_daughter_len_to_mother_ratio_boundary,
                                                        max_daughter_len_to_mother_ratio_boundary,
                                                        min_life_history_of_bacteria,
                                                        source_bac_next_time_step_dict[candidate_source_bac_ndx],
                                                        source_bac_daughters_dict[candidate_source_bac_ndx])

            division_cost_df = pd.DataFrame.from_dict(division_cost_dict, orient='index')
            redundant_link_division_df = pd.DataFrame.from_dict(redundant_link_dict_division, orient='index')

            # now we should transform
            division_cost_df = division_cost_df.transpose()
            # redundant_link_division_df = redundant_link_division_df.transpose()

        else:
            division_cost_df = pd.DataFrame()
            redundant_link_division_df = pd.DataFrame()
    else:
        same_link_cost_df = pd.DataFrame()
        division_cost_df = pd.DataFrame()
        redundant_link_division_df = pd.DataFrame()
        maintenance_cost_df = pd.DataFrame()

    return same_link_cost_df, division_cost_df, redundant_link_division_df, maintenance_cost_df


def daughter_cost(df, neighbors_df, neighbor_list_array, df_source_daughter, center_coordinate_columns, col_source,
                  col_target, parent_image_number_col, parent_object_number_col, divided_bac_model,
                  comparing_divided_non_divided_model, min_life_history_of_bacteria_time_step, maintenance_cost_df,
                  maintenance_to_be_check='target', coordinate_array=None):
    df_source_daughter['index_prev' + col_target] = df_source_daughter['index' + col_target]
    df_source_daughter['index2' + col_target] = df_source_daughter.index.values

    df_source_daughter['LengthChangeRatio' + col_target] = \
        (df_source_daughter['AreaShape_MajorAxisLength' + col_target] /
         df_source_daughter['AreaShape_MajorAxisLength' + col_source])

    df_source_daughter = iou_calc(df_source_daughter, col_source='prev_index' + col_source,
                                  col_target='prev_index' + col_target, stat='div', coordinate_array=coordinate_array,
                                  both=True)

    df_source_daughter = calc_distance(df_source_daughter, center_coordinate_columns, postfix_target=col_target,
                                       postfix_source=col_source, stat='div')

    df_source_daughter['prev_time_step_index' + col_target] = \
        df_source_daughter['index' + col_source].astype('int64')

    df_source_daughter['difference_neighbors' + col_target] = np.nan
    df_source_daughter['other_daughter_index' + col_target] = np.nan
    df_source_daughter['parent_id' + col_target] = df_source_daughter['id' + col_source]

    df_source_daughter = \
        neighbor_checking(df_source_daughter, neighbor_list_array,
                          parent_image_number_col, parent_object_number_col,
                          selected_rows_df=df_source_daughter, selected_time_step_df=df,
                          return_common_elements=True, col_target=col_target)

    df_source_daughter['diff_common_diff' + col_target] = \
        df_source_daughter['difference_neighbors' + col_target] - \
        df_source_daughter['common_neighbors' + col_target]

    df_source_daughter.loc[df_source_daughter['diff_common_diff' + col_target] < 0, 'diff_common_diff' + col_target] = 0

    df_source_daughter['angle_mother_daughter' + col_target] = \
        calculate_orientation_angle_batch(df_source_daughter['bacteria_slope' + col_source].values,
                                          df_source_daughter['bacteria_slope' + col_target].values)

    df_source_daughter['prev_time_step_center_x' + col_target] = \
        df_source_daughter[center_coordinate_columns['x'] + col_source]
    df_source_daughter['prev_time_step_center_y' + col_target] = \
        df_source_daughter[center_coordinate_columns['y'] + col_source]

    direction_of_motion = calculate_trajectory_direction_angle_all(df_source_daughter,
                                                                   center_coordinate_columns, col_target=col_target)

    df_source_daughter['direction_of_motion' + col_target] = direction_of_motion

    # motion alignment
    # calculated for original df and we should calc for new df
    df_source_daughter["MotionAlignmentAngle" + col_target] = np.nan
    df_source_daughter = \
        calc_MotionAlignmentAngle(df, neighbors_df, center_coordinate_columns,
                                  selected_rows=df_source_daughter, col_target=col_target,
                                  col_source=col_source)

    df_source_daughter['adjusted_common_neighbors' + col_target] = np.where(
        df_source_daughter['common_neighbors' + col_target] == 0,
        df_source_daughter['common_neighbors' + col_target] + 1,
        df_source_daughter['common_neighbors' + col_target]
    )

    df_source_daughter['neighbor_ratio' + col_target] = \
        (df_source_daughter['difference_neighbors' + col_target] / (
            df_source_daughter['adjusted_common_neighbors' + col_target]))

    raw_features = ['iou', 'iou_same', 'min_distance', 'min_distance_same', 'difference_neighbors' + col_target,
                    'neighbor_ratio' + col_target, 'common_neighbors' + col_target,
                    'angle_mother_daughter' + col_target, 'direction_of_motion' + col_target,
                    'MotionAlignmentAngle' + col_target, 'LengthChangeRatio' + col_target, 'index' + col_source,
                    'index_prev' + col_target, 'divideFlag' + col_target, 'LifeHistory' + col_target,
                    'age' + col_target, 'diff_common_diff' + col_target]

    df_source_daughter = df_source_daughter[raw_features].copy()

    df_source_daughter = df_source_daughter.rename(
        {
            'difference_neighbors' + col_target: 'difference_neighbors',
            'common_neighbors' + col_target: 'common_neighbors',
            'neighbor_ratio' + col_target: 'neighbor_ratio',
            'angle_mother_daughter' + col_target: 'angle_mother_daughter',
            'direction_of_motion' + col_target: 'direction_of_motion',
            'MotionAlignmentAngle' + col_target: 'MotionAlignmentAngle',
            'LengthChangeRatio' + col_target: 'LengthChangeRatio',
            'index_prev' + col_target: 'index_prev',
            'divideFlag' + col_target: 'divideFlag',
            'LifeHistory' + col_target: 'LifeHistory',
            'age' + col_target: 'age',
            'diff_common_diff' + col_target: 'diff_common_diff'
        }, axis=1)

    # 'difference_neighbors'
    feature_list_divided_bac_model = ['iou', 'min_distance', 'neighbor_ratio',
                                      'angle_mother_daughter']

    y_prob_divided_bac_model = divided_bac_model.predict_proba(df_source_daughter[feature_list_divided_bac_model])[:, 1]
    df_source_daughter['prob_divided_bac_model'] = y_prob_divided_bac_model

    # df_source_daughter = df_source_daughter.rename({'iou': 'iou_div', 'iou_same': 'iou',
    #                                                'min_distance': 'min_distance_div',
    #                                                'min_distance_same': 'min_distance'}, axis=1)
    # difference_neighbors
    feature_list_for_compare = \
        ['iou', 'min_distance', 'neighbor_ratio', 'direction_of_motion', 'MotionAlignmentAngle',
         'LengthChangeRatio']

    # division is class 0
    y_prob_compare = \
        comparing_divided_non_divided_model.predict_proba(df_source_daughter[feature_list_for_compare])[:, 0]
    df_source_daughter['prob_compare'] = y_prob_compare

    if comparing_divided_non_divided_model is not None:
        # if t is not None:
        #     df_source_daughter.to_csv('df_source_daughter.csv')
        #    breakpoint()
        # & (df_source_daughter['difference_neighbors'] <= df_source_daughter['common_neighbors'])
        df_source_daughter = df_source_daughter.loc[(df_source_daughter['prob_divided_bac_model'] > 0.5) &
                                                    (df_source_daughter['prob_compare'] > 0.5)]

        # now we should check life history of target bacteria
        # df_source_daughter = \
        # df_source_daughter.loc[(df_source_daughter['divideFlag'] == False) |
        # (df_source_daughter['LifeHistory'] - df_source_daughter['age'] + 1 >
        # min_life_history_of_bacteria_time_step)]

        if df_source_daughter.shape[0] > 0:
            # Pivot this DataFrame to get the desired structure
            division_cost_df = \
                df_source_daughter[['index' + col_source, 'index_prev', 'prob_compare']].pivot(
                    index='index' + col_source, columns='index_prev', values='prob_compare')
            division_cost_df.columns.name = None
            division_cost_df.index.name = None

            if maintenance_to_be_check == 'target':
                # I want to measure is it possible to remove the current link of target bacterium?
                candidate_target_maintenance_cost = \
                    maintenance_cost_df.loc[:, maintenance_cost_df.columns.isin(
                        df_source_daughter['index_prev'].unique())]

                # don't need to convert probability to 1 - probability
                for col in candidate_target_maintenance_cost.columns.values:
                    non_na_probability = candidate_target_maintenance_cost[col].dropna().iloc[0]
                    division_cost_df.loc[division_cost_df[col] <= non_na_probability, col] = np.nan

            # for maintenance_to_be_check = source we can not check, and we should check it after

            division_cost_df = 1 - division_cost_df

        else:
            division_cost_df = pd.DataFrame()
    else:
        # Pivot this DataFrame to get the desired structure
        division_cost_df = \
            df_source_daughter[['index' + col_source, 'index_prev', 'prob_divided_bac_model']].pivot(
                index='index' + col_source, columns='index_prev', values='prob_divided_bac_model')
        division_cost_df.columns.name = None
        division_cost_df.index.name = None

    return division_cost_df


def daughter_cost_for_final_step(df, neighbors_df, neighbor_list_array, df_source_daughter, center_coordinate_columns,
                                 col_source, col_target,
                                 parent_image_number_col, parent_object_number_col, divided_bac_model,
                                 maintenance_cost_df,
                                 maintenance_to_be_check='target', coordinate_array=None):
    df_source_daughter['index_prev' + col_target] = df_source_daughter['index' + col_target]
    df_source_daughter['index2' + col_target] = df_source_daughter.index.values

    df_source_daughter['LengthChangeRatio' + col_target] = \
        (df_source_daughter['AreaShape_MajorAxisLength' + col_target] /
         df_source_daughter['AreaShape_MajorAxisLength' + col_source])

    df_source_daughter = iou_calc(df_source_daughter, col_source='prev_index' + col_source,
                                  col_target='prev_index' + col_target, stat='div', coordinate_array=coordinate_array,
                                  both=True)

    df_source_daughter = calc_distance(df_source_daughter, center_coordinate_columns, postfix_target=col_target,
                                       postfix_source=col_source, stat='div')

    df_source_daughter['prev_time_step_index' + col_target] = \
        df_source_daughter['index' + col_source].fillna(-1).astype('int64')

    df_source_daughter['difference_neighbors' + col_target] = np.nan
    df_source_daughter['other_daughter_index' + col_target] = np.nan
    df_source_daughter['parent_id' + col_target] = df_source_daughter['id' + col_source]

    df_source_daughter = \
        neighbor_checking(df_source_daughter, neighbor_list_array,
                          parent_image_number_col, parent_object_number_col,
                          selected_rows_df=df_source_daughter, selected_time_step_df=df,
                          return_common_elements=True, col_target=col_target)

    df_source_daughter['diff_common_diff' + col_target] = \
        df_source_daughter['difference_neighbors' + col_target] - \
        df_source_daughter['common_neighbors' + col_target]

    df_source_daughter.loc[df_source_daughter['diff_common_diff' + col_target] < 0, 'diff_common_diff' + col_target] = 0

    df_source_daughter['angle_mother_daughter' + col_target] = \
        calculate_orientation_angle_batch(df_source_daughter['bacteria_slope' + col_source].values,
                                          df_source_daughter['bacteria_slope' + col_target].values)

    df_source_daughter['prev_time_step_center_x' + col_target] = \
        df_source_daughter[center_coordinate_columns['x'] + col_source]
    df_source_daughter['prev_time_step_center_y' + col_target] = \
        df_source_daughter[center_coordinate_columns['y'] + col_source]

    direction_of_motion = calculate_trajectory_direction_angle_all(df_source_daughter,
                                                                   center_coordinate_columns, col_target=col_target)

    df_source_daughter['direction_of_motion' + col_target] = direction_of_motion

    # motion alignment
    # calculated for original df and we should calc for new df
    df_source_daughter["MotionAlignmentAngle" + col_target] = np.nan
    df_source_daughter = \
        calc_MotionAlignmentAngle(df, neighbors_df, center_coordinate_columns,
                                  selected_rows=df_source_daughter, col_target=col_target,
                                  col_source=col_source)

    df_source_daughter['adjusted_common_neighbors' + col_target] = np.where(
        df_source_daughter['common_neighbors' + col_target] == 0,
        df_source_daughter['common_neighbors' + col_target] + 1,
        df_source_daughter['common_neighbors' + col_target]
    )

    df_source_daughter['neighbor_ratio' + col_target] = \
        (df_source_daughter['difference_neighbors' + col_target] / (
            df_source_daughter['adjusted_common_neighbors' + col_target]))

    raw_features = ['iou', 'iou_same', 'min_distance', 'min_distance_same', 'difference_neighbors' + col_target,
                    'common_neighbors' + col_target, 'neighbor_ratio' + col_target,
                    'angle_mother_daughter' + col_target, 'direction_of_motion' + col_target,
                    'MotionAlignmentAngle' + col_target, 'LengthChangeRatio' + col_target, 'index' + col_source,
                    'index_prev' + col_target, 'divideFlag' + col_target, 'LifeHistory' + col_target,
                    'age' + col_target, 'diff_common_diff' + col_target]

    df_source_daughter = df_source_daughter[raw_features].copy()

    df_source_daughter = df_source_daughter.rename(
        {
            'difference_neighbors' + col_target: 'difference_neighbors',
            'common_neighbors' + col_target: 'common_neighbors',
            'neighbor_ratio' + col_target: 'neighbor_ratio',
            'angle_mother_daughter' + col_target: 'angle_mother_daughter',
            'direction_of_motion' + col_target: 'direction_of_motion',
            'MotionAlignmentAngle' + col_target: 'MotionAlignmentAngle',
            'LengthChangeRatio' + col_target: 'LengthChangeRatio',
            'index_prev' + col_target: 'index_prev',
            'divideFlag' + col_target: 'divideFlag',
            'LifeHistory' + col_target: 'LifeHistory',
            'age' + col_target: 'age',
            'diff_common_diff' + col_target: 'diff_common_diff'
        }, axis=1)

    # 'difference_neighbors'
    feature_list_divided_bac_model = ['iou', 'min_distance', 'neighbor_ratio',
                                      'angle_mother_daughter']

    y_prob_divided_bac_model = divided_bac_model.predict_proba(df_source_daughter[feature_list_divided_bac_model])[:, 1]
    df_source_daughter['prob_divided_bac_model'] = y_prob_divided_bac_model

    df_source_daughter = df_source_daughter.loc[df_source_daughter['prob_divided_bac_model'] > 0.5]

    if df_source_daughter.shape[0] > 0:
        # Pivot this DataFrame to get the desired structure
        division_cost_df = \
            df_source_daughter[['index' + col_source, 'index_prev', 'prob_divided_bac_model']].pivot(
                index='index' + col_source, columns='index_prev', values='prob_divided_bac_model')
        division_cost_df.columns.name = None
        division_cost_df.index.name = None

        if maintenance_to_be_check == 'target':
            # I want to measure is it possible to remove the current link of target bacterium?
            candidate_target_maintenance_cost = \
                maintenance_cost_df.loc[:, maintenance_cost_df.columns.isin(
                    df_source_daughter['index_prev'].unique())]

            # don't need to convert probability to 1 - probability
            for col in candidate_target_maintenance_cost.columns.values:
                non_na_probability = candidate_target_maintenance_cost[col].dropna().iloc[0]
                division_cost_df.loc[division_cost_df[col] <= non_na_probability, col] = np.nan

        # for maintenance_to_be_check = source we can not check, and we should check it after

        division_cost_df = 1 - division_cost_df

    else:
        division_cost_df = pd.DataFrame()

    return division_cost_df


def division_detection_cost(df, neighbors_df, neighbor_list_array, candidate_source_daughter_df,
                            min_life_history_of_bacteria_time_step, center_coordinate_columns, parent_image_number_col,
                            parent_object_number_col, divided_bac_model, comparing_divided_non_divided_model,
                            maintenance_cost_df, coordinate_array):
    # source has one link, and we check if we can add another link to source bacteria (mother  - daughter relation)

    max_daughter_len_to_mother_ratio_boundary = find_max_daughter_len_to_mother_ratio_boundary(df)
    sum_daughter_len_to_mother_ratio_boundary = find_sum_daughter_len_to_mother_ratio_boundary(df)

    candidate_source_daughter_df['max_target_neighbor_len_to_source'] = \
        np.max([candidate_source_daughter_df['AreaShape_MajorAxisLength_target'],
                candidate_source_daughter_df['AreaShape_MajorAxisLength']], axis=0) / \
        candidate_source_daughter_df['AreaShape_MajorAxisLength_source'].values

    candidate_source_daughter_df['sum_target_neighbor_len_to_source'] = \
        (candidate_source_daughter_df['AreaShape_MajorAxisLength_target'] +
         candidate_source_daughter_df['AreaShape_MajorAxisLength']) / \
        candidate_source_daughter_df['AreaShape_MajorAxisLength_source']

    candidate_source_daughter_df['LengthChangeRatio'] = \
        (candidate_source_daughter_df['AreaShape_MajorAxisLength'] /
         candidate_source_daughter_df['AreaShape_MajorAxisLength_source'])

    upper_bound_max_daughter_len = find_upper_bound(max_daughter_len_to_mother_ratio_boundary)

    lower_bound_sum_daughter_len = find_lower_bound(sum_daughter_len_to_mother_ratio_boundary)

    upper_bound_sum_daughter_len = find_upper_bound(sum_daughter_len_to_mother_ratio_boundary)

    # now check division conditions
    cond1 = ((candidate_source_daughter_df['max_target_neighbor_len_to_source'] < 1) |
             (candidate_source_daughter_df['max_target_neighbor_len_to_source'] <= upper_bound_max_daughter_len))

    cond2 = ((candidate_source_daughter_df['sum_target_neighbor_len_to_source'] >= lower_bound_sum_daughter_len)
             & (candidate_source_daughter_df['sum_target_neighbor_len_to_source'] <= upper_bound_sum_daughter_len))

    # cond3 = (candidate_source_daughter_df['LifeHistory_source'] >=
    #         (candidate_source_daughter_df['age'] + min_life_history_of_bacteria_time_step))
    cond3 = candidate_source_daughter_df['age_source'] > min_life_history_of_bacteria_time_step

    # cond4 = candidate_source_daughter_df['divideFlag'] == False
    cond4 = candidate_source_daughter_df['unexpected_beginning_source'] == True

    # cond5 = candidate_source_daughter_df['age'] > min_life_history_of_bacteria_time_step

    # cond6 = candidate_source_daughter_df['unexpected_beginning'] == True

    incorrect_same_link_in_this_time_step_with_candidate_neighbors = \
        candidate_source_daughter_df.loc[cond1 & cond2]

    # if incorrect_same_link_in_this_time_step_with_candidate_neighbors['ImageNumber'].values[0] == 68:
    #    incorrect_same_link_in_this_time_step_with_candidate_neighbors.to_csv('incorrect_same_link_in_this_time_step_with_candidate_neighbors1.csv')

    incorrect_same_link_in_this_time_step_with_candidate_neighbors = \
        incorrect_same_link_in_this_time_step_with_candidate_neighbors.loc[cond3 | cond4]

    # incorrect_same_link_in_this_time_step_with_candidate_neighbors = \
    #     incorrect_same_link_in_this_time_step_with_candidate_neighbors.loc[cond5 | cond6]

    # if incorrect_same_link_in_this_time_step_with_candidate_neighbors['ImageNumber'].values[0] == 68:
    #    incorrect_same_link_in_this_time_step_with_candidate_neighbors.to_csv('incorrect_same_link_in_this_time_step_with_candidate_neighbors.csv')
    #    breakpoint()

    if incorrect_same_link_in_this_time_step_with_candidate_neighbors.shape[0] > 0:

        cost_df = \
            daughter_cost(df, neighbors_df, neighbor_list_array,
                          incorrect_same_link_in_this_time_step_with_candidate_neighbors.copy(),
                          center_coordinate_columns, '_source', '',
                          parent_image_number_col, parent_object_number_col, divided_bac_model,
                          comparing_divided_non_divided_model, min_life_history_of_bacteria_time_step,
                          maintenance_cost_df, maintenance_to_be_check='target', coordinate_array=coordinate_array)

    else:
        # result_df = pd.DataFrame()
        cost_df = pd.DataFrame()

    return cost_df


def same_link_cost(df, neighbors_df, neighbor_list_array, source_bac_with_can_target, center_coordinate_columns,
                   col_source, col_target, parent_image_number_col, parent_object_number_col, non_divided_bac_model,
                   comparing_divided_non_divided_model, maintenance_cost_df, maintenance_to_be_check='target',
                   coordinate_array=None):
    bac_len_to_bac_ratio_boundary = find_bac_len_to_bac_ratio_boundary(df)

    lower_bound_threshold = find_lower_bound(bac_len_to_bac_ratio_boundary)

    source_bac_with_can_target['index_prev' + col_target] = source_bac_with_can_target['index' + col_target]
    source_bac_with_can_target['index2' + col_target] = source_bac_with_can_target.index.values

    source_bac_with_can_target['LengthChangeRatio' + col_target] = (
            source_bac_with_can_target['AreaShape_MajorAxisLength' + col_target] /
            source_bac_with_can_target['AreaShape_MajorAxisLength' + col_source])

    source_bac_with_can_target = \
        source_bac_with_can_target.loc[source_bac_with_can_target['LengthChangeRatio' + col_target]
                                       >= lower_bound_threshold].copy()

    if source_bac_with_can_target.shape[0] > 0:

        source_bac_with_can_target = iou_calc(source_bac_with_can_target, col_source='prev_index' + col_source,
                                              col_target='prev_index' + col_target, stat='same',
                                              coordinate_array=coordinate_array)

        source_bac_with_can_target = calc_distance(source_bac_with_can_target, center_coordinate_columns,
                                                   postfix_target=col_target, postfix_source=col_source, stat=None)

        source_bac_with_can_target['prev_time_step_index' + col_target] = \
            source_bac_with_can_target['index' + col_source].astype('int64')

        source_bac_with_can_target['difference_neighbors' + col_target] = np.nan
        source_bac_with_can_target['other_daughter_index' + col_target] = np.nan
        source_bac_with_can_target['parent_id' + col_target] = source_bac_with_can_target['id' + col_source]

        source_bac_with_can_target = \
            neighbor_checking(source_bac_with_can_target, neighbor_list_array,
                              parent_image_number_col, parent_object_number_col,
                              selected_rows_df=source_bac_with_can_target, selected_time_step_df=df,
                              return_common_elements=True, col_target=col_target)

        source_bac_with_can_target = check_len_ratio(df, source_bac_with_can_target, col_target=col_target,
                                                     col_source=col_source)

        # motion alignment
        # calculated for original df and we should calc for new df
        source_bac_with_can_target["MotionAlignmentAngle" + col_target] = np.nan
        source_bac_with_can_target = \
            calc_MotionAlignmentAngle(df, neighbors_df, center_coordinate_columns,
                                      selected_rows=source_bac_with_can_target, col_target=col_target,
                                      col_source=col_source)

        source_bac_with_can_target['prev_time_step_center_x' + col_target] = \
            source_bac_with_can_target[center_coordinate_columns['x'] + col_source]
        source_bac_with_can_target['prev_time_step_center_y' + col_target] = \
            source_bac_with_can_target[center_coordinate_columns['y'] + col_source]

        direction_of_motion = calculate_trajectory_direction_angle_all(source_bac_with_can_target,
                                                                       center_coordinate_columns, col_target=col_target)

        source_bac_with_can_target['direction_of_motion' + col_target] = direction_of_motion

        source_bac_with_can_target['adjusted_common_neighbors' + col_target] = np.where(
            source_bac_with_can_target['common_neighbors' + col_target] == 0,
            source_bac_with_can_target['common_neighbors' + col_target] + 1,
            source_bac_with_can_target['common_neighbors' + col_target]
        )

        source_bac_with_can_target['neighbor_ratio' + col_target] = \
            (source_bac_with_can_target['difference_neighbors' + col_target] / (
                source_bac_with_can_target['adjusted_common_neighbors' + col_target]))

        raw_feature_list = ['iou', 'min_distance', 'difference_neighbors' + col_target, 'common_neighbors' + col_target,
                            'neighbor_ratio' + col_target, 'length_dynamic' + col_target,
                            'MotionAlignmentAngle' + col_target,
                            'direction_of_motion' + col_target, 'LengthChangeRatio' + col_target,
                            'index' + col_source, 'index_prev' + col_target]

        source_bac_with_can_target = source_bac_with_can_target[raw_feature_list].copy()
        source_bac_with_can_target = source_bac_with_can_target.rename(
            {
                'common_neighbors' + col_target: 'common_neighbors',
                'difference_neighbors' + col_target: 'difference_neighbors',
                'neighbor_ratio' + col_target: 'neighbor_ratio',
                'length_dynamic' + col_target: 'length_dynamic',
                'MotionAlignmentAngle' + col_target: 'MotionAlignmentAngle',
                'direction_of_motion' + col_target: 'direction_of_motion',
                'LengthChangeRatio' + col_target: 'LengthChangeRatio',
                'index_prev' + col_target: 'index_prev',
            }, axis=1)

        # difference_neighbors
        feature_list_for_non_divided_bac_model = \
            ['iou', 'min_distance', 'neighbor_ratio', 'length_dynamic', 'MotionAlignmentAngle']

        # difference_neighbors
        feature_list_for_compare = \
            ['iou', 'min_distance', 'neighbor_ratio', 'direction_of_motion', 'MotionAlignmentAngle',
             'LengthChangeRatio']

        y_prob_non_divided_bac_model = \
            non_divided_bac_model.predict_proba(source_bac_with_can_target[
                                                    feature_list_for_non_divided_bac_model])[:, 1]
        source_bac_with_can_target['prob_non_divided_bac_model'] = y_prob_non_divided_bac_model

        # non divided class 1
        y_prob_compare = \
            comparing_divided_non_divided_model.predict_proba(source_bac_with_can_target[
                                                                  feature_list_for_compare])[:, 1]
        source_bac_with_can_target['prob_compare'] = y_prob_compare

        source_bac_with_can_target = \
            source_bac_with_can_target.loc[(source_bac_with_can_target['prob_non_divided_bac_model'] > 0.5) &
                                           (source_bac_with_can_target['prob_compare'] > 0.5) &
                                           (source_bac_with_can_target['difference_neighbors'] <=
                                            source_bac_with_can_target['common_neighbors'])]

        # Pivot this DataFrame to get the desired structure
        same_link_cost_df = \
            source_bac_with_can_target[['index' + col_source, 'index_prev', 'prob_compare']].pivot(
                index='index' + col_source, columns='index_prev', values='prob_compare')
        same_link_cost_df.columns.name = None
        same_link_cost_df.index.name = None

        if maintenance_to_be_check == 'target':
            # I want to measure is it possible to remove the current link of target bacterium?
            candidate_target_maintenance_cost = \
                maintenance_cost_df.loc[:, maintenance_cost_df.columns.isin(
                    source_bac_with_can_target['index_prev'].unique())]

            # don't need to convert probability to 1 - probability
            for col in candidate_target_maintenance_cost.columns.values:
                this_target_bac_maintenance_cost = candidate_target_maintenance_cost[col].dropna().iloc[0]
                same_link_cost_df.loc[same_link_cost_df[col] <= this_target_bac_maintenance_cost, col] = np.nan

        elif maintenance_to_be_check == 'source':

            # I want to measure is it possible to remove the current link of target bacterium?
            candidate_target_maintenance_cost = \
                maintenance_cost_df.loc[maintenance_cost_df.index.isin(
                    source_bac_with_can_target['index' + col_source].unique())]

            selected_candidate_target_maintenance_cost = \
                candidate_target_maintenance_cost[candidate_target_maintenance_cost.min(axis=1) > 0.5]

            for source_idx in selected_candidate_target_maintenance_cost.index.values:
                max_probability_value = candidate_target_maintenance_cost.loc[source_idx].max()

                same_link_cost_df.loc[source_idx, same_link_cost_df.loc[source_idx] <= max_probability_value] = np.nan

        same_link_cost_df = 1 - same_link_cost_df

    else:
        same_link_cost_df = pd.DataFrame()

    return same_link_cost_df


def same_link_cost_for_final_checking(df, neighbors_df, neighbor_list_array, source_bac_with_can_target,
                                      center_coordinate_columns, col_source, col_target, parent_image_number_col,
                                      parent_object_number_col,
                                      non_divided_bac_model, maintenance_cost_df, maintenance_to_be_check='target',
                                      coordinate_array=None):
    bac_len_to_bac_ratio_boundary = find_bac_len_to_bac_ratio_boundary(df)

    source_bac_with_can_target['index_prev' + col_target] = source_bac_with_can_target['index' + col_target]
    source_bac_with_can_target['index2' + col_target] = source_bac_with_can_target.index.values

    source_bac_with_can_target['LengthChangeRatio' + col_target] = (
            source_bac_with_can_target['AreaShape_MajorAxisLength' + col_target] /
            source_bac_with_can_target['AreaShape_MajorAxisLength' + col_source])

    if source_bac_with_can_target.shape[0] > 0:

        source_bac_with_can_target = iou_calc(source_bac_with_can_target, col_source='prev_index' + col_source,
                                              col_target='prev_index' + col_target, stat='same',
                                              coordinate_array=coordinate_array)

        source_bac_with_can_target = calc_distance(source_bac_with_can_target, center_coordinate_columns,
                                                   postfix_target=col_target, postfix_source=col_source, stat=None)

        source_bac_with_can_target['prev_time_step_index' + col_target] = \
            source_bac_with_can_target['index' + col_source].fillna(-1).astype('int64')

        source_bac_with_can_target['difference_neighbors' + col_target] = np.nan
        source_bac_with_can_target['other_daughter_index' + col_target] = np.nan
        source_bac_with_can_target['parent_id' + col_target] = source_bac_with_can_target['id' + col_source]

        source_bac_with_can_target = \
            neighbor_checking(source_bac_with_can_target, neighbor_list_array,
                              parent_image_number_col, parent_object_number_col,
                              selected_rows_df=source_bac_with_can_target, selected_time_step_df=df,
                              return_common_elements=True, col_target=col_target)

        source_bac_with_can_target = check_len_ratio(df, source_bac_with_can_target, col_target=col_target,
                                                     col_source=col_source)

        # motion alignment
        # calculated for original df and we should calc for new df
        source_bac_with_can_target["MotionAlignmentAngle" + col_target] = np.nan
        source_bac_with_can_target = \
            calc_MotionAlignmentAngle(df, neighbors_df, center_coordinate_columns,
                                      selected_rows=source_bac_with_can_target, col_target=col_target,
                                      col_source=col_source)

        source_bac_with_can_target['prev_time_step_center_x' + col_target] = \
            source_bac_with_can_target[center_coordinate_columns['x'] + col_source]
        source_bac_with_can_target['prev_time_step_center_y' + col_target] = \
            source_bac_with_can_target[center_coordinate_columns['y'] + col_source]

        direction_of_motion = calculate_trajectory_direction_angle_all(source_bac_with_can_target,
                                                                       center_coordinate_columns, col_target=col_target)

        source_bac_with_can_target['direction_of_motion' + col_target] = direction_of_motion

        source_bac_with_can_target['adjusted_common_neighbors' + col_target] = np.where(
            source_bac_with_can_target['common_neighbors' + col_target] == 0,
            source_bac_with_can_target['common_neighbors' + col_target] + 1,
            source_bac_with_can_target['common_neighbors' + col_target]
        )

        source_bac_with_can_target['neighbor_ratio' + col_target] = \
            (source_bac_with_can_target['difference_neighbors' + col_target] / (
                source_bac_with_can_target['adjusted_common_neighbors' + col_target]))

        raw_feature_list = ['iou', 'min_distance', 'difference_neighbors' + col_target, 'common_neighbors' + col_target,
                            'length_dynamic' + col_target,
                            'neighbor_ratio' + col_target,
                            'MotionAlignmentAngle' + col_target,
                            'direction_of_motion' + col_target, 'LengthChangeRatio' + col_target,
                            'index' + col_source, 'index_prev' + col_target]

        source_bac_with_can_target = source_bac_with_can_target[raw_feature_list].copy()
        source_bac_with_can_target = source_bac_with_can_target.rename(
            {
                'common_neighbors' + col_target: 'common_neighbors',
                'difference_neighbors' + col_target: 'difference_neighbors',
                'neighbor_ratio' + col_target: 'neighbor_ratio',
                'length_dynamic' + col_target: 'length_dynamic',
                'MotionAlignmentAngle' + col_target: 'MotionAlignmentAngle',
                'direction_of_motion' + col_target: 'direction_of_motion',
                'LengthChangeRatio' + col_target: 'LengthChangeRatio',
                'index_prev' + col_target: 'index_prev',
            }, axis=1)

        # difference_neighbors
        feature_list_for_non_divided_bac_model = \
            ['iou', 'min_distance', 'neighbor_ratio', 'length_dynamic', 'MotionAlignmentAngle']

        y_prob_non_divided_bac_model = \
            non_divided_bac_model.predict_proba(source_bac_with_can_target[
                                                    feature_list_for_non_divided_bac_model])[:, 1]
        source_bac_with_can_target['prob_non_divided_bac_model'] = y_prob_non_divided_bac_model

        source_bac_with_can_target = \
            source_bac_with_can_target.loc[(source_bac_with_can_target['prob_non_divided_bac_model'] > 0.5) &
                                           (source_bac_with_can_target['difference_neighbors'] <=
                                            source_bac_with_can_target['common_neighbors'])]

        # Pivot this DataFrame to get the desired structure
        same_link_cost_df = \
            source_bac_with_can_target[['index' + col_source, 'index_prev', 'prob_non_divided_bac_model']].pivot(
                index='index' + col_source, columns='index_prev', values='prob_non_divided_bac_model')
        same_link_cost_df.columns.name = None
        same_link_cost_df.index.name = None

        if maintenance_to_be_check == 'target':
            # I want to measure is it possible to remove the current link of target bacterium?
            candidate_target_maintenance_cost = \
                maintenance_cost_df.loc[:, maintenance_cost_df.columns.isin(
                    source_bac_with_can_target['index_prev'].unique())]

            # don't need to convert probability to 1 - probability
            for col in candidate_target_maintenance_cost.columns.values:
                # maintenance cost
                this_target_bac_maintenance_cost = candidate_target_maintenance_cost[col].dropna().iloc[0]
                same_link_cost_df.loc[same_link_cost_df[col] <= this_target_bac_maintenance_cost, col] = np.nan

        elif maintenance_to_be_check == 'source':

            # I want to measure is it possible to remove the current link of target bacterium?
            candidate_target_maintenance_cost = \
                maintenance_cost_df.loc[maintenance_cost_df.index.isin(
                    source_bac_with_can_target['index' + col_source].unique())]

            selected_candidate_target_maintenance_cost = \
                candidate_target_maintenance_cost[candidate_target_maintenance_cost.min(axis=1) > 0.5]

            for source_idx in selected_candidate_target_maintenance_cost.index.values:
                max_probability_value = candidate_target_maintenance_cost.loc[source_idx].max()

                same_link_cost_df.loc[source_idx, same_link_cost_df.loc[source_idx] <= max_probability_value] = np.nan

        same_link_cost_df = 1 - same_link_cost_df

    else:
        same_link_cost_df = pd.DataFrame()

    return same_link_cost_df


def adding_new_link_cost(df, neighbors_df, neighbor_list_array,
                         incorrect_same_link_in_this_time_step_with_target_neighbors_features,
                         center_coordinate_columns, parent_image_number_col, parent_object_number_col,
                         non_divided_bac_model, comparing_divided_non_divided_model, maintenance_cost_df,
                         coordinate_array):
    cost_df = \
        same_link_cost(df, neighbors_df, neighbor_list_array,
                       incorrect_same_link_in_this_time_step_with_target_neighbors_features.copy(),
                       center_coordinate_columns,
                       '_source', '', parent_image_number_col, parent_object_number_col,
                       non_divided_bac_model, comparing_divided_non_divided_model, maintenance_cost_df,
                       maintenance_to_be_check='target', coordinate_array=coordinate_array)

    return cost_df


def calc_maintenance_cost(df, all_bac_in_source_time_step, sel_source_bacteria_info,
                          all_bac_in_target_time_step, neighbor_df, sel_target_bacteria_info,
                          center_coordinate_columns, parent_image_number_col, parent_object_number_col,
                          comparing_divided_non_divided_model, coordinate_array):
    if sel_target_bacteria_info.shape[0] > 0 and sel_source_bacteria_info.shape[0] > 0:

        # difference_neighbors,
        feature_list = ['iou', 'min_distance', 'neighbor_ratio', 'direction_of_motion',
                        'MotionAlignmentAngle', 'LengthChangeRatio']

        important_cols_division = ['id', 'parent_id', 'prev_index',
                                   center_coordinate_columns['x'], center_coordinate_columns['y'],
                                   'endpoint1_X', 'endpoint1_Y', 'endpoint2_X', 'endpoint2_Y',
                                   'AreaShape_MajorAxisLength', 'common_neighbors',
                                   'difference_neighbors', 'direction_of_motion', 'MotionAlignmentAngle', 'index']

        important_cols_same = ['id', 'parent_id', 'prev_index',
                               center_coordinate_columns['x'], center_coordinate_columns['y'],
                               'endpoint1_X', 'endpoint1_Y', 'endpoint2_X', 'endpoint2_Y',
                               'AreaShape_MajorAxisLength', 'common_neighbors',
                               'difference_neighbors', 'direction_of_motion', 'MotionAlignmentAngle', 'index',
                               'LengthChangeRatio']

        division_merged_df = \
            sel_source_bacteria_info[important_cols_division].merge(
                sel_target_bacteria_info[important_cols_division],
                left_on='id', right_on='parent_id', how='inner', suffixes=('_parent', '_daughter'))

        same_bac_merged_df = \
            sel_source_bacteria_info[important_cols_same].merge(
                sel_target_bacteria_info[important_cols_same], left_on='id', right_on='id', how='inner',
                suffixes=('_bac1', '_bac2'))

        # IOU
        division_merged_df = iou_calc(division_merged_df,
                                      col_source='prev_index_parent', col_target='prev_index_daughter', stat='div',
                                      coordinate_array=coordinate_array, both=False)

        same_bac_merged_df = iou_calc(same_bac_merged_df,
                                      col_source='prev_index_bac1', col_target='prev_index_bac2', stat='same',
                                      coordinate_array=coordinate_array)

        # distance
        division_merged_df = calc_distance(division_merged_df, center_coordinate_columns, '_daughter',
                                           '_parent', stat='div')

        same_bac_merged_df = calc_distance(same_bac_merged_df, center_coordinate_columns, '_bac2',
                                           '_bac1')

        division_merged_df['LengthChangeRatio'] = (division_merged_df['AreaShape_MajorAxisLength_daughter'] /
                                                   division_merged_df['AreaShape_MajorAxisLength_parent'])

        division_merged_df['adjusted_common_neighbors_daughter'] = np.where(
            division_merged_df['common_neighbors_daughter'] == 0,
            division_merged_df['common_neighbors_daughter'] + 1,
            division_merged_df['common_neighbors_daughter']
        )

        division_merged_df['neighbor_ratio_daughter'] = \
            (division_merged_df['difference_neighbors_daughter'] / (
                division_merged_df['adjusted_common_neighbors_daughter']))

        same_bac_merged_df['adjusted_common_neighbors_bac2'] = np.where(
            same_bac_merged_df['common_neighbors_bac2'] == 0,
            same_bac_merged_df['common_neighbors_bac2'] + 1,
            same_bac_merged_df['common_neighbors_bac2']
        )

        same_bac_merged_df['neighbor_ratio_bac2'] = \
            (same_bac_merged_df['difference_neighbors_bac2'] / (
                same_bac_merged_df['adjusted_common_neighbors_bac2']))

        # rename columns
        division_merged_df = division_merged_df.rename(
            {'difference_neighbors_daughter': 'difference_neighbors',
             'neighbor_ratio_daughter': 'neighbor_ratio',
             'direction_of_motion_daughter': 'direction_of_motion',
             'MotionAlignmentAngle_daughter': 'MotionAlignmentAngle'}, axis=1)

        same_bac_merged_df = same_bac_merged_df.rename(
            {'difference_neighbors_bac2': 'difference_neighbors',
             'neighbor_ratio_bac2': 'neighbor_ratio',
             'direction_of_motion_bac2': 'direction_of_motion',
             'MotionAlignmentAngle_bac2': 'MotionAlignmentAngle',
             'LengthChangeRatio_bac2': 'LengthChangeRatio'}, axis=1)

        if division_merged_df.shape[0] > 0:
            y_prob_compare_division = \
                comparing_divided_non_divided_model.predict_proba(division_merged_df[feature_list])[:, 0]

            division_merged_df['prob_compare'] = y_prob_compare_division

            # Pivot this DataFrame to get the desired structure
            cost_df_division = \
                division_merged_df[['index_parent', 'index_daughter', 'prob_compare']].pivot(
                    index='index_parent', columns='index_daughter', values='prob_compare')
            cost_df_division.columns.name = None
            cost_df_division.index.name = None

        if same_bac_merged_df.shape[0] > 0:
            y_prob_compare_same_bac = \
                comparing_divided_non_divided_model.predict_proba(same_bac_merged_df[feature_list])[:, 1]

            same_bac_merged_df['prob_compare'] = y_prob_compare_same_bac

            cost_df_same = \
                same_bac_merged_df[['index_bac1', 'index_bac2', 'prob_compare']].pivot(
                    index='index_bac1', columns='index_bac2', values='prob_compare')
            cost_df_same.columns.name = None
            cost_df_same.index.name = None

        if same_bac_merged_df.shape[0] > 0 and division_merged_df.shape[0] > 0:
            final_maintenance_cost = pd.concat([cost_df_division, cost_df_same], axis=1, sort=False)
        elif same_bac_merged_df.shape[0] > 0:
            final_maintenance_cost = cost_df_same
        elif division_merged_df.shape[0] > 0:
            final_maintenance_cost = cost_df_division

    else:
        final_maintenance_cost = pd.DataFrame()

    return final_maintenance_cost


def feature_space_adding_new_link_to_unexpected(df, neighbors_df, unexpected_end_bac_in_current_time_step,
                                                all_bac_in_current_time_step, all_bac_in_next_time_step,
                                                center_coordinate_columns, parent_object_number_col):
    # neighbors bacteria to unexpected bac
    unexpected_end_bac_neighbors = \
        unexpected_end_bac_in_current_time_step.merge(neighbors_df, left_on=['ImageNumber', 'ObjectNumber'],
                                                      right_on=['First Image Number', 'First Object Number'],
                                                      how='inner')

    unexpected_end_bac_neighbors_link_next_time_step = \
        unexpected_end_bac_neighbors.merge(all_bac_in_next_time_step, left_on='Second Object Number',
                                           right_on=parent_object_number_col, how='inner',
                                           suffixes=('_unexpected_bac', '_neighbors_next_time_step'))

    neighbors_of_neighbors_link = \
        neighbors_df.loc[(neighbors_df['First Image Number'] ==
                          all_bac_in_next_time_step['ImageNumber'].values[0]) &
                         (neighbors_df['First Object Number'].isin(
                             unexpected_end_bac_neighbors_link_next_time_step['ObjectNumber_neighbors_next_time_step']))
                         ]

    neighbors_to_neighbors_link_info = neighbors_of_neighbors_link.merge(
        all_bac_in_next_time_step, left_on='Second Object Number', right_on='ObjectNumber', how='inner')

    neighbors_to_neighbors_link_info.columns = [v + '_neighbors_neighbors_next_time_step' for v
                                                in neighbors_to_neighbors_link_info.columns]

    total_neighbors = \
        unexpected_end_bac_neighbors_link_next_time_step.merge(
            neighbors_to_neighbors_link_info,
            left_on='ObjectNumber_neighbors_next_time_step',
            right_on='First Object Number_neighbors_neighbors_next_time_step', how='inner')

    final_neighbors_unexpected_bac_df = total_neighbors[['ImageNumber_unexpected_bac', 'ObjectNumber_unexpected_bac',
                                                         center_coordinate_columns['x'] + '_unexpected_bac',
                                                         center_coordinate_columns['y'] + '_unexpected_bac',
                                                         'prev_index_unexpected_bac', 'index_unexpected_bac',
                                                         'endpoint1_X_unexpected_bac', 'endpoint1_Y_unexpected_bac',
                                                         'endpoint2_X_unexpected_bac', 'endpoint2_Y_unexpected_bac',
                                                         'ImageNumber_neighbors_next_time_step',
                                                         'ObjectNumber_neighbors_next_time_step',
                                                         center_coordinate_columns['x'] + '_neighbors_next_time_step',
                                                         center_coordinate_columns['y'] + '_neighbors_next_time_step',
                                                         'prev_index_neighbors_next_time_step',
                                                         'index_neighbors_next_time_step',
                                                         'endpoint1_X_neighbors_next_time_step',
                                                         'endpoint1_Y_neighbors_next_time_step',
                                                         'endpoint2_X_neighbors_next_time_step',
                                                         'endpoint2_Y_neighbors_next_time_step']]

    final_neighbors_unexpected_bac_df.columns = ['ImageNumber_unexpected_bac', 'ObjectNumber_unexpected_bac',
                                                 center_coordinate_columns['x'] + '_unexpected_bac',
                                                 center_coordinate_columns['y'] + '_unexpected_bac',
                                                 'prev_index_unexpected_bac', 'index_unexpected_bac',
                                                 'endpoint1_X_unexpected_bac', 'endpoint1_Y_unexpected_bac',
                                                 'endpoint2_X_unexpected_bac', 'endpoint2_Y_unexpected_bac',
                                                 'ImageNumber', 'ObjectNumber',
                                                 center_coordinate_columns['x'], center_coordinate_columns['y'],
                                                 'prev_index', 'index', 'endpoint1_X', 'endpoint1_Y',
                                                 'endpoint2_X', 'endpoint2_Y']

    final_neighbors_neighbors_unexpected_bac_df = \
        total_neighbors[['ImageNumber_unexpected_bac', 'ObjectNumber_unexpected_bac',
                         center_coordinate_columns['x'] + '_unexpected_bac',
                         center_coordinate_columns['y'] + '_unexpected_bac', 'prev_index_unexpected_bac',
                         'index_unexpected_bac', 'endpoint1_X_unexpected_bac', 'endpoint1_Y_unexpected_bac',
                         'endpoint2_X_unexpected_bac', 'endpoint2_Y_unexpected_bac',
                         'ImageNumber_neighbors_neighbors_next_time_step',
                         'ObjectNumber_neighbors_neighbors_next_time_step',
                         center_coordinate_columns['x'] + '_neighbors_neighbors_next_time_step',
                         center_coordinate_columns['y'] + '_neighbors_neighbors_next_time_step',
                         'prev_index_neighbors_neighbors_next_time_step', 'index_neighbors_neighbors_next_time_step',
                         'endpoint1_X_neighbors_neighbors_next_time_step',
                         'endpoint1_Y_neighbors_neighbors_next_time_step',
                         'endpoint2_X_neighbors_neighbors_next_time_step',
                         'endpoint2_Y_neighbors_neighbors_next_time_step']]

    final_neighbors_neighbors_unexpected_bac_df.columns = \
        ['ImageNumber_unexpected_bac', 'ObjectNumber_unexpected_bac',
         center_coordinate_columns['x'] + '_unexpected_bac',
         center_coordinate_columns['y'] + '_unexpected_bac',
         'prev_index_unexpected_bac', 'index_unexpected_bac',
         'endpoint1_X_unexpected_bac', 'endpoint1_Y_unexpected_bac',
         'endpoint2_X_unexpected_bac', 'endpoint2_Y_unexpected_bac',
         'ImageNumber', 'ObjectNumber',
         center_coordinate_columns['x'], center_coordinate_columns['y'],
         'prev_index', 'index', 'endpoint1_X', 'endpoint1_Y',
         'endpoint2_X', 'endpoint2_Y']

    final_candidate_bac = pd.concat([final_neighbors_neighbors_unexpected_bac_df,
                                     final_neighbors_unexpected_bac_df])

    final_candidate_bac = final_candidate_bac.drop_duplicates(subset=['ImageNumber', 'ObjectNumber'])

    bac_under_invest = df.loc[final_candidate_bac['index'].values]

    return bac_under_invest, final_candidate_bac


def adding_new_link_to_unexpected_end(df, neighbors_df, neighbor_list_array, unexpected_end_bac_in_current_time_step,
                                      all_bac_in_unexpected_end_bac_time_step,
                                      all_bac_in_next_time_step_to_unexpected_end_bac, center_coordinate_columns,
                                      parent_image_number_col, parent_object_number_col,
                                      min_life_history_of_bacteria, comparing_divided_non_divided_model,
                                      non_divided_bac_model, divided_bac_model, color_array, coordinate_array):
    max_daughter_len_to_mother_ratio_boundary = find_max_daughter_len_to_mother_ratio_boundary(df)
    sum_daughter_len_to_mother_ratio_boundary = find_sum_daughter_len_to_mother_ratio_boundary(df)

    upper_bound_max_daughter_len = find_upper_bound(max_daughter_len_to_mother_ratio_boundary)

    lower_bound_sum_daughter_len = find_lower_bound(sum_daughter_len_to_mother_ratio_boundary)

    upper_bound_sum_daughter_len = find_upper_bound(sum_daughter_len_to_mother_ratio_boundary)

    bac_under_invest, final_candidate_bac = \
        feature_space_adding_new_link_to_unexpected_end(neighbors_df, unexpected_end_bac_in_current_time_step,
                                                        all_bac_in_unexpected_end_bac_time_step,
                                                        all_bac_in_next_time_step_to_unexpected_end_bac,
                                                        center_coordinate_columns, parent_object_number_col,
                                                        color_array, coordinate_array)

    if final_candidate_bac.shape[0] > 0:

        source_of_bac_under_invest_link = \
            all_bac_in_unexpected_end_bac_time_step.loc[
                all_bac_in_unexpected_end_bac_time_step['ObjectNumber'].isin(
                    bac_under_invest[parent_object_number_col].values)]

        # now check the cost of maintaining the link
        maintenance_cost_df = calc_maintenance_cost(df, all_bac_in_unexpected_end_bac_time_step,
                                                    source_of_bac_under_invest_link,
                                                    all_bac_in_next_time_step_to_unexpected_end_bac,
                                                    neighbors_df, bac_under_invest,
                                                    center_coordinate_columns, parent_image_number_col,
                                                    parent_object_number_col, comparing_divided_non_divided_model,
                                                    coordinate_array=coordinate_array)

        same_link_cost_df = \
            same_link_cost(df, neighbors_df, neighbor_list_array, final_candidate_bac.copy(), center_coordinate_columns,
                           col_source='', col_target='_candidate', parent_image_number_col=parent_image_number_col,
                           parent_object_number_col=parent_object_number_col,
                           non_divided_bac_model=non_divided_bac_model,
                           comparing_divided_non_divided_model=comparing_divided_non_divided_model,
                           maintenance_cost_df=maintenance_cost_df, maintenance_to_be_check='target',
                           coordinate_array=coordinate_array)

        raw_same_link_cost_df = same_link_cost_df.copy()

        division_cost_df = daughter_cost(df, neighbors_df, neighbor_list_array, final_candidate_bac.copy(),
                                         center_coordinate_columns, '', '_candidate',
                                         parent_image_number_col, parent_object_number_col, divided_bac_model,
                                         comparing_divided_non_divided_model, min_life_history_of_bacteria,
                                         maintenance_cost_df=maintenance_cost_df, maintenance_to_be_check='target',
                                         coordinate_array=coordinate_array)

        # now I want to check division chance
        # ba careful the probability reported in division_cost_df is 1 - probability
        # conditions for division: 1. sum daughter length to mother 2. max daughter length to mother
        # 3. min life history of source
        # first I check life history of source bac (unexpected end)
        candidate_unexpected_end_bac_for_division = \
            final_candidate_bac.loc[(final_candidate_bac['age'] > min_life_history_of_bacteria) |
                                    (final_candidate_bac['unexpected_beginning'] == True)]

        # unexpected bacteria index is index of division_cost_df
        division_cost_df = \
            division_cost_df.loc[
                division_cost_df.index.isin(candidate_unexpected_end_bac_for_division['index'].values)]

        division_cost_dict = {}

        # < 2: # it means that division can not possible
        filtered_division_cost_df = division_cost_df[division_cost_df[division_cost_df < 0.5].count(axis=1) >= 2]

        for row_idx, row in filtered_division_cost_df.iterrows():

            candidate_daughters = row[row < 0.5].index.values

            unexpected_bac_length = df.at[row_idx, 'AreaShape_MajorAxisLength']

            if len(candidate_daughters) == 2:

                daughter1_length = df.at[candidate_daughters[0], 'AreaShape_MajorAxisLength']
                daughter2_length = df.at[candidate_daughters[1], 'AreaShape_MajorAxisLength']

                sum_daughter_to_mother = ((daughter1_length + daughter2_length) / unexpected_bac_length)

                max_daughter_to_mother = max(daughter1_length, daughter2_length) / unexpected_bac_length

                if max_daughter_to_mother < 1 and max_daughter_to_mother <= upper_bound_max_daughter_len and \
                        lower_bound_sum_daughter_len <= sum_daughter_to_mother <= upper_bound_sum_daughter_len:
                    division_cost_dict[row_idx] = {candidate_daughters[0]: row[candidate_daughters[0]],
                                                   candidate_daughters[1]: row[candidate_daughters[1]]}

            elif len(candidate_daughters) > 2:

                num_candidate_daughters = len(candidate_daughters)

                division_cost_for_compare = []
                division_cost_daughters = []

                for i_idx in range(num_candidate_daughters):

                    i = candidate_daughters[i_idx]
                    daughter1_length = df.at[i, 'AreaShape_MajorAxisLength']

                    for j_idx in range(i_idx + 1, num_candidate_daughters):

                        j = candidate_daughters[j_idx]

                        daughter2_length = df.at[j, 'AreaShape_MajorAxisLength']

                        sum_daughter_to_mother = ((daughter1_length + daughter2_length) / unexpected_bac_length)

                        max_daughter_to_mother = max(daughter1_length, daughter2_length) / unexpected_bac_length

                        if (max_daughter_to_mother < 1 and
                                max_daughter_to_mother <= upper_bound_max_daughter_len and
                                lower_bound_sum_daughter_len <=
                                sum_daughter_to_mother <= upper_bound_sum_daughter_len):
                            # daughters
                            # compare basen on average probability (1 - probability)
                            division_cost_for_compare.append(np.average([row[i], row[j]]))
                            division_cost_daughters.append((i, j))

                    if division_cost_daughters:
                        min_div_cost_idx = np.argmin(division_cost_for_compare)
                        selected_daughters = division_cost_daughters[min_div_cost_idx]

                        division_cost_dict[row_idx] = {
                            selected_daughters[0]: row[selected_daughters[0]],
                            selected_daughters[1]: row[selected_daughters[1]]}

        final_division_cost_df = pd.DataFrame.from_dict(division_cost_dict, orient='index')

        # now I want to check is there any column with more than one value? if yes, it means that two different source
        # want to have same daughter
        non_nan_counts = final_division_cost_df.count()
        # Identify columns with more than one non-NaN value
        # it means that this daughter can participate in two different division
        columns_with_multiple_non_nan = non_nan_counts[non_nan_counts > 1].index

        if len(columns_with_multiple_non_nan) > 0:
            for col in columns_with_multiple_non_nan:
                # if final_division_cost_df[col].count() > 1:
                # I want to compare average probability
                division_cost_check = final_division_cost_df.loc[~ final_division_cost_df[col].isna()]

                # find division with lower chance
                avg_division_probability_per_unexpected_end_bac = division_cost_check.mean(axis=1)

                incorrect_division_cond = (avg_division_probability_per_unexpected_end_bac >
                                           avg_division_probability_per_unexpected_end_bac.min())

                # Align the boolean condition index with the original DataFrame
                condition_aligned = incorrect_division_cond.reindex(final_division_cost_df.index, fill_value=False)

                final_division_cost_df.loc[condition_aligned] = np.nan

        # now we should compare division with same cost because one target bac can be daughter of
        common_columns = np.intersect1d(final_division_cost_df.columns, same_link_cost_df.columns)

        if len(common_columns) > 0:

            # Calculate the minimum values for each column in same_link_cost_df
            incorrect_same_link_nan_col = []
            incorrect_division_row_ndx = []

            for col in common_columns:
                min_same_bac_cost = same_link_cost_df[col].min()

                division_related_to_this_col = final_division_cost_df.loc[~ final_division_cost_df[col].isna()]
                min_div_probability = min(division_related_to_this_col.mean())

                if min_same_bac_cost > min_div_probability:
                    # it means 1 - same probability is lower than 1 - division probability
                    incorrect_same_link_nan_col.append(col)
                else:
                    incorrect_division_row_ndx.extend(division_related_to_this_col.index.values)
                    final_division_cost_df.loc[division_related_to_this_col.index.values] = np.nan

            same_link_cost_df[incorrect_same_link_nan_col] = np.nan
            final_division_cost_df.loc[incorrect_division_row_ndx] = np.nan

        # remove rows with all value equal to nan
        same_link_cost_df = same_link_cost_df.dropna(axis=1, how='all')
        final_division_cost_df = final_division_cost_df.dropna(axis=1, how='all')
        final_division_cost_df_after_merge_prob = final_division_cost_df.copy()

        un_selected_col = []
        for row_idx, row in final_division_cost_df_after_merge_prob.iterrows():
            un_selected_col.extend(row[row > row.min()].index.values.tolist())

        final_division_cost_df_after_merge_prob[un_selected_col] = np.nan
        final_division_cost_df_after_merge_prob = final_division_cost_df_after_merge_prob.dropna(axis=1, how='all')

    else:
        same_link_cost_df = pd.DataFrame()
        final_division_cost_df = pd.DataFrame()
        final_division_cost_df_after_merge_prob = pd.DataFrame()
        maintenance_cost_df = pd.DataFrame()
        raw_same_link_cost_df = pd.DataFrame()
        division_cost_df = pd.DataFrame()

    return (raw_same_link_cost_df, same_link_cost_df, division_cost_df, final_division_cost_df_after_merge_prob,
            final_division_cost_df, maintenance_cost_df)
