import numpy as np
from Trackrefiner.strain.correction.action.findOverlap import (find_overlap_object_to_next_frame,
                                                               find_overlap_object_to_next_frame_maintain,
                                                               find_overlap_object_to_next_frame_unexpected,
                                                               find_overlap_mother_bad_daughters,
                                                               find_overlap_object_for_division_chance)
from Trackrefiner.strain.correction.action.findOutlier import find_upper_bound, find_lower_bound
import pandas as pd


def dist_for_maintain_same(same_df, center_coordinate_columns):
    # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)
    same_df['center_distance'] = \
        np.linalg.norm(same_df[[center_coordinate_columns['x'] + '_1', center_coordinate_columns['y'] + '_1']].values -
                       same_df[[center_coordinate_columns['x'] + '_2', center_coordinate_columns['y'] + '_2']].values,
                       axis=1)

    same_df['endpoint1_1_distance'] = \
        np.linalg.norm(same_df[['endpoint1_X_1', 'endpoint1_Y_1']].values -
                       same_df[['endpoint1_X_2', 'endpoint1_Y_2']].values, axis=1)

    same_df['endpoint2_2_distance'] = \
        np.linalg.norm(same_df[['endpoint2_X_1', 'endpoint2_Y_1']].values -
                       same_df[['endpoint2_X_2', 'endpoint2_Y_2']].values, axis=1)

    same_df['min_distance'] = same_df[['center_distance', 'endpoint1_1_distance', 'endpoint2_2_distance']].min(axis=1)

    # Pivot this DataFrame to get the desired structure
    same_df_distance_df = \
        same_df[['index_1', 'index_2', 'min_distance']].pivot(index='index_1', columns='index_2',
                                                              values='min_distance')
    same_df_distance_df.columns.name = None
    same_df_distance_df.index.name = None

    return same_df_distance_df


def dist_for_maintain_division(division_df, center_coordinate_columns):
    # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)
    division_df['center_distance'] = \
        np.linalg.norm(division_df[[center_coordinate_columns['x'] + '_parent',
                                    center_coordinate_columns['y'] + '_parent']].values -
                       division_df[[center_coordinate_columns['x'] + '_daughter',
                                    center_coordinate_columns['y'] + '_daughter']].values, axis=1)

    division_df['endpoint1_1_distance'] = \
        np.linalg.norm(division_df[['endpoint1_X_parent', 'endpoint1_Y_parent']].values -
                       division_df[['endpoint1_X_daughter', 'endpoint1_Y_daughter']].values, axis=1)

    division_df['endpoint2_2_distance'] = \
        np.linalg.norm(division_df[['endpoint2_X_parent', 'endpoint2_Y_parent']].values -
                       division_df[['endpoint2_X_daughter', 'endpoint2_Y_daughter']].values, axis=1)

    # relation: mother & daughter
    division_df['endpoint12_distance'] = \
        np.linalg.norm(division_df[['endpoint1_X_parent', 'endpoint1_Y_parent']].values -
                       division_df[['endpoint2_X_daughter', 'endpoint2_Y_daughter']].values, axis=1)

    division_df['endpoint21_distance'] = \
        np.linalg.norm(division_df[['endpoint2_X_parent', 'endpoint2_Y_parent']].values -
                       division_df[['endpoint1_X_daughter', 'endpoint1_Y_daughter']].values, axis=1)

    division_df['center_endpoint1_distance'] = \
        np.linalg.norm(division_df[[center_coordinate_columns['x'] + '_parent',
                                    center_coordinate_columns['y'] + '_parent']].values -
                       division_df[['endpoint1_X_daughter', 'endpoint1_Y_daughter']].values, axis=1)

    division_df['center_endpoint2_distance'] = \
        np.linalg.norm(division_df[[center_coordinate_columns['x'] + '_parent',
                                    center_coordinate_columns['y'] + '_parent']].values -
                       division_df[['endpoint2_X_daughter', 'endpoint2_Y_daughter']].values, axis=1)

    division_df['min_distance'] = (
        division_df)[['center_distance', 'endpoint1_1_distance', 'endpoint2_2_distance', 'endpoint12_distance',
                      'endpoint21_distance', 'center_endpoint1_distance', 'center_endpoint2_distance']].min(axis=1)

    # Pivot this DataFrame to get the desired structure
    division_distance_df = \
        division_df[['index_parent', 'index_daughter', 'min_distance']].pivot(index='index_parent',
                                                                              columns='index_daughter',
                                                                              values='min_distance')
    division_distance_df.columns.name = None
    division_distance_df.index.name = None

    return division_distance_df


def make_initial_distance_matrix_unexpected(final_candidate_bac, center_coordinate_columns, coordinate_array):

    overlap_df = \
        find_overlap_object_to_next_frame_unexpected(final_candidate_bac, coordinate_array)

    # distance
    # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)
    final_candidate_bac['center_distance'] = \
        np.linalg.norm(final_candidate_bac[[center_coordinate_columns['x'],
                                            center_coordinate_columns['y']]].values -
                       final_candidate_bac[[center_coordinate_columns['x'] + '_candidate',
                                            center_coordinate_columns['y'] + '_candidate']].values, axis=1)

    final_candidate_bac['endpoint1_1_distance'] = \
        np.linalg.norm(final_candidate_bac[['endpoint1_X', 'endpoint1_Y']].values -
                       final_candidate_bac[['endpoint1_X_candidate', 'endpoint1_Y_candidate']].values, axis=1)

    final_candidate_bac['endpoint2_2_distance'] = \
        np.linalg.norm(final_candidate_bac[['endpoint2_X', 'endpoint2_Y']].values -
                       final_candidate_bac[['endpoint2_X_candidate', 'endpoint2_Y_candidate']].values, axis=1)

    final_candidate_bac['min_distance'] = \
        final_candidate_bac[['center_distance', 'endpoint1_1_distance', 'endpoint2_2_distance']].min(axis=1)

    # Pivot this DataFrame to get the desired structure
    unexpected_bac_distance_df = \
        final_candidate_bac[['index', 'index_candidate', 'min_distance']].pivot(index='index',
                                                                                columns='index_candidate',
                                                                                values='min_distance')
    unexpected_bac_distance_df.columns.name = None
    unexpected_bac_distance_df.index.name = None

    return unexpected_bac_distance_df, overlap_df


def make_initial_distance_matrix_bad_daughters(mother_bad_daughters_df, center_coordinate_columns,
                                               coordinate_array):

    overlap_df = find_overlap_mother_bad_daughters(mother_bad_daughters_df, coordinate_array)

    # distance
    # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)
    mother_bad_daughters_df['center_distance'] = \
        np.linalg.norm(mother_bad_daughters_df[[center_coordinate_columns['x'] + '_mother',
                                                center_coordinate_columns['y'] + '_mother']].values -
                       mother_bad_daughters_df[[center_coordinate_columns['x'] + '_daughter',
                                                center_coordinate_columns['y'] + '_daughter']].values, axis=1)

    mother_bad_daughters_df['endpoint1_1_distance'] = \
        np.linalg.norm(mother_bad_daughters_df[['endpoint1_X_mother', 'endpoint1_Y_mother']].values -
                       mother_bad_daughters_df[['endpoint1_X_daughter', 'endpoint1_Y_daughter']].values, axis=1)

    mother_bad_daughters_df['endpoint2_2_distance'] = \
        np.linalg.norm(mother_bad_daughters_df[['endpoint2_X_mother', 'endpoint2_Y_mother']].values -
                       mother_bad_daughters_df[['endpoint2_X_daughter', 'endpoint2_Y_daughter']].values, axis=1)

    mother_bad_daughters_df['min_distance'] = \
        mother_bad_daughters_df[['center_distance', 'endpoint1_1_distance', 'endpoint2_2_distance']].min(axis=1)

    # Pivot this DataFrame to get the desired structure
    mother_bad_daughters_distance_df = \
        mother_bad_daughters_df[['index_mother', 'index_daughter', 'min_distance']].pivot(
            index='index_mother', columns='index_daughter', values='min_distance')
    mother_bad_daughters_distance_df.columns.name = None
    mother_bad_daughters_distance_df.index.name = None

    return mother_bad_daughters_distance_df, overlap_df


def make_initial_distance_matrix_for_division_chance(source_with_candidate_neighbors, center_coordinate_columns,
                                                     col1, col2, daughter_flag=True, coordinate_array=None):
    overlap_df = \
        find_overlap_object_for_division_chance(source_with_candidate_neighbors, center_coordinate_columns,
                                                col1, col2, daughter_flag, coordinate_array)

    # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)

    dis_cols = ['center_distance', 'endpoint1_1_distance', 'endpoint2_2_distance']

    source_with_candidate_neighbors['center_distance'] = \
        np.linalg.norm(source_with_candidate_neighbors[[center_coordinate_columns['x'] + col1,
                                                        center_coordinate_columns['y'] + col1]].values -
                       source_with_candidate_neighbors[[center_coordinate_columns['x'] + col2,
                                                        center_coordinate_columns['y'] + col2]].values, axis=1)

    source_with_candidate_neighbors['endpoint1_1_distance'] = \
        np.linalg.norm(source_with_candidate_neighbors[['endpoint1_X' + col1, 'endpoint1_Y' + col1]].values -
                       source_with_candidate_neighbors[['endpoint1_X' + col2, 'endpoint1_Y' + col2]].values, axis=1)

    source_with_candidate_neighbors['endpoint2_2_distance'] = \
        np.linalg.norm(source_with_candidate_neighbors[['endpoint2_X' + col1, 'endpoint2_Y' + col1]].values -
                       source_with_candidate_neighbors[['endpoint2_X' + col2, 'endpoint2_Y' + col2]].values, axis=1)

    if daughter_flag:
        dis_cols.extend(['endpoint1_2_distance', 'endpoint2_1_distance', 'center_endpoint1_distance',
                         'center_endpoint2_distance'])

        source_with_candidate_neighbors['endpoint1_2_distance'] = \
            np.linalg.norm(source_with_candidate_neighbors[['endpoint1_X' + col1, 'endpoint1_Y' + col1]].values -
                           source_with_candidate_neighbors[['endpoint2_X' + col2, 'endpoint2_Y' + col2]].values,
                           axis=1)

        source_with_candidate_neighbors['endpoint2_1_distance'] = \
            np.linalg.norm(source_with_candidate_neighbors[['endpoint2_X' + col1, 'endpoint2_Y' + col1]].values -
                           source_with_candidate_neighbors[['endpoint1_X' + col2, 'endpoint1_Y' + col2]].values,
                           axis=1)

        source_with_candidate_neighbors['center_endpoint1_distance'] = \
            np.linalg.norm(source_with_candidate_neighbors[[center_coordinate_columns['x'] + col1,
                                                            center_coordinate_columns['y'] + col1]].values -
                           source_with_candidate_neighbors[['endpoint1_X' + col2, 'endpoint1_Y' + col2]].values,
                           axis=1)

        source_with_candidate_neighbors['center_endpoint2_distance'] = \
            np.linalg.norm(source_with_candidate_neighbors[[center_coordinate_columns['x'] + col1,
                                                            center_coordinate_columns['y'] + col1]].values -
                           source_with_candidate_neighbors[['endpoint2_X' + col2, 'endpoint2_Y' + col2]].values, axis=1)

    source_with_candidate_neighbors['min_distance'] = source_with_candidate_neighbors[dis_cols].min(axis=1)

    # Pivot this DataFrame to get the desired structure
    distance_df = \
        source_with_candidate_neighbors[['index' + col1, 'index' + col2, 'min_distance']].pivot(
            index='index' + col1, columns='index' + col2, values='min_distance')
    distance_df.columns.name = None
    distance_df.index.name = None

    return overlap_df, distance_df


def make_initial_distance_matrix(source_time_step_df, sel_source_bacteria, bacteria_in_target_time_step,
                                 sel_target_bacteria, center_coordinate_columns, color_array,
                                 daughter_flag=False, maintain=False, coordinate_array=None):

    if sel_target_bacteria.shape[0] > 0 and sel_source_bacteria.shape[0] > 0:

        if maintain:

            division_df = \
                sel_source_bacteria.merge(sel_target_bacteria, left_on='id', right_on='parent_id', how='inner',
                                          suffixes=('_parent', '_daughter'))
            same_df = sel_source_bacteria.merge(sel_target_bacteria, on='id', how='inner', suffixes=('_1', '_2'))

            division_distance_df = dist_for_maintain_division(division_df, center_coordinate_columns)
            same_distance_df = dist_for_maintain_same(same_df, center_coordinate_columns)

            distance_df = pd.concat([division_distance_df, same_distance_df], axis=0)
            overlap_df = find_overlap_object_to_next_frame_maintain(division_df, same_df, coordinate_array)

            distance_df = distance_df.fillna(999)
            overlap_df = overlap_df.fillna(0)

        else:

            overlap_df, product_df = \
                find_overlap_object_to_next_frame(source_time_step_df, sel_source_bacteria,
                                                  bacteria_in_target_time_step, sel_target_bacteria,
                                                  center_coordinate_columns,
                                                  color_array=color_array,
                                                  daughter_flag=daughter_flag,
                                                  coordinate_array=coordinate_array)

            # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)

            dis_cols = ['center_distance', 'endpoint1_1_distance', 'endpoint2_2_distance']

            # _current', '_next
            product_df['center_distance'] = \
                np.linalg.norm(product_df[[center_coordinate_columns['x'] + '_current',
                                           center_coordinate_columns['y'] + '_current']].values -
                               product_df[[center_coordinate_columns['x'] + '_next',
                                           center_coordinate_columns['y'] + '_next']].values, axis=1)

            product_df['endpoint1_1_distance'] = \
                np.linalg.norm(product_df[['endpoint1_X_current', 'endpoint1_Y_current']].values -
                               product_df[['endpoint1_X_next', 'endpoint1_Y_next']].values, axis=1)

            product_df['endpoint2_2_distance'] = \
                np.linalg.norm(product_df[['endpoint2_X_current', 'endpoint2_Y_current']].values -
                               product_df[['endpoint2_X_next', 'endpoint2_Y_next']].values, axis=1)

            if daughter_flag:
                dis_cols.extend(['endpoint1_2_distance', 'endpoint2_1_distance', 'center_endpoint1_distance',
                                 'center_endpoint2_distance'])

                product_df['endpoint1_2_distance'] = \
                    np.linalg.norm(product_df[['endpoint1_X_current', 'endpoint1_Y_current']].values -
                                   product_df[['endpoint2_X_next', 'endpoint2_Y_next']].values, axis=1)

                product_df['endpoint2_1_distance'] = \
                    np.linalg.norm(product_df[['endpoint2_X_current', 'endpoint2_Y_current']].values -
                                   product_df[['endpoint1_X_next', 'endpoint1_Y_next']].values, axis=1)

                product_df['center_endpoint1_distance'] = \
                    np.linalg.norm(product_df[[center_coordinate_columns['x'] + '_current',
                                               center_coordinate_columns['y'] + '_current']].values -
                                   product_df[['endpoint1_X_next', 'endpoint1_Y_next']].values, axis=1)

                product_df['center_endpoint2_distance'] = \
                    np.linalg.norm(product_df[[center_coordinate_columns['x'] + '_current',
                                               center_coordinate_columns['y'] + '_current']].values -
                                   product_df[['endpoint2_X_next', 'endpoint2_Y_next']].values, axis=1)

            product_df['min_distance'] = product_df[dis_cols].min(axis=1)

            # Pivot this DataFrame to get the desired structure
            distance_df = \
                product_df[['index_current', 'index_next', 'min_distance']].pivot(
                    index='index_current', columns='index_next', values='min_distance')
            distance_df.columns.name = None
            distance_df.index.name = None

    else:
        if sel_target_bacteria.shape[0] > 0:
            overlap_df = pd.DataFrame(columns=[sel_target_bacteria.index], index=[0], data=999)
            distance_df = pd.DataFrame(columns=[sel_target_bacteria.index], index=[0], data=999)
        elif sel_source_bacteria.shape[0] > 0:
            overlap_df = pd.DataFrame(columns=[0], index=sel_source_bacteria.index, data=999)
            distance_df = pd.DataFrame(columns=[0], index=sel_source_bacteria.index, data=999)

    overlap_df = overlap_df.apply(pd.to_numeric)
    distance_df = distance_df.apply(pd.to_numeric)

    return overlap_df, distance_df


def create_division_link_instead_cont_life_cost(df_source_daughter_cost, source_bac_ndx, source_bac, target_bac_ndx,
                                                target_bac, source_bac_next_time_step,
                                                sum_daughter_len_to_mother_ratio_boundary,
                                                max_daughter_len_to_mother_ratio_boundary,
                                                min_life_history_of_bacteria):

    daughters_bac_len_to_source = ((target_bac['AreaShape_MajorAxisLength'] +
                                    source_bac_next_time_step['AreaShape_MajorAxisLength'].values[0]) /
                                   source_bac['AreaShape_MajorAxisLength'])

    max_daughters_bac_len_to_source = (max(target_bac['AreaShape_MajorAxisLength'],
                                           source_bac_next_time_step['AreaShape_MajorAxisLength'].values[0]) /
                                       source_bac['AreaShape_MajorAxisLength'])

    upper_bound_sum_daughter_len = find_upper_bound(sum_daughter_len_to_mother_ratio_boundary)

    lower_bound_sum_daughter_len = find_lower_bound(sum_daughter_len_to_mother_ratio_boundary)

    upper_bound_max_daughter_len = find_upper_bound(max_daughter_len_to_mother_ratio_boundary)

    if (lower_bound_sum_daughter_len <= daughters_bac_len_to_source <= upper_bound_sum_daughter_len and
            max_daughters_bac_len_to_source < 1 and max_daughters_bac_len_to_source <= upper_bound_max_daughter_len):

        # if source_bac['divideFlag']:
        #    if source_bac['LifeHistory'] > source_bac['age'] + min_life_history_of_bacteria:
        #        if source_bac['age'] > min_life_history_of_bacteria or source_bac['unexpected_beginning'] == True:
        #            calc_cost = True
        #        else:
        #            calc_cost = False
        #    else:
        #        calc_cost = False
        # else:
        #    calc_cost = True

        if source_bac['age'] > min_life_history_of_bacteria or source_bac['unexpected_beginning'] == True:
            calc_cost = True
        else:
            calc_cost = False

        if calc_cost:
            division_cost = df_source_daughter_cost.at[source_bac_ndx, target_bac_ndx]
        else:
            # prob = 0 so 1 - prob = 1
            division_cost = 1
    else:
        # prob = 0 so 1 - prob = 1
        division_cost = 1

    return division_cost


def link_to_source_with_already_has_one_link(cost_df, division_cost_dict,
                                             source_bac_ndx, source_bac, target_bac_ndx,
                                             target_bac, source_bac_next_time_step,
                                             sum_daughter_len_to_mother_ratio_boundary,
                                             max_daughter_len_to_mother_ratio_boundary, min_life_history_of_bacteria):

    # 2 possible modes:
    # 1. new link cost

    # 2. Maintain the previous link and create a new link (cell division)
    division_cost = \
        create_division_link_instead_cont_life_cost(cost_df, source_bac_ndx, source_bac, target_bac_ndx,
                                                    target_bac, source_bac_next_time_step,
                                                    sum_daughter_len_to_mother_ratio_boundary,
                                                    max_daughter_len_to_mother_ratio_boundary,
                                                    min_life_history_of_bacteria)

    # for new link cost
    division_cost_dict[target_bac_ndx][source_bac_ndx] = division_cost

    return division_cost_dict


def replacing_new_link_to_one_of_daughters(maintenance_cost_df, prev_daughters_idx, target_bac_ndx,
                                           source_bac_ndx, new_daughter_prob, redundant_link_dict_division):

    # we should find bad daughter
    all_bac_prob_dict = {
        prev_daughters_idx[0]: maintenance_cost_df.at[source_bac_ndx, prev_daughters_idx[0]],
        prev_daughters_idx[1]: maintenance_cost_df.at[source_bac_ndx, prev_daughters_idx[1]],
        target_bac_ndx: 1 - new_daughter_prob,
    }

    # bac with minimum probability
    wrong_daughter_index = list(all_bac_prob_dict.keys())[np.argmin(list(all_bac_prob_dict.values()))]

    if wrong_daughter_index != target_bac_ndx:

        adding_new_daughter_cost = new_daughter_prob
        redundant_link_dict_division[target_bac_ndx][source_bac_ndx] = wrong_daughter_index

    else:
        # probability = 0 so 1 - probability = 1
        adding_new_daughter_cost = 1

    return adding_new_daughter_cost, redundant_link_dict_division


def replacing_new_link_to_division(maintenance_cost_df, source_bac_ndx, source_bac, target_bac_ndx, target_bac,
                                   new_daughter_cost, source_bac_daughters,
                                   max_daughter_len_to_mother_ratio_boundary,
                                   sum_daughter_len_to_mother_ratio_boundary,
                                   redundant_link_dict_division):

    upper_bound_max_daughter_len = find_upper_bound(max_daughter_len_to_mother_ratio_boundary)

    upper_bound_sum_daughter_len = find_upper_bound(sum_daughter_len_to_mother_ratio_boundary)

    lower_bound_sum_daughter_len = find_lower_bound(sum_daughter_len_to_mother_ratio_boundary)

    source_bac_daughters['new_sum_daughters_len_to_source'] = \
        (source_bac_daughters['AreaShape_MajorAxisLength'] + target_bac['AreaShape_MajorAxisLength']) / \
        source_bac['AreaShape_MajorAxisLength']

    source_bac_daughters['new_max_daughters_len_to_source'] = \
        np.maximum(source_bac_daughters['AreaShape_MajorAxisLength'], target_bac['AreaShape_MajorAxisLength']) / \
        source_bac['AreaShape_MajorAxisLength']

    filtered_daughters = \
        source_bac_daughters.loc[(source_bac_daughters['new_max_daughters_len_to_source'] < 1) &
                                 (source_bac_daughters['new_max_daughters_len_to_source'] <=
                                  upper_bound_max_daughter_len) &
                                 (lower_bound_sum_daughter_len <=
                                  source_bac_daughters['new_sum_daughters_len_to_source']) &
                                 (source_bac_daughters['new_sum_daughters_len_to_source'] <=
                                  upper_bound_sum_daughter_len)]

    if filtered_daughters.shape[0] > 0:

        if filtered_daughters.shape[0] == 1:

            other_daughter_ndx = [v for v in source_bac_daughters['index'].values if v not in
                                  filtered_daughters['index'].values][0]

            other_daughter_prob = maintenance_cost_df.at[source_bac_ndx, other_daughter_ndx]
            target_bac_prob = 1 - new_daughter_cost

            if target_bac_prob > other_daughter_prob:
                redundant_link_dict_division[target_bac_ndx][source_bac_ndx] = other_daughter_ndx

                adding_new_daughter_cost = new_daughter_cost
            else:
                # probability = 0 so 1 - probability = 1
                adding_new_daughter_cost = 1

        else:

            adding_new_daughter_cost, redundant_link_dict_new_link = \
                replacing_new_link_to_one_of_daughters(maintenance_cost_df, filtered_daughters['index'].values.tolist(),
                                                       target_bac_ndx, source_bac_ndx, new_daughter_cost,
                                                       redundant_link_dict_division)
    else:
        # probability = 0 so 1 - probability = 1
        adding_new_daughter_cost = 1

    return adding_new_daughter_cost, redundant_link_dict_division


def link_to_source_with_two_links(division_cost_df, maintenance_cost_df, division_cost_dict,
                                  source_bac_ndx,
                                  source_bac, target_bac_ndx, target_bac,
                                  redundant_link_dict_division, source_bac_daughters,
                                  max_daughter_len_to_mother_ratio_boundary, sum_daughter_len_to_mother_ratio_boundary):

    new_daughter_cost = division_cost_df.at[source_bac_ndx, target_bac_ndx]

    (adding_new_daughter_cost, redundant_link_dict_division) = \
        replacing_new_link_to_division(maintenance_cost_df,
                                       source_bac_ndx, source_bac, target_bac_ndx, target_bac,
                                       new_daughter_cost, source_bac_daughters,
                                       max_daughter_len_to_mother_ratio_boundary,
                                       sum_daughter_len_to_mother_ratio_boundary,
                                       redundant_link_dict_division)

    division_cost_dict[target_bac_ndx][source_bac_ndx] = adding_new_daughter_cost

    return division_cost_dict, redundant_link_dict_division


def adding_new_terms_to_cost_matrix(division_cost_df, maintenance_cost_df, division_cost_dict,
                                    source_bac_ndx, source_bac, target_bac_ndx, target_bac,
                                    redundant_link_dict_division,
                                    sum_daughter_len_to_mother_ratio_boundary,
                                    max_daughter_len_to_mother_ratio_boundary, min_life_history_of_bacteria,
                                    source_bac_next_time_step, source_bac_daughters):

    if source_bac_next_time_step.shape[0] > 0:

        # if we want to make new link, we should remove prev link of source bac, so we should write the index of prev
        # target of source bac in redundant dict
        division_cost_dict = \
            link_to_source_with_already_has_one_link(division_cost_df,
                                                     division_cost_dict,
                                                     source_bac_ndx, source_bac,
                                                     target_bac_ndx,
                                                     target_bac, source_bac_next_time_step,
                                                     sum_daughter_len_to_mother_ratio_boundary,
                                                     max_daughter_len_to_mother_ratio_boundary,
                                                     min_life_history_of_bacteria)

    elif source_bac_daughters.shape[0] > 0:

        # my idea: at least one of a daughter-mother links can be wrong
        division_cost_dict, redundant_link_dict_division = \
            link_to_source_with_two_links(division_cost_df, maintenance_cost_df, division_cost_dict,
                                          source_bac_ndx, source_bac, target_bac_ndx, target_bac,
                                          redundant_link_dict_division, source_bac_daughters,
                                          max_daughter_len_to_mother_ratio_boundary,
                                          sum_daughter_len_to_mother_ratio_boundary)

    return division_cost_dict, redundant_link_dict_division
