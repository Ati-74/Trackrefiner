import numpy as np
from CellProfilerAnalysis.strain.correction.action.helperFunctions import calculate_trajectory_direction, \
    calc_neighbors_dir_motion, calc_normalized_angle_between_motion, distance_normalization, \
    calculate_orientation_angle, calc_distance_matrix
from CellProfilerAnalysis.strain.correction.action.findOverlap import find_overlap_object_to_next_frame
from CellProfilerAnalysis.strain.correction.neighborChecking import check_num_neighbors
import pandas as pd


def make_initial_distance_matrix(masks_dict, source_time_step_df, sel_source_bacteria, bacteria_in_target_time_step,
                                 sel_target_bacteria, daughter_flag=False, maintain=False):

    if sel_target_bacteria.shape[0] > 0 and sel_source_bacteria.shape[0] > 0:

        overlap_df = find_overlap_object_to_next_frame(masks_dict, source_time_step_df, sel_source_bacteria,
                                                       bacteria_in_target_time_step, sel_target_bacteria, daughter_flag,
                                                       maintain)

        # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)
        try:
            center_distance_df = calc_distance_matrix(sel_source_bacteria, sel_target_bacteria,
                                                      'Location_Center_X', 'Location_Center_Y')

        except KeyError:
            center_distance_df = calc_distance_matrix(sel_source_bacteria, sel_target_bacteria,
                                                      'AreaShape_Center_X', 'AreaShape_Center_Y')

        endpoint1_1_distance_df = calc_distance_matrix(sel_source_bacteria, sel_target_bacteria,
                                                       'endppoint1_X', 'endppoint1_Y')
        endpoint2_2_distance_df = calc_distance_matrix(sel_source_bacteria, sel_target_bacteria, 'endppoint2_X',
                                                       'endppoint2_Y')

        if daughter_flag:
            endpoint12_distance_df = calc_distance_matrix(sel_source_bacteria, sel_target_bacteria, 'endppoint1_X',
                                                          'endppoint1_Y', 'endppoint2_X', 'endppoint2_Y')
            endpoint21_distance_df = calc_distance_matrix(sel_source_bacteria, sel_target_bacteria,
                                                          'endppoint2_X', 'endppoint2_Y',
                                                          'endppoint1_X', 'endppoint1_Y')

            try:
                center_endpoint1_distance_df = calc_distance_matrix(sel_source_bacteria, sel_target_bacteria,
                                                                    'Location_Center_X', 'Location_Center_Y',
                                                                    'endppoint1_X', 'endppoint1_Y')
                center_endpoint2_distance_df = calc_distance_matrix(sel_source_bacteria, sel_target_bacteria,
                                                                    'Location_Center_X', 'Location_Center_Y',
                                                                    'endppoint2_X', 'endppoint2_Y')

            except KeyError:

                center_endpoint1_distance_df = calc_distance_matrix(sel_source_bacteria, sel_target_bacteria,
                                                                    'AreaShape_Center_X', 'AreaShape_Center_Y',
                                                                    'endppoint1_X', 'endppoint1_Y')

                center_endpoint2_distance_df = calc_distance_matrix(sel_source_bacteria, sel_target_bacteria,
                                                                    'AreaShape_Center_X', 'AreaShape_Center_Y',
                                                                    'endppoint2_X', 'endppoint2_Y')

            # Concatenate the DataFrames
            combined_df = pd.concat([center_distance_df, endpoint2_2_distance_df, endpoint1_1_distance_df,
                                     endpoint12_distance_df, endpoint21_distance_df, center_endpoint1_distance_df,
                                     center_endpoint2_distance_df])
        else:
            # Concatenate the DataFrames
            combined_df = pd.concat([center_distance_df, endpoint2_2_distance_df, endpoint1_1_distance_df])

        # Group by index and find the min value for each cell
        distance_df = combined_df.groupby(level=0).min()

        if maintain:
            for source_bac_ndx in distance_df.index:
                for target_bac_ndx in distance_df.columns:
                    if sel_target_bacteria.loc[target_bac_ndx]['parent_id'] == \
                            sel_source_bacteria.loc[source_bac_ndx]['id']:
                        # relation: mother & daughter
                        this_link_endpoint12_distance = np.sqrt(
                            (sel_source_bacteria.loc[source_bac_ndx]['endppoint1_X'] - \
                             sel_target_bacteria.loc[target_bac_ndx]['endppoint2_X']) ** 2 + \
                            (sel_source_bacteria.loc[source_bac_ndx]['endppoint1_Y'] - \
                             sel_target_bacteria.loc[target_bac_ndx]['endppoint2_Y']) ** 2
                        )

                        this_link_endpoint21_distance = np.sqrt(
                            (sel_source_bacteria.loc[source_bac_ndx]['endppoint2_X'] - \
                             sel_target_bacteria.loc[target_bac_ndx]['endppoint1_X']) ** 2 + \
                            (sel_source_bacteria.loc[source_bac_ndx]['endppoint2_Y'] - \
                             sel_target_bacteria.loc[target_bac_ndx]['endppoint1_Y']) ** 2
                        )

                        try:

                            this_link_center_endpoint1_distance = np.sqrt(
                                (sel_source_bacteria.loc[source_bac_ndx]['Location_Center_X'] - \
                                 sel_target_bacteria.loc[target_bac_ndx]['endppoint1_X']) ** 2 + \
                                (sel_source_bacteria.loc[source_bac_ndx]['Location_Center_Y'] - \
                                 sel_target_bacteria.loc[target_bac_ndx]['endppoint1_Y']) ** 2
                            )

                            this_link_center_endpoint2_distance = np.sqrt(
                                (sel_source_bacteria.loc[source_bac_ndx]['Location_Center_X'] - \
                                 sel_target_bacteria.loc[target_bac_ndx]['endppoint2_X']) ** 2 + \
                                (sel_source_bacteria.loc[source_bac_ndx]['Location_Center_Y'] - \
                                 sel_target_bacteria.loc[target_bac_ndx]['endppoint2_Y']) ** 2
                            )

                        except KeyError:

                            this_link_center_endpoint1_distance = np.sqrt(
                                (sel_source_bacteria.loc[source_bac_ndx]['AreaShape_Center_X'] - \
                                 sel_target_bacteria.loc[target_bac_ndx]['endppoint1_X']) ** 2 + \
                                (sel_source_bacteria.loc[source_bac_ndx]['AreaShape_Center_Y'] - \
                                 sel_target_bacteria.loc[target_bac_ndx]['endppoint1_Y']) ** 2
                            )

                            this_link_center_endpoint2_distance = np.sqrt(
                                (sel_source_bacteria.loc[source_bac_ndx]['AreaShape_Center_X'] - \
                                 sel_target_bacteria.loc[target_bac_ndx]['endppoint2_X']) ** 2 + \
                                (sel_source_bacteria.loc[source_bac_ndx]['AreaShape_Center_Y'] - \
                                 sel_target_bacteria.loc[target_bac_ndx]['endppoint2_Y']) ** 2
                            )

                        distance_df.at[source_bac_ndx, target_bac_ndx] = \
                            min(distance_df.loc[source_bac_ndx][target_bac_ndx], this_link_endpoint12_distance,
                                this_link_endpoint21_distance, this_link_center_endpoint1_distance,
                                this_link_center_endpoint2_distance)

        # Convert all elements to float, coercing errors to NaN
        distance_df = distance_df.applymap(lambda x: pd.to_numeric(x, errors='coerce'))

    else:
        if sel_target_bacteria.shape[0] > 0:
            overlap_df = pd.DataFrame(columns=[sel_target_bacteria.index],
                                               index=[0], data=999)
            distance_df = pd.DataFrame(columns=[sel_target_bacteria.index],
                                               index=[0], data=999)
        elif sel_source_bacteria.shape[0] > 0:
            overlap_df = pd.DataFrame(columns=[0],
                                      index=sel_source_bacteria.index, data=999)
            distance_df = pd.DataFrame(columns=[0],
                                       index=sel_source_bacteria.index, data=999)


    return overlap_df, distance_df


def cal_division_cost(related_cost, angle_between_motion, difference_neighbors, max_neighbor_changes):
    division_cost = np.sqrt(
        np.power(related_cost, 2) +
        np.power(angle_between_motion, 2) +
        np.power(difference_neighbors / max_neighbor_changes, 2))

    return division_cost


def create_new_link_cost(target_bac_len_to_source, cost_df, source_bac_ndx, target_bac_ndx, maintenance_cost_this_link,
                         angle_between_motion, source_bac_bac_length_to_back_avg, difference_neighbors,
                         max_neighbor_changes, bac_len_to_bac_ratio_boundary):
    new_link_cost = 999

    if target_bac_len_to_source >= bac_len_to_bac_ratio_boundary['avg'] - \
            1.96 * bac_len_to_bac_ratio_boundary['std']:

        if target_bac_len_to_source < 1:
            new_link_cost = \
                np.sqrt(
                    np.power(cost_df.loc[target_bac_ndx][source_bac_ndx], 2) +
                    np.power(angle_between_motion, 2) +
                    np.power(source_bac_bac_length_to_back_avg - target_bac_len_to_source, 2) +
                    np.power(difference_neighbors / max_neighbor_changes, 2))

        else:
            new_link_cost = np.sqrt(
                np.power(cost_df.loc[target_bac_ndx][source_bac_ndx], 2) +
                np.power(angle_between_motion, 2) +
                np.power(difference_neighbors / max_neighbor_changes, 2))

    if new_link_cost >= maintenance_cost_this_link:
        new_link_cost = 999

    return new_link_cost


def create_division_link_instead_cont_life_cost(df, cost_df, source_bac_ndx, source_bac, target_bac_ndx, target_bac,
                                                source_bac_next_time_step, angle_between_motion, difference_neighbors,
                                                max_neighbor_changes, sum_daughter_len_to_mother_ratio_boundary,
                                                max_daughter_len_to_mother_ratio_boundary,
                                                min_life_history_of_bacteria):
    division_cost = 999

    daughters_bac_len_to_source = ((target_bac['AreaShape_MajorAxisLength'] +
                                    source_bac_next_time_step['AreaShape_MajorAxisLength'].values.tolist()[0]) /
                                   source_bac['AreaShape_MajorAxisLength'])

    max_daughters_bac_len_to_source = (max(target_bac['AreaShape_MajorAxisLength'],
                                           source_bac_next_time_step['AreaShape_MajorAxisLength'].values.tolist()[0]) /
                                       source_bac['AreaShape_MajorAxisLength'])

    if daughters_bac_len_to_source <= sum_daughter_len_to_mother_ratio_boundary['avg'] + \
            1.96 * sum_daughter_len_to_mother_ratio_boundary['std'] and max_daughters_bac_len_to_source < 1 and \
            max_daughters_bac_len_to_source <= \
            max_daughter_len_to_mother_ratio_boundary['avg'] + \
            1.96 * max_daughter_len_to_mother_ratio_boundary['std']:

        # second mode is possible
        nex_generation_source_bac = df.loc[df['parent_id'] == source_bac['id']]

        if nex_generation_source_bac.shape[0] > 0:
            if nex_generation_source_bac['ImageNumber'].values.tolist()[0] > \
                    target_bac['ImageNumber'] + min_life_history_of_bacteria:
                division_cost = cal_division_cost(cost_df.loc[target_bac_ndx][source_bac_ndx], angle_between_motion,
                                                  difference_neighbors, max_neighbor_changes)
        else:
            division_cost = cal_division_cost(cost_df.loc[target_bac_ndx][source_bac_ndx], angle_between_motion,
                                              difference_neighbors, max_neighbor_changes)

    else:
        division_cost = 999

    return division_cost


def link_to_source_with_already_has_one_link(df, cost_df, maintenance_cost_df,
                                             redundant_link_dict, source_bac_ndx, source_bac, target_bac_ndx,
                                             target_bac, source_bac_next_time_step, target_bac_len_to_source,
                                             angle_between_motion, source_bac_bac_length_to_back_avg,
                                             difference_neighbors, max_neighbor_changes, bac_len_to_bac_ratio_boundary,
                                             sum_daughter_len_to_mother_ratio_boundary,
                                             max_daughter_len_to_mother_ratio_boundary, min_life_history_of_bacteria):

    maintenance_cost_this_link = \
        maintenance_cost_df.loc[source_bac_ndx][source_bac_next_time_step.index.values.tolist()[0]]

    # 2 possible modes:

    # 1. Create a new link and delete the previous link
    new_link_cost = create_new_link_cost(target_bac_len_to_source, cost_df, source_bac_ndx, target_bac_ndx,
                                         maintenance_cost_this_link, angle_between_motion,
                                         source_bac_bac_length_to_back_avg, difference_neighbors, max_neighbor_changes,
                                         bac_len_to_bac_ratio_boundary)
    # 2. Maintain the previous link and create a new link (cell division)
    division_cost = \
        create_division_link_instead_cont_life_cost(df, cost_df, source_bac_ndx, source_bac, target_bac_ndx,
                                                    target_bac, source_bac_next_time_step, angle_between_motion,
                                                    difference_neighbors, max_neighbor_changes,
                                                    sum_daughter_len_to_mother_ratio_boundary,
                                                    max_daughter_len_to_mother_ratio_boundary,
                                                    min_life_history_of_bacteria)

    if division_cost == 999 and new_link_cost == 999:
        cost_df.at[target_bac_ndx, source_bac_ndx] = 999

    elif new_link_cost < division_cost:
        cost_df.at[target_bac_ndx, source_bac_ndx] = new_link_cost
        redundant_link_dict[target_bac_ndx] = [source_bac_ndx, [source_bac_next_time_step.index.values.tolist()[0]]]

    else:
        cost_df.at[target_bac_ndx, source_bac_ndx] = division_cost

    return cost_df, redundant_link_dict


def replacing_new_link_to_inappropriate_daughter(new_conditions, maintenance_cost_daughters_link,
                                                 new_daughter_cost, target_bac_ndx,
                                                 source_bac_ndx, redundant_link_dict):
    adding_new_daughter_cost = 999

    correct_daughter = list(new_conditions.keys())[0]
    other_daughter_ndx = [ndx for ndx in list(maintenance_cost_daughters_link.keys()) if \
                          ndx != correct_daughter][0]
    other_daughter_cost = maintenance_cost_daughters_link[other_daughter_ndx]

    if new_daughter_cost < other_daughter_cost:
        adding_new_daughter_cost = new_daughter_cost

        redundant_link_dict[target_bac_ndx] = [source_bac_ndx, [other_daughter_cost]]

    return adding_new_daughter_cost, redundant_link_dict


def replacing_new_link_to_one_of_daughters(df, masks_dict, new_conditions, target_bac_ndx,
                                           source_bac_ndx, source_bac, new_daughter_cost, redundant_link_dict,
                                           all_bacteria_in_source_time_step, all_bac_in_target_time_step_df):
    adding_new_daughter_cost = 999

    # we should find bad daughter
    candidate_daughters_ndx = list(new_conditions.keys())
    candidate_daughters_ndx.append(target_bac_ndx)
    daughters_df = df.loc[df.index.isin(candidate_daughters_ndx)]

    daughters_overlap_df, daughters_distance_df = \
        make_initial_distance_matrix(masks_dict, all_bacteria_in_source_time_step,
                                     source_bac.to_frame().transpose(), all_bac_in_target_time_step_df, daughters_df)

    daughters_normalized_distance_df = distance_normalization(df, daughters_distance_df)

    daughters_cost_df = \
        np.sqrt((1 - daughters_overlap_df) ** 2 + daughters_normalized_distance_df ** 2)

    for daughter_ndx in candidate_daughters_ndx:
        # Calculate orientation angle
        orientation_angle = calculate_orientation_angle(source_bac['bacteria_slope'],
                                                        df.loc[daughter_ndx]['bacteria_slope'])

        daughters_cost_df[daughter_ndx] = \
            np.sqrt(np.power(daughters_cost_df[daughter_ndx], 2) +
                    np.power(orientation_angle, 2))

    wrong_daughter_index = daughters_cost_df.max().idxmax()

    if wrong_daughter_index != target_bac_ndx:
        adding_new_daughter_cost = new_daughter_cost
        redundant_link_dict[target_bac_ndx] = [source_bac_ndx, [wrong_daughter_index]]

    return adding_new_daughter_cost, redundant_link_dict


def replacing_new_link_to_division(df, masks_dict, cost_df, maintenance_cost_df,
                                   source_bac_ndx, source_bac, target_bac_ndx, target_bac, target_bac_len_to_source,
                                   new_daughter_cost, candidate_source_bac_daughters,
                                   max_daughter_len_to_mother_ratio_boundary,
                                   sum_daughter_len_to_mother_ratio_boundary, redundant_link_dict,
                                   all_bacteria_in_source_time_step, all_bac_in_target_time_step_df,
                                   angle_between_motion, source_bac_bac_length_to_back_avg, difference_neighbors,
                                   max_neighbor_changes, bac_len_to_bac_ratio_boundary):
    adding_new_daughter_cost = 999
    maintenance_cost_this_division = 0

    # division occurs
    maintenance_cost_daughters_link = {}
    new_conditions = {}

    for daughter_ndx, daughter_bac in candidate_source_bac_daughters.iterrows():

        maintenance_cost_daughters_link[daughter_ndx] = maintenance_cost_df.loc[source_bac_ndx][daughter_ndx]
        maintenance_cost_this_division += maintenance_cost_df.loc[source_bac_ndx][daughter_ndx]

        new_sum_daughters_len_to_source = \
            (daughter_bac['AreaShape_MajorAxisLength'] + target_bac['AreaShape_MajorAxisLength']) / \
            source_bac['AreaShape_MajorAxisLength']

        new_max_daughters_len_to_source = \
            max(daughter_bac['AreaShape_MajorAxisLength'], target_bac['AreaShape_MajorAxisLength']) / \
            source_bac['AreaShape_MajorAxisLength']

        if new_max_daughters_len_to_source < 1 and new_max_daughters_len_to_source <= \
                max_daughter_len_to_mother_ratio_boundary['avg'] + \
                1.96 * max_daughter_len_to_mother_ratio_boundary['std'] and \
                new_sum_daughters_len_to_source <= sum_daughter_len_to_mother_ratio_boundary['avg'] + \
                1.96 * sum_daughter_len_to_mother_ratio_boundary['std']:
            new_conditions[daughter_ndx] = {'sum': new_sum_daughters_len_to_source,
                                            'max': new_max_daughters_len_to_source}

    if len(new_conditions.keys()) > 0:
        if len(new_conditions.keys()) == 1:
            adding_new_daughter_cost, redundant_link_dict = \
                replacing_new_link_to_inappropriate_daughter(new_conditions, maintenance_cost_daughters_link,
                                                             new_daughter_cost,
                                                             target_bac_ndx, source_bac_ndx, redundant_link_dict)
        else:
            adding_new_daughter_cost, redundant_link_dict = \
                replacing_new_link_to_one_of_daughters(df, masks_dict, new_conditions,
                                                       target_bac_ndx, source_bac_ndx, source_bac, new_daughter_cost,
                                                       redundant_link_dict, all_bacteria_in_source_time_step,
                                                       all_bac_in_target_time_step_df)

    new_link_cost = create_new_link_cost(target_bac_len_to_source, cost_df, source_bac_ndx, target_bac_ndx,
                                         maintenance_cost_this_division, angle_between_motion,
                                         source_bac_bac_length_to_back_avg, difference_neighbors, max_neighbor_changes,
                                         bac_len_to_bac_ratio_boundary)

    return new_link_cost, adding_new_daughter_cost, redundant_link_dict, maintenance_cost_daughters_link


def link_to_source_with_two_links(df, masks_dict, cost_df, maintenance_cost_df,
                                  source_bac_ndx,
                                  source_bac, target_bac_ndx, target_bac, redundant_link_dict,
                                  target_bac_len_to_source, angle_between_motion, difference_neighbors,
                                  max_neighbor_changes, candidate_source_bac_daughters,
                                  max_daughter_len_to_mother_ratio_boundary, sum_daughter_len_to_mother_ratio_boundary,
                                  all_bacteria_in_source_time_step,
                                  all_bac_in_target_time_step_df, source_bac_bac_length_to_back_avg,
                                  bac_len_to_bac_ratio_boundary):
    new_daughter_cost = np.sqrt(
        np.power(cost_df.loc[target_bac_ndx][source_bac_ndx], 2) + np.power(angle_between_motion, 2) +
        np.power(difference_neighbors / max_neighbor_changes, 2))

    new_link_instead_daughters_cost, adding_new_daughter_cost, redundant_link_dict, maintenance_cost_daughters_link = \
        replacing_new_link_to_division(df, masks_dict, cost_df, maintenance_cost_df,
                                       source_bac_ndx, source_bac, target_bac_ndx, target_bac, target_bac_len_to_source,
                                       new_daughter_cost, candidate_source_bac_daughters,
                                       max_daughter_len_to_mother_ratio_boundary,
                                       sum_daughter_len_to_mother_ratio_boundary, redundant_link_dict,
                                       all_bacteria_in_source_time_step,
                                       all_bac_in_target_time_step_df, angle_between_motion,
                                       source_bac_bac_length_to_back_avg, difference_neighbors, max_neighbor_changes,
                                       bac_len_to_bac_ratio_boundary)

    if adding_new_daughter_cost < new_link_instead_daughters_cost:
        cost_df.at[target_bac_ndx, source_bac_ndx] = adding_new_daughter_cost

    elif new_link_instead_daughters_cost < max(list(maintenance_cost_daughters_link.values())):
        cost_df.at[target_bac_ndx, source_bac_ndx] = new_link_instead_daughters_cost
        redundant_link_dict[target_bac_ndx] = [source_bac_ndx, list(maintenance_cost_daughters_link.keys())]
    else:
        cost_df.at[target_bac_ndx, source_bac_ndx] = 999

    return cost_df, redundant_link_dict


def adding_new_terms_to_cost_matrix(df, masks_dict, cost_df, maintenance_cost_df,
                                    source_bac_ndx, source_bac, target_bac_ndx, target_bac, neighbors_df,
                                    redundant_link_dict, max_neighbor_changes, bac_len_to_bac_ratio_boundary,
                                    sum_daughter_len_to_mother_ratio_boundary,
                                    max_daughter_len_to_mother_ratio_boundary, min_life_history_of_bacteria,
                                    all_bacteria_in_source_time_step, all_bac_in_target_time_step_df):
    target_bac_len_to_source = target_bac['AreaShape_MajorAxisLength'] / source_bac['AreaShape_MajorAxisLength']

    source_bac_life_history = df.loc[(df['id'] == source_bac['id']) & (df['ImageNumber'] < target_bac['ImageNumber'])]

    source_bac_bac_length_to_back = [v for v in source_bac_life_history['bac_length_to_back'] if v != '']

    if len(source_bac_bac_length_to_back) > 0:
        source_bac_bac_length_to_back_avg = np.average(source_bac_bac_length_to_back)
    else:
        source_bac_bac_length_to_back_avg = 1

    neighbors_dir_motion = calc_neighbors_dir_motion(df, source_bac, neighbors_df)

    direction_of_motion = \
        calculate_trajectory_direction(np.array([source_bac["AreaShape_Center_X"],
                                                 source_bac["AreaShape_Center_Y"]]),
                                       np.array([target_bac["AreaShape_Center_X"],
                                                 target_bac["AreaShape_Center_Y"]]))

    if str(neighbors_dir_motion[0]) != 'nan':
        angle_between_motion = calc_normalized_angle_between_motion(neighbors_dir_motion,
                                                                    direction_of_motion)
    else:
        angle_between_motion = 0

    # check neighbors
    difference_neighbors, common_neighbors = check_num_neighbors(df, neighbors_df, df.iloc[source_bac_ndx],
                                                                 df.iloc[target_bac_ndx], return_common_elements=True)

    # Did the bacteria survive or divide?
    source_bac_next_time_step = df.loc[(df['id'] == source_bac['id']) &
                                       (df['ImageNumber'] == source_bac['ImageNumber'] + 1)]

    candidate_source_bac_daughters = df.loc[(df['parent_id'] == source_bac['id']) &
                                            (df['ImageNumber'] == source_bac['ImageNumber'] + 1)]

    if difference_neighbors > common_neighbors:
        cost_df.at[target_bac_ndx, source_bac_ndx] = 999

    elif source_bac_next_time_step.shape[0] > 0:

        cost_df, redundant_link_dict = \
            link_to_source_with_already_has_one_link(df, cost_df, maintenance_cost_df,
                                                     redundant_link_dict, source_bac_ndx, source_bac, target_bac_ndx,
                                                     target_bac, source_bac_next_time_step, target_bac_len_to_source,
                                                     angle_between_motion, source_bac_bac_length_to_back_avg,
                                                     difference_neighbors, max_neighbor_changes,
                                                     bac_len_to_bac_ratio_boundary,
                                                     sum_daughter_len_to_mother_ratio_boundary,
                                                     max_daughter_len_to_mother_ratio_boundary,
                                                     min_life_history_of_bacteria)

    elif candidate_source_bac_daughters.shape[0] > 0:
        # my idea: at least one of a daughter-mother links can be wrong
        cost_df, redundant_link_dict = \
            link_to_source_with_two_links(df, masks_dict, cost_df, maintenance_cost_df,
                                          source_bac_ndx, source_bac, target_bac_ndx, target_bac, redundant_link_dict,
                                          target_bac_len_to_source, angle_between_motion, difference_neighbors,
                                          max_neighbor_changes, candidate_source_bac_daughters,
                                          max_daughter_len_to_mother_ratio_boundary,
                                          sum_daughter_len_to_mother_ratio_boundary,
                                          all_bacteria_in_source_time_step,
                                          all_bac_in_target_time_step_df, source_bac_bac_length_to_back_avg,
                                          bac_len_to_bac_ratio_boundary)

    else:
        # source bac is unexpected end
        cost_df.at[target_bac_ndx, source_bac_ndx] = 999

    return cost_df, redundant_link_dict
