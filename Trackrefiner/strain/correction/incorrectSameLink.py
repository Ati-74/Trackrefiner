import numpy as np
import pandas as pd
from Trackrefiner.strain.correction.action.findOutlier import find_bac_change_length_ratio_outliers
from Trackrefiner.strain.correction.action.compareBacteria import (division_detection_cost,
                                                                   adding_new_link_cost, calc_maintenance_cost)
from Trackrefiner.strain.correction.action.bacteriaModification import bacteria_modification, \
    new_transition_bacteria


def assign_new_link(df, neighbors_df, source_bac_index, source_bac, division_cost_df, new_link_cost_df,
                    all_bac_in_target_time_step, maintenance_cost, parent_image_number_col, parent_object_number_col,
                    label_col, center_coordinate_columns):

    source_bac_division_cost = division_cost_df.loc[division_cost_df['without parent index'] == source_bac_index]
    source_bac_new_link_cost = new_link_cost_df.loc[new_link_cost_df['without parent index'] == source_bac_index]

    if source_bac_division_cost.shape[0] > 0 and source_bac_new_link_cost.shape[0] > 0:
        if (source_bac_division_cost['Cost'].values.tolist()[0] != 999 or
                source_bac_new_link_cost['Cost'].values.tolist()[0] != 999):

            if source_bac_division_cost['Cost'].values.tolist()[0] < source_bac_new_link_cost['Cost'].values.tolist()[
                0]:
                # it means division occurs
                daughter_indx = \
                source_bac_division_cost['Candida bacteria index in previous time step'].values.tolist()[0]
                new_daughter = df.iloc[daughter_indx]
                new_daughter_life_history = df.loc[df['id'] == new_daughter['id']]

                # check the maintenance cost of prev link
                new_daughter_prev_link_source = df.loc[(df['ImageNumber'] == new_daughter[parent_image_number_col]) &
                                                       (df['ObjectNumber'] == new_daughter[parent_object_number_col])]

                if new_daughter_prev_link_source.shape[0] > 0:
                    maintenance_cost_this_link = \
                        maintenance_cost.loc[new_daughter_prev_link_source.index.values.tolist()[0]][daughter_indx]

                    if maintenance_cost_this_link > source_bac_division_cost['Cost'].values.tolist()[0]:
                        df = bacteria_modification(df, source_bac, new_daughter_life_history, all_bac_in_target_time_step,
                                                   neighbors_df, parent_image_number_col, parent_object_number_col,
                                                   label_col, center_coordinate_columns)
                else:
                    df = bacteria_modification(df, source_bac, new_daughter_life_history, all_bac_in_target_time_step,
                                               neighbors_df, parent_image_number_col, parent_object_number_col,
                                               label_col, center_coordinate_columns)

            else:
                # incorrect link
                incorrect_link_life_history = df.loc[(df['id'] == source_bac['id']) &
                                                     (df['ImageNumber'] >= source_bac['ImageNumber'] + 1)]
                new_bac_ndx = source_bac_new_link_cost['Candida bacteria index in previous time step'].values.tolist()[
                    0]
                # same bac
                new_bac = df.iloc[new_bac_ndx]
                new_bac_life_history = df.loc[df['id'] == new_bac['id']]

                # check the maintenance cost of prev link
                new_bac_prev_link_source = df.loc[(df['ImageNumber'] == new_bac[parent_image_number_col]) &
                                                  (df['ObjectNumber'] == new_bac[parent_object_number_col])]

                maintenance_cost_this_link = maintenance_cost.loc[new_bac_prev_link_source.index.values.tolist()[0]][
                    new_bac_ndx]

                if maintenance_cost_this_link > source_bac_new_link_cost['Cost'].values.tolist()[0]:
                    df = new_transition_bacteria(df, incorrect_link_life_history, parent_image_number_col,
                                                 parent_object_number_col, label_col)

                    df = bacteria_modification(df, source_bac, new_bac_life_history, all_bac_in_target_time_step,
                                               neighbors_df, parent_image_number_col, parent_object_number_col,
                                               label_col, center_coordinate_columns)

    elif source_bac_division_cost.shape[0] > 0:
        if source_bac_division_cost['Cost'].values.tolist()[0] != 999:
            # it means division occurs
            daughter_indx = source_bac_division_cost['Candida bacteria index in previous time step'].values.tolist()[0]
            new_daughter = df.iloc[daughter_indx]
            new_daughter_life_history = df.loc[df['id'] == new_daughter['id']]

            # check the maintenance cost of prev link
            new_daughter_prev_link_source = df.loc[(df['ImageNumber'] == new_daughter[parent_image_number_col]) &
                                                   (df['ObjectNumber'] == new_daughter[parent_object_number_col])]
            maintenance_cost_this_link = \
                maintenance_cost.loc[new_daughter_prev_link_source.index.values.tolist()[0]][daughter_indx]

            if maintenance_cost_this_link > source_bac_division_cost['Cost'].values.tolist()[0]:
                df = bacteria_modification(df, source_bac, new_daughter_life_history, all_bac_in_target_time_step,
                                           neighbors_df, parent_image_number_col, parent_object_number_col, label_col,
                                           center_coordinate_columns)

    elif source_bac_new_link_cost.shape[0] > 0:
        if source_bac_new_link_cost['Cost'].values.tolist()[0] != 999:
            # incorrect link
            incorrect_link_life_history = df.loc[(df['id'] == source_bac['id']) &
                                                 (df['ImageNumber'] >= source_bac['ImageNumber'] + 1)]

            # same bac
            new_bac_ndx = source_bac_new_link_cost['Candida bacteria index in previous time step'].values.tolist()[0]
            new_bac = df.iloc[new_bac_ndx]
            new_bac_life_history = df.loc[df['id'] == new_bac['id']]

            # check the maintenance cost of prev link
            new_bac_prev_link_source = df.loc[(df['ImageNumber'] == new_bac[parent_image_number_col]) &
                                              (df['ObjectNumber'] == new_bac[parent_object_number_col])]

            maintenance_cost_this_link = maintenance_cost.loc[new_bac_prev_link_source.index.values.tolist()[0]][
                new_bac_ndx]

            if maintenance_cost_this_link > source_bac_new_link_cost['Cost'].values.tolist()[0]:
                df = new_transition_bacteria(df, incorrect_link_life_history, parent_image_number_col,
                                             parent_object_number_col, label_col)
                df = bacteria_modification(df, source_bac, new_bac_life_history, all_bac_in_target_time_step,
                                           neighbors_df, parent_image_number_col, parent_object_number_col, label_col,
                                           center_coordinate_columns)

    return df


def incorrect_same_link(df, neighbors_df, sorted_npy_files_list, min_life_history_of_bacteria, interval_time,
                        parent_image_number_col, parent_object_number_col, label_col, center_coordinate_columns,
                        logs_df):

    num_incorrect_same_links = None
    prev_bacteria_with_wrong_same_link = None
    n_iterate = 0

    # min life history of bacteria
    min_life_history_of_bacteria_time_step = np.round_(min_life_history_of_bacteria / interval_time)

    while num_incorrect_same_links != 0:

        # check incorrect same link
        bac_to_bac_len_ratio_list_outliers = find_bac_change_length_ratio_outliers(df)

        bacteria_with_wrong_same_link = \
            df.loc[(df["bac_length_to_back"].isin(bac_to_bac_len_ratio_list_outliers))]

        if n_iterate > 0:
            if prev_bacteria_with_wrong_same_link.values.all() == bacteria_with_wrong_same_link.values.all():
                num_incorrect_same_links = 0
            else:
                num_incorrect_same_links = bacteria_with_wrong_same_link.shape[0]

        prev_bacteria_with_wrong_same_link = bacteria_with_wrong_same_link

        if bacteria_with_wrong_same_link.shape[0] > 0:

            # possible states:
            # 1. division occurs (at least one link of other neighbours is wrong)
            # 2. this link is incorrect
            for incorrect_same_link_bacteria_time_step in bacteria_with_wrong_same_link['ImageNumber'].unique():
                neighbors_indx_dict = {}
                neighbors_bacteria_obj_num_list = []

                target_incorrect_same_link = \
                    bacteria_with_wrong_same_link.loc[
                        bacteria_with_wrong_same_link['ImageNumber'] == incorrect_same_link_bacteria_time_step]

                for target_bac_ndx, target_bac in target_incorrect_same_link.iterrows():
                    neighbors_bacteria_obj_nums = \
                        neighbors_df.loc[
                            (neighbors_df['First Image Number'] == incorrect_same_link_bacteria_time_step) &
                            (neighbors_df['First Object Number'] ==
                             target_bac['ObjectNumber'])]['Second Object Number'].values.tolist()

                    target_bac_neighbors_info = df.loc[(df['ImageNumber'] == incorrect_same_link_bacteria_time_step) &
                                                       (df['ObjectNumber'].isin(neighbors_bacteria_obj_nums))]

                    neighbors_indx_dict[target_bac_ndx] = target_bac_neighbors_info.index.values.tolist()

                    neighbors_bacteria_obj_num_list.extend(neighbors_bacteria_obj_nums)

                neighbors_bacteria_info = df.loc[(df['ImageNumber'] == incorrect_same_link_bacteria_time_step) &
                                                 (df['ObjectNumber'].isin(neighbors_bacteria_obj_num_list))]

                neighbors_parents_obj_num = neighbors_bacteria_info[parent_object_number_col].values.tolist()

                parent_of_neighbors_info = df.loc[(df['ImageNumber'] == incorrect_same_link_bacteria_time_step - 1) &
                                                  (df['ObjectNumber'].isin(neighbors_parents_obj_num))]

                # ATTENTION: it's incorrect for gap
                source_incorrect_same_link = df.loc[(df['ImageNumber'] == incorrect_same_link_bacteria_time_step - 1) &
                                                    (df['ObjectNumber'].isin(
                                                        target_incorrect_same_link[
                                                            parent_object_number_col].values.tolist()
                                                    ))]

                all_bac_in_source_time_step = df.loc[df['ImageNumber'] == incorrect_same_link_bacteria_time_step - 1]
                all_bac_in_target_time_step = df.loc[df['ImageNumber'] == incorrect_same_link_bacteria_time_step]

                # try to detect division
                division_cost_df = division_detection_cost(df, sorted_npy_files_list, source_incorrect_same_link,
                                                           all_bac_in_source_time_step,
                                                           min_life_history_of_bacteria_time_step,
                                                           target_incorrect_same_link,
                                                           all_bac_in_target_time_step, neighbors_bacteria_info,
                                                           neighbors_indx_dict, center_coordinate_columns,
                                                           parent_object_number_col)

                new_link_cost_df = adding_new_link_cost(df, neighbors_df, sorted_npy_files_list, source_incorrect_same_link,
                                                        all_bac_in_source_time_step, target_incorrect_same_link,
                                                        all_bac_in_target_time_step, neighbors_bacteria_info,
                                                        neighbors_indx_dict, center_coordinate_columns,
                                                        parent_image_number_col, parent_object_number_col)

                maintenance_cost_df = calc_maintenance_cost(df, sorted_npy_files_list, all_bac_in_source_time_step,
                                                            parent_of_neighbors_info,
                                                            all_bac_in_target_time_step, neighbors_df,
                                                            neighbors_bacteria_info, center_coordinate_columns)

                # try to modify the links
                for source_bac_index, source_bac in source_incorrect_same_link.iterrows():
                    df = assign_new_link(df, neighbors_df, source_bac_index, source_bac, division_cost_df,
                                         new_link_cost_df, all_bac_in_target_time_step, maintenance_cost_df,
                                         parent_image_number_col, parent_object_number_col, label_col,
                                         center_coordinate_columns)

                    logs_df = pd.concat([logs_df, df.iloc[source_bac_index].to_frame().transpose()],
                                        ignore_index=True)

        n_iterate += 1
    return df, logs_df
