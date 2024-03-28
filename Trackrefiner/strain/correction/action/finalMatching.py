import numpy as np
import pandas as pd
from Trackrefiner.strain.correction.action.bacteriaModification import bacteria_modification
from Trackrefiner.strain.correction.action.findOutlier import find_final_bac_change_length_ratio_outliers
from Trackrefiner.strain.correction.action.compareBacteria import final_division_detection_cost


def assign_new_link(df, neighbors_df, source_bac_index, source_bac, division_cost_df, all_bac_in_target_time_step,
                    parent_image_number_col, parent_object_number_col, label_col, center_coordinate_columns):

    source_bac_division_cost = division_cost_df.loc[division_cost_df['without parent index'] == source_bac_index]

    if source_bac_division_cost.shape[0] > 0:
        if source_bac_division_cost['Cost'].values.tolist()[0] != 999:
            # it means division occurs
            daughter_indx = source_bac_division_cost['Candida bacteria index in previous time step'].values.tolist()[0]
            new_daughter = df.iloc[daughter_indx]
            new_daughter_life_history = df.loc[df['id'] == new_daughter['id']]

            df = bacteria_modification(df, source_bac, new_daughter_life_history, all_bac_in_target_time_step,
                                       neighbors_df, parent_image_number_col, parent_object_number_col, label_col,
                                       center_coordinate_columns)

    return df


def final_matching(df, neighbors_df, min_life_history_of_bacteria, interval_time, sorted_npy_files_list,
                   parent_image_number_col, parent_object_number_col, label_col, center_coordinate_columns, logs_df):

    num_incorrect_same_links = None
    prev_bacteria_with_wrong_same_link = None
    n_iterate = 0

    # min life history of bacteria
    min_life_history_of_bacteria_time_step = np.round_(min_life_history_of_bacteria / interval_time)

    while num_incorrect_same_links != 0:

        # check incorrect same link
        bac_to_bac_len_ratio_list_outliers = find_final_bac_change_length_ratio_outliers(df)

        bacteria_with_wrong_same_link = \
            df.loc[(df["bac_length_to_back"].isin(bac_to_bac_len_ratio_list_outliers)) & (df['noise_bac'] == False)]

        if n_iterate > 0:
            if prev_bacteria_with_wrong_same_link.values.all() == bacteria_with_wrong_same_link.values.all():
                num_incorrect_same_links = 0
            else:
                num_incorrect_same_links = bacteria_with_wrong_same_link.shape[0]

        prev_bacteria_with_wrong_same_link = bacteria_with_wrong_same_link

        if bacteria_with_wrong_same_link.shape[0] > 0:
            for incorrect_same_link_bacteria_time_step in bacteria_with_wrong_same_link['ImageNumber'].unique():

                neighbors_indx_dict = {}
                neighbors_bacteria_obj_num_list = []

                target_incorrect_same_link = \
                    bacteria_with_wrong_same_link.loc[
                        bacteria_with_wrong_same_link['ImageNumber'] == incorrect_same_link_bacteria_time_step]

                for target_bac_ndx, target_bac in target_incorrect_same_link.iterrows():

                    target_bac_neighbors_obj_nums = \
                        neighbors_df.loc[
                            (neighbors_df['First Image Number'] == incorrect_same_link_bacteria_time_step) &
                            (neighbors_df['First Object Number'] ==
                             target_bac['ObjectNumber'])]['Second Object Number'].values.tolist()

                    target_bac_neighbors_info = df.loc[(df['ImageNumber'] == incorrect_same_link_bacteria_time_step) &
                                                       (df['ObjectNumber'].isin(target_bac_neighbors_obj_nums))]

                    without_link_neighbors = \
                        target_bac_neighbors_info.loc[target_bac_neighbors_info['transition'] == True]

                    neighbors_indx_dict[target_bac_ndx] = without_link_neighbors.index.values.tolist()

                    neighbors_bacteria_obj_num_list.extend(without_link_neighbors['ObjectNumber'].values.tolist())

                neighbors_bacteria_info = df.loc[(df['ImageNumber'] == incorrect_same_link_bacteria_time_step) &
                                                 (df['ObjectNumber'].isin(neighbors_bacteria_obj_num_list))]

                source_incorrect_same_link = df.loc[(df['ImageNumber'] == incorrect_same_link_bacteria_time_step - 1) &
                                                    (df['ObjectNumber'].isin(
                                                        target_incorrect_same_link[
                                                            parent_object_number_col].values.tolist()
                                                    ))]

                all_bac_in_source_time_step = df.loc[df['ImageNumber'] == incorrect_same_link_bacteria_time_step - 1]
                all_bac_in_target_time_step = df.loc[df['ImageNumber'] == incorrect_same_link_bacteria_time_step]

                # try to detect division
                division_cost_df = final_division_detection_cost(df, sorted_npy_files_list, source_incorrect_same_link,
                                                                 all_bac_in_source_time_step,
                                                                 min_life_history_of_bacteria_time_step,
                                                                 target_incorrect_same_link,
                                                                 all_bac_in_target_time_step, neighbors_bacteria_info,
                                                                 neighbors_indx_dict, center_coordinate_columns,
                                                                 parent_object_number_col)

                # try to modify the links
                for source_bac_index, source_bac in source_incorrect_same_link.iterrows():
                    df = assign_new_link(df, neighbors_df, source_bac_index, source_bac, division_cost_df,
                                         all_bac_in_target_time_step, parent_image_number_col, parent_object_number_col,
                                         label_col, center_coordinate_columns)
                    logs_df = pd.concat([logs_df, df.iloc[source_bac_index].to_frame().transpose()],
                                        ignore_index=True)

        n_iterate += 1
    return df, logs_df
