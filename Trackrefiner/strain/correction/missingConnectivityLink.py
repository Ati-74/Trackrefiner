import numpy as np
from Trackrefiner.strain.correction.action.findOutlier import find_bac_change_length_ratio_outliers
from Trackrefiner.strain.correction.action.compareBacteria import division_detection_cost, adding_new_link_cost, \
    calc_maintenance_cost, optimize_assignment
from Trackrefiner.strain.correction.action.bacteriaModification import bacteria_modification, remove_redundant_link
import pandas as pd


def assign_new_link(df, neighbors_df, neighbor_list_array, source_bac_index, source_bac, prob_val, new_target_bac_idx,
                    stat, all_bac_in_target_time_step, parent_image_number_col, parent_object_number_col,
                    label_col, center_coordinate_columns):
    if stat == 'div':

        # probability is 1 - probability
        if 1 - prob_val > 0.5:

            # it means division occurs
            new_daughter = df.loc[new_target_bac_idx]
            new_daughter_life_history = df.loc[(df['id'] == new_daughter['id'])
                                               & (df['ImageNumber'] >= new_daughter['ImageNumber'])]

            if new_daughter[parent_image_number_col] != 0:
                df = remove_redundant_link(df, new_daughter_life_history, neighbors_df, neighbor_list_array,
                                           parent_image_number_col, parent_object_number_col,
                                           center_coordinate_columns, label_col)

                # update info
                new_daughter_life_history = df.loc[new_daughter_life_history.index]

            df = bacteria_modification(df, df.loc[source_bac_index], new_daughter_life_history,
                                       all_bac_in_target_time_step, neighbors_df, neighbor_list_array,
                                       parent_image_number_col, parent_object_number_col,
                                       center_coordinate_columns, label_col)

    elif stat == 'same':

        # probability is 1 - probability
        if 1 - prob_val > 0.5:

            # same bac
            new_bac = df.loc[new_target_bac_idx]
            new_bac_life_history = df.loc[(df['id'] == new_bac['id']) &
                                          (df['ImageNumber'] >= new_bac['ImageNumber'])]

            # incorrect link
            target_incorrect_same_link_life_history = \
                df.loc[(df['id'] == df.at[source_bac_index, 'id']) &
                       (df['ImageNumber'] >= source_bac['ImageNumber'] + 1)]

            if target_incorrect_same_link_life_history.shape[0] > 0:
                df = remove_redundant_link(df, target_incorrect_same_link_life_history, neighbors_df,
                                           neighbor_list_array, parent_image_number_col,
                                           parent_object_number_col, center_coordinate_columns, label_col)

            # update info
            new_bac_life_history = df.loc[new_bac_life_history.index]

            if new_bac[parent_image_number_col] != 0:

                df = remove_redundant_link(df, new_bac_life_history, neighbors_df, neighbor_list_array,
                                           parent_image_number_col, parent_object_number_col,
                                           center_coordinate_columns, label_col)

                # update info
                new_bac_life_history = df.loc[new_bac_life_history.index]

            df = bacteria_modification(df, df.loc[source_bac_index], new_bac_life_history,
                                       all_bac_in_target_time_step,
                                       neighbors_df, neighbor_list_array, parent_image_number_col,
                                       parent_object_number_col, center_coordinate_columns, label_col)

    return df


def detect_missing_connectivity_link(df, parent_image_number_col, parent_object_number_col):

    num_incorrect_same_links = None

    while num_incorrect_same_links != 0 and len(df["LengthChangeRatio"].dropna().values) > 0:

        # check incorrect same link
        check_mcl_df = df.loc[df['target_mcl'] == False]
        max_bac_to_bac_len_ratio_list_outliers_min_boundary = find_bac_change_length_ratio_outliers(check_mcl_df)

        if str(max_bac_to_bac_len_ratio_list_outliers_min_boundary) != 'nan':

            bacteria_with_wrong_same_link = \
                df.loc[df["LengthChangeRatio"] <= max_bac_to_bac_len_ratio_list_outliers_min_boundary]

            df.loc[df['index'].isin(bacteria_with_wrong_same_link['index'].values), 'target_mcl'] = True

            bacteria_with_wrong_same_link_with_source = \
                bacteria_with_wrong_same_link.merge(df,
                                                    left_on=[parent_image_number_col, parent_object_number_col],
                                                    right_on=['ImageNumber', 'ObjectNumber'], suffixes=('_target',
                                                                                                        '_source'))

            df.loc[df['index'].isin(
                bacteria_with_wrong_same_link_with_source['index_source'].values), 'source_mcl'] = True

        else:
            num_incorrect_same_links = 0

    # breakpoint()
    return df


def missing_connectivity_link(df, neighbors_df, neighbor_list_array, min_life_history_of_bacteria, interval_time,
                              parent_image_number_col, parent_object_number_col, label_col, center_coordinate_columns,
                              comparing_divided_non_divided_model, non_divided_bac_model, divided_bac_model,
                              coordinate_array):

    num_incorrect_same_links = None
    prev_bacteria_with_wrong_same_link = None
    n_iterate = 0

    # min life history of bacteria
    min_life_history_of_bacteria_time_step = np.round(min_life_history_of_bacteria / interval_time)

    while num_incorrect_same_links != 0 and len(df["LengthChangeRatio"].dropna().values) > 0:

        # check incorrect same link
        max_bac_to_bac_len_ratio_list_outliers_min_boundary = find_bac_change_length_ratio_outliers(df)

        if str(max_bac_to_bac_len_ratio_list_outliers_min_boundary) != 'nan':

            bacteria_with_wrong_same_link = \
                df.loc[df["LengthChangeRatio"] <= max_bac_to_bac_len_ratio_list_outliers_min_boundary]

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

                    all_bac_in_target_time_step = df.loc[df['ImageNumber'] == incorrect_same_link_bacteria_time_step]
                    all_bac_in_source_time_step = \
                        df.loc[df['ImageNumber'] == (incorrect_same_link_bacteria_time_step - 1)]

                    # it's the target of incorrect same link (because the bac_bac_len of target to source has been
                    # written for target bac not source bac)
                    incorrect_same_link_target_bac = \
                        bacteria_with_wrong_same_link.loc[
                            bacteria_with_wrong_same_link['ImageNumber'] == incorrect_same_link_bacteria_time_step]

                    incorrect_same_link_target_bac_with_source = \
                        incorrect_same_link_target_bac.merge(all_bac_in_source_time_step,
                                                             left_on=[parent_image_number_col,
                                                                      parent_object_number_col],
                                                             right_on=['ImageNumber', 'ObjectNumber'], how='inner',
                                                             suffixes=('_target', '_source'))

                    neighbors_df_this_time_step = \
                        neighbors_df.loc[neighbors_df['First Image Number'] == incorrect_same_link_bacteria_time_step]

                    incorrect_same_link_in_this_time_step_with_neighbors = \
                        incorrect_same_link_target_bac_with_source.merge(neighbors_df_this_time_step,
                                                                         left_on=['ImageNumber_target',
                                                                                  'ObjectNumber_target'],
                                                                         right_on=['First Image Number',
                                                                                   'First Object Number'], how='inner')

                    # I didn't define suffix because target and source bac has it
                    incorrect_same_link_in_this_time_step_with_neighbors_features = \
                        incorrect_same_link_in_this_time_step_with_neighbors.merge(all_bac_in_target_time_step,
                                                                                   left_on=['Second Image Number',
                                                                                            'Second Object Number'],
                                                                                   right_on=['ImageNumber',
                                                                                             'ObjectNumber'])

                    incorrect_same_link_in_this_time_step_with_neighbors_features = \
                        incorrect_same_link_in_this_time_step_with_neighbors_features.rename(
                            {'First Image Number': 'First Image Number_target',
                             'First Object Number': 'First Object Number_target',
                             'Second Image Number': 'Second Image Number_neighbor',
                             'Second Object Number': 'Second Object Number_neighbor'}, axis=1)

                    columns_related_to_neighbors = \
                        [v for v in incorrect_same_link_in_this_time_step_with_neighbors_features.columns if
                         v.count('_target') == 0 and v.count('_source') == 0]

                    neighbors_bac_to_target = \
                        incorrect_same_link_in_this_time_step_with_neighbors_features.drop_duplicates(
                            subset=['ImageNumber', 'ObjectNumber'], keep='last')[columns_related_to_neighbors]

                    neighbors_bac_to_target_with_source = \
                        neighbors_bac_to_target.merge(all_bac_in_source_time_step,
                                                      left_on=[parent_image_number_col,
                                                               parent_object_number_col],
                                                      right_on=['ImageNumber', 'ObjectNumber'],
                                                      how='inner', suffixes=('_neighbor_target',
                                                                             '_neighbor_source'))

                    neighbors_source_bac = \
                        all_bac_in_source_time_step.loc[
                            neighbors_bac_to_target_with_source['index_neighbor_source'].unique()]

                    neighbors_target_bac = \
                        all_bac_in_target_time_step.loc[neighbors_bac_to_target_with_source['index_neighbor_target']]

                    maintenance_cost_df = calc_maintenance_cost(df, all_bac_in_source_time_step,
                                                                neighbors_source_bac,
                                                                all_bac_in_target_time_step, neighbors_df,
                                                                neighbors_target_bac, center_coordinate_columns,
                                                                parent_image_number_col, parent_object_number_col,
                                                                comparing_divided_non_divided_model, coordinate_array)
                    # try to detect division
                    division_cost_df = \
                        division_detection_cost(df, neighbors_df, neighbor_list_array,
                                                incorrect_same_link_in_this_time_step_with_neighbors_features,
                                                min_life_history_of_bacteria_time_step, center_coordinate_columns,
                                                parent_image_number_col, parent_object_number_col,
                                                divided_bac_model, comparing_divided_non_divided_model,
                                                maintenance_cost_df, coordinate_array)

                    new_link_cost_df = \
                        adding_new_link_cost(df, neighbors_df, neighbor_list_array,
                                             incorrect_same_link_in_this_time_step_with_neighbors_features,
                                             center_coordinate_columns, parent_image_number_col,
                                             parent_object_number_col, non_divided_bac_model,
                                             comparing_divided_non_divided_model, maintenance_cost_df, coordinate_array)

                    # if incorrect_same_link_bacteria_time_step == 68:
                    #    incorrect_same_link_in_this_time_step_with_neighbors_features.to_csv(
                    #        'incorrect_same_link_in_this_time_step_with_neighbors_features.' + str(n_iterate) + '.csv')
                    #    division_cost_df.to_csv('division_cost_df.' + str(n_iterate) + '.csv')
                    #    new_link_cost_df.to_csv('new_link_cost_df.' + str(n_iterate) + '.csv')
                    #    maintenance_cost_df.to_csv('maintenance_cost_df.' + str(n_iterate) + '.csv')

                    if division_cost_df.shape[0] > 0 and new_link_cost_df.shape[0] > 0:

                        # Merge the dataframes using outer join to combine all columns and indices
                        df_combined = pd.concat([division_cost_df, new_link_cost_df], axis=1, sort=False).fillna(1)

                        # Create a third DataFrame to hold the max values for common rows and columns
                        total_cost_df = pd.DataFrame(index=df_combined.index, columns=df_combined.columns.unique())

                        # Iterate through the columns and fill the third DataFrame with max values
                        for column in df_combined.columns.unique():
                            total_cost_df[column] = df_combined[[column]].min(axis=1)

                        continue_flag = True

                    elif division_cost_df.shape[0] > 0:

                        total_cost_df = division_cost_df

                        continue_flag = True

                    elif new_link_cost_df.shape[0] > 0:

                        total_cost_df = new_link_cost_df

                        continue_flag = True
                    else:
                        continue_flag = False

                    if continue_flag:

                        total_cost_df = total_cost_df.fillna(1)
                        optimized_df = optimize_assignment(total_cost_df)

                        # try to modify the links
                        for row_index, row in optimized_df.iterrows():
                            source_bac_index = row['without parent index']
                            source_bac = df.loc[source_bac_index]

                            new_target_bac_idx = int(row['Candida bacteria index in previous time step'])

                            cost_val = row['Cost']

                            find_stat = False

                            if source_bac_index in division_cost_df.index.values.tolist() and \
                                    new_target_bac_idx in division_cost_df.columns.values.tolist():

                                if cost_val == division_cost_df.at[source_bac_index, new_target_bac_idx]:
                                    stat = 'div'
                                    find_stat = True

                            if source_bac_index in new_link_cost_df.index.values.tolist() and \
                                    new_target_bac_idx in new_link_cost_df.columns.values.tolist():

                                if cost_val == new_link_cost_df.at[source_bac_index, new_target_bac_idx]:
                                    stat = 'same'
                                    find_stat = True

                            if not find_stat:
                                if cost_val == 1:
                                    stat = 'same'

                            df = assign_new_link(df, neighbors_df, neighbor_list_array, source_bac_index, source_bac,
                                                 cost_val, new_target_bac_idx, stat,
                                                 all_bac_in_target_time_step,
                                                 parent_image_number_col, parent_object_number_col, label_col,
                                                 center_coordinate_columns)

            n_iterate += 1

        else:
            num_incorrect_same_links = 0

    # breakpoint()
    return df
