from Trackrefiner.strain.correction.action.compareBacteria import adding_new_link_to_unexpected_end
from Trackrefiner.strain.correction.action.compareBacteria import optimize_assignment
from Trackrefiner.strain.correction.action.bacteriaModification import bacteria_modification, remove_redundant_link
import pandas as pd


def adding_new_link(df, neighbors_df, neighbor_list_array, stat, unexpected_end_bac_idx, unexpected_end_bac,
                    target_bac_idx, target_bac,
                    another_daughter_bac_idx, another_daughter_bac, parent_image_number_col,
                    parent_object_number_col, center_coordinate_columns, label_col,
                    all_bac_in_next_time_step_to_unexpected_end_bac, prob_val, another_daughter_bac_prob_val):

    if stat == 'div':

        if 1 - prob_val > 0.5 and 1 - another_daughter_bac_prob_val > 0.5:
            source_bac = df.loc[unexpected_end_bac_idx]

            # remove incorrect link
            if target_bac[parent_image_number_col] != 0:

                incorrect_bac_life_history_target_bac = \
                    df.loc[(df['id'] == df.at[target_bac_idx, 'id']) &
                           (df['ImageNumber'] >= source_bac['ImageNumber'] + 1)]

                df = remove_redundant_link(df, incorrect_bac_life_history_target_bac, neighbors_df, neighbor_list_array,
                                           parent_image_number_col,
                                           parent_object_number_col, center_coordinate_columns, label_col)

            if another_daughter_bac[parent_image_number_col] != 0:
                incorrect_bac_life_history_another_target_bac = \
                    df.loc[(df['id'] == df.at[another_daughter_bac_idx, 'id']) &
                           (df['ImageNumber'] >= source_bac['ImageNumber'] + 1)]

                df = remove_redundant_link(df, incorrect_bac_life_history_another_target_bac, neighbors_df,
                                           neighbor_list_array,
                                           parent_image_number_col,
                                           parent_object_number_col, center_coordinate_columns, label_col)

            # now we should add link
            # update info
            source_bac = df.loc[unexpected_end_bac_idx]
            target_bac_life_history = df.loc[df['id'] == df.at[target_bac_idx, 'id']]

            df = bacteria_modification(df, source_bac, target_bac_life_history,
                                       all_bac_in_next_time_step_to_unexpected_end_bac,
                                       neighbors_df, neighbor_list_array, parent_image_number_col,
                                       parent_object_number_col,
                                       center_coordinate_columns, label_col)

            source_bac = df.loc[unexpected_end_bac_idx]
            another_target_bac_life_history = df.loc[df['id'] == df.at[another_daughter_bac_idx, 'id']]

            df = bacteria_modification(df, source_bac, another_target_bac_life_history,
                                       all_bac_in_next_time_step_to_unexpected_end_bac,
                                       neighbors_df, neighbor_list_array, parent_image_number_col,
                                       parent_object_number_col,
                                       center_coordinate_columns, label_col)

    elif stat == 'same':

        if 1 - prob_val > 0.5:
            # there is two scenario: first: source bac with only one bac in next time step: so we compare
            # the probability of that with new bac
            # the second scenario is source bac has two daughters. I think it's good idea two compare
            # max daughter probability with new bac probability

            # be careful: maintenance cost should compare with 1 - cost value

            source_bac = df.loc[unexpected_end_bac_idx]

            # remove incorrect link
            if target_bac[parent_image_number_col] != 0:
                incorrect_bac_life_history_target_bac = \
                    df.loc[(df['id'] == df.at[target_bac_idx, 'id']) &
                           (df['ImageNumber'] >= source_bac['ImageNumber'] + 1)]

                df = remove_redundant_link(df, incorrect_bac_life_history_target_bac, neighbors_df,
                                           neighbor_list_array,
                                           parent_image_number_col,
                                           parent_object_number_col, center_coordinate_columns, label_col)

            # update info
            source_bac = df.loc[unexpected_end_bac_idx]
            target_bac_life_history = df.loc[df['id'] == df.at[target_bac_idx, 'id']]

            df = bacteria_modification(df, source_bac, target_bac_life_history,
                                       all_bac_in_next_time_step_to_unexpected_end_bac,
                                       neighbors_df, neighbor_list_array, parent_image_number_col,
                                       parent_object_number_col,
                                       center_coordinate_columns, label_col)

    return df


def unexpected_end_bacteria(df, neighbors_df, neighbor_list_array, min_life_history_of_bacteria, interval_time,
                            parent_image_number_col, parent_object_number_col, label_col, center_coordinate_columns,
                            comparing_divided_non_divided_model, non_divided_bac_model,
                            divided_bac_model, color_array, coordinate_array):
    num_incorrect_same_links = None
    prev_bacteria_with_wrong_same_link = None
    n_iterate = 0

    # min life history of bacteria
    # min_life_history_of_bacteria_time_step = np.round_(min_life_history_of_bacteria / interval_time)

    while num_incorrect_same_links != 0:

        df_unexpected_end_bacteria = df.loc[df['unexpected_end']]

        if n_iterate > 0:
            if prev_bacteria_with_wrong_same_link.values.all() == df_unexpected_end_bacteria.values.all():
                num_incorrect_same_links = 0
            else:
                num_incorrect_same_links = df_unexpected_end_bacteria.shape[0]

        prev_bacteria_with_wrong_same_link = df_unexpected_end_bacteria

        if df_unexpected_end_bacteria.shape[0] > 0:
            for unexpected_end_bacteria_time_step in df_unexpected_end_bacteria['ImageNumber'].unique():

                unexpected_end_bac_in_current_time_step_df = \
                    df_unexpected_end_bacteria.loc[
                        df_unexpected_end_bacteria['ImageNumber'] == unexpected_end_bacteria_time_step]

                unexpected_end_bac_in_current_time_step_df = \
                    df.loc[unexpected_end_bac_in_current_time_step_df['index'].values]

                all_bac_in_unexpected_end_bac_time_step = \
                    df.loc[df['ImageNumber'] == unexpected_end_bacteria_time_step]
                all_bac_in_next_time_step_to_unexpected_end_bac = \
                    df.loc[df['ImageNumber'] == unexpected_end_bacteria_time_step + 1]

                (raw_same_link_cost_df, new_link_cost_df, division_cost_df, final_division_cost_df_after_merge_prob,
                 final_division_cost_df, maintenance_cost_df) = \
                    adding_new_link_to_unexpected_end(df, neighbors_df, neighbor_list_array,
                                                      unexpected_end_bac_in_current_time_step_df,
                                                      all_bac_in_unexpected_end_bac_time_step,
                                                      all_bac_in_next_time_step_to_unexpected_end_bac,
                                                      center_coordinate_columns, parent_image_number_col,
                                                      parent_object_number_col, min_life_history_of_bacteria,
                                                      comparing_divided_non_divided_model, non_divided_bac_model,
                                                      divided_bac_model, color_array, coordinate_array)

                if final_division_cost_df_after_merge_prob.shape[0] > 0 and new_link_cost_df.shape[0] > 0:

                    # Merge the dataframes using outer join to combine all columns and indices
                    total_cost_df = pd.concat([final_division_cost_df_after_merge_prob, new_link_cost_df],
                                              axis=1, sort=False).fillna(1)

                    continue_flag = True

                elif final_division_cost_df_after_merge_prob.shape[0] > 0:

                    total_cost_df = final_division_cost_df_after_merge_prob

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

                        unexpected_end_bac_idx = row['without parent index']
                        unexpected_end_bac = df.loc[unexpected_end_bac_idx]

                        target_bac_idx = int(row['Candida bacteria index in previous time step'])
                        target_bac = df.loc[target_bac_idx]

                        cost_val = row['Cost']
                        find_stat = False

                        if target_bac_idx in final_division_cost_df.columns.values.tolist() and \
                                unexpected_end_bac_idx in final_division_cost_df.index.values.tolist():

                            if cost_val == final_division_cost_df.at[unexpected_end_bac_idx, target_bac_idx]:
                                stat = 'div'

                                another_daughter_bac_idx = [idx for idx in
                                                            final_division_cost_df.loc[
                                                                unexpected_end_bac_idx].dropna().index
                                                            if idx != target_bac_idx][0]

                                another_daughter_bac = df.loc[another_daughter_bac_idx]

                                another_daughter_bac_prob_val = \
                                    final_division_cost_df.at[unexpected_end_bac_idx, another_daughter_bac_idx]
                                find_stat = True

                        if target_bac_idx in new_link_cost_df.columns.values.tolist() and \
                                unexpected_end_bac_idx in new_link_cost_df.index.values.tolist():

                            if cost_val == new_link_cost_df.at[unexpected_end_bac_idx, target_bac_idx]:
                                stat = 'same'
                                another_daughter_bac_idx = None
                                another_daughter_bac = None
                                another_daughter_bac_prob_val = None
                                find_stat = True

                        if not find_stat:
                            if cost_val == 1:
                                stat = 'same'
                                another_daughter_bac_idx = None
                                another_daughter_bac = None
                                another_daughter_bac_prob_val = None

                        df = adding_new_link(df, neighbors_df, neighbor_list_array, stat, unexpected_end_bac_idx,
                                             unexpected_end_bac,
                                             target_bac_idx, target_bac, another_daughter_bac_idx,
                                             another_daughter_bac, parent_image_number_col,
                                             parent_object_number_col, center_coordinate_columns, label_col,
                                             all_bac_in_next_time_step_to_unexpected_end_bac, cost_val,
                                             another_daughter_bac_prob_val)

        n_iterate += 1

    return df
