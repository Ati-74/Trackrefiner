import numpy as np
from Trackrefiner.strain.correction.action.bacteriaModification import bacteria_modification, remove_redundant_link
from Trackrefiner.strain.correction.action.compareBacteria import optimization_unexpected_beginning_cost, \
    optimize_assignment
import pandas as pd


def assign_new_link(df, neighbors_df, neighbor_list_array, unexpected_beginning_bac_idx, unexpected_beginning_bac,
                    prob_val, source_bac_idx,
                    stat, all_bac_in_unexpected_beginning_time_step, maintenance_cost, parent_image_number_col,
                    parent_object_number_col, label_col, center_coordinate_columns, redundant_link_division_df):
    if stat == 'div':

        # be careful: maintenance cost should compare with 1 - cost value

        # probability is 1 - probability
        if 1 - prob_val > 0.5:

            # check if probability is higher than min prob of one of daughters

            if len(maintenance_cost.loc[source_bac_idx].dropna()) == 1:

                source_bac = df.loc[source_bac_idx]
                unexpected_beginning_bac_life_history = df.loc[df['id'] == unexpected_beginning_bac['id']]

                df = bacteria_modification(df, source_bac, unexpected_beginning_bac_life_history,
                                           all_bac_in_unexpected_beginning_time_step,
                                           neighbors_df, neighbor_list_array, parent_image_number_col,
                                           parent_object_number_col,
                                           center_coordinate_columns, label_col)
            else:

                # it means source bacterium has two daughters

                redundant_daughter_idx = redundant_link_division_df.at[unexpected_beginning_bac_idx, source_bac_idx]

                # incorrect link (incorrect previous daughter)
                incorrect_daughter_life_history = df.loc[df['id'] == df.at[redundant_daughter_idx, 'id']]

                if incorrect_daughter_life_history.shape[0] > 0:
                    df = remove_redundant_link(df, incorrect_daughter_life_history, neighbors_df, neighbor_list_array,
                                               parent_image_number_col,
                                               parent_object_number_col, center_coordinate_columns, label_col)

                # update info
                source_bac = df.loc[source_bac_idx]
                unexpected_beginning_bac_life_history = df.loc[df['id'] == unexpected_beginning_bac['id']]

                df = bacteria_modification(df, source_bac, unexpected_beginning_bac_life_history,
                                           all_bac_in_unexpected_beginning_time_step,
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

            source_bac = df.loc[source_bac_idx]

            # remove incorrect link
            for prev_target_bac_to_source_idx in maintenance_cost.loc[source_bac_idx].dropna().index:
                incorrect_bac_life_history = \
                    df.loc[(df['id'] == df.at[prev_target_bac_to_source_idx, 'id']) &
                           (df['ImageNumber'] >= source_bac['ImageNumber'] + 1)]

                if incorrect_bac_life_history.shape[0] > 0:
                    df = remove_redundant_link(df, incorrect_bac_life_history, neighbors_df, neighbor_list_array,
                                               parent_image_number_col,
                                               parent_object_number_col, center_coordinate_columns, label_col)

            # update info
            source_bac = df.loc[source_bac_idx]
            unexpected_beginning_bac_life_history = df.loc[df['id'] == unexpected_beginning_bac['id']]

            df = bacteria_modification(df, source_bac, unexpected_beginning_bac_life_history,
                                       all_bac_in_unexpected_beginning_time_step,
                                       neighbors_df, neighbor_list_array, parent_image_number_col,
                                       parent_object_number_col,
                                       center_coordinate_columns, label_col)
    return df


def correction_unexpected_beginning(df, neighbors_df, neighbor_list_array, check_cell_type, interval_time,
                                    min_life_history_of_bacteria, parent_image_number_col, parent_object_number_col,
                                    label_col, center_coordinate_columns, comparing_divided_non_divided_model,
                                    non_divided_bac_model, divided_bac_model, color_array, coordinate_array):
    """
        goal: For bacteria without parent, assign labels, ParentImageNumber, and ParentObjectNumber
        @param df    dataframe   bacteria dataframe
        @param number_of_gap int I define a gap number to find parent in other previous time steps
        in last time step of its life history before unexpected_beginning bacterium to length of candidate parent
        bacterium in investigated time step
        output: df   dataframe   modified dataframe (without any unexpected_beginnings)
    """

    # min life history of bacteria
    min_life_history_of_bacteria_time_step = np.round(min_life_history_of_bacteria / interval_time)

    num_unexpected_beginning_bac = None
    prev_unexpected_beginning_bac = None
    n_iterate = 0

    while num_unexpected_beginning_bac != 0:

        unexpected_beginning_bacteria = df.loc[df["unexpected_beginning"]]

        if n_iterate > 0:
            if prev_unexpected_beginning_bac.values.all() == unexpected_beginning_bacteria.values.all():
                num_unexpected_beginning_bac = 0
            else:
                num_unexpected_beginning_bac = unexpected_beginning_bacteria.shape[0]

        prev_unexpected_beginning_bac = unexpected_beginning_bacteria

        for i, unexpected_beginning_bac_time_step in enumerate(unexpected_beginning_bacteria['ImageNumber'].unique()):

            # all bacteria in selected unexpected_beginning bacteria time step
            all_bac_in_unexpected_beginning_bac_time_step = \
                df.loc[df['ImageNumber'] == unexpected_beginning_bac_time_step]

            # filter unexpected beginning bacteria features value
            sel_unexpected_beginning_bac = \
                unexpected_beginning_bacteria.loc[unexpected_beginning_bacteria['ImageNumber'] ==
                                                  unexpected_beginning_bac_time_step]

            # filter consider time step bacteria information
            all_bac_in_source_time_step = df.loc[df['ImageNumber'] == (unexpected_beginning_bac_time_step - 1)]

            # optimized cost dataframe
            # (rows: next time step sudden bacteria, columns: consider time step bacteria)
            new_link_cost_df, division_cost_df, redundant_link_division_df, maintenance_cost_df = \
                optimization_unexpected_beginning_cost(df, all_bac_in_unexpected_beginning_bac_time_step,
                                                       sel_unexpected_beginning_bac, all_bac_in_source_time_step,
                                                       check_cell_type, neighbors_df, neighbor_list_array,
                                                       min_life_history_of_bacteria_time_step,
                                                       parent_image_number_col, parent_object_number_col,
                                                       center_coordinate_columns,
                                                       comparing_divided_non_divided_model,
                                                       non_divided_bac_model, divided_bac_model, color_array,
                                                       coordinate_array)

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
                    source_bac_idx = row['without parent index']

                    unexpected_beginning_bac_idx = int(row['Candida bacteria index in previous time step'])
                    unexpected_beginning_bac = df.loc[unexpected_beginning_bac_idx]

                    cost_val = row['Cost']

                    find_stat = False

                    if unexpected_beginning_bac_idx in division_cost_df.columns.values.tolist() and \
                            source_bac_idx in division_cost_df.index.values.tolist():

                        if cost_val == division_cost_df.at[source_bac_idx, unexpected_beginning_bac_idx]:
                            stat = 'div'
                            find_stat = True

                    if unexpected_beginning_bac_idx in new_link_cost_df.columns.values.tolist() and \
                            source_bac_idx in new_link_cost_df.index.values.tolist():

                        if cost_val == new_link_cost_df.at[source_bac_idx, unexpected_beginning_bac_idx]:
                            stat = 'same'
                            find_stat = True

                    if not find_stat:
                        if cost_val == 1:
                            stat = 'same'

                    df = assign_new_link(df, neighbors_df, neighbor_list_array, unexpected_beginning_bac_idx,
                                         unexpected_beginning_bac, cost_val,
                                         source_bac_idx, stat,
                                         all_bac_in_unexpected_beginning_bac_time_step, maintenance_cost_df,
                                         parent_image_number_col, parent_object_number_col, label_col,
                                         center_coordinate_columns, redundant_link_division_df)

        n_iterate += 1

    return df
