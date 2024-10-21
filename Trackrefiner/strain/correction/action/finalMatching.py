import pandas as pd
from Trackrefiner.strain.correction.action.bacteriaModification import bacteria_modification
from Trackrefiner.strain.correction.action.compareBacteria import optimize_assignment
from Trackrefiner.strain.correction.action.compareBacteria import daughter_cost_for_final_step, \
    same_link_cost_for_final_checking


def adding_new_link(df, neighbors_df, neighbor_list_array, stat, source_bac_idx, source_bac, target_bac_idx, target_bac,
                    parent_image_number_col, parent_object_number_col, center_coordinate_columns, label_col,
                    all_bac_in_target_bac_time_step, prob_val):

    if 1 - prob_val > 0.5:
        # there is two scenario: first: source bac with only one bac in next time step: so we compare
        # the probability of that with new bac
        # the second scenario is source bac has two daughters. I think it's good idea two compare
        # max daughter probability with new bac probability

        # update info
        source_bac = df.loc[source_bac_idx]
        # unexpected beginning
        target_bac_life_history = df.loc[df['id'] == df.at[target_bac_idx, 'id']]

        df = bacteria_modification(df, source_bac, target_bac_life_history,
                                   all_bac_in_target_bac_time_step,
                                   neighbors_df, neighbor_list_array, parent_image_number_col, parent_object_number_col,
                                   center_coordinate_columns, label_col)

    return df


def final_matching(df, neighbors_df, neighbor_list_array, min_life_history_of_bacteria, interval_time,
                   parent_image_number_col, parent_object_number_col, label_col, center_coordinate_columns,
                   df_before_more_detection_and_removing, non_divided_bac_model, divided_bac_model, coordinate_array):

    # only we should check unexpected beginning bacteria and compare with previous links to make sure
    # it can be possible to restore links or not

    unexpected_beginning_bacteria = df.loc[df['unexpected_beginning'] == True]

    division_df = pd.DataFrame()
    same_df = pd.DataFrame()

    for unexpected_bac_idx in unexpected_beginning_bacteria.index.values:

        unexpected_bac = df.loc[[unexpected_bac_idx]]

        bac_related_to_this_bac_in_raw_df = \
            df_before_more_detection_and_removing.loc[df_before_more_detection_and_removing['prev_index'] ==
                                                      unexpected_bac['prev_index'].values[0]]

        if bac_related_to_this_bac_in_raw_df[parent_image_number_col].values[0] != 0:

            # it means that this bacterium had a previous link
            source_link = df.loc[
                (df['ImageNumber'] == bac_related_to_this_bac_in_raw_df[parent_image_number_col].values[0]) &
                (df['ObjectNumber'] == bac_related_to_this_bac_in_raw_df[parent_object_number_col].values[0])]

            if source_link['unexpected_end'].values[0]:
                check_prob = True
                stat = 'same'

            else:
                if str(source_link['daughter_length_to_mother'].values[0]) != 'nan':
                    # it means that source bacterium has two links right now, and we can not restore links
                    check_prob = False
                else:
                    # discuses more about min life history for this
                    check_prob = True
                    other_daughter = df.loc[(df['id'] == source_link['id'].values[0]) & (
                            df['ImageNumber'] == unexpected_bac['ImageNumber'].values[0])]
                    stat = 'div'

            if check_prob:

                df_bac_with_source = unexpected_bac.merge(source_link, how='cross', suffixes=('', '_source'))

                if stat == 'same':
                    same_df = pd.concat([same_df, df_bac_with_source], ignore_index=True)

                elif stat == 'div':
                    max_daughter_len = (max(other_daughter['AreaShape_MajorAxisLength'].values[0],
                                            unexpected_bac['AreaShape_MajorAxisLength'].values[0]) /
                                        source_link['AreaShape_MajorAxisLength'].values[0])
                    if max_daughter_len < 1:
                        division_df = pd.concat([division_df, df_bac_with_source], ignore_index=True)

    if same_df.shape[0] > 0:
        bac_with_same_source = same_df[same_df.duplicated(['ImageNumber_source', 'ObjectNumber_source'],
                                                          keep=False)]

        if bac_with_same_source.shape[0] > 0:

            # Group by column 2 and aggregate
            agg_df = bac_with_same_source.groupby(['ImageNumber_source', 'ObjectNumber_source']).agg(
                {'AreaShape_MajorAxisLength': 'max', 'AreaShape_MajorAxisLength_source': 'mean'}).reset_index()

            # Rename columns for clarity
            agg_df.columns = ['ImageNumber_source', 'ObjectNumber_source', 'max_daughter_len_same', 'source_len_same']
            agg_df['max_daughter_len_to_mother_same'] = agg_df['max_daughter_len_same'] / agg_df['source_len_same']
            merged_df = pd.merge(bac_with_same_source, agg_df, on=['ImageNumber_source', 'ObjectNumber_source'],
                                 how='inner')

            # now we should check daughters condition
            daughters_passed_condition = merged_df.loc[merged_df['max_daughter_len_to_mother_same'] < 1]
            daughters_passed_condition = daughters_passed_condition.drop(columns=['max_daughter_len_same',
                                                                                  'source_len_same'])

            division_df = pd.concat([division_df, daughters_passed_condition], ignore_index=True)

            # now we should remove them from same df
            same_df = same_df.loc[~ same_df.index.isin(bac_with_same_source.index.values)]

    if division_df.shape[0] > 0:

        division_cost_df = daughter_cost_for_final_step(df, neighbors_df, neighbor_list_array, division_df,
                                                        center_coordinate_columns, col_source='_source',
                                                        col_target='', parent_image_number_col=parent_image_number_col,
                                                        parent_object_number_col=parent_object_number_col,
                                                        divided_bac_model=divided_bac_model,
                                                        maintenance_cost_df=None, maintenance_to_be_check=None,
                                                        coordinate_array=coordinate_array)

        if division_cost_df.shape[0] > 0:

            division_cost_df = division_cost_df.fillna(1)
            # optimization
            optimized_df = optimize_assignment(division_cost_df)

            for row_index, row in optimized_df.iterrows():
                source_bac_idx = row['without parent index']
                source_bac = df.loc[source_bac_idx]

                target_bac_idx = int(row['Candida bacteria index in previous time step'])
                target_bac = df.loc[target_bac_idx]

                cost_val = row['Cost']

                all_bac_in_target_bac_time_step = df.loc[df['ImageNumber'] == target_bac['ImageNumber']]

                df = adding_new_link(df, neighbors_df, neighbor_list_array, stat, source_bac_idx, source_bac,
                                     target_bac_idx, target_bac, parent_image_number_col,
                                     parent_object_number_col, center_coordinate_columns, label_col,
                                     all_bac_in_target_bac_time_step, cost_val)

    if same_df.shape[0] > 0:

        same_df['id_source'] = df.loc[df['index'].isin(same_df['index_source'].values), 'id'].values
        same_df['age_source'] = df.loc[df['index'].isin(same_df['index_source'].values), 'age'].values

        same_cost_df = same_link_cost_for_final_checking(df, neighbors_df, neighbor_list_array, same_df,
                                                         center_coordinate_columns,
                                                         col_source='_source', col_target='',
                                                         parent_image_number_col=parent_image_number_col,
                                                         parent_object_number_col=parent_object_number_col,
                                                         non_divided_bac_model=non_divided_bac_model,
                                                         maintenance_cost_df=None, maintenance_to_be_check=None,
                                                         coordinate_array=coordinate_array)

        if same_cost_df.shape[0] > 0:

            same_cost_df = same_cost_df.fillna(1)
            # optimization
            optimized_df = optimize_assignment(same_cost_df)

            for row_index, row in optimized_df.iterrows():
                source_bac_idx = row['without parent index']
                source_bac = df.loc[source_bac_idx]

                target_bac_idx = int(row['Candida bacteria index in previous time step'])
                target_bac = df.loc[target_bac_idx]

                cost_val = row['Cost']

                all_bac_in_target_bac_time_step = df.loc[df['ImageNumber'] == target_bac['ImageNumber']]

                df = adding_new_link(df, neighbors_df, neighbor_list_array, stat, source_bac_idx, source_bac,
                                     target_bac_idx, target_bac, parent_image_number_col,
                                     parent_object_number_col, center_coordinate_columns, label_col,
                                     all_bac_in_target_bac_time_step, cost_val)
    return df
