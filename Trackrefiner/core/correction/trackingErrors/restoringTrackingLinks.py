import pandas as pd
from Trackrefiner.core.correction.action.bacteriaTrackingUpdate import bacteria_modification
from Trackrefiner.core.correction.action.trackingCost.calculateCreateLinkCost import optimize_assignment_using_hungarian
from Trackrefiner.core.correction.action.trackingCost.calculateCreateLinkCost import \
    calc_division_link_cost_for_restoring_links, calc_continuity_link_cost_for_restoring_links


def add_tracking_link(df, neighbors_df, neighbor_list_array, source_bac_idx, target_bac_idx,
                      parent_image_number_col, parent_object_number_col, center_coord_cols,
                      all_bac_in_target_bac_time_step, prob_val):

    """
    Adds a new tracking link between a source bacterium and a target bacterium if conditions are met.

    :param pandas.DataFrame df:
        The main DataFrame containing tracking information for bacteria.
    :param pandas.DataFrame neighbors_df:
        DataFrame representing the neighbor relationships between bacteria.
    :param scipy.sparse.csr_matrix neighbor_list_array:
        Sparse matrix representing neighborhood connectivity for all bacteria.
    :param int source_bac_idx:
        Index of the source bacterium in the current DataFrame.
    :param int target_bac_idx:
        Index of the target bacterium in the current DataFrame.
    :param str parent_image_number_col:
        Column name for the parent bacterium's image number.
    :param str parent_object_number_col:
        Column name for the parent bacterium's object number.
    :param dict center_coord_cols:
        Dictionary specifying the column names for the x and y centroid coordinates,
        e.g., `{'x': 'Center_X', 'y': 'Center_Y'}`.
    :param pandas.DataFrame all_bac_in_target_bac_time_step:
        Subset of the DataFrame containing all bacteria in the target bacterium's time step.
    :param float prob_val:
        Probability value indicating the likelihood of the new link being valid. A lower probability (`1 - prob_val`)
        must exceed a threshold (0.5) for the link to be considered.

    **Returns**:
        pandas.DataFrame:
            The updated DataFrame with the new tracking link added if the conditions are met.
    """

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
                                   center_coord_cols)

    return df


def restore_tracking_links(df, neighbors_df, neighbor_list_array, parent_image_number_col, parent_object_number_col,
                           center_coord_cols, df_raw_before_rpl_errors, continuity_links_model, division_links_model,
                           coordinate_array):

    """
    Attempts to restore previously removed tracking links for bacteria by re-evaluating conditions and costs.
    This function focuses on resolving links for unexpected beginning bacteria while maintaining tracking integrity.

    **Process**:
        - Identifies bacteria with `unexpected_beginning` status.
        - Cross-checks these bacteria against the raw tracking data to find potential links that were removed.
        - Evaluates potential links based on:
            - Continuity (single-link bacteria).
            - Division (bacteria dividing into daughters).
        - Applies machine learning models to calculate costs for restoring links and uses
          an optimization algorithm (Hungarian method) to finalize the link restoration.
        - Updates the DataFrame with restored links if conditions are satisfied.

    **Key Features**:
        - **Continuity Links**:
            - Checks whether the bacterium in question can continue its life history based on its source link.
        - **Division Links**:
            - Ensures division links meet criteria, such as the daughter-to-mother length ratio being below a threshold.
        - **Optimization**:
            - Uses a cost matrix and optimization algorithm to assign links with minimal cost.

    :param pandas.DataFrame df:
        The current DataFrame containing bacteria tracking data.
    :param pandas.DataFrame neighbors_df:
        DataFrame representing the neighbor relationships between bacteria.
    :param lil_matrix neighbor_list_array:
        Sparse matrix representing neighborhood connectivity for all bacteria.
    :param str parent_image_number_col:
        Column name for the parent bacterium's image number.
    :param str parent_object_number_col:
        Column name for the parent bacterium's object number.
    :param dict center_coord_cols:
        Dictionary containing the column names for x and y centroid coordinates,
        e.g., `{'x': 'Center_X', 'y': 'Center_Y'}`.
    :param pandas.DataFrame df_raw_before_rpl_errors:
        Original tracking data before resolving RPL errors, used to verify previously existing links.
    :param sklearn.Model continuity_links_model:
        Machine learning model to evaluate the validity of non-divided bacteria continuity links.
    :param sklearn.Model division_links_model:
        Machine learning model to evaluate division-based links for bacteria.
    :param numpy.ndarray coordinate_array:
        Array containing precomputed spatial coordinates for bacteria.

    **Returns**:
        pandas.DataFrame:
            The updated DataFrame with restored tracking links where applicable.
    """

    # only we should check unexpected beginning bacteria and compare with previous links to make sure
    # it can be possible to restore links or not

    unexpected_beginning_bacteria = df.loc[df['Unexpected_Beginning'] == True]

    division_df = pd.DataFrame()
    continuity_links_df = pd.DataFrame()

    for unexpected_bac_idx in unexpected_beginning_bacteria.index.values:

        unexpected_bac = df.loc[[unexpected_bac_idx]]

        bac_related_to_this_bac_in_raw_df = \
            df_raw_before_rpl_errors.loc[df_raw_before_rpl_errors['prev_index'] ==
                                         unexpected_bac['prev_index'].values[0]]

        if bac_related_to_this_bac_in_raw_df[parent_image_number_col].values[0] != 0:

            # it means that this bacterium had a previous link
            source_link = df.loc[
                (df['ImageNumber'] == bac_related_to_this_bac_in_raw_df[parent_image_number_col].values[0]) &
                (df['ObjectNumber'] == bac_related_to_this_bac_in_raw_df[parent_object_number_col].values[0])]

            if source_link['Unexpected_End'].values[0]:
                check_prob = True
                link_type = 'continuity'

            else:
                if str(source_link['Total_Daughter_Mother_Length_Ratio'].values[0]) != 'nan':
                    # it means that source bacterium has two links right now, and we can not restore links
                    check_prob = False
                else:
                    # discuses more about min life history for this
                    check_prob = True
                    other_daughter = df.loc[(df['id'] == source_link['id'].values[0]) & (
                            df['ImageNumber'] == unexpected_bac['ImageNumber'].values[0])]
                    link_type = 'div'

            if check_prob:

                df_bac_with_source = unexpected_bac.merge(source_link, how='cross', suffixes=('', '_source'))

                if link_type == 'continuity':
                    continuity_links_df = pd.concat([continuity_links_df, df_bac_with_source], ignore_index=True)

                elif link_type == 'div':
                    max_daughter_len = (max(other_daughter['AreaShape_MajorAxisLength'].values[0],
                                            unexpected_bac['AreaShape_MajorAxisLength'].values[0]) /
                                        source_link['AreaShape_MajorAxisLength'].values[0])
                    if max_daughter_len < 1:
                        division_df = pd.concat([division_df, df_bac_with_source], ignore_index=True)

    if continuity_links_df.shape[0] > 0:
        bac_with_same_source = continuity_links_df[
            continuity_links_df.duplicated(['ImageNumber_source', 'ObjectNumber_source'],
                                           keep=False)]

        if bac_with_same_source.shape[0] > 0:
            # Group by column 2 and aggregate
            agg_df = bac_with_same_source.groupby(['ImageNumber_source', 'ObjectNumber_source']).agg(
                {'AreaShape_MajorAxisLength': 'max', 'AreaShape_MajorAxisLength_source': 'mean'}).reset_index()

            # Rename columns for clarity
            agg_df.columns = ['ImageNumber_source', 'ObjectNumber_source', 'max_daughter_len_same', 'source_len_same']
            agg_df['Max_Daughter_Mother_Length_Ratio_same'] = (agg_df['max_daughter_len_same'] /
                                                               agg_df['source_len_same'])
            merged_df = pd.merge(bac_with_same_source, agg_df, on=['ImageNumber_source', 'ObjectNumber_source'],
                                 how='inner')

            # now we should check daughters condition
            daughters_passed_condition = merged_df.loc[merged_df['Max_Daughter_Mother_Length_Ratio_same'] < 1]
            daughters_passed_condition = daughters_passed_condition.drop(columns=['max_daughter_len_same',
                                                                                  'source_len_same'])

            division_df = pd.concat([division_df, daughters_passed_condition], ignore_index=True)

            # now we should remove them from continuity link df
            continuity_links_df = continuity_links_df.loc[~ continuity_links_df.index.isin(
                bac_with_same_source.index.values)]

    if division_df.shape[0] > 0:

        division_cost_df = \
            calc_division_link_cost_for_restoring_links(df, neighbors_df, neighbor_list_array, division_df,
                                                        center_coord_cols, col_source='_source', col_target='',
                                                        parent_image_number_col=parent_image_number_col,
                                                        parent_object_number_col=parent_object_number_col,
                                                        division_links_model=division_links_model,
                                                        coordinate_array=coordinate_array)

        if division_cost_df.shape[0] > 0:

            division_cost_df = division_cost_df.fillna(1)
            # optimization
            optimized_df = optimize_assignment_using_hungarian(division_cost_df)

            for row_index, row in optimized_df.iterrows():
                source_bac_idx = row['idx1']

                target_bac_idx = int(row['idx2'])
                target_bac = df.loc[target_bac_idx]

                cost_val = row['cost']

                all_bac_in_target_bac_time_step = df.loc[df['ImageNumber'] == target_bac['ImageNumber']]

                df = add_tracking_link(df, neighbors_df, neighbor_list_array, source_bac_idx, target_bac_idx,
                                       parent_image_number_col, parent_object_number_col, center_coord_cols,
                                       all_bac_in_target_bac_time_step, cost_val)

    if continuity_links_df.shape[0] > 0:

        continuity_links_df['id_source'] = df.loc[
            df['index'].isin(continuity_links_df['index_source'].values), 'id'].values
        continuity_links_df['age_source'] = df.loc[
            df['index'].isin(continuity_links_df['index_source'].values), 'age'].values

        continuity_cost_df = \
            calc_continuity_link_cost_for_restoring_links(df, neighbors_df, neighbor_list_array, continuity_links_df,
                                                          center_coord_cols, col_source='_source', col_target='',
                                                          parent_image_number_col=parent_image_number_col,
                                                          parent_object_number_col=parent_object_number_col,
                                                          continuity_links_model=continuity_links_model,
                                                          coordinate_array=coordinate_array)

        if continuity_cost_df.shape[0] > 0:

            continuity_cost_df = continuity_cost_df.fillna(1)
            # optimization
            optimized_df = optimize_assignment_using_hungarian(continuity_cost_df)

            for row_index, row in optimized_df.iterrows():
                source_bac_idx = row['idx1']

                target_bac_idx = int(row['idx2'])
                target_bac = df.loc[target_bac_idx]

                cost_val = row['cost']

                all_bac_in_target_bac_time_step = df.loc[df['ImageNumber'] == target_bac['ImageNumber']]

                df = add_tracking_link(df, neighbors_df, neighbor_list_array, source_bac_idx,
                                       target_bac_idx, parent_image_number_col,
                                       parent_object_number_col, center_coord_cols,
                                       all_bac_in_target_bac_time_step, cost_val)
    return df
