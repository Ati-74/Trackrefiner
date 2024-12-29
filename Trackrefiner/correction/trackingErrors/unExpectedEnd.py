from Trackrefiner.correction.action.trackingCost.calculateCreateLinkCostUE import \
    compute_create_link_from_unexpected_end_cost
from Trackrefiner.correction.action.trackingCost.calculateCreateLinkCost import optimize_assignment_using_hungarian
from Trackrefiner.correction.action.bacteriaTrackingUpdate import bacteria_modification, remove_redundant_link
import pandas as pd
import numpy as np


def resolve_unexpected_end_bacteria(df, neighbors_df, neighbor_matrix, link_type, unexpected_end_bac_idx,
                                    target_bac_idx, target_bac, daughter_bac_idx, daughter_bac,
                                    parent_image_number_col, parent_object_number_col, center_coord_cols,
                                    next_frame_bacteria, target_link_probability, daughter_link_probability):
    """
    Resolve unexpected end bacteria by creating new tracking links.

    This function handles unexpected end bacteria by creating new tracking
    links to the next frame. It supports two types of links:
    - `'continuity'`: The source bacterium's life history continues to the target bacterium in the next frame.
    - `'division'`: The source bacterium divides, creating links to both a target bacterium and a daughter bacterium.

    The function uses probabilities from an ML model to validate the links and removes previous
    links if necessary.

    :param pd.DataFrame df:
        Dataframe containing bacterial tracking data, including IDs, life histories, and coordinates.
    :param pd.DataFrame neighbors_df:
        Dataframe containing neighbor relationship information for bacteria.
    :param lil_matrix neighbor_matrix:
        Sparse matrix representing neighbor relationships.
    :param str link_type:
        Type of link to establish:
        - `'continuity'`: Continuity link to the target bacterium in the next frame.
        - `'division'`: Division link to the target bacterium and a daughter bacterium in the next frame.
    :param int unexpected_end_bac_idx:
        Index of the unexpected end bacterium (source bacterium).
    :param int target_bac_idx:
        Index of the target bacterium in the next frame.
    :param pd.Series target_bac:
        Data for the target bacterium in the next frame.
    :param int daughter_bac_idx:
        Index of the daughter bacterium, used for division links.
    :param pd.Series daughter_bac:
        Data for the daughter bacterium, used for division links.
    :param str parent_image_number_col:
        Column name for the parent image number in the dataframe.
    :param str parent_object_number_col:
        Column name for the parent object number in the dataframe.
    :param dict center_coord_cols:
        Dictionary specifying the column names for x and y coordinates of bacterial centers.
        Example: {'x': 'Center_X', 'y': 'Center_Y'}
    :param pd.DataFrame next_frame_bacteria:
        Dataframe containing bacteria in the next frame relative to the source bacterium.
    :param float target_link_probability:
        Probability value from the ML model for the link to the target bacterium.
        A valid link is considered if `1 - target_link_probability > 0.5`.
    :param float daughter_link_probability:
        Probability value from the ML model for the link to the daughter bacterium (division).
        A valid link is considered if `1 - daughter_link_probability > 0.5`.

    :return:
        pd.DataFrame

        Updated dataframe with new links added and previous links removed.
    """

    if link_type == 'division':

        if 1 - target_link_probability > 0.5 and 1 - daughter_link_probability > 0.5:
            source_bac = df.loc[unexpected_end_bac_idx]

            # remove incorrect link
            if target_bac[parent_image_number_col] != 0:
                incorrect_bac_life_history_target_bac = \
                    df.loc[(df['id'] == df.at[target_bac_idx, 'id']) &
                           (df['ImageNumber'] >= source_bac['ImageNumber'] + 1)]

                df = remove_redundant_link(df, incorrect_bac_life_history_target_bac, neighbors_df, neighbor_matrix,
                                           parent_image_number_col, parent_object_number_col, center_coord_cols)

            if daughter_bac[parent_image_number_col] != 0:
                incorrect_bac_life_history_another_target_bac = \
                    df.loc[(df['id'] == df.at[daughter_bac_idx, 'id']) &
                           (df['ImageNumber'] >= source_bac['ImageNumber'] + 1)]

                df = remove_redundant_link(df, incorrect_bac_life_history_another_target_bac, neighbors_df,
                                           neighbor_matrix, parent_image_number_col, parent_object_number_col,
                                           center_coord_cols)

            # now we should add link
            # update info
            source_bac = df.loc[unexpected_end_bac_idx]
            target_bac_life_history = df.loc[df['id'] == df.at[target_bac_idx, 'id']]

            df = bacteria_modification(df, source_bac, target_bac_life_history, next_frame_bacteria, neighbors_df,
                                       neighbor_matrix, parent_image_number_col, parent_object_number_col,
                                       center_coord_cols)

            source_bac = df.loc[unexpected_end_bac_idx]
            another_target_bac_life_history = df.loc[df['id'] == df.at[daughter_bac_idx, 'id']]

            df = bacteria_modification(df, source_bac, another_target_bac_life_history, next_frame_bacteria,
                                       neighbors_df, neighbor_matrix, parent_image_number_col, parent_object_number_col,
                                       center_coord_cols)

    elif link_type == 'continuity':

        if 1 - target_link_probability > 0.5:
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

                df = remove_redundant_link(df, incorrect_bac_life_history_target_bac, neighbors_df, neighbor_matrix,
                                           parent_image_number_col, parent_object_number_col, center_coord_cols)

            # update info
            source_bac = df.loc[unexpected_end_bac_idx]
            target_bac_life_history = df.loc[df['id'] == df.at[target_bac_idx, 'id']]

            df = bacteria_modification(df, source_bac, target_bac_life_history, next_frame_bacteria, neighbors_df,
                                       neighbor_matrix, parent_image_number_col, parent_object_number_col,
                                       center_coord_cols)
    else:
        raise ValueError("Link type is None.")

    return df


def handle_unexpected_end_bacteria(df, neighbors_df, neighbor_matrix, interval_time, doubling_time,
                                   parent_image_number_col, parent_object_number_col, center_coord_cols,
                                   divided_vs_non_divided_model, non_divided_model, division_model, color_array,
                                   coordinate_array):

    """
    Handle unexpected end bacteria by finding and validating candidate targets, and creating new links.

    This function processes bacteria flagged with `unexpected_end` by identifying candidate target bacteria
    in the next frame. It validates potential links using statistical conditions, biological plausibility,
    and machine learning models. If validated, it creates new links to resolve tracking gaps.

    The function handles two scenarios:
    - **continuity**: The bacterium's life history continues to the next frame.
    - **division**: The bacterium divides, linking to two daughter bacteria.

    :param pd.DataFrame df:
        Dataframe containing bacterial measured bacterial features.
    :param pd.DataFrame neighbors_df:
        Dataframe containing neighbor relationship information for bacteria.
    :param lil_matrix neighbor_matrix:
        Sparse matrix representing neighbor relationships.
    :param float interval_time:
        Time interval between frames in the tracking data.
    :param float doubling_time:
        Estimated doubling time for bacteria, used to determine plausible division events.
    :param str parent_image_number_col:
        Column name for the parent image number in the dataframe.
    :param str parent_object_number_col:
        Column name for the parent object number in the dataframe.
    :param dict center_coord_cols:
        Dictionary specifying the column names for x and y coordinates of bacterial centers.
        Example: {'x': 'Center_X', 'y': 'Center_Y'}
    :param sklearn.Model divided_vs_non_divided_model:
        Machine learning model used to compare divided and non-divided states for bacteria.
    :param sklearn.Model non_divided_model:
        Machine learning model used to evaluate candidate links for continuity events.
    :param sklearn.Model division_model:
        Machine learning model used to validate division links.
    :param np.ndarray color_array:
        Array representing the colors of objects in the tracking data. This is used for mapping
        bacteria from the dataframe to the coordinate array for spatial analysis.
    :param csr_matrix coordinate_array:
        Array of spatial coordinates used for evaluating candidate links.

    :return:
        pd.DataFrame

        Updated dataframe with resolved links for unexpected end bacteria.
    """

    num_unexpected_end_bac = None
    prev_unexpected_end_bac = None
    n_iterate = 0

    # min life history of bacteria
    min_life_history_time_step = int(np.round(doubling_time / interval_time))

    while num_unexpected_end_bac != 0:

        df_unexpected_end_bacteria = df.loc[df['unexpected_end']]

        if n_iterate > 0:
            if prev_unexpected_end_bac.values.all() == df_unexpected_end_bacteria.values.all():
                num_unexpected_end_bac = 0
            else:
                num_unexpected_end_bac = df_unexpected_end_bacteria.shape[0]

        prev_unexpected_end_bac = df_unexpected_end_bacteria

        if df_unexpected_end_bacteria.shape[0] > 0:
            for unexpected_end_bacteria_time_step in df_unexpected_end_bacteria['ImageNumber'].unique():

                sel_unexpected_end_bacteria = \
                    df_unexpected_end_bacteria.loc[
                        df_unexpected_end_bacteria['ImageNumber'] == unexpected_end_bacteria_time_step]

                sel_unexpected_end_bacteria = \
                    df.loc[sel_unexpected_end_bacteria['index'].values]

                unexpected_end_bac_time_step_bacteria = \
                    df.loc[df['ImageNumber'] == unexpected_end_bacteria_time_step]
                next_time_step_bacteria = \
                    df.loc[df['ImageNumber'] == unexpected_end_bacteria_time_step + 1]

                filtered_continuity_link_cost_df, validated_division_cost_df, final_division_cost_df = \
                    compute_create_link_from_unexpected_end_cost(df, neighbors_df, neighbor_matrix,
                                                                 unexpected_end_bac_time_step_bacteria,
                                                                 sel_unexpected_end_bacteria, next_time_step_bacteria,
                                                                 center_coord_cols, parent_image_number_col,
                                                                 parent_object_number_col, min_life_history_time_step,
                                                                 divided_vs_non_divided_model, non_divided_model,
                                                                 division_model, color_array, coordinate_array)

                if (validated_division_cost_df.shape[0] > 0 and
                        filtered_continuity_link_cost_df.shape[0] > 0):

                    # Merge the dataframes using outer join to combine all columns and indices
                    total_cost_df = pd.concat([validated_division_cost_df, filtered_continuity_link_cost_df],
                                              axis=1, sort=False).fillna(1)

                    continue_flag = True

                elif validated_division_cost_df.shape[0] > 0:

                    total_cost_df = validated_division_cost_df

                    continue_flag = True

                elif filtered_continuity_link_cost_df.shape[0] > 0:

                    total_cost_df = filtered_continuity_link_cost_df

                    continue_flag = True
                else:
                    continue_flag = False
                    total_cost_df = None

                if continue_flag:

                    total_cost_df = total_cost_df.fillna(1)
                    optimized_df = optimize_assignment_using_hungarian(total_cost_df)

                    # try to modify the links
                    for row_index, row in optimized_df.iterrows():

                        unexpected_end_bac_idx = row['idx1']

                        target_bac_idx = int(row['idx2'])
                        target_bac = df.loc[target_bac_idx]

                        cost_val = row['cost']
                        find_stat = False

                        link_type = None
                        another_daughter_bac_idx = None
                        another_daughter_bac = None
                        another_daughter_bac_prob_val = None

                        if target_bac_idx in final_division_cost_df.columns.values.tolist() and \
                                unexpected_end_bac_idx in final_division_cost_df.index.values.tolist():

                            if cost_val == final_division_cost_df.at[unexpected_end_bac_idx, target_bac_idx]:
                                link_type = 'division'

                                another_daughter_bac_idx = [idx for idx in
                                                            final_division_cost_df.loc[
                                                                unexpected_end_bac_idx].dropna().index
                                                            if idx != target_bac_idx][0]

                                another_daughter_bac = df.loc[another_daughter_bac_idx]

                                another_daughter_bac_prob_val = \
                                    final_division_cost_df.at[unexpected_end_bac_idx, another_daughter_bac_idx]
                                find_stat = True

                        if target_bac_idx in filtered_continuity_link_cost_df.columns.values.tolist() and \
                                unexpected_end_bac_idx in filtered_continuity_link_cost_df.index.values.tolist():

                            if cost_val == filtered_continuity_link_cost_df.at[unexpected_end_bac_idx, target_bac_idx]:
                                link_type = 'continuity'
                                find_stat = True

                        if not find_stat:
                            if cost_val == 1:
                                link_type = 'continuity'

                        df = \
                            resolve_unexpected_end_bacteria(df, neighbors_df, neighbor_matrix, link_type,
                                                            unexpected_end_bac_idx, target_bac_idx, target_bac,
                                                            another_daughter_bac_idx, another_daughter_bac,
                                                            parent_image_number_col, parent_object_number_col,
                                                            center_coord_cols, next_time_step_bacteria, cost_val,
                                                            another_daughter_bac_prob_val)

        n_iterate += 1

    return df
