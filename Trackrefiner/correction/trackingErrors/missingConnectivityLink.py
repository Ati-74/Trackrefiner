import numpy as np
from Trackrefiner.correction.action.findOutlier import detect_length_change_ratio_outliers
from Trackrefiner.correction.action.trackingCost.calculateCreateLinkCost import optimize_assignment_using_hungarian
from Trackrefiner.correction.action.trackingCost.calculateCreateLinkCostMCC import division_detection_cost, \
    create_continuity_link_cost
from Trackrefiner.correction.action.trackingCost.calculateMaintainingLinkCost import calc_maintain_exist_link_cost
from Trackrefiner.correction.action.bacteriaTrackingUpdate import bacteria_modification, remove_redundant_link
import pandas as pd


def resolve_missing_connectivity_links(df, neighbors_df, neighbor_matrix, source_bac_idx, source_bac,
                                       link_probability, new_target_bac_idx, link_type, target_frame_bacteria,
                                       parent_image_number_col, parent_object_number_col, center_coord_cols):
    """
    Resolve missing connectivity links by removing incorrect links and establishing new ones.

    This function adjusts bacterial tracking data to resolve missing connectivity by
    removing redundant or incorrect links and creating new ones. Based on the
    `link_probability` and the `link_type`, it establishes or removes connections
    for division links (`'division'`) or continuity links (`'continuity'`).

    :param pd.DataFrame df:
        Dataframe containing bacterial tracking data, including IDs, relationships, and coordinates.
    :param pd.DataFrame neighbors_df:
        Dataframe containing neighbor relationship information for bacteria.
    :param lil_matrix neighbor_matrix:
        Sparse matrix representing neighbor relationships.
    :param int source_bac_idx:
        Index of the source bacteria whose tracking link is being resolved.
    :param pd.Series source_bac:
        Data for the source bacteria whose link needs to be adjusted.
    :param float link_probability:
        Probability value from a machine learning model. The value `1 - link_probability` represents
        the likelihood that the proposed new link is valid.
    :param int new_target_bac_idx:
        Index of the target bacteria to establish a new link with.
    :param str link_type:
        Type of link to handle:
        - `'division'`: Handles division events.
        - `'continuity'`: Handles continuity of the same bacteria across frames.
    :param pd.DataFrame target_frame_bacteria:
        Subset of the dataframe containing bacteria in the target time frame.
    :param str parent_image_number_col:
        Column name for the parent image number in the dataframe.
    :param str parent_object_number_col:
        Column name for the parent object number in the dataframe.
    :param dict center_coord_cols:
        Dictionary specifying the column names for x and y coordinates of bacterial centers.
        Example: {'x': 'Center_X', 'y': 'Center_Y'}

    :return:
        pd.DataFrame:

        Updated dataframe with resolved missing connectivity link and adjusted bacterial tracking features.
    """

    if link_type == 'division':

        # probability is 1 - probability
        if 1 - link_probability > 0.5:

            # it means division occurs
            new_daughter = df.loc[new_target_bac_idx]
            new_daughter_life_history = df.loc[(df['id'] == new_daughter['id'])
                                               & (df['ImageNumber'] >= new_daughter['ImageNumber'])]

            if new_daughter[parent_image_number_col] != 0:
                df = remove_redundant_link(df, new_daughter_life_history, neighbors_df, neighbor_matrix,
                                           parent_image_number_col, parent_object_number_col, center_coord_cols)

                # update info
                new_daughter_life_history = df.loc[new_daughter_life_history.index]

            df = bacteria_modification(df, df.loc[source_bac_idx], new_daughter_life_history, target_frame_bacteria,
                                       neighbors_df, neighbor_matrix, parent_image_number_col, parent_object_number_col,
                                       center_coord_cols)

    elif link_type == 'continuity':

        # probability is 1 - probability
        if 1 - link_probability > 0.5:

            # same bac
            new_bac = df.loc[new_target_bac_idx]
            new_bac_life_history = df.loc[(df['id'] == new_bac['id']) &
                                          (df['ImageNumber'] >= new_bac['ImageNumber'])]

            # incorrect link
            target_incorrect_continuity_link_life_history = \
                df.loc[(df['id'] == df.at[source_bac_idx, 'id']) &
                       (df['ImageNumber'] >= source_bac['ImageNumber'] + 1)]

            if target_incorrect_continuity_link_life_history.shape[0] > 0:
                df = remove_redundant_link(df, target_incorrect_continuity_link_life_history, neighbors_df,
                                           neighbor_matrix, parent_image_number_col, parent_object_number_col,
                                           center_coord_cols)

            # update info
            new_bac_life_history = df.loc[new_bac_life_history.index]

            if new_bac[parent_image_number_col] != 0:
                df = remove_redundant_link(df, new_bac_life_history, neighbors_df, neighbor_matrix,
                                           parent_image_number_col, parent_object_number_col, center_coord_cols)

                # update info
                new_bac_life_history = df.loc[new_bac_life_history.index]

            df = bacteria_modification(df, df.loc[source_bac_idx], new_bac_life_history, target_frame_bacteria,
                                       neighbors_df, neighbor_matrix, parent_image_number_col,
                                       parent_object_number_col, center_coord_cols)

    return df


def detect_and_resolve_missing_connectivity_link(df, neighbors_df, neighbor_matrix, doubling_time,
                                                 interval_time, parent_image_number_col, parent_object_number_col,
                                                 center_coord_cols, division_vs_continuity_model,
                                                 continuity_links_model, division_links_model, coordinate_array):
    """
    Detect and resolve missing connectivity links in bacterial tracking data.

    This function identifies missing connectivity links in bacterial tracking, finds candidate
    sources for linking with target bacteria, and evaluates candidates based on statistical
    conditions, biological plausibility, and machine learning models. If valid candidates are
    found, the function modifies the tracking data to establish new links.

    :param pd.DataFrame df:
        Dataframe containing bacterial tracking data, including IDs, life histories, and coordinates.
    :param pd.DataFrame neighbors_df:
        Dataframe containing information about neighboring bacteria.
    :param lil_matrix neighbor_matrix:
        Sparse matrix representing neighbor relationships.
    :param float doubling_time:
        Estimated doubling time for bacteria, used to set thresholds for division and growth.
    :param float interval_time:
        Time interval between frames in the tracking data.
    :param str parent_image_number_col:
        Column name for the parent image number in the dataframe.
    :param str parent_object_number_col:
        Column name for the parent object number in the dataframe.
    :param dict center_coord_cols:
        Dictionary specifying the column names for x and y coordinates of bacterial centers.
        Example: {'x': 'Center_X', 'y': 'Center_Y'}
    :param sklearn.Model division_vs_continuity_model:
        Machine learning model used to compare divided and non-divided states for bacteria.
    :param sklearn.Model continuity_links_model:
        Machine learning model used to evaluate candidate links for non-divided bacteria.
    :param sklearn.Model division_links_model:
        Machine learning model used to validate division links.
    :param csr_matrix coordinate_array:
        Array of spatial coordinates used for evaluating candidate links (calculation of IOU).

    :return:
        pd.DataFrame:

        Updated dataframe with resolved connectivity links and adjusted bacterial life histories.
    """

    num_incorrect_continuity_links = None
    prev_bacteria_with_wrong_continuity_link = None
    n_iterate = 0

    # min life history of bacteria
    min_life_history_of_bacteria_time_step = int(np.round(doubling_time / interval_time))

    while num_incorrect_continuity_links != 0 and len(df["LengthChangeRatio"].dropna().values) > 0:

        # check incorrect continuity link
        max_bac_to_bac_len_ratio_list_outliers_min_boundary = detect_length_change_ratio_outliers(df)

        if str(max_bac_to_bac_len_ratio_list_outliers_min_boundary) != 'nan':

            mcc_links = \
                df.loc[df["LengthChangeRatio"] <= max_bac_to_bac_len_ratio_list_outliers_min_boundary]

            if n_iterate > 0:
                if (prev_bacteria_with_wrong_continuity_link.values.all() ==
                        mcc_links.values.all()):
                    num_incorrect_continuity_links = 0
                else:
                    num_incorrect_continuity_links = mcc_links.shape[0]

            prev_bacteria_with_wrong_continuity_link = mcc_links

            if mcc_links.shape[0] > 0:

                # possible states:
                # 1. division occurs (at least one link of other neighbours is wrong)
                # 2. this link is incorrect
                for mccs_target_time_step in mcc_links['ImageNumber'].unique():

                    target_frame_bacteria = df.loc[df['ImageNumber'] == mccs_target_time_step]
                    source_frame_bacteria = df.loc[df['ImageNumber'] == (mccs_target_time_step - 1)]

                    # it's the target of incorrect continuity link (because the bac_bac_len of target to source has been
                    # written for target bac not source bac)
                    sel_mccs_targets = mcc_links.loc[mcc_links['ImageNumber'] == mccs_target_time_step]

                    mcc_target_source = \
                        sel_mccs_targets.merge(source_frame_bacteria, left_on=[parent_image_number_col,
                                                                               parent_object_number_col],
                                               right_on=['ImageNumber', 'ObjectNumber'], how='inner',
                                               suffixes=('_target', '_source'))

                    mccs_target_neighbors_df = \
                        neighbors_df.loc[neighbors_df['First Image Number'] == mccs_target_time_step]

                    mcc_target_neighbors = \
                        mcc_target_source.merge(mccs_target_neighbors_df, left_on=['ImageNumber_target',
                                                                                   'ObjectNumber_target'],
                                                right_on=['First Image Number', 'First Object Number'], how='inner')

                    # I didn't define suffix because target and source bac has it
                    mcc_target_neighbors_features = mcc_target_neighbors.merge(
                        target_frame_bacteria, left_on=['Second Image Number', 'Second Object Number'],
                        right_on=['ImageNumber', 'ObjectNumber'])

                    mcc_target_neighbors_features = mcc_target_neighbors_features.rename(
                            {'First Image Number': 'First Image Number_target',
                             'First Object Number': 'First Object Number_target',
                             'Second Image Number': 'Second Image Number_neighbor',
                             'Second Object Number': 'Second Object Number_neighbor'}, axis=1)

                    columns_related_to_neighbors = [v for v in mcc_target_neighbors_features.columns if
                                                    v.count('_target') == 0 and v.count('_source') == 0]

                    neighbors_bac_to_target = mcc_target_neighbors_features.drop_duplicates(
                        subset=['ImageNumber', 'ObjectNumber'], keep='last')[columns_related_to_neighbors]

                    neighbors_bac_to_target_with_source = \
                        neighbors_bac_to_target.merge(source_frame_bacteria, left_on=[parent_image_number_col,
                                                                                      parent_object_number_col],
                                                      right_on=['ImageNumber', 'ObjectNumber'],
                                                      how='inner', suffixes=('_neighbor_target', '_neighbor_source'))

                    neighbors_source_bac = source_frame_bacteria.loc[
                        neighbors_bac_to_target_with_source['index_neighbor_source'].unique()]

                    neighbors_target_bac = \
                        target_frame_bacteria.loc[neighbors_bac_to_target_with_source['index_neighbor_target']]

                    maintenance_cost_df = \
                        calc_maintain_exist_link_cost(neighbors_source_bac, neighbors_target_bac, center_coord_cols,
                                                      division_vs_continuity_model, coordinate_array)
                    # try to detect division
                    division_cost_df = \
                        division_detection_cost(df, neighbors_df, neighbor_matrix, mcc_target_neighbors_features,
                                                min_life_history_of_bacteria_time_step, center_coord_cols,
                                                parent_image_number_col, parent_object_number_col, division_links_model,
                                                division_vs_continuity_model, maintenance_cost_df, coordinate_array)

                    new_link_cost_df = \
                        create_continuity_link_cost(df, neighbors_df, neighbor_matrix, mcc_target_neighbors_features,
                                                    center_coord_cols, parent_image_number_col,
                                                    parent_object_number_col, continuity_links_model,
                                                    division_vs_continuity_model, maintenance_cost_df, coordinate_array)

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
                        optimized_df = optimize_assignment_using_hungarian(total_cost_df)

                        # try to modify the links
                        for row_index, row in optimized_df.iterrows():
                            source_bac_idx = row['idx1']
                            source_bac = df.loc[source_bac_idx]

                            new_target_bac_idx = int(row['idx2'])

                            cost_val = row['cost']

                            find_stat = False

                            if source_bac_idx in division_cost_df.index.values and \
                                    new_target_bac_idx in division_cost_df.columns.values:

                                if cost_val == division_cost_df.at[source_bac_idx, new_target_bac_idx]:
                                    link_type = 'division'
                                    find_stat = True

                            if source_bac_idx in new_link_cost_df.index.values and \
                                    new_target_bac_idx in new_link_cost_df.columns.values:

                                if cost_val == new_link_cost_df.at[source_bac_idx, new_target_bac_idx]:
                                    link_type = 'continuity'
                                    find_stat = True

                            if not find_stat:
                                if cost_val == 1:
                                    link_type = 'continuity'
                                else:
                                    raise ValueError("Unable to determine the link type for the given source and "
                                                     "target bacteria. Check the input data or model outputs.")

                            df = \
                                resolve_missing_connectivity_links(df, neighbors_df, neighbor_matrix,
                                                                   source_bac_idx, source_bac,
                                                                   cost_val, new_target_bac_idx, link_type,
                                                                   target_frame_bacteria,
                                                                   parent_image_number_col, parent_object_number_col,
                                                                   center_coord_cols)

            n_iterate += 1

        else:
            num_incorrect_continuity_links = 0

    return df
