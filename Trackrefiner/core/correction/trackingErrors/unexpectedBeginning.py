import numpy as np
from Trackrefiner.core.correction.action.bacteriaTrackingUpdate import bacteria_modification, remove_redundant_link
from Trackrefiner.core.correction.action.trackingCost.calculateCreateLinkCostUB import \
    compute_create_link_to_unexpected_beginning_cost
from Trackrefiner.core.correction.action.trackingCost.calculateCreateLinkCost import optimize_assignment_using_hungarian
import pandas as pd


def resolve_unexpected_beginning_bacteria(df, neighbors_df, neighbor_list_array, unexpected_beginning_bac_idx,
                                          unexpected_beginning_bac, link_probability, source_bac_idx, link_type,
                                          unexpected_begging_bac_time_step_bacteria, maintenance_cost,
                                          parent_image_number_col, parent_object_number_col, center_coord_col,
                                          redundant_link_division_df):
    """
    Resolves unexpected beginning bacteria by establishing new links to them and, if needed, modifying or removing
    existing links to source bacteria, based on the specified link type and probabilities.

    :param pandas.DataFrame df:
        DataFrame containing bacterial life history and tracking information.
    :param pandas.DataFrame neighbors_df:
        DataFrame containing neighboring relationships between bacteria, including their image and object numbers.
    :param lil_matrix neighbor_list_array:
        Sparse matrix representing neighbor connections between bacteria.
    :param int unexpected_beginning_bac_idx:
        Index of the unexpected beginning bacterium being resolved.
    :param pandas.Series unexpected_beginning_bac:
        Series containing data for the unexpected beginning bacterium.
    :param float link_probability:
        Probability associated with the proposed link between the source and unexpected beginning bacterium.
        A valid link is established if `1 - link_probability > 0.5`.
    :param int source_bac_idx:
        Index of the source bacterium being linked to the unexpected beginning bacterium.
    :param str link_type:
        Type of link being evaluated:
            - 'division': Division link (source bacterium divides into two daughters, including the
              unexpected beginning bacterium).
            - 'continuity': Continuity link (unexpected beginning bacterium continues the life history of
              the source bacterium).
    :param pandas.DataFrame unexpected_begging_bac_time_step_bacteria:
        DataFrame containing all bacteria present in the time step of the unexpected beginning bacterium.
    :param pandas.DataFrame maintenance_cost:
        Cost DataFrame representing the cost of maintaining existing links for source bacteria.
    :param str parent_image_number_col:
        Column name in the DataFrame for parent image numbers.
    :param str parent_object_number_col:
        Column name in the DataFrame for parent object numbers.
    :param dict center_coord_col:
        Dictionary containing column names for x and y center coordinates.
    :param pandas.DataFrame redundant_link_division_df:
        DataFrame containing redundant links between candidate source bacteria and their current targets.
        These links are removed to allow the unexpected beginning bacterium to link to the candidate bacterium.

    :returns:
        pandas.DataFrame:

        Updated DataFrame with resolved links for the unexpected beginning bacterium.
    """

    if link_type == 'division':

        # be careful: maintenance cost should compare with 1 - cost value

        # probability is 1 - probability
        if 1 - link_probability > 0.5:

            # check if probability is higher than min prob of one of daughters

            if len(maintenance_cost.loc[source_bac_idx].dropna()) == 1:

                source_bac = df.loc[source_bac_idx]
                unexpected_beginning_bac_life_history = df.loc[df['id'] == unexpected_beginning_bac['id']]

                df = bacteria_modification(df, source_bac, unexpected_beginning_bac_life_history,
                                           unexpected_begging_bac_time_step_bacteria, neighbors_df,
                                           neighbor_list_array, parent_image_number_col, parent_object_number_col,
                                           center_coord_col)
            else:

                # it means source bacterium has two daughters

                redundant_daughter_idx = redundant_link_division_df.at[unexpected_beginning_bac_idx, source_bac_idx]

                # incorrect link (incorrect previous daughter)
                incorrect_daughter_life_history = df.loc[df['id'] == df.at[redundant_daughter_idx, 'id']]

                if incorrect_daughter_life_history.shape[0] > 0:
                    df = remove_redundant_link(df, incorrect_daughter_life_history, neighbors_df, neighbor_list_array,
                                               parent_image_number_col, parent_object_number_col, center_coord_col)

                # update info
                source_bac = df.loc[source_bac_idx]
                unexpected_beginning_bac_life_history = df.loc[df['id'] == Unexpected_Beginning_bac['id']]

                df = bacteria_modification(df, source_bac, unexpected_beginning_bac_life_history,
                                           unexpected_begging_bac_time_step_bacteria, neighbors_df,
                                           neighbor_list_array, parent_image_number_col, parent_object_number_col,
                                           center_coord_col)

    elif link_type == 'continuity':

        if 1 - link_probability > 0.5:
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
                                               parent_image_number_col, parent_object_number_col, center_coord_col)

            # update info
            source_bac = df.loc[source_bac_idx]
            unexpected_beginning_bac_life_history = df.loc[df['id'] == unexpected_beginning_bac['id']]

            df = bacteria_modification(df, source_bac, unexpected_beginning_bac_life_history,
                                       unexpected_begging_bac_time_step_bacteria, neighbors_df, neighbor_list_array,
                                       parent_image_number_col, parent_object_number_col, center_coord_col)

    else:
        raise ValueError("Link type is None.")

    return df


def handle_unexpected_beginning_bacteria(df, neighbors_df, neighbor_matrix, interval_time,
                                         doubling_time, parent_image_number_col, parent_object_number_col,
                                         center_coord_cols, division_vs_continuity_model,
                                         continuity_links_model, division_links_model, color_array, coordinate_array):
    """
    Handle unexpected beginning bacteria by finding and validating candidate sources, and creating new links.

    This function processes bacteria flagged with `unexpected_beginning` by identifying candidate target bacteria
    in the next frame. It validates potential links using statistical conditions, biological plausibility,
    and machine learning models. If validated, it creates new links to resolve tracking gaps.

    The function handles two scenarios:
        - **continuity**: The source bacterium's life history continues to the next frame.
        - **division**: The source bacterium divides, linking to two daughter bacteria.

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
    :param sklearn.Model division_vs_continuity_model:
        Machine learning model used to compare divided and non-divided states for bacteria.
    :param sklearn.Model continuity_links_model:
        Machine learning model used to evaluate candidate links for continuity events.
    :param sklearn.Model division_links_model:
        Machine learning model used to validate division links.
    :param np.ndarray color_array:
        Array representing the colors of objects in the tracking data. This is used for mapping
        bacteria from the dataframe to the coordinate array for spatial analysis.
    :param csr_matrix coordinate_array:
        Array of spatial coordinates used for evaluating candidate links.

    :return:
        pd.DataFrame

        Updated dataframe with resolved links for unexpected beginning bacteria.
    """

    # min life history of bacteria
    min_life_history_bac = int(np.round(doubling_time / interval_time))

    num_unexpected_beginning_bac = None
    prev_unexpected_beginning_bac = None
    n_iterate = 0

    while num_unexpected_beginning_bac != 0:

        unexpected_beginning_bacteria = df.loc[df["Unexpected_Beginning"]]

        if n_iterate > 0:
            if prev_unexpected_beginning_bac.values.all() == unexpected_beginning_bacteria.values.all():
                num_unexpected_beginning_bac = 0
            else:
                num_unexpected_beginning_bac = unexpected_beginning_bacteria.shape[0]

        prev_unexpected_beginning_bac = unexpected_beginning_bacteria

        for i, unexpected_beginning_bac_time_step in enumerate(unexpected_beginning_bacteria['ImageNumber'].unique()):

            # all bacteria in selected unexpected_beginning bacteria time step
            unexpected_begging_bac_time_step_bacteria = \
                df.loc[df['ImageNumber'] == unexpected_beginning_bac_time_step]

            # filter unexpected beginning bacteria features value
            sel_unexpected_beginning_bac = \
                unexpected_beginning_bacteria.loc[unexpected_beginning_bacteria['ImageNumber'] ==
                                                  unexpected_beginning_bac_time_step]

            # filter consider time step bacteria information
            previous_time_step_bacteria = df.loc[df['ImageNumber'] == (unexpected_beginning_bac_time_step - 1)]

            # optimized cost dataframe
            continuity_link_cost_df, division_cost_df, redundant_link_division_df, maintenance_cost_df = \
                compute_create_link_to_unexpected_beginning_cost(df, unexpected_begging_bac_time_step_bacteria,
                                                                 sel_unexpected_beginning_bac,
                                                                 previous_time_step_bacteria, neighbors_df,
                                                                 neighbor_matrix, min_life_history_bac,
                                                                 parent_image_number_col, parent_object_number_col,
                                                                 center_coord_cols, division_vs_continuity_model,
                                                                 continuity_links_model, division_links_model, color_array,
                                                                 coordinate_array)

            if division_cost_df.shape[0] > 0 and continuity_link_cost_df.shape[0] > 0:

                # Merge the dataframes using outer join to combine all columns and indices
                df_combined = pd.concat([division_cost_df, continuity_link_cost_df], axis=1, sort=False).fillna(1)

                # Create a third DataFrame to hold the max values for common rows and columns
                total_cost_df = pd.DataFrame(index=df_combined.index, columns=df_combined.columns.unique())

                # Iterate through the columns and fill the third DataFrame with max values

                for column in df_combined.columns.unique():
                    total_cost_df[column] = df_combined[[column]].min(axis=1)

                continue_flag = True

            elif division_cost_df.shape[0] > 0:

                total_cost_df = division_cost_df

                continue_flag = True

            elif continuity_link_cost_df.shape[0] > 0:

                total_cost_df = continuity_link_cost_df

                continue_flag = True
            else:
                continue_flag = False
                total_cost_df = None

            if continue_flag:

                total_cost_df = total_cost_df.fillna(1)
                optimized_df = optimize_assignment_using_hungarian(total_cost_df)

                # try to modify the links
                for row_index, row in optimized_df.iterrows():
                    source_bac_idx = row['idx1']

                    unexpected_beginning_bac_idx = int(row['idx2'])
                    unexpected_beginning_bac = df.loc[unexpected_beginning_bac_idx]

                    cost_val = row['cost']

                    find_stat = False
                    link_type = None

                    if unexpected_beginning_bac_idx in division_cost_df.columns.values.tolist() and \
                            source_bac_idx in division_cost_df.index.values.tolist():

                        if cost_val == division_cost_df.at[source_bac_idx, unexpected_beginning_bac_idx]:
                            link_type = 'division'
                            find_stat = True

                    if unexpected_beginning_bac_idx in continuity_link_cost_df.columns.values.tolist() and \
                            source_bac_idx in continuity_link_cost_df.index.values.tolist():

                        if cost_val == continuity_link_cost_df.at[source_bac_idx, unexpected_beginning_bac_idx]:
                            link_type = 'continuity'
                            find_stat = True

                    if not find_stat:
                        if cost_val == 1:
                            link_type = 'continuity'

                    df = resolve_unexpected_beginning_bacteria(df, neighbors_df, neighbor_matrix,
                                                               unexpected_beginning_bac_idx, unexpected_beginning_bac,
                                                               cost_val, source_bac_idx, link_type,
                                                               unexpected_begging_bac_time_step_bacteria,
                                                               maintenance_cost_df, parent_image_number_col,
                                                               parent_object_number_col, center_coord_cols,
                                                               redundant_link_division_df)

        n_iterate += 1

    return df
