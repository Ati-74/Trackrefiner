import pandas as pd
from Trackrefiner.correction.action.trackingCost.calculateInitialCostTerms import calc_distance_and_overlap_matrices


def find_candidates_for_unexpected_beginning(
        neighbors_df, unexpected_begging_bac, unexpected_beginning_bac_time_step_bacteria, previous_time_step_bacteria,
        center_coord_cols, color_array, coordinate_array):
    """
    Finds candidate bacteria from the previous time step to link with unexpected beginning bacteria.

    :param pandas.DataFrame neighbors_df:
        DataFrame containing neighboring relationships between bacteria, including their image and object numbers.
    :param pandas.DataFrame unexpected_begging_bac:
        DataFrame containing information about unexpected beginning bacteria.
    :param pandas.DataFrame unexpected_beginning_bac_time_step_bacteria:
        DataFrame containing all bacteria in the time step of the unexpected beginning bacteria.
    :param pandas.DataFrame previous_time_step_bacteria:
        DataFrame containing all bacteria from the previous time step.
    :param dict center_coord_cols:
        Dictionary mapping x and y center coordinates to their respective column names.
    :param numpy.ndarray color_array:
        Array storing color-coded information for bacterial objects.
    :param csr_matrix coordinate_array:
        Sparse boolean array representing encoded spatial coordinates of bacteria.

    :returns:
        tuple:
            - **final_candidate_bacteria_info** (*pandas.DataFrame*): Subset of bacteria from the previous time step
              that can potentially link to unexpected beginning bacteria and are selected for further analysis.
            - **final_candidate_bacteria** (*pandas.DataFrame*): DataFrame containing detailed information about
              candidate bacteria and their neighbors, combined with unexpected beginning bacteria data.
    """

    overlap_df, distance_df = calc_distance_and_overlap_matrices(unexpected_beginning_bac_time_step_bacteria,
                                                                 unexpected_begging_bac, previous_time_step_bacteria,
                                                                 previous_time_step_bacteria, center_coord_cols,
                                                                 color_array=color_array,
                                                                 coordinate_array=coordinate_array)

    # remove columns with all values equal to zero
    overlap_df_replaced = overlap_df[(overlap_df != 0).any(axis=1)]
    max_overlap_bac_idx = overlap_df_replaced.idxmax(axis=1)
    min_distance_bac_ndx = distance_df.idxmin(axis=1)

    df_max_overlap_bac_idx = max_overlap_bac_idx.reset_index()
    df_min_distance_bac_ndx = min_distance_bac_ndx.reset_index()

    df_max_overlap_bac_idx.columns = ['unexpected beginning bac idx', 'candidate bac idx']
    df_min_distance_bac_ndx.columns = ['unexpected beginning bac idx', 'candidate bac idx']

    df_primary_candidate_bac_ndx = pd.concat([df_max_overlap_bac_idx, df_min_distance_bac_ndx],
                                             ignore_index=True).drop_duplicates()

    df_unexpected_bac_indo_with_candidates = df_primary_candidate_bac_ndx.merge(unexpected_begging_bac,
                                                                                left_on='unexpected beginning bac idx',
                                                                                right_on='index', how='inner')

    df_primary_candidate_bac_ndx_info = df_unexpected_bac_indo_with_candidates.merge(previous_time_step_bacteria,
                                                                                     left_on='candidate bac idx',
                                                                                     right_on='index', how='inner',
                                                                                     suffixes=('', '_candidate'))

    df_primary_candidate_bac_ndx_info_neighbors = \
        df_primary_candidate_bac_ndx_info.merge(neighbors_df,
                                                left_on=['ImageNumber_candidate', 'ObjectNumber_candidate'],
                                                right_on=['First Image Number', 'First Object Number'], how='left')

    df_primary_candidate_bac_ndx_info_neighbors_info = \
        df_primary_candidate_bac_ndx_info_neighbors.merge(previous_time_step_bacteria,
                                                          left_on=['Second Image Number', 'Second Object Number'],
                                                          right_on=['ImageNumber', 'ObjectNumber'], how='left',
                                                          suffixes=('', '_candidate_neighbors'))

    raw_cols = previous_time_step_bacteria.columns.tolist()
    candidate_columns = [v + '_candidate' for v in raw_cols]
    neighbor_candidate_columns = [v + '_candidate_neighbors' for v in raw_cols]

    candidate_df_cols = raw_cols.copy()
    candidate_df_cols.extend(candidate_columns)

    neighbor_candidate_df_cols = raw_cols.copy()
    neighbor_candidate_df_cols.extend(neighbor_candidate_columns)

    rename_neighbor_candidate_df_cols_dict = {}

    for i, col in enumerate(neighbor_candidate_columns):
        rename_neighbor_candidate_df_cols_dict[col] = raw_cols[i] + '_candidate'

    # Create DataFrames for the two groups of columns to be combined
    df_unexpected_beginning_with_can_bac = df_primary_candidate_bac_ndx_info_neighbors_info[candidate_df_cols]

    df_unexpected_beginning_with_can_neighbor_bac = \
        df_primary_candidate_bac_ndx_info_neighbors_info[neighbor_candidate_df_cols].rename(
            columns=rename_neighbor_candidate_df_cols_dict)

    # Concatenate the DataFrames
    final_candidate_bacteria = pd.concat([df_unexpected_beginning_with_can_bac,
                                          df_unexpected_beginning_with_can_neighbor_bac]).sort_index(
        kind='merge').reset_index(drop=True)

    # we don't check unexpected end bacteria in this phase
    final_candidate_bacteria = final_candidate_bacteria.loc[~ final_candidate_bacteria['ImageNumber_candidate'].isna()]

    final_candidate_bacteria = final_candidate_bacteria.loc[final_candidate_bacteria['unexpected_end_candidate']
                                                            == False]

    final_candidate_bacteria[['index', 'index_candidate']] = final_candidate_bacteria[['index',
                                                                                       'index_candidate']].astype(int)

    final_candidate_bacteria = final_candidate_bacteria.drop_duplicates(subset=['index', 'index_candidate'],
                                                                        keep='first').reset_index(drop=True)

    final_candidate_bacteria_info = previous_time_step_bacteria.loc[
        final_candidate_bacteria['index_candidate'].unique()]

    return final_candidate_bacteria_info, final_candidate_bacteria


def find_candidates_for_unexpected_end(neighbors_df, unexpected_end_bac, unexpected_end_bac_time_step_bacteria,
                                       next_time_step_bacteria, center_coord_cols, color_array, coordinate_array):
    """
    Finds candidate bacteria from the next time step to link with unexpected end bacteria.

    :param pandas.DataFrame neighbors_df:
        DataFrame containing neighboring relationships between bacteria, including their image and object numbers.
    :param pandas.DataFrame unexpected_end_bac:
        DataFrame containing information about unexpected end bacteria.
    :param pandas.DataFrame unexpected_end_bac_time_step_bacteria:
        DataFrame containing all bacteria in the time step of the unexpected end bacteria.
    :param pandas.DataFrame next_time_step_bacteria:
        DataFrame containing all bacteria from the next time step.
    :param dict center_coord_cols:
        Dictionary mapping x and y center coordinates to their respective column names.
    :param numpy.ndarray color_array:
        Array storing color-coded information for bacterial objects.
    :param csr_matrix coordinate_array:
        Sparse boolean array representing encoded spatial coordinates of bacteria.

    :returns:
        tuple:
            - **final_candidate_bacteria_info** (*pandas.DataFrame*): Subset of bacteria from the previous time step
              that can potentially link to unexpected end bacteria and are selected for further analysis.
            - **final_candidate_bacteria** (*pandas.DataFrame*): DataFrame containing detailed information about
              candidate bacteria and their neighbors, combined with unexpected end bacteria data.
    """

    overlap_df, distance_df = calc_distance_and_overlap_matrices(unexpected_end_bac_time_step_bacteria,
                                                                 unexpected_end_bac, next_time_step_bacteria,
                                                                 next_time_step_bacteria, center_coord_cols,
                                                                 color_array=color_array,
                                                                 coordinate_array=coordinate_array)

    # remove columns with all values equal to zero
    overlap_df_replaced = overlap_df[(overlap_df != 0).any(axis=1)]
    max_overlap_bac_idx = overlap_df_replaced.idxmax(axis=1)
    min_distance_bac_ndx = distance_df.idxmin(axis=1)

    df_max_overlap_bac_idx = max_overlap_bac_idx.reset_index()
    df_min_distance_bac_ndx = min_distance_bac_ndx.reset_index()

    df_max_overlap_bac_idx.columns = ['unexpected end bac idx', 'candidate bac idx']
    df_min_distance_bac_ndx.columns = ['unexpected end bac idx', 'candidate bac idx']

    df_primary_candidate_bac_ndx = pd.concat([df_max_overlap_bac_idx, df_min_distance_bac_ndx],
                                             ignore_index=True).drop_duplicates()

    df_unexpected_bac_info_with_candidates = \
        df_primary_candidate_bac_ndx.merge(unexpected_end_bac,
                                           left_on='unexpected end bac idx',
                                           right_on='index', how='inner')

    df_primary_candidate_bac_ndx_info = \
        df_unexpected_bac_info_with_candidates.merge(next_time_step_bacteria,
                                                     left_on='candidate bac idx',
                                                     right_on='index', how='inner',
                                                     suffixes=('', '_candidate'))

    df_primary_candidate_bac_ndx_info_neighbors = \
        df_primary_candidate_bac_ndx_info.merge(neighbors_df,
                                                left_on=['ImageNumber_candidate', 'ObjectNumber_candidate'],
                                                right_on=['First Image Number', 'First Object Number'], how='left')

    df_primary_candidate_bac_ndx_info_neighbors_info = \
        df_primary_candidate_bac_ndx_info_neighbors.merge(next_time_step_bacteria,
                                                          left_on=['Second Image Number', 'Second Object Number'],
                                                          right_on=['ImageNumber', 'ObjectNumber'], how='left',
                                                          suffixes=('', '_candidate_neighbors'))

    raw_cols = next_time_step_bacteria.columns.tolist()
    candidate_columns = [v + '_candidate' for v in raw_cols]
    neighbor_candidate_columns = [v + '_candidate_neighbors' for v in raw_cols]

    candidate_df_cols = raw_cols.copy()
    candidate_df_cols.extend(candidate_columns)

    neighbor_candidate_df_cols = raw_cols.copy()
    neighbor_candidate_df_cols.extend(neighbor_candidate_columns)

    rename_neighbor_candidate_df_cols_dict = {}

    for i, col in enumerate(neighbor_candidate_columns):
        rename_neighbor_candidate_df_cols_dict[col] = raw_cols[i] + '_candidate'

    # Create DataFrames for the two groups of columns to be combined
    df_unexpected_beginning_with_can_bac = df_primary_candidate_bac_ndx_info_neighbors_info[candidate_df_cols]

    df_unexpected_beginning_with_can_neighbor_bac = \
        df_primary_candidate_bac_ndx_info_neighbors_info[neighbor_candidate_df_cols].rename(
            columns=rename_neighbor_candidate_df_cols_dict)

    # Concatenate the DataFrames
    final_candidate_bacteria = pd.concat([df_unexpected_beginning_with_can_bac,
                                          df_unexpected_beginning_with_can_neighbor_bac]).sort_index(
        kind='merge').reset_index(drop=True)

    # remove Nas because we used left join for neighbors
    final_candidate_bacteria = final_candidate_bacteria.dropna(subset=['index', 'index_candidate'])

    final_candidate_bacteria[['index', 'index_candidate']] = \
        final_candidate_bacteria[['index', 'index_candidate']].astype(int)

    final_candidate_bacteria = final_candidate_bacteria.drop_duplicates(subset=['index', 'index_candidate'],
                                                                        keep='first').reset_index(drop=True)

    final_candidate_bacteria_df = next_time_step_bacteria.loc[final_candidate_bacteria['index_candidate'].unique()]

    return final_candidate_bacteria_df, final_candidate_bacteria
