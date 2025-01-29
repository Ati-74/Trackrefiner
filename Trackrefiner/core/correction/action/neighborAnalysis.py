import numpy as np
from itertools import chain


def compare_neighbor_sets(df, neighbor_list_array, parent_image_number_col, parent_object_number_col,
                          selected_rows_df=None, selected_time_step_df=None, return_common_elements=True, col_target='',
                          index2=True):

    """
    Compares neighbor sets of source and target bacteria for each link to calculate differences and commonalities.

    **This function determines**:
        - How many neighbors are different between the source and target bacteria.
        - How many neighbors are shared (common) between the source and target bacteria.

    The results are stored as new columns in the DataFrame (`difference_neighbors` and `common_neighbors`).

    **Key Operations**:
        1. **Neighbor Extraction**:
           - Extracts the neighbor sets for each bacterium (source and target) from the `neighbor_list_array`.
           - Accounts for neighbors in the current and previous time steps.

        2. **Comparison**:
           - Calculates the difference in neighbors (i.e., neighbors present in one bacterium but not in the other).
           - Identifies common neighbors (i.e., neighbors shared between source and target).

        3. **Handling Special Cases**:
           - Accounts for bacteria with unexpected beginnings or ends.
           - Ensures proper handling of bacteria flagged with parent-child relationships.

        4. **Updating DataFrame**:
            - Updates `df` with calculated values for:
                - `difference_neighbors_<col_target>`: Number of differing neighbors between source and target.
                - `common_neighbors_<col_target>`: Number of shared neighbors.

    :param pandas.DataFrame df:
        The main DataFrame containing bacterial data, including neighbor information and flags.
    :param scipy.sparse.lil_matrix neighbor_list_array:
        A sparse matrix representing neighbor relationships, where rows correspond to bacteria and columns
        represent their neighbors.
    :param str parent_image_number_col:
        The column name in `df` representing the parent bacterium's image number.
    :param str parent_object_number_col:
        The column name in `df` representing the parent bacterium's object number.
    :param pandas.DataFrame selected_rows_df:
        A subset of rows from `df` to analyze specific bacteria and their neighbors. If `None`, the entire
        DataFrame is considered. Default is `None`.
    :param pandas.DataFrame selected_time_step_df:
        A subset of rows representing bacteria from a specific time step for reference. If `None`, the entire
        DataFrame is used as the reference. Default is `None`.
    :param bool return_common_elements:
        Whether to calculate and return the number of shared neighbors between source and target bacteria.
        Default is `True`.
    :param str col_target:
        A suffix added to column names to differentiate data for specific bacteria subsets. Default is `''`.
    :param bool index2:
        Whether to use a secondary index (`index2`) for calculations. Default is `True`.

    :returns:
        pandas.DataFrame: Updated DataFrame with columns:
            - `difference_neighbors_<col_target>`: Number of neighbors differing between source and target bacteria.
            - `common_neighbors_<col_target>`: Number of shared neighbors.
    """

    if selected_rows_df is not None:

        # 'prev_time_step_NeighborIndexList' + col_target
        # 'NeighborIndexList' + col_target
        if index2:
            important_info_list = ['ImageNumber' + col_target, 'ObjectNumber' + col_target,
                                   'Total_Daughter_Mother_Length_Ratio' + col_target, 'index' + col_target,
                                   'index2' + col_target,
                                   'prev_time_step_index' + col_target, 'Unexpected_Beginning' + col_target,
                                   'Unexpected_End' + col_target, parent_image_number_col + col_target,
                                   parent_object_number_col + col_target, 'id' + col_target, 'parent_id' + col_target,
                                   'other_daughter_index' + col_target]

            selected_rows_df = selected_rows_df[important_info_list]

            # 'NeighborIndexList' + col_target
            # 'prev_time_step_NeighborIndexList' + col_target
            bacteria_info_dict = selected_rows_df[['index' + col_target, 'index2' + col_target, 'id' + col_target,
                                                   'parent_id' + col_target, 'prev_time_step_index' + col_target,
                                                   'Unexpected_Beginning' + col_target,
                                                   'Unexpected_End' + col_target, 'other_daughter_index' + col_target,
                                                   parent_image_number_col + col_target]].to_dict(orient='index')
        else:
            important_info_list = ['ImageNumber' + col_target, 'ObjectNumber' + col_target,
                                   'Total_Daughter_Mother_Length_Ratio' + col_target, 'index' + col_target,
                                   'prev_time_step_index' + col_target, 'Unexpected_Beginning' + col_target,
                                   'Unexpected_End' + col_target, parent_image_number_col + col_target,
                                   parent_object_number_col + col_target, 'id' + col_target, 'parent_id' + col_target,
                                   'other_daughter_index' + col_target]

            selected_rows_df = selected_rows_df[important_info_list]

            # 'NeighborIndexList' + col_target
            # 'prev_time_step_NeighborIndexList' + col_target
            bacteria_info_dict = selected_rows_df[['index' + col_target, 'id' + col_target,
                                                   'parent_id' + col_target, 'prev_time_step_index' + col_target,
                                                   'Unexpected_Beginning' + col_target,
                                                   'Unexpected_End' + col_target, 'other_daughter_index' + col_target,
                                                   parent_image_number_col + col_target]].to_dict(orient='index')

    else:

        # it means: division doesn't occurs
        cond1_1 = df['Total_Daughter_Mother_Length_Ratio'].isna()

        # finding ids of daughters for each mother
        # 'ImageNumber', 'ObjectNumber' --> for last time step of mother
        # parent_image_number_col, parent_object_number_col --> daughters
        # finding mother and daughters
        df_mother_daughters = \
            df[~ cond1_1].merge(df, left_on=['ImageNumber', 'ObjectNumber'],
                                right_on=[parent_image_number_col, parent_object_number_col], how='inner',
                                suffixes=('', '_daughter'))

        df['prev_time_step_index'] = df.groupby('id')['index'].shift(1).fillna(-1).astype('int64')

        # for daughters its equal to mother index
        df.loc[df_mother_daughters['index_daughter'].values, 'prev_time_step_index'] = \
            df_mother_daughters['index'].values.astype('int64')

        # 'NeighborIndexList' + col_target
        # 'prev_time_step_NeighborIndexList' + col_target
        bacteria_info_dict = df[['index' + col_target, 'id' + col_target, 'parent_id' + col_target,
                                 'Unexpected_Beginning' + col_target, 'Unexpected_End' + col_target,
                                 'other_daughter_index' + col_target,
                                 'prev_time_step_index' + col_target,
                                 parent_image_number_col + col_target]].to_dict(orient='index')

    if selected_time_step_df is not None:

        # 'NeighborIndexList'
        # 'prev_time_step_NeighborIndexList'
        ref_bacteria_info_dict = \
            selected_time_step_df[['index', 'id', 'parent_id', 'Unexpected_Beginning',
                                   'Unexpected_End', 'other_daughter_index', 'prev_time_step_index',
                                   parent_image_number_col]].to_dict(orient='index')
    else:

        ref_bacteria_info_dict = bacteria_info_dict

    ref_bacteria_info_dict_keys = ref_bacteria_info_dict.keys()

    diff_neighbor_list = []
    common_neighbor_list = []

    for row_idx in bacteria_info_dict:

        sel_bac = bacteria_info_dict[row_idx]

        if sel_bac[parent_image_number_col + col_target] != 0:

            # prev_time_step_NeighborIndexList
            # NAN = -1
            if sel_bac['prev_time_step_index' + col_target] != -1:

                prev_time_step_neighbor_index_list = \
                    list(chain.from_iterable((
                        neighbor_list_array.rows[[sel_bac['prev_time_step_index' + col_target]]]
                    )))

                if prev_time_step_neighbor_index_list:

                    prev_time_step_neighbor_index_list = \
                        [v for v in prev_time_step_neighbor_index_list if v in ref_bacteria_info_dict_keys]

                    prev_time_step_neighbor_id_list = \
                        [ref_bacteria_info_dict[v]['id'] for v in prev_time_step_neighbor_index_list if
                         ref_bacteria_info_dict[v]['Unexpected_End'] == False]

                else:
                    prev_time_step_neighbor_id_list = []
            else:
                prev_time_step_neighbor_id_list = []

            neighbor_index_list = neighbor_list_array.rows[sel_bac['index' + col_target]]

            if neighbor_index_list:

                neighbor_index_list = [v for v in neighbor_index_list if v in ref_bacteria_info_dict_keys]

                neighbor_id_and_parent_id_list = \
                    [(ref_bacteria_info_dict[v]['id'], ref_bacteria_info_dict[v]['parent_id'])
                     for v in neighbor_index_list if ref_bacteria_info_dict[v]['Unexpected_Beginning'] == False]

                neighbor_id_list = [(v[1] if v[1] in prev_time_step_neighbor_id_list else v[0]) for v
                                    in neighbor_id_and_parent_id_list]

            else:
                neighbor_id_list = []

            # other daughter

            if neighbor_id_list and prev_time_step_neighbor_id_list:

                all_ids = []
                all_ids.extend(neighbor_id_list)
                all_ids.extend(prev_time_step_neighbor_id_list)

                diff_id_list = [v for v in all_ids if all_ids.count(v) < 2 and
                                v != sel_bac['parent_id' + col_target]]

                common_id_list = np.unique([v for v in all_ids if all_ids.count(v) >= 2])

                diff_neighbor_list.append(len(diff_id_list))
                common_neighbor_list.append(len(common_id_list))

            elif neighbor_id_list:

                diff_id_list = [v for v in neighbor_id_list if v != sel_bac['parent_id' + col_target]]

                common_id_list = [v for v in neighbor_id_list if v == sel_bac['parent_id' + col_target]]

                diff_neighbor_list.append(len(diff_id_list))
                common_neighbor_list.append(len(common_id_list))

            elif prev_time_step_neighbor_id_list:

                diff_neighbor_list.append(0)
                common_neighbor_list.append(0)

            else:
                # len(prev_time_step_neighbor_id_list) == 0 and len(neighbor_id_list) == 0
                diff_neighbor_list.append(0)
                common_neighbor_list.append(0)

        else:
            # For unexpected beginning bacteria, we don't calculate neighbour changes
            diff_neighbor_list.append(0)
            common_neighbor_list.append(0)

    # now update difference list
    if selected_rows_df is not None:

        if index2:
            idx = selected_rows_df['index2' + col_target].values
        else:
            idx = selected_rows_df['index' + col_target].values

        df.loc[idx, "Neighbor_Difference_Count" + col_target] = diff_neighbor_list

        if return_common_elements:
            df.loc[idx, 'Neighbor_Shared_Count' + col_target] = common_neighbor_list
    else:

        df["Neighbor_Difference_Count" + col_target] = diff_neighbor_list
        if return_common_elements:
            df["Neighbor_Shared_Count" + col_target] = common_neighbor_list

    return df


def compare_neighbors_single_pair(df, neighbor_list_array, bac1, bac2, return_common_elements=False):

    """
    Compares the neighbor sets of two bacteria (bac1 and bac2) to calculate the number of differing
    and shared neighbors.

    **Key Operations**:
        - Extracts the neighbor indices for `bac1` and `bac2` from `neighbor_list_array`.
        - Filters neighbors based on conditions like unexpected beginnings and ends.
        - Compares the neighbor sets to compute the differences and commonalities.

    :param pandas.DataFrame df:
        The main DataFrame containing bacterial data.
    :param scipy.sparse.lil_matrix neighbor_list_array:
        Sparse matrix representing neighbor relationships for bacteria.
    :param pandas.Series bac1:
        Data for the first bacterium (source) in the relationship.
    :param pandas.Series bac2:
        Data for the second bacterium (target) in the relationship.
    :param bool return_common_elements:
        If `True`, returns the number of shared neighbors in addition to differing neighbors.
        Default is `False`.

    :returns:
        tuple:
            - `diff_neighbor` (int): Number of differing neighbors.
            - `common_neighbor` (int, optional): Number of shared neighbors if `return_common_elements=True`.
    """

    # Note: bac2 is after bac1 and there is no relation between them (we want to check can we make a relation?)

    prev_time_step_neighbor_index_list = neighbor_list_array.rows[bac1['index']]

    if prev_time_step_neighbor_index_list:
        prev_time_step_neighbor_df = df.loc[prev_time_step_neighbor_index_list]
        prev_time_step_neighbor_id_list = \
            prev_time_step_neighbor_df.loc[~ prev_time_step_neighbor_df['Unexpected_End']]['id'].values
    else:
        prev_time_step_neighbor_id_list = []

    neighbor_index_list = neighbor_list_array.rows[bac2['index']]

    if neighbor_index_list:

        neighbor_bac_df = df.loc[neighbor_index_list]
        neighbor_id_and_parent_id_list = neighbor_bac_df.loc[~ neighbor_bac_df['Unexpected_Beginning']][
            ['id', 'parent_id']].values
        neighbor_id_list = [(v[1] if v[1] in prev_time_step_neighbor_id_list else v[0]) for v
                            in neighbor_id_and_parent_id_list]

    else:
        neighbor_id_list = []

    if neighbor_id_list and prev_time_step_neighbor_id_list:

        all_ids = []
        all_ids.extend(neighbor_id_list)
        all_ids.extend(prev_time_step_neighbor_id_list)

        diff_id_list = [v for v in all_ids if all_ids.count(v) < 2]
        common_id_list = np.unique([v for v in all_ids if all_ids.count(v) >= 2])

        diff_neighbor = len(diff_id_list)
        common_neighbor = len(common_id_list)

    elif neighbor_id_list:

        diff_id_list = [v for v in neighbor_id_list if v != bac2['parent_id']]
        common_id_list = [v for v in neighbor_id_list if v == bac2['parent_id']]

        diff_neighbor = len(diff_id_list)
        common_neighbor = len(common_id_list)

    elif prev_time_step_neighbor_id_list:

        diff_neighbor = 0
        common_neighbor = 0

    else:
        # len(prev_time_step_neighbor_id_list) == 0 and len(neighbor_id_list) == 0
        diff_neighbor = 0
        common_neighbor = 0

    if not return_common_elements:
        return diff_neighbor
    else:
        return diff_neighbor, common_neighbor


def compare_neighbors_batch(df, neighbor_list_array, bac1, bac2_batch, return_common_elements=False):

    """
    Compares the neighbor sets of a single bacterium (`bac1`) with multiple bacteria (`bac2_batch`)
    (candidate target bacteria) to calculate the number of differing and shared neighbors for each pair.

    **Key Operations**:
        - Extracts the neighbor indices for `bac1` and all bacteria in `bac2_batch` from `neighbor_list_array`.
        - Filters neighbors based on conditions like unexpected beginnings and ends.
        - Compares neighbor sets for each pair to compute the differences and commonalities.

    :param pandas.DataFrame df:
        The main DataFrame containing bacterial data.
    :param scipy.sparse.lil_matrix neighbor_list_array:
        Sparse matrix representing neighbor relationships for bacteria.
    :param pandas.Series bac1:
        Data for the first bacterium (source) in the relationship.
    :param pandas.DataFrame bac2_batch:
        Data for multiple target bacteria in the relationship.
    :param bool return_common_elements:
        If `True`, returns the number of shared neighbors in addition to differing neighbors.
        Default is `False`.

    :returns:
        tuple:
            - `diff_neighbor_list` (np.ndarray): Array of differing neighbor counts for each pair.
            - `common_neighbor_list` (np.ndarray, optional): Array of shared neighbor counts for each pair if
              `return_common_elements=True`.
    """

    # bac2 is after bac1
    # Note: bac2 is after bac1 and there is no relation between them (we want to check can we make a relation?)

    diff_neighbor_list = []
    common_neighbor_list = []

    prev_time_step_neighbor_index_list = neighbor_list_array.rows[bac1['index']]

    if prev_time_step_neighbor_index_list:

        prev_time_step_neighbor_df = df.loc[prev_time_step_neighbor_index_list]
        prev_time_step_neighbor_id_list = \
            prev_time_step_neighbor_df.loc[~ prev_time_step_neighbor_df['Unexpected_End']]['id'].values
    else:
        prev_time_step_neighbor_id_list = []

    for bac2_ndx, bac2 in bac2_batch.iterrows():
        neighbor_index_list = neighbor_list_array.rows[bac2['index']]

        if neighbor_index_list:

            neighbor_bac_df = df.loc[neighbor_index_list]
            neighbor_id_and_parent_id_list = neighbor_bac_df.loc[~ neighbor_bac_df['Unexpected_Beginning']][
                ['id', 'parent_id']].values
            neighbor_id_list = [(v[1] if v[1] in prev_time_step_neighbor_id_list else v[0]) for v
                                in neighbor_id_and_parent_id_list]

        else:
            neighbor_id_list = []

        if neighbor_id_list and prev_time_step_neighbor_id_list:
            all_ids = []
            all_ids.extend(neighbor_id_list)
            all_ids.extend(prev_time_step_neighbor_id_list)

            diff_id_list = [v for v in all_ids if all_ids.count(v) < 2]
            common_id_list = np.unique([v for v in all_ids if all_ids.count(v) >= 2])

            diff_neighbor_list.append(len(diff_id_list))
            common_neighbor_list.append(len(common_id_list))

        elif neighbor_id_list:

            diff_id_list = [v for v in neighbor_id_list if v != bac2['parent_id']]
            common_id_list = [v for v in neighbor_id_list if v == bac2['parent_id']]

            diff_neighbor_list.append(len(diff_id_list))
            common_neighbor_list.append(len(common_id_list))

        elif prev_time_step_neighbor_id_list:

            diff_neighbor_list.append(0)
            common_neighbor_list.append(0)
        else:
            # len(prev_time_step_neighbor_id_list) == 0 and len(neighbor_id_list) == 0
            diff_neighbor_list.append(0)
            common_neighbor_list.append(0)

    if not return_common_elements:
        return np.array(diff_neighbor_list, dtype=float)
    else:
        return np.array(diff_neighbor_list, dtype=float), np.array(common_neighbor_list, dtype=float)
