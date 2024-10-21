import numpy as np
from itertools import chain


def neighbor_checking(df, neighbor_list_array, parent_image_number_col, parent_object_number_col,
                      selected_rows_df=None, selected_time_step_df=None, return_common_elements=True, col_target='',
                      index2=True, sig=False):
    if selected_rows_df is not None:

        # 'prev_time_step_NeighborIndexList' + col_target
        # 'NeighborIndexList' + col_target
        if index2:
            important_info_list = ['ImageNumber' + col_target, 'ObjectNumber' + col_target,
                                   'daughter_length_to_mother' + col_target, 'index' + col_target,
                                   'index2' + col_target,
                                   'prev_time_step_index' + col_target, 'unexpected_beginning' + col_target,
                                   'unexpected_end' + col_target, parent_image_number_col + col_target,
                                   parent_object_number_col + col_target, 'id' + col_target, 'parent_id' + col_target,
                                   'other_daughter_index' + col_target]

            selected_rows_df = selected_rows_df[important_info_list]

            # 'NeighborIndexList' + col_target
            # 'prev_time_step_NeighborIndexList' + col_target
            bacteria_info_dict = selected_rows_df[['index' + col_target, 'index2' + col_target, 'id' + col_target,
                                                   'parent_id' + col_target, 'prev_time_step_index' + col_target,
                                                   'unexpected_beginning' + col_target,
                                                   'unexpected_end' + col_target, 'other_daughter_index' + col_target,
                                                   parent_image_number_col + col_target]].to_dict(orient='index')
        else:
            important_info_list = ['ImageNumber' + col_target, 'ObjectNumber' + col_target,
                                   'daughter_length_to_mother' + col_target, 'index' + col_target,
                                   'prev_time_step_index' + col_target, 'unexpected_beginning' + col_target,
                                   'unexpected_end' + col_target, parent_image_number_col + col_target,
                                   parent_object_number_col + col_target, 'id' + col_target, 'parent_id' + col_target,
                                   'other_daughter_index' + col_target]

            selected_rows_df = selected_rows_df[important_info_list]

            # 'NeighborIndexList' + col_target
            # 'prev_time_step_NeighborIndexList' + col_target
            bacteria_info_dict = selected_rows_df[['index' + col_target, 'id' + col_target,
                                                   'parent_id' + col_target, 'prev_time_step_index' + col_target,
                                                   'unexpected_beginning' + col_target,
                                                   'unexpected_end' + col_target, 'other_daughter_index' + col_target,
                                                   parent_image_number_col + col_target]].to_dict(orient='index')

    else:

        # it means: division doesn't occurs
        cond1_1 = df['daughter_length_to_mother'].isna()

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
                                 'unexpected_beginning' + col_target, 'unexpected_end' + col_target,
                                 'other_daughter_index' + col_target,
                                 'prev_time_step_index' + col_target,
                                 parent_image_number_col + col_target]].to_dict(orient='index')

    if selected_time_step_df is not None:

        # 'NeighborIndexList'
        # 'prev_time_step_NeighborIndexList'
        ref_bacteria_info_dict = \
            selected_time_step_df[['index', 'id', 'parent_id', 'unexpected_beginning',
                                   'unexpected_end', 'other_daughter_index', 'prev_time_step_index',
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
                         ref_bacteria_info_dict[v]['unexpected_end'] == False]

                else:
                    prev_time_step_neighbor_id_list = []
            else:
                prev_time_step_neighbor_id_list = []

            neighbor_index_list = neighbor_list_array.rows[sel_bac['index' + col_target]]

            if neighbor_index_list:

                neighbor_index_list = [v for v in neighbor_index_list if v in ref_bacteria_info_dict_keys]

                neighbor_id_and_parent_id_list = \
                    [(ref_bacteria_info_dict[v]['id'], ref_bacteria_info_dict[v]['parent_id'])
                     for v in neighbor_index_list if ref_bacteria_info_dict[v]['unexpected_beginning'] == False]

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

                """
                if sig:
                    if sel_bac['index'] == 26:
                        print(sel_bac)
                        print(prev_time_step_neighbor_index_list)
                        print(prev_time_step_neighbor_id_list)
                        print(neighbor_index_list)
                        print(neighbor_id_list)
                        print(diff_id_list)
                        print(common_id_list)
                        print(all_ids)
                        print(sel_bac['parent_id' + col_target])
                        breakpoint()
                """

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

        df.loc[idx, "difference_neighbors" + col_target] = diff_neighbor_list

        if return_common_elements:
            df.loc[idx, 'common_neighbors' + col_target] = common_neighbor_list
    else:

        df["difference_neighbors" + col_target] = diff_neighbor_list
        if return_common_elements:
            df["common_neighbors" + col_target] = common_neighbor_list

    return df


def check_num_neighbors(df, neighbor_list_array, bac1, bac2, parent_image_number_col, return_common_elements=False):
    # Note: bac2 is after bac1 and there is no relation between them (we want to check can we make a relation?)

    prev_time_step_neighbor_index_list = neighbor_list_array.rows[bac1['index']]

    if prev_time_step_neighbor_index_list:
        prev_time_step_neighbor_df = df.loc[prev_time_step_neighbor_index_list]
        prev_time_step_neighbor_id_list = \
            prev_time_step_neighbor_df.loc[~ prev_time_step_neighbor_df['unexpected_end']]['id'].values
    else:
        prev_time_step_neighbor_id_list = []

    neighbor_index_list = neighbor_list_array.rows[bac2['index']]

    if neighbor_index_list:

        neighbor_bac_df = df.loc[neighbor_index_list]
        neighbor_id_and_parent_id_list = neighbor_bac_df.loc[~ neighbor_bac_df['unexpected_beginning']][
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


def check_num_neighbors_batch(df, neighbor_list_array, bac1, bac2_batch, parent_image_number_col,
                              return_common_elements=False):
    # bac2 is after bac1
    # Note: bac2 is after bac1 and there is no relation between them (we want to check can we make a relation?)

    diff_neighbor_list = []
    common_neighbor_list = []

    prev_time_step_neighbor_index_list = neighbor_list_array.rows[bac1['index']]

    if prev_time_step_neighbor_index_list:

        prev_time_step_neighbor_df = df.loc[prev_time_step_neighbor_index_list]
        prev_time_step_neighbor_id_list = \
            prev_time_step_neighbor_df.loc[~ prev_time_step_neighbor_df['unexpected_end']]['id'].values
    else:
        prev_time_step_neighbor_id_list = []

    for bac2_ndx, bac2 in bac2_batch.iterrows():
        neighbor_index_list = neighbor_list_array.rows[bac2['index']]

        if neighbor_index_list:

            neighbor_bac_df = df.loc[neighbor_index_list]
            neighbor_id_and_parent_id_list = neighbor_bac_df.loc[~ neighbor_bac_df['unexpected_beginning']][
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
