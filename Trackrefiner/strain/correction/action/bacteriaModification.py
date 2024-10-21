import numpy as np
from Trackrefiner.strain.correction.action.helperFunctions import (calculate_trajectory_direction_daughters_mother,
                                                                   adding_features_to_continues_life_history,
                                                                   calculate_orientation_angle_batch)
from Trackrefiner.strain.correction.neighborChecking import neighbor_checking


def calc_modified_features(df, selected_rows_df, neighbor_df, neighbor_list_array, center_coordinate_columns,
                           parent_image_number_col, parent_object_number_col):
    """
    goal: assign new features like: `id`, `divideFlag`, `daughters_index`, `bad_division_flag`,
    `unexpected_end`, `division_time`, `unexpected_beginning`, `LifeHistory`, `parent_id` to bacteria and find errors

    @param df dataframe bacteria features value
    """

    first_time_step_selected_rows = selected_rows_df['ImageNumber'].min()
    last_time_step_selected_rows = selected_rows_df['ImageNumber'].max()

    selected_rows_df = selected_rows_df.drop(["daughter_length_to_mother", "max_daughter_len_to_mother",
                                              'avg_daughters_TrajectoryX', 'avg_daughters_TrajectoryY'], axis=1)

    selected_rows_df_first_time_step = \
        selected_rows_df.loc[(selected_rows_df['ImageNumber'] == selected_rows_df['ImageNumber'].min()) &
                             (selected_rows_df[parent_image_number_col] != 0)]

    selected_rows = selected_rows_df['index'].values

    df.loc[selected_rows, ["id", "divideFlag", 'unexpected_end', 'unexpected_beginning', 'daughters_index',
                           'division_time', "difference_neighbors", "parent_id", "LengthChangeRatio",
                           "bacteria_movement", "direction_of_motion", "TrajectoryX",
                           "TrajectoryY", "daughter_length_to_mother", "max_daughter_len_to_mother",
                           'LifeHistory', 'age', 'MotionAlignmentAngle', 'slope_bac_bac',
                           'prev_time_step_index', 'parent_index', 'other_daughter_index',
                           'daughter_mother_LengthChangeRatio', 'prev_bacteria_slope',
                           'avg_daughters_TrajectoryX', 'avg_daughters_TrajectoryY']] = \
        [0,  # id
         False,  # divideFlag
         False,  # unexpected_end
         False,  # unexpected_beginning
         '',  # daughters_index
         np.nan,  # division_time
         0,  # difference_neighbors
         np.nan,  # parent_id
         np.nan,  # LengthChangeRatio
         np.nan,  # bacteria_movement
         np.nan,  # direction_of_motion
         np.nan,  # TrajectoryX
         np.nan,  # TrajectoryY
         np.nan,  # daughter_length_to_mother
         np.nan,  # max_daughter_len_to_mother
         0,  # LifeHistory
         0,  # age
         np.nan,  # MotionAlignmentAngle
         np.nan,  # slope_bac_bac
         -1,  # prev_time_step_index
         np.nan,  # parent_index
         np.nan,  # daughter_index
         np.nan,  # daughter_mother_LengthChangeRatio
         np.nan,  # prev_bacteria_slope
         np.nan,  # avg_daughters_TrajectoryX
         np.nan,  # avg_daughters_TrajectoryY
         ]

    # unexpected_beginning_bacteria
    cond1 = selected_rows_df[parent_image_number_col] == 0
    cond2 = selected_rows_df['ImageNumber'] > 1
    cond3 = selected_rows_df['ImageNumber'] == 1

    df.loc[selected_rows_df[cond1 & cond2]['index'].values, ['checked', 'unexpected_beginning', 'parent_id']] = \
        [
            True,
            True,
            0
        ]

    df.loc[selected_rows_df[cond1 & cond3]['index'].values, ['checked', 'parent_id']] = \
        [
            True,
            0
        ]

    selected_rows_df.loc[cond1, 'checked'] = True

    max_bac_id = df['id'].max()
    df.loc[selected_rows_df[cond1]['index'].values, 'id'] = max_bac_id + selected_rows_df[cond1]['index'].values + 1

    selected_rows_df_and_neighbors = \
        selected_rows_df[['ImageNumber', 'ObjectNumber', 'index']].merge(neighbor_df,
                                                                         left_on=['ImageNumber', 'ObjectNumber'],
                                                                         right_on=['First Image Number',
                                                                                   'First Object Number'],
                                                                         how='left')
    selected_rows_df_and_neighbors_info = \
        selected_rows_df_and_neighbors.merge(df[['ImageNumber', 'ObjectNumber', 'index']],
                                             left_on=['Second Image Number', 'Second Object Number'],
                                             right_on=['ImageNumber', 'ObjectNumber'], suffixes=('', '_neighbor'),
                                             how='inner')

    selected_rows_df_and_neighbors_info_idx = \
        np.unique(np.append(selected_rows_df['index'].values,
                            selected_rows_df_and_neighbors_info['index_neighbor'].values))

    # check division
    # _2: bac2, _1: bac1 (source bac)
    # selected_rows_df
    merged_df = \
        df[['ImageNumber', 'ObjectNumber', parent_image_number_col, parent_object_number_col, 'index',
            'AreaShape_MajorAxisLength', 'bacteria_slope',
            center_coordinate_columns['x'], center_coordinate_columns['y']]].merge(
            df.loc[selected_rows_df_and_neighbors_info_idx][['ImageNumber', 'ObjectNumber', 'index',
                                                             'AreaShape_MajorAxisLength', 'bacteria_slope',
                                                             center_coordinate_columns['x'],
                                                             center_coordinate_columns['y'],
                                                             parent_image_number_col, parent_object_number_col]],
            left_on=[parent_image_number_col, parent_object_number_col],
            right_on=['ImageNumber', 'ObjectNumber'], how='inner', suffixes=('_2', '_1'))

    division_bac_and_neighbors = \
        merged_df[merged_df.duplicated(subset='index_1', keep=False)][['ImageNumber_1', 'ObjectNumber_1',
                                                                       'index_1', 'index_2',
                                                                       'ImageNumber_2', 'ObjectNumber_2',
                                                                       'AreaShape_MajorAxisLength_1',
                                                                       'AreaShape_MajorAxisLength_2',
                                                                       'bacteria_slope_1', 'bacteria_slope_2',
                                                                       center_coordinate_columns['x'] + '_1',
                                                                       center_coordinate_columns['y'] + '_1',
                                                                       center_coordinate_columns['x'] + '_2',
                                                                       center_coordinate_columns['y'] + '_2',
                                                                       parent_image_number_col + '_2',
                                                                       parent_object_number_col + '_2']].copy()

    division = division_bac_and_neighbors.loc[
        (division_bac_and_neighbors['index_2'].isin(selected_rows_df['index'].values)) |
        (division_bac_and_neighbors['index_1'].isin(selected_rows_df['index'].values))].copy()

    daughter_to_daughter = division.merge(division, on=[parent_image_number_col + '_2',
                                                        parent_object_number_col + '_2'],
                                          suffixes=('_daughter1', '_daughter2'))

    daughter_to_daughter = daughter_to_daughter.loc[daughter_to_daughter['ObjectNumber_2_daughter1'] !=
                                                    daughter_to_daughter['ObjectNumber_2_daughter2']]

    daughters_are_not_in_selected_rows = division.loc[~ division['index_2'].isin(selected_rows_df['index'].values)]

    division['daughters_index'] = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['index_2'].transform(lambda x: ', '.join(x.astype(str)))

    division['daughter_mother_slope'] = \
        calculate_orientation_angle_batch(division['bacteria_slope_2'].values,
                                          division['bacteria_slope_1'])

    division['daughter_mother_LengthChangeRatio'] = (division['AreaShape_MajorAxisLength_2'] /
                                                     division['AreaShape_MajorAxisLength_1'])

    mothers_df_last_time_step = division.drop_duplicates(subset='index_1', keep='first')

    division['daughters_TrajectoryX'] = \
        division[center_coordinate_columns['x'] + '_2'] - division[center_coordinate_columns['x'] + '_1']

    division['daughters_TrajectoryY'] = \
        division[center_coordinate_columns['y'] + '_2'] - division[center_coordinate_columns['y'] + '_1']

    direction_of_motion = calculate_trajectory_direction_daughters_mother(division, center_coordinate_columns)

    division["direction_of_motion"] = direction_of_motion

    sum_daughters_len = division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['AreaShape_MajorAxisLength_2'].sum()
    max_daughters_len = division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['AreaShape_MajorAxisLength_2'].max()

    sum_daughters_len_to_mother = \
        sum_daughters_len / division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['AreaShape_MajorAxisLength_1'].mean()
    max_daughters_len_to_mother = \
        max_daughters_len / division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['AreaShape_MajorAxisLength_1'].mean()

    avg_daughters_trajectory_x = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['daughters_TrajectoryX'].mean()

    avg_daughters_trajectory_y = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['daughters_TrajectoryY'].mean()

    # Create a temporary DataFrame from sum_daughters_len_to_mother for easier merging
    temp_df_sum_daughters = sum_daughters_len_to_mother.reset_index()
    temp_df_sum_daughters.columns = ['ImageNumber', 'ObjectNumber', 'daughter_length_to_mother']

    temp_df_max_daughters = max_daughters_len_to_mother.reset_index()
    temp_df_max_daughters.columns = ['ImageNumber', 'ObjectNumber', 'max_daughter_len_to_mother']

    temp_df_avg_daughters_trajectory_x = avg_daughters_trajectory_x.reset_index()
    temp_df_avg_daughters_trajectory_x.columns = ['ImageNumber', 'ObjectNumber', 'avg_daughters_TrajectoryX']

    temp_df_avg_daughters_trajectory_y = avg_daughters_trajectory_y.reset_index()
    temp_df_avg_daughters_trajectory_y.columns = ['ImageNumber', 'ObjectNumber', 'avg_daughters_TrajectoryY']

    selected_rows_df = selected_rows_df.merge(temp_df_sum_daughters, on=['ImageNumber', 'ObjectNumber'], how='outer')
    selected_rows_df = selected_rows_df.merge(temp_df_max_daughters, on=['ImageNumber', 'ObjectNumber'], how='outer')
    selected_rows_df = selected_rows_df.merge(temp_df_avg_daughters_trajectory_x,
                                              on=['ImageNumber', 'ObjectNumber'], how='outer')
    selected_rows_df = selected_rows_df.merge(temp_df_avg_daughters_trajectory_y,
                                              on=['ImageNumber', 'ObjectNumber'], how='outer')

    mother_idxs = division['index_1'].unique()
    mother_cols_should_update = ['daughters_index', 'division_time']
    related_to_mothers_update = ['daughters_index', 'ImageNumber_2']

    df.loc[mother_idxs, mother_cols_should_update] = mothers_df_last_time_step[related_to_mothers_update].values

    df.loc[mother_idxs, "divideFlag"] = True

    selected_rows_idx = selected_rows_df['index'].values
    selected_rows_cols_should_update = ['daughter_length_to_mother', 'max_daughter_len_to_mother',
                                        'avg_daughters_TrajectoryX', 'avg_daughters_TrajectoryY']
    df.loc[selected_rows_idx, selected_rows_cols_should_update] = \
        selected_rows_df[selected_rows_cols_should_update].values

    # for daughters
    daughters_idxs = division['index_2'].values
    daughter_cols_should_update = ["direction_of_motion", "TrajectoryX", "TrajectoryY",
                                   'slope_bac_bac', 'prev_bacteria_slope', 'parent_index',
                                   'daughter_mother_LengthChangeRatio', 'prev_time_step_index']

    related_cols_daughter_update = ["direction_of_motion", 'daughters_TrajectoryX', "daughters_TrajectoryY",
                                    'daughter_mother_slope', 'bacteria_slope_1', 'index_1',
                                    'daughter_mother_LengthChangeRatio', 'index_1']

    df.loc[daughters_idxs, daughter_cols_should_update] = \
        division[related_cols_daughter_update].values

    df.loc[daughters_idxs, 'prev_time_step_index'] = division['index_1'].values.astype('int64')

    df.loc[daughter_to_daughter['index_2_daughter1'].values, "other_daughter_index"] = \
        daughter_to_daughter['index_2_daughter2'].values

    # other bacteria
    other_bac_idx = selected_rows_df.loc[selected_rows_df['checked'] == False]['index'].values
    other_bac_df = df.loc[other_bac_idx]

    temp_df = df.loc[(df['ImageNumber'] >= first_time_step_selected_rows - 1) &
                     (df['ImageNumber'] <= last_time_step_selected_rows)][['ImageNumber', 'ObjectNumber',
                                                                           'index']].copy()

    temp_df.index = (temp_df['ImageNumber'].astype(str) + '_' + temp_df['ObjectNumber'].astype(str))

    bac_index_dict = temp_df['index'].to_dict()

    id_list = []
    parent_id_list = []

    same_bac_dict = {}

    last_bac_id = df['id'].max() + 1

    for row_index, row in other_bac_df.iterrows():

        image_number, object_number, parent_img_num, parent_obj_num = \
            row[['ImageNumber', 'ObjectNumber', parent_image_number_col, parent_object_number_col]]

        if str(int(parent_img_num)) + '_' + str(parent_obj_num) not in same_bac_dict.keys():

            source_link = df.iloc[bac_index_dict[str(int(parent_img_num)) + '_' + str(parent_obj_num)]]

            # life history continues
            parent_id = source_link['parent_id']
            source_bac_id = source_link['id']
            division_stat = source_link['divideFlag']

        else:

            parent_id, source_bac_id, division_stat = \
                same_bac_dict[str(int(parent_img_num)) + '_' + str(parent_obj_num)]

        if division_stat:

            # division occurs
            new_bac_id = last_bac_id
            last_bac_id += 1
            same_bac_dict[str(int(image_number)) + '_' + str(object_number)] = \
                [source_bac_id, new_bac_id, row['divideFlag']]

            parent_id_list.append(source_bac_id)
            id_list.append(new_bac_id)

        else:
            parent_id_list.append(parent_id)
            id_list.append(source_bac_id)

            # same bacteria
            same_bac_dict[str(int(image_number)) + '_' + str(object_number)] = \
                [parent_id, source_bac_id, row['divideFlag']]

    other_bac_df_idx = other_bac_df['index'].values
    df.loc[other_bac_df_idx, 'id'] = id_list
    df.loc[other_bac_df_idx, 'parent_id'] = parent_id_list

    # for adding updated features
    updated_selected_rows_df = df.loc[selected_rows_df['index'].values]
    updated_selected_rows_df['LifeHistory'] = updated_selected_rows_df.groupby('id')['id'].transform('size')
    df.loc[updated_selected_rows_df['index'].values, 'LifeHistory'] = updated_selected_rows_df['LifeHistory'].values

    division = \
        division.merge(df[['ImageNumber', 'ObjectNumber', 'id', 'index']], left_on=['ImageNumber_1', 'ObjectNumber_1'],
                       right_on=['ImageNumber', 'ObjectNumber'], how='inner')

    division_mothers_life_history = division.merge(df[['id', 'index']],
                                                   on='id', how='inner', suffixes=('', '_life_history'))

    df.loc[division_mothers_life_history['index_life_history'].unique(), "divideFlag"] = True

    # update daughters parent id
    df.loc[division['index_2'].values, 'parent_id'] = division['id'].values

    if daughters_are_not_in_selected_rows.shape[0] > 0:
        daughters_with_updated_values = \
            df.loc[daughters_are_not_in_selected_rows['index_2'].values][[parent_image_number_col,
                                                                          parent_object_number_col,
                                                                          'ImageNumber', 'ObjectNumber',
                                                                          'id']]

        daughters_with_updated_id_mother = \
            daughters_with_updated_values.merge(df[['ImageNumber', 'ObjectNumber', 'id']],
                                                left_on=[parent_image_number_col, parent_object_number_col],
                                                right_on=['ImageNumber', 'ObjectNumber'], how='inner',
                                                suffixes=('_daughter', '_mother'))

        daughters_with_life_history = \
            daughters_with_updated_id_mother.merge(df[['id', 'index']], left_on='id_daughter', right_on='id',
                                                   how='inner', suffixes=('', '_daughter_life'))

        df.loc[daughters_with_life_history['index'].values, 'parent_id'] = \
            daughters_with_life_history['id_mother'].values

    bac_in_selected_rows_time_step = df.loc[selected_rows_df_and_neighbors_info_idx]
    division_bac_and_neighbors = \
        division_bac_and_neighbors.loc[
            (division_bac_and_neighbors['ImageNumber_2'] >= bac_in_selected_rows_time_step['ImageNumber'].min() - 1) &
            (division_bac_and_neighbors['ImageNumber_2'] <= bac_in_selected_rows_time_step['ImageNumber'].max())]

    last_time_step_df = df.loc[df['ImageNumber'] == df['ImageNumber'].max()]

    updated_selected_rows_df = df.loc[selected_rows_df['index'].values]

    # for adding updated features
    # if df['ImageNumber'].max() == last_time_step_selected_rows:
    bac_without_division = updated_selected_rows_df.loc[updated_selected_rows_df["divideFlag"] == False]

    if bac_without_division.shape[0] > 0:
        bac_without_division = \
            bac_without_division.loc[~ bac_without_division['id'].isin(last_time_step_df['id'].values)]
        bac_without_division_last_time_step = bac_without_division.drop_duplicates(subset='id', keep='last')

        df.loc[bac_without_division_last_time_step['index'].values, 'unexpected_end'] = True

    updated_selected_rows_df['division_time'] = \
        updated_selected_rows_df.groupby('id')['division_time'].transform(lambda x: x.ffill().bfill())
    updated_selected_rows_df['daughters_index'] = \
        updated_selected_rows_df.groupby('id')['daughters_index'].transform(lambda x: x.ffill().bfill())

    # set age
    updated_selected_rows_df['age'] = updated_selected_rows_df.groupby('id').cumcount() + 1

    updated_selected_rows_df['prev_time_step_index'] = \
        updated_selected_rows_df.groupby('id')['index'].shift(1)

    updated_selected_rows_df['division_time'] = updated_selected_rows_df['division_time'].fillna(0)
    updated_selected_rows_df['daughters_index'] = updated_selected_rows_df['daughters_index'].fillna('')

    updated_selected_rows_df_cols_should_update = ['age', 'division_time', 'daughters_index']
    df.loc[updated_selected_rows_df['index'].values, updated_selected_rows_df_cols_should_update] = \
        updated_selected_rows_df[updated_selected_rows_df_cols_should_update].values

    # other bac (except daughters)
    bac_idx_needed_to_update = df.loc[(df.index.isin(updated_selected_rows_df['index'].values)) &
                                      (df['prev_time_step_index'] == -1)].index.values

    # it means only daughters
    bac_idx_not_needed_to_update = df.loc[(df.index.isin(updated_selected_rows_df['index'].values)) &
                                          (df['prev_time_step_index'] != -1)].index.values

    df.loc[bac_idx_needed_to_update, 'prev_time_step_index'] = \
        updated_selected_rows_df.loc[bac_idx_needed_to_update, 'prev_time_step_index'].fillna(-1).values.astype('int64')

    updated_selected_rows_df['prev_bacteria_slope'] = updated_selected_rows_df.groupby('id')['bacteria_slope'].shift(1)
    updated_selected_rows_df.loc[bac_idx_not_needed_to_update, 'prev_bacteria_slope'] = \
        df.loc[bac_idx_not_needed_to_update, 'prev_bacteria_slope'].values

    df.loc[updated_selected_rows_df['index'].values, 'prev_bacteria_slope'] = \
        updated_selected_rows_df['prev_bacteria_slope'].values

    bac_need_to_cal_dir_motion = \
        updated_selected_rows_df.loc[(updated_selected_rows_df['slope_bac_bac'].isna()) &
                                     (updated_selected_rows_df['unexpected_beginning'] == False) &
                                     (updated_selected_rows_df['ImageNumber'] != 1)][['index', 'bacteria_slope',
                                                                                      'prev_bacteria_slope']].copy()

    df.loc[bac_need_to_cal_dir_motion['index'].values, 'slope_bac_bac'] = \
        calculate_orientation_angle_batch(bac_need_to_cal_dir_motion['bacteria_slope'].values,
                                          bac_need_to_cal_dir_motion['prev_bacteria_slope'])

    if selected_rows_df_first_time_step.shape[0] > 0:
        selected_rows_df_first_time_step_idx = selected_rows_df_first_time_step['index'].values

        cols_should_update = ['MotionAlignmentAngle', 'parent_index', 'direction_of_motion',
                              'TrajectoryX', 'TrajectoryY', 'slope_bac_bac', 'prev_bacteria_slope',
                              'other_daughter_index', 'daughter_mother_LengthChangeRatio', 'prev_time_step_index']

        df.loc[selected_rows_df_first_time_step_idx, cols_should_update] = \
            selected_rows_df_first_time_step[cols_should_update].fillna({'prev_time_step_index': -1}).values

    # updated_selected_rows_df
    updated_selected_rows_df = df.loc[selected_rows_df['index'].values]
    df = adding_features_to_continues_life_history(df, neighbor_df, division_bac_and_neighbors,
                                                   center_coordinate_columns,
                                                   parent_image_number_col, parent_object_number_col,
                                                   calc_all=True, selected_rows=updated_selected_rows_df,
                                                   bac_with_neighbors=division_bac_and_neighbors)

    updated_selected_rows_df['AverageLengthChangeRatio'] = \
        updated_selected_rows_df.groupby('id')['LengthChangeRatio'].transform('mean')

    df.loc[updated_selected_rows_df['index'].values, 'AverageLengthChangeRatio'] = \
        updated_selected_rows_df['AverageLengthChangeRatio'].values

    df['checked'] = True

    bac_in_selected_rows_time_step = df.loc[selected_rows_df_and_neighbors_info_idx]

    selected_time_step_df = df.loc[(df['ImageNumber'] >= bac_in_selected_rows_time_step['ImageNumber'].min() - 1) &
                                   (df['ImageNumber'] <= bac_in_selected_rows_time_step['ImageNumber'].max())]

    df = neighbor_checking(df, neighbor_list_array, parent_image_number_col, parent_object_number_col,
                           selected_rows_df=bac_in_selected_rows_time_step,
                           selected_time_step_df=selected_time_step_df, index2=False)

    return df


def bacteria_modification(df, bac1, bac2_life_history, all_bac_undergo_phase_change, neighbor_df, neighbor_list_array,
                          parent_image_number_col, parent_object_number_col, center_coordinate_columns, label_col):
    # I want to assign bacterium 2 (bac2) to bacterium 1 (bac1)
    # bacterium 1 (bac1) is parent or same bacterium

    if bac1 is not None:

        index_should_be_checked = []
        index_should_be_checked.extend(bac2_life_history.index.values.tolist())

        bac1_life_history = df.loc[df['id'] == bac1['id']]
        index_should_be_checked.extend(bac1_life_history.index.values.tolist())

        bac2_in_current_time_step = \
            bac2_life_history.loc[bac2_life_history['ImageNumber'] ==
                                  all_bac_undergo_phase_change['ImageNumber'].values[0]]

        # change bac2 parent info
        df.at[bac2_in_current_time_step.index[0], parent_image_number_col] = bac1['ImageNumber']
        df.at[bac2_in_current_time_step.index[0], parent_object_number_col] = bac1['ObjectNumber']

        df.loc[index_should_be_checked, "checked"] = False
        selected_rows_df = df.loc[sorted(set(index_should_be_checked))]

        df = calc_modified_features(df, selected_rows_df, neighbor_df, neighbor_list_array, center_coordinate_columns,
                                    parent_image_number_col, parent_object_number_col)

    else:

        # unexpected_beginning
        index_should_be_checked = []

        bac2_mother_life_history = df.loc[df['id'] == bac2_life_history['parent_id'].values[0]]

        if bac2_mother_life_history.shape[0] > 0:
            bac2_other_daughter_life_history = df.loc[(df['parent_id'] == bac2_mother_life_history['id'].values[0]) &
                                                      (df['id'] != bac2_life_history['id'].values[0])]

            index_should_be_checked.extend(bac2_other_daughter_life_history.index.values.tolist())

        bac2_life_history_before_selected_first_time_step = \
            df.loc[(df['id'] == bac2_life_history['id'].values[0]) &
                   (df['ImageNumber'] < bac2_life_history['ImageNumber'].min())]

        index_should_be_checked.extend(bac2_mother_life_history.index.values.tolist())
        index_should_be_checked.extend(bac2_life_history_before_selected_first_time_step.index.values.tolist())
        index_should_be_checked.extend(bac2_life_history.index.values.tolist())

        # change bac2 parent info
        df.at[bac2_life_history['ImageNumber'].index[0], parent_image_number_col] = 0
        df.at[bac2_life_history['ImageNumber'].index[0], parent_object_number_col] = 0

        df.loc[index_should_be_checked, "checked"] = False
        selected_rows_df = df.loc[sorted(set(index_should_be_checked))]

        df = calc_modified_features(df, selected_rows_df, neighbor_df, neighbor_list_array, center_coordinate_columns,
                                    parent_image_number_col, parent_object_number_col)

    return df


def remove_redundant_link(dataframe, wrong_daughter_life_history, neighbor_df, neighbor_list_array,
                          parent_image_number_col, parent_object_number_col, center_coordinate_columns, label_col):
    dataframe = bacteria_modification(dataframe, None, wrong_daughter_life_history, None,
                                      neighbor_df=neighbor_df, neighbor_list_array=neighbor_list_array,
                                      parent_image_number_col=parent_image_number_col,
                                      parent_object_number_col=parent_object_number_col,
                                      center_coordinate_columns=center_coordinate_columns, label_col=label_col)

    return dataframe
