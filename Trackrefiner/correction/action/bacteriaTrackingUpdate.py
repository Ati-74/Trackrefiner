import numpy as np
from Trackrefiner.correction.action.helper import calculate_trajectory_angles, \
    calculate_bacterial_life_history_features, calculate_angles_between_slopes
from Trackrefiner.correction.action.neighborAnalysis import compare_neighbor_sets


def calc_modified_features(df, selected_bacteria, neighbor_df, neighbor_list_array, center_coord_cols,
                           parent_image_number_col, parent_object_number_col):
    """
    Modifies and updates bacterial tracking data by calculating various features related to bacterial life history.

    **Calculated Features**:
    - `id`: Unique identifier for each bacterium.
    - `divideFlag`: Whether a bacterium has divided.
    - `unexpected_end` and `unexpected_beginning`: Flags indicating unexpected lifecycle events.
    - `daughters_index`: List of indices for daughter bacteria.
    - `division_time`: Time of division.
    - `LengthChangeRatio`: Ratio of daughter to mother length.
    - `bacteria_movement`: Distance traveled between time steps.
    - `TrajectoryX` and `TrajectoryY`: Components of movement trajectory.
    - `MotionAlignmentAngle`: Angle of motion alignment.
    - `LifeHistory`: Duration of a bacterium's lifecycle.
    - `direction_of_motion`: Direction of movement in degrees.
    - Neighbor-related features such as `difference_neighbors`.
    - Slope and trajectory information (`bacteria_slope`, `prev_bacteria_slope`).
    - Parent-daughter relationships and alignment angles.

    :param pandas.DataFrame df:
        The main DataFrame containing tracking data and measured bacterial features for all bacteria.
        This DataFrame is updated in place with calculated features.
    :param pandas.DataFrame selected_bacteria:
        A DataFrame containing the subset of rows (bacteria) to be modified. It includes tracking and
        measured bacterial features for selected bacteria.
    :param pandas.DataFrame neighbor_df:
        A DataFrame containing information about neighboring bacteria. Used for calculating neighbor-related
        features.
    :param lil_matrix neighbor_list_array:
        Sparse matrix indicating neighboring relationships between bacteria.
    :param dict center_coord_cols:
        A dictionary specifying the column names for the x and y coordinates of bacterial centroids
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param str parent_image_number_col:
        Column name for the parent image number in the dataframe.
    :param str parent_object_number_col:
        Column name for the parent object number in the dataframe.

    :returns:
        pandas.DataFrame:

        The updated DataFrame `df` with modified features.
    """

    first_time_step_selected_rows = selected_bacteria['ImageNumber'].min()
    last_time_step_selected_rows = selected_bacteria['ImageNumber'].max()

    selected_bacteria = selected_bacteria.drop(["daughter_length_to_mother", "max_daughter_len_to_mother",
                                                'avg_daughters_TrajectoryX', 'avg_daughters_TrajectoryY'], axis=1)

    selected_rows_df_first_time_step = \
        selected_bacteria.loc[(selected_bacteria['ImageNumber'] == selected_bacteria['ImageNumber'].min()) &
                              (selected_bacteria[parent_image_number_col] != 0)]

    selected_rows = selected_bacteria['index'].values

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
    cond1 = selected_bacteria[parent_image_number_col] == 0
    cond2 = selected_bacteria['ImageNumber'] > 1
    cond3 = selected_bacteria['ImageNumber'] == 1

    df.loc[selected_bacteria[cond1 & cond2]['index'].values, ['checked', 'unexpected_beginning', 'parent_id']] = \
        [
            True,
            True,
            0
        ]

    df.loc[selected_bacteria[cond1 & cond3]['index'].values, ['checked', 'parent_id']] = \
        [
            True,
            0
        ]

    selected_bacteria.loc[cond1, 'checked'] = True

    max_bac_id = df['id'].max()
    df.loc[selected_bacteria[cond1]['index'].values, 'id'] = max_bac_id + selected_bacteria[cond1]['index'].values + 1

    selected_rows_df_and_neighbors = \
        selected_bacteria[['ImageNumber', 'ObjectNumber', 'index']].merge(neighbor_df,
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
        np.unique(np.append(selected_bacteria['index'].values,
                            selected_rows_df_and_neighbors_info['index_neighbor'].values))

    # check division
    # _2: bac2, _1: bac1 (source bac)
    # selected_rows_df
    merged_df = \
        df[['ImageNumber', 'ObjectNumber', parent_image_number_col, parent_object_number_col, 'index',
            'AreaShape_MajorAxisLength', 'bacteria_slope',
            center_coord_cols['x'], center_coord_cols['y']]].merge(
            df.loc[selected_rows_df_and_neighbors_info_idx][['ImageNumber', 'ObjectNumber', 'index',
                                                             'AreaShape_MajorAxisLength', 'bacteria_slope',
                                                             center_coord_cols['x'],
                                                             center_coord_cols['y'],
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
                                                                       f"{center_coord_cols['x']}_1",
                                                                       f"{center_coord_cols['y']}_1",
                                                                       f"{center_coord_cols['x']}_2",
                                                                       f"{center_coord_cols['y']}_2",
                                                                       f"{parent_image_number_col}_2",
                                                                       f"{parent_object_number_col}_2"]].copy()

    division = division_bac_and_neighbors.loc[
        (division_bac_and_neighbors['index_2'].isin(selected_bacteria['index'].values)) |
        (division_bac_and_neighbors['index_1'].isin(selected_bacteria['index'].values))].copy()

    daughter_to_daughter = division.merge(division, on=[f"{parent_image_number_col}_2",
                                                        f'{parent_object_number_col}_2'],
                                          suffixes=('_daughter1', '_daughter2'))

    daughter_to_daughter = daughter_to_daughter.loc[daughter_to_daughter['ObjectNumber_2_daughter1'] !=
                                                    daughter_to_daughter['ObjectNumber_2_daughter2']]

    daughters_are_not_in_selected_rows = division.loc[~ division['index_2'].isin(selected_bacteria['index'].values)]

    division['daughters_index'] = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['index_2'].transform(lambda x: ', '.join(x.astype(str)))

    division['daughter_mother_slope'] = \
        calculate_angles_between_slopes(division['bacteria_slope_2'].values,
                                        division['bacteria_slope_1'])

    division['daughter_mother_LengthChangeRatio'] = (division['AreaShape_MajorAxisLength_2'] /
                                                     division['AreaShape_MajorAxisLength_1'])

    mothers_df_last_time_step = division.drop_duplicates(subset='index_1', keep='first')

    division['daughters_TrajectoryX'] = \
        division[f"{center_coord_cols['x']}_2"] - division[f"{center_coord_cols['x']}_1"]

    division['daughters_TrajectoryY'] = \
        division[f"{center_coord_cols['y']}_2"] - division[f"{center_coord_cols['y']}_1"]

    direction_of_motion = calculate_trajectory_angles(division, center_coord_cols, suffix1='_2', suffix2='_1')

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

    selected_bacteria = selected_bacteria.merge(temp_df_sum_daughters, on=['ImageNumber', 'ObjectNumber'], how='outer')
    selected_bacteria = selected_bacteria.merge(temp_df_max_daughters, on=['ImageNumber', 'ObjectNumber'], how='outer')
    selected_bacteria = selected_bacteria.merge(temp_df_avg_daughters_trajectory_x,
                                                on=['ImageNumber', 'ObjectNumber'], how='outer')
    selected_bacteria = selected_bacteria.merge(temp_df_avg_daughters_trajectory_y,
                                                on=['ImageNumber', 'ObjectNumber'], how='outer')

    mother_idx = division['index_1'].unique()
    mother_cols_should_update = ['daughters_index', 'division_time']
    related_to_mothers_update = ['daughters_index', 'ImageNumber_2']

    df.loc[mother_idx, mother_cols_should_update] = mothers_df_last_time_step[related_to_mothers_update].values

    df.loc[mother_idx, "divideFlag"] = True

    selected_rows_idx = selected_bacteria['index'].values
    selected_rows_cols_should_update = ['daughter_length_to_mother', 'max_daughter_len_to_mother',
                                        'avg_daughters_TrajectoryX', 'avg_daughters_TrajectoryY']
    df.loc[selected_rows_idx, selected_rows_cols_should_update] = \
        selected_bacteria[selected_rows_cols_should_update].values

    # for daughters
    daughters_idx = division['index_2'].values
    daughter_cols_should_update = ["direction_of_motion", "TrajectoryX", "TrajectoryY",
                                   'slope_bac_bac', 'prev_bacteria_slope', 'parent_index',
                                   'daughter_mother_LengthChangeRatio', 'prev_time_step_index']

    related_cols_daughter_update = ["direction_of_motion", 'daughters_TrajectoryX', "daughters_TrajectoryY",
                                    'daughter_mother_slope', 'bacteria_slope_1', 'index_1',
                                    'daughter_mother_LengthChangeRatio', 'index_1']

    df.loc[daughters_idx, daughter_cols_should_update] = \
        division[related_cols_daughter_update].values

    df.loc[daughters_idx, 'prev_time_step_index'] = division['index_1'].values.astype('int64')

    df.loc[daughter_to_daughter['index_2_daughter1'].values, "other_daughter_index"] = \
        daughter_to_daughter['index_2_daughter2'].values

    # other bacteria
    other_bac_idx = selected_bacteria.loc[selected_bacteria['checked'] == False]['index'].values
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

        if f'{int(parent_img_num)}_{int(parent_obj_num)}' not in same_bac_dict.keys():

            source_link = df.iloc[bac_index_dict[f'{int(parent_img_num)}_{int(parent_obj_num)}']]

            # life history continues
            parent_id = source_link['parent_id']
            source_bac_id = source_link['id']
            division_stat = source_link['divideFlag']

        else:

            parent_id, source_bac_id, division_stat = same_bac_dict[f"{int(parent_img_num)}_{int(parent_obj_num)}"]

        if division_stat:

            # division occurs
            new_bac_id = last_bac_id
            last_bac_id += 1
            same_bac_dict[f"{int(image_number)}_{int(object_number)}"] = [source_bac_id, new_bac_id, row['divideFlag']]

            parent_id_list.append(source_bac_id)
            id_list.append(new_bac_id)

        else:
            parent_id_list.append(parent_id)
            id_list.append(source_bac_id)

            # same bacteria
            same_bac_dict[f"{int(image_number)}_{int(object_number)}"] = [parent_id, source_bac_id, row['divideFlag']]

    other_bac_df_idx = other_bac_df['index'].values
    df.loc[other_bac_df_idx, 'id'] = id_list
    df.loc[other_bac_df_idx, 'parent_id'] = parent_id_list

    # for adding updated features
    updated_selected_rows_df = df.loc[selected_bacteria['index'].values]
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

    updated_selected_rows_df = df.loc[selected_bacteria['index'].values]

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
        calculate_angles_between_slopes(bac_need_to_cal_dir_motion['bacteria_slope'].values,
                                        bac_need_to_cal_dir_motion['prev_bacteria_slope'])

    if selected_rows_df_first_time_step.shape[0] > 0:
        selected_rows_df_first_time_step_idx = selected_rows_df_first_time_step['index'].values

        cols_should_update = ['MotionAlignmentAngle', 'parent_index', 'direction_of_motion',
                              'TrajectoryX', 'TrajectoryY', 'slope_bac_bac', 'prev_bacteria_slope',
                              'other_daughter_index', 'daughter_mother_LengthChangeRatio', 'prev_time_step_index']

        df.loc[selected_rows_df_first_time_step_idx, cols_should_update] = \
            selected_rows_df_first_time_step[cols_should_update].fillna({'prev_time_step_index': -1}).values

    # updated_selected_rows_df
    updated_selected_rows_df = df.loc[selected_bacteria['index'].values]
    df = calculate_bacterial_life_history_features(updated_selected_rows_df, calc_all_features=True,
                                                   neighbor_df=neighbor_df, division_df=division_bac_and_neighbors,
                                                   center_coord_cols=center_coord_cols,
                                                   use_selected_rows=True, original_df=df)

    updated_selected_rows_df['AverageLengthChangeRatio'] = \
        updated_selected_rows_df.groupby('id')['LengthChangeRatio'].transform('mean')

    df.loc[updated_selected_rows_df['index'].values, 'AverageLengthChangeRatio'] = \
        updated_selected_rows_df['AverageLengthChangeRatio'].values

    df['checked'] = True

    bac_in_selected_rows_time_step = df.loc[selected_rows_df_and_neighbors_info_idx]

    selected_time_step_df = df.loc[(df['ImageNumber'] >= bac_in_selected_rows_time_step['ImageNumber'].min() - 1) &
                                   (df['ImageNumber'] <= bac_in_selected_rows_time_step['ImageNumber'].max())]

    df = compare_neighbor_sets(df, neighbor_list_array, parent_image_number_col, parent_object_number_col,
                               selected_rows_df=bac_in_selected_rows_time_step,
                               selected_time_step_df=selected_time_step_df, index2=False)

    return df


def bacteria_modification(df, source_bac, target_bac_life_history, target_frame_bacteria, neighbor_df,
                          neighbor_list_array, parent_image_number_col, parent_object_number_col, center_coord_cols):

    """
    Modifies bacterial tracking data by assigning target bacterium to source bacterium, either as its child or
    the same bacterium, and recalculates features to ensure consistency in the tracking data.

    :param pandas.DataFrame df:
        The main DataFrame containing tracking data and measured bacterial features for all bacteria.
        This DataFrame is updated in place.
    :param dict source_bac:
        Dictionary representing the parent or the bacterium to which target is being assigned. Contains keys such as
        `ImageNumber`, `ObjectNumber`, and `id`. If `None`, target bacterium is treated as an unexpected beginning.
    :param pandas.DataFrame target_bac_life_history:
        Subset of the DataFrame containing the lifecycle data of target bacterium.
    :param pandas.DataFrame target_frame_bacteria:
        DataFrame containing bacteria present in the same time step as the target bacterium.
    :param pandas.DataFrame neighbor_df:
        DataFrame containing information about neighboring bacteria for calculating neighbor-specific features.
    :param lil_matrix neighbor_list_array:
        Sparse matrix indicating neighbor relationships between bacteria.
    :param str parent_image_number_col:
        Column name for the parent image number in the DataFrame.
    :param str parent_object_number_col:
        Column name for the parent object number in the DataFrame.
    :param dict center_coord_cols:
        Dictionary specifying column names for bacterial centroid coordinates
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).

    :returns:
        pandas.DataFrame:

        The updated DataFrame `df` with modified bacterial tracking data.
    """

    # I want to assign bacterium 2 (bac2) to bacterium 1 (bac1)
    # bacterium 1 (bac1) is parent or same bacterium

    if source_bac is not None:

        index_should_be_checked = []
        index_should_be_checked.extend(target_bac_life_history.index.values.tolist())

        bac1_life_history = df.loc[df['id'] == source_bac['id']]
        index_should_be_checked.extend(bac1_life_history.index.values.tolist())

        bac2_in_current_time_step = \
            target_bac_life_history.loc[target_bac_life_history['ImageNumber'] ==
                                        target_frame_bacteria['ImageNumber'].values[0]]

        # change bac2 parent info
        df.at[bac2_in_current_time_step.index[0], parent_image_number_col] = source_bac['ImageNumber']
        df.at[bac2_in_current_time_step.index[0], parent_object_number_col] = source_bac['ObjectNumber']

        df.loc[index_should_be_checked, "checked"] = False
        selected_rows_df = df.loc[sorted(set(index_should_be_checked))]

        df = calc_modified_features(df, selected_rows_df, neighbor_df, neighbor_list_array, center_coord_cols,
                                    parent_image_number_col, parent_object_number_col)

    else:

        # unexpected_beginning
        index_should_be_checked = []

        bac2_mother_life_history = df.loc[df['id'] == target_bac_life_history['parent_id'].values[0]]

        if bac2_mother_life_history.shape[0] > 0:
            bac2_other_daughter_life_history = df.loc[(df['parent_id'] == bac2_mother_life_history['id'].values[0]) &
                                                      (df['id'] != target_bac_life_history['id'].values[0])]

            index_should_be_checked.extend(bac2_other_daughter_life_history.index.values.tolist())

        bac2_life_history_before_selected_first_time_step = \
            df.loc[(df['id'] == target_bac_life_history['id'].values[0]) &
                   (df['ImageNumber'] < target_bac_life_history['ImageNumber'].min())]

        index_should_be_checked.extend(bac2_mother_life_history.index.values.tolist())
        index_should_be_checked.extend(bac2_life_history_before_selected_first_time_step.index.values.tolist())
        index_should_be_checked.extend(target_bac_life_history.index.values.tolist())

        # change bac2 parent info
        df.at[target_bac_life_history['ImageNumber'].index[0], parent_image_number_col] = 0
        df.at[target_bac_life_history['ImageNumber'].index[0], parent_object_number_col] = 0

        df.loc[index_should_be_checked, "checked"] = False
        selected_rows_df = df.loc[sorted(set(index_should_be_checked))]

        df = calc_modified_features(df, selected_rows_df, neighbor_df, neighbor_list_array, center_coord_cols,
                                    parent_image_number_col, parent_object_number_col)

    return df


def remove_redundant_link(dataframe, incorrect_target_bac_life_history, neighbor_df, neighbor_list_array,
                          parent_image_number_col, parent_object_number_col, center_coord_cols):

    """
    Removes redundant or incorrect link in the bacterial tracking data by modifying the lifecycle information
    of a target bacterium.

    :param pandas.DataFrame dataframe:
        The main DataFrame containing tracking data and measured bacterial features for all bacteria.
        This DataFrame is updated in place.
    :param pandas.DataFrame incorrect_target_bac_life_history:
        Subset of the DataFrame containing the lifecycle data of the target bacterium with incorrect link.
    :param pandas.DataFrame neighbor_df:
        DataFrame containing information about neighboring bacteria for calculating neighbor-specific features.
    :param lil_matrix neighbor_list_array:
        Sparse matrix indicating neighbor relationships between bacteria.
    :param str parent_image_number_col:
        Column name for the parent image number in the DataFrame.
    :param str parent_object_number_col:
        Column name for the parent object number in the DataFrame.
    :param dict center_coord_cols:
        Dictionary specifying the column names for bacterial centroid coordinates
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).

    :returns:
        pandas.DataFrame:

        The updated DataFrame with redundant link removed.
    """

    dataframe = bacteria_modification(
        dataframe, None, incorrect_target_bac_life_history, None, neighbor_df=neighbor_df,
        neighbor_list_array=neighbor_list_array, parent_image_number_col=parent_image_number_col,
        parent_object_number_col=parent_object_number_col, center_coord_cols=center_coord_cols)

    return dataframe
