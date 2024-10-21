import pandas as pd
import numpy as np
import sys
from scipy.spatial import distance_matrix


def print_progress_bar(iteration, total=10, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

    if iteration == total:
        print('\n')


def calc_neighbors_avg_dir_motion(df, source_bac, neighbor_df, center_coordinate_columns,
                                  parent_image_number_col, parent_object_number_col):
    neighbors_of_source_bac_info = find_neighbors_info(df, neighbor_df, source_bac)

    if neighbors_of_source_bac_info.shape[0] > 0:

        neighbor_of_source_bac_info_with_next_time_step = \
            neighbors_of_source_bac_info.merge(df, left_on=['ImageNumber', 'ObjectNumber'],
                                               right_on=[parent_image_number_col, parent_object_number_col],
                                               how='inner', suffixes=("_current", '_next'))

        if neighbor_of_source_bac_info_with_next_time_step.shape[0] > 0:

            average_first = np.mean(neighbor_of_source_bac_info_with_next_time_step['TrajectoryX_next'].dropna())
            average_second = np.mean(neighbor_of_source_bac_info_with_next_time_step['TrajectoryY_next'].dropna())

        else:
            average_first = np.nan
            average_second = np.nan
    else:
        average_first = np.nan
        average_second = np.nan

    return [average_first, average_second]


def calc_neighbors_dir_motion_all(df, neighbor_df, division_df, parent_image_number_col,
                                  parent_object_number_col, selected_rows=None):

    temp_df = df[['ImageNumber', 'ObjectNumber', 'id', "TrajectoryX", "TrajectoryY",
                  "daughter_length_to_mother", 'avg_daughters_TrajectoryX', 'avg_daughters_TrajectoryY']].copy()

    temp_df['source_target_TrajectoryX'] = df.groupby('id')["TrajectoryX"].shift(-1)
    temp_df['source_target_TrajectoryY'] = df.groupby('id')["TrajectoryY"].shift(-1)

    temp_df.loc[~ temp_df['daughter_length_to_mother'].isna(), 'source_target_TrajectoryX'] = \
        temp_df.loc[~ temp_df['daughter_length_to_mother'].isna(), 'avg_daughters_TrajectoryX']

    temp_df.loc[~ temp_df['daughter_length_to_mother'].isna(), 'source_target_TrajectoryY'] = \
        temp_df.loc[~ temp_df['daughter_length_to_mother'].isna(), 'avg_daughters_TrajectoryY']

    if selected_rows is None:
        all_bac_neighbors = df.merge(neighbor_df, left_on=['ImageNumber', 'ObjectNumber'],
                                     right_on=['First Image Number', 'First Object Number'], how='left')

    else:
        all_bac_neighbors = selected_rows.merge(neighbor_df, left_on=['ImageNumber', 'ObjectNumber'],
                                                right_on=['First Image Number', 'First Object Number'], how='left')

    # be careful: 'source_target_TrajectoryX', 'source_target_TrajectoryY' is only for neighbors
    all_bac_neighbors_info = all_bac_neighbors.merge(temp_df, left_on=['Second Image Number', 'Second Object Number'],
                                                     right_on=['ImageNumber', 'ObjectNumber'], how='left',
                                                     suffixes=('_bac', '_neighbor'))

    all_bac_neighbors_info['source_neighbors_TrajectoryX'] = \
        all_bac_neighbors_info.groupby(['ImageNumber_bac',
                                        'ObjectNumber_bac'])['source_target_TrajectoryX'].transform('mean')

    all_bac_neighbors_info['source_neighbors_TrajectoryY'] = \
        all_bac_neighbors_info.groupby(['ImageNumber_bac',
                                        'ObjectNumber_bac'])['source_target_TrajectoryY'].transform('mean')

    all_bac_neighbors_info = all_bac_neighbors_info.drop_duplicates(subset=['ImageNumber_bac', 'ObjectNumber_bac'],
                                                                    keep='first')

    all_bac_neighbors_info.index = all_bac_neighbors_info['index'].values

    if len(np.unique(all_bac_neighbors_info['index'].values)) != len(all_bac_neighbors_info['index'].values):
        breakpoint()

    # update dataframe
    df.loc[all_bac_neighbors_info['index'].values, 'bac_source_neighbors_TrajectoryX'] = \
        all_bac_neighbors_info['source_neighbors_TrajectoryX'].values

    df.loc[all_bac_neighbors_info['index'].values, 'bac_source_neighbors_TrajectoryY'] = \
        all_bac_neighbors_info['source_neighbors_TrajectoryY'].values

    # now we should focus on target bac
    all_bac_neighbors_info['source_neighbors_TrajectoryX_for_target'] = \
        all_bac_neighbors_info.groupby('id_bac')["source_neighbors_TrajectoryX"].shift(1)

    all_bac_neighbors_info['source_neighbors_TrajectoryY_for_target'] = \
        all_bac_neighbors_info.groupby('id_bac')["source_neighbors_TrajectoryY"].shift(1)

    # daughters at the division time
    all_bac_neighbors_info.loc[division_df['index_2'].values, "source_neighbors_TrajectoryX_for_target"] = \
        df.loc[division_df['index_1'].values, "bac_source_neighbors_TrajectoryX"].values

    all_bac_neighbors_info.loc[division_df['index_2'].values, "source_neighbors_TrajectoryY_for_target"] = \
        df.loc[division_df['index_1'].values, "bac_source_neighbors_TrajectoryY"].values

    if selected_rows is None:
        df["MotionAlignmentAngle"] = \
            calc_normalized_angle_between_motion_all(
                all_bac_neighbors_info[['source_neighbors_TrajectoryX_for_target',
                                        'source_neighbors_TrajectoryY_for_target']].values,
                all_bac_neighbors_info[['TrajectoryX_bac', 'TrajectoryY_bac']].values)
    else:

        # bac with nan
        bac_with_nan = all_bac_neighbors_info.loc[
            (all_bac_neighbors_info["source_neighbors_TrajectoryX_for_target"].isna()) & (
                    all_bac_neighbors_info['unexpected_beginning'] == False)]

        prev_motion_alignment_angle = df.loc[bac_with_nan['index'].values, "MotionAlignmentAngle"].values

        df.loc[all_bac_neighbors_info['index'].values, "MotionAlignmentAngle"] = \
            calc_normalized_angle_between_motion_all(
                all_bac_neighbors_info[['source_neighbors_TrajectoryX_for_target',
                                        'source_neighbors_TrajectoryY_for_target']].values,
                all_bac_neighbors_info[['TrajectoryX_bac', 'TrajectoryY_bac']].values)

        df.loc[bac_with_nan['index'].values, "MotionAlignmentAngle"] = prev_motion_alignment_angle

    df["MotionAlignmentAngle"] = df["MotionAlignmentAngle"].fillna(0)

    return df


def update_df_based_on_selected_rows(dataframe, selected_rows):
    dataframe.loc[selected_rows.index, 'prev_time_step_MajorAxisLength'] = \
        selected_rows['prev_time_step_MajorAxisLength'].values

    dataframe.loc[selected_rows.index, 'LengthChangeRatio'] = selected_rows['LengthChangeRatio'].values

    dataframe.loc[selected_rows.index, 'prev_time_step_center_x'] = selected_rows['prev_time_step_center_x'].values
    dataframe.loc[selected_rows.index, 'prev_time_step_center_y'] = selected_rows['prev_time_step_center_y'].values
    dataframe.loc[selected_rows.index, 'prev_time_step_endpoint1_X'] = \
        selected_rows['prev_time_step_endpoint1_X'].values
    dataframe.loc[selected_rows.index, 'prev_time_step_endpoint1_Y'] = \
        selected_rows['prev_time_step_endpoint1_Y'].values
    dataframe.loc[selected_rows.index, 'prev_time_step_endpoint2_X'] = \
        selected_rows['prev_time_step_endpoint2_X'].values
    dataframe.loc[selected_rows.index, 'prev_time_step_endpoint2_Y'] = \
        selected_rows['prev_time_step_endpoint2_Y'].values

    dataframe.loc[selected_rows.index, "bacteria_movement"] = selected_rows["bacteria_movement"].values

    dataframe.loc[selected_rows.index, "direction_of_motion"] = selected_rows["direction_of_motion"].values
    dataframe.loc[selected_rows.index, "TrajectoryX"] = selected_rows["TrajectoryX"].values
    dataframe.loc[selected_rows.index, "TrajectoryY"] = selected_rows["TrajectoryY"].values

    return dataframe


def calc_features_features_to_continues_life_history(dataframe, calc_all, neighbor_df, division_df,
                                                     parent_image_number_col,
                                                     parent_object_number_col, center_coordinate_columns,
                                                     flag_selected_rows, original_df=None):
    if calc_all:
        dataframe['prev_time_step_MajorAxisLength'] = dataframe.groupby('id')["AreaShape_MajorAxisLength"].shift(1)
        dataframe['LengthChangeRatio'] = (dataframe["AreaShape_MajorAxisLength"] /
                                          dataframe['prev_time_step_MajorAxisLength'])

    dataframe['prev_time_step_center_x'] = dataframe.groupby('id')[center_coordinate_columns['x']].shift(1)
    dataframe['prev_time_step_center_y'] = dataframe.groupby('id')[center_coordinate_columns['y']].shift(1)
    dataframe['prev_time_step_endpoint1_X'] = dataframe.groupby('id')['endpoint1_X'].shift(1)
    dataframe['prev_time_step_endpoint1_Y'] = dataframe.groupby('id')["endpoint1_Y"].shift(1)
    dataframe['prev_time_step_endpoint2_X'] = dataframe.groupby('id')["endpoint2_X"].shift(1)
    dataframe['prev_time_step_endpoint2_Y'] = dataframe.groupby('id')["endpoint2_Y"].shift(1)

    center_movement = \
        np.linalg.norm(dataframe[[center_coordinate_columns['x'], center_coordinate_columns['y']]].values -
                       dataframe[['prev_time_step_center_x', 'prev_time_step_center_y']].values, axis=1)

    endpoint1_1_movement = \
        np.linalg.norm(dataframe[['endpoint1_X', 'endpoint1_Y']].values -
                       dataframe[['prev_time_step_endpoint1_X', 'prev_time_step_endpoint1_Y']].values, axis=1)

    endpoint1_endpoint2_movement = \
        np.linalg.norm(dataframe[['endpoint1_X', 'endpoint1_Y']].values -
                       dataframe[['prev_time_step_endpoint2_X', 'prev_time_step_endpoint2_Y']].values, axis=1)

    endpoint2_2_movement = \
        np.linalg.norm(dataframe[['endpoint2_X', 'endpoint2_Y']].values -
                       dataframe[['prev_time_step_endpoint2_X', 'prev_time_step_endpoint2_Y']].values, axis=1)

    endpoint2_endpoint1_movement = \
        np.linalg.norm(dataframe[['endpoint2_X', 'endpoint2_Y']].values -
                       dataframe[['prev_time_step_endpoint1_X', 'prev_time_step_endpoint1_Y']].values, axis=1)

    dataframe["bacteria_movement"] = \
        np.minimum.reduce([center_movement, endpoint1_1_movement, endpoint2_2_movement,
                           endpoint1_endpoint2_movement, endpoint2_endpoint1_movement])

    if calc_all:

        bac_need_to_cal_dir_motion_condition = dataframe["direction_of_motion"].isna()

        direction_of_motion = \
            calculate_trajectory_direction_angle_all(dataframe[bac_need_to_cal_dir_motion_condition],
                                                     center_coordinate_columns)

        dataframe.loc[bac_need_to_cal_dir_motion_condition, "direction_of_motion"] = direction_of_motion

        calculated_trajectory_x = calculate_trajectory_direction_all(
            dataframe.loc[bac_need_to_cal_dir_motion_condition],
            center_coordinate_columns, mode='x')

        calculated_trajectory_y = calculate_trajectory_direction_all(
            dataframe.loc[bac_need_to_cal_dir_motion_condition],
            center_coordinate_columns, mode='y')

        dataframe.loc[bac_need_to_cal_dir_motion_condition, 'TrajectoryX'] = calculated_trajectory_x
        dataframe.loc[bac_need_to_cal_dir_motion_condition, 'TrajectoryY'] = calculated_trajectory_y

        if flag_selected_rows:

            original_df = update_df_based_on_selected_rows(original_df, dataframe)
            selected_rows = original_df.loc[(original_df['ImageNumber'] >= dataframe['ImageNumber'].min()) &
                                            (original_df['ImageNumber'] <= dataframe['ImageNumber'].max())]

            dataframe = calc_neighbors_dir_motion_all(original_df, neighbor_df, division_df, parent_image_number_col,
                                                      parent_object_number_col, selected_rows)

        else:
            dataframe = calc_neighbors_dir_motion_all(dataframe, neighbor_df, division_df, parent_image_number_col,
                                                      parent_object_number_col)

    return dataframe


def adding_features_to_continues_life_history(dataframe, neighbor_df, division_df, center_coordinate_columns,
                                              parent_image_number_col, parent_object_number_col,
                                              calc_all=True, selected_rows=None, bac_with_neighbors=None):
    if selected_rows is not None:

        dataframe = calc_features_features_to_continues_life_history(selected_rows, calc_all, neighbor_df, division_df,
                                                                     parent_image_number_col,
                                                                     parent_object_number_col,
                                                                     center_coordinate_columns,
                                                                     flag_selected_rows=True, original_df=dataframe)

    else:
        dataframe = calc_features_features_to_continues_life_history(dataframe, calc_all, neighbor_df, division_df,
                                                                     parent_image_number_col,
                                                                     parent_object_number_col,
                                                                     center_coordinate_columns,
                                                                     flag_selected_rows=False)

    return dataframe


def adding_features_to_continues_life_history_after_oad(dataframe, neighbor_df, division_df, center_coordinate_columns,
                                                        parent_image_number_col, parent_object_number_col):
    dataframe['prev_time_step_MajorAxisLength'] = dataframe.groupby('id')["AreaShape_MajorAxisLength"].shift(1)

    dataframe['LengthChangeRatio'] = (dataframe["AreaShape_MajorAxisLength"] /
                                      dataframe['prev_time_step_MajorAxisLength'])

    dataframe['prev_time_step_center_x'] = dataframe.groupby('id')[center_coordinate_columns['x']].shift(1)
    dataframe['prev_time_step_center_y'] = dataframe.groupby('id')[center_coordinate_columns['y']].shift(1)
    dataframe['prev_time_step_endpoint1_X'] = dataframe.groupby('id')['endpoint1_X'].shift(1)
    dataframe['prev_time_step_endpoint1_Y'] = dataframe.groupby('id')["endpoint1_Y"].shift(1)
    dataframe['prev_time_step_endpoint2_X'] = dataframe.groupby('id')["endpoint2_X"].shift(1)
    dataframe['prev_time_step_endpoint2_Y'] = dataframe.groupby('id')["endpoint2_Y"].shift(1)

    center_movement = \
        np.linalg.norm(dataframe[[center_coordinate_columns['x'], center_coordinate_columns['y']]].values -
                       dataframe[['prev_time_step_center_x', 'prev_time_step_center_y']].values, axis=1)

    endpoint1_1_movement = \
        np.linalg.norm(dataframe[['endpoint1_X', 'endpoint1_Y']].values -
                       dataframe[['prev_time_step_endpoint1_X', 'prev_time_step_endpoint1_Y']].values, axis=1)

    endpoint1_endpoint2_movement = \
        np.linalg.norm(dataframe[['endpoint1_X', 'endpoint1_Y']].values -
                       dataframe[['prev_time_step_endpoint2_X', 'prev_time_step_endpoint2_Y']].values, axis=1)

    endpoint2_2_movement = \
        np.linalg.norm(dataframe[['endpoint2_X', 'endpoint2_Y']].values -
                       dataframe[['prev_time_step_endpoint2_X', 'prev_time_step_endpoint2_Y']].values, axis=1)

    endpoint2_endpoint1_movement = \
        np.linalg.norm(dataframe[['endpoint2_X', 'endpoint2_Y']].values -
                       dataframe[['prev_time_step_endpoint1_X', 'prev_time_step_endpoint1_Y']].values, axis=1)

    dataframe["bacteria_movement"] = \
        np.minimum.reduce([center_movement, endpoint1_1_movement, endpoint2_2_movement,
                           endpoint1_endpoint2_movement, endpoint2_endpoint1_movement])

    bac_need_to_cal_dir_motion_condition = dataframe["direction_of_motion"].isna()

    direction_of_motion = \
        calculate_trajectory_direction_angle_all(dataframe.loc[bac_need_to_cal_dir_motion_condition],
                                                 center_coordinate_columns)

    dataframe.loc[bac_need_to_cal_dir_motion_condition, "direction_of_motion"] = direction_of_motion

    calculated_trajectory_x = calculate_trajectory_direction_all(dataframe.loc[bac_need_to_cal_dir_motion_condition],
                                                                 center_coordinate_columns, mode='x')

    calculated_trajectory_y = calculate_trajectory_direction_all(dataframe.loc[bac_need_to_cal_dir_motion_condition],
                                                                 center_coordinate_columns, mode='y')

    dataframe.loc[bac_need_to_cal_dir_motion_condition, 'TrajectoryX'] = calculated_trajectory_x

    dataframe.loc[bac_need_to_cal_dir_motion_condition, 'TrajectoryY'] = calculated_trajectory_y

    dataframe = calc_neighbors_dir_motion_all(dataframe, neighbor_df, division_df, parent_image_number_col,
                                              parent_object_number_col)

    return dataframe


def calc_new_features_after_rpl(df, neighbor_df, center_coordinate_columns, parent_image_number_col,
                                parent_object_number_col, label_col):
    """
    goal: assign new features like: `id`, `divideFlag`, `daughters_index`, `bad_division_flag`,
    `unexpected_end`, `division_time`, `unexpected_beginning`, `LifeHistory`, `parent_id` to bacteria and find errors

    @param df dataframe bacteria features value
    """

    df["divideFlag"] = False
    df['daughters_index'] = ''
    df['division_time'] = np.nan
    df['parent_index'] = np.nan
    df['other_daughter_index'] = np.nan

    # check division
    # _2: bac2, _1: bac1 (source bac)
    merged_df = df.merge(df, left_on=[parent_image_number_col, parent_object_number_col],
                         right_on=['ImageNumber', 'ObjectNumber'], how='inner', suffixes=('_2', '_1'))

    division = \
        merged_df[merged_df.duplicated(subset='index_1', keep=False)][['ImageNumber_1', 'ObjectNumber_1',
                                                                       'index_1', 'id_1', 'index_2',
                                                                       'ImageNumber_2', 'ObjectNumber_2',
                                                                       'AreaShape_MajorAxisLength_1',
                                                                       'AreaShape_MajorAxisLength_2',
                                                                       'TrackObjects_Label_50_1',
                                                                       'bacteria_slope_1', 'bacteria_slope_2',
                                                                       center_coordinate_columns['x'] + '_1',
                                                                       center_coordinate_columns['y'] + '_1',
                                                                       center_coordinate_columns['x'] + '_2',
                                                                       center_coordinate_columns['y'] + '_2',
                                                                       parent_image_number_col + '_2',
                                                                       parent_object_number_col + '_2']].copy()

    division['daughters_index'] = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['index_2'].transform(lambda x: ', '.join(x.astype(str)))

    division['daughter_mother_slope'] = \
        calculate_orientation_angle_batch(division['bacteria_slope_2'].values,
                                          division['bacteria_slope_1'].values)

    division['daughter_mother_LengthChangeRatio'] = (division['AreaShape_MajorAxisLength_2'] /
                                                     division['AreaShape_MajorAxisLength_1'])

    daughter_to_daughter = division.merge(division,
                                          on=[parent_image_number_col + '_2', parent_object_number_col + '_2'],
                                          suffixes=('_daughter1', '_daughter2'))

    daughter_to_daughter = daughter_to_daughter.loc[daughter_to_daughter['ObjectNumber_2_daughter1'] !=
                                                    daughter_to_daughter['ObjectNumber_2_daughter2']]

    mothers_df_last_time_step = division.drop_duplicates(subset='index_1', keep='first')

    df.loc[division['index_1'].unique(), 'daughters_index'] = mothers_df_last_time_step['daughters_index'].values
    df.loc[division['index_1'].unique(), 'division_time'] = mothers_df_last_time_step['ImageNumber_2'].values
    df.loc[division['index_2'], 'parent_index'] = division['index_1'].values
    df.loc[division['index_2'].values, 'daughter_mother_LengthChangeRatio'] = \
        division['daughter_mother_LengthChangeRatio'].values

    df.loc[daughter_to_daughter['index_2_daughter1'].values, "other_daughter_index"] = \
        daughter_to_daughter['index_2_daughter2'].values

    df.loc[division['index_2'].values, 'prev_bacteria_slope'] = division['bacteria_slope_1'].values
    df.loc[division['index_2'].values, 'slope_bac_bac'] = division['daughter_mother_slope'].values

    df['LifeHistory'] = df.groupby('id')['id'].transform('size')

    mothers_id = division['id_1'].unique()
    df.loc[df['id'].isin(mothers_id), "divideFlag"] = True

    df['division_time'] = df.groupby('id')['division_time'].transform(lambda x: x.ffill().bfill())
    df['daughters_index'] = df.groupby('id')['daughters_index'].transform(lambda x: x.ffill().bfill())

    bac_idx_not_needed_to_update = df.loc[(~ df['prev_bacteria_slope'].isna())].index.values
    temporal_df = df.loc[bac_idx_not_needed_to_update]

    df['prev_bacteria_slope'] = df.groupby('id')['bacteria_slope'].shift(1)
    df.loc[bac_idx_not_needed_to_update, 'prev_bacteria_slope'] = temporal_df['prev_bacteria_slope'].values

    bac_need_to_cal_dir_motion_condition = (df['slope_bac_bac'].isna() &
                                            df['unexpected_beginning'] == False &
                                            df['ImageNumber'] != 1)

    df.loc[bac_need_to_cal_dir_motion_condition, 'slope_bac_bac'] = \
        calculate_orientation_angle_batch(df[bac_need_to_cal_dir_motion_condition, 'bacteria_slope'].values,
                                          df[bac_need_to_cal_dir_motion_condition, 'prev_bacteria_slope'])

    df = calc_neighbors_dir_motion_all(df, neighbor_df, division, parent_image_number_col, parent_object_number_col)

    df['checked'] = True

    df['division_time'] = df['division_time'].fillna(0)
    df['daughters_index'] = df['daughters_index'].fillna('')

    # set age
    df['age'] = df.groupby('id').cumcount() + 1

    return df


def find_neighbors_info(df, neighbors_df, bac):
    neighbors = neighbors_df.loc[(neighbors_df['First Image Number'] == bac['ImageNumber']) &
                                 (neighbors_df['First Object Number'] == bac['ObjectNumber'])]

    if neighbors.shape[0] > 0:
        neighbors_df = \
            neighbors.merge(df, left_on=['Second Image Number', 'Second Object Number'],
                            right_on=['ImageNumber', 'ObjectNumber'], how='inner')
    else:
        neighbors_df = pd.DataFrame()

    return neighbors_df


def calculate_trajectory_direction(previous_position, current_position):
    # Calculate the direction vector from the previous position to the current position
    direction = current_position - previous_position

    return direction


def calculate_trajectory_direction_all(df, center_coordinate_columns, mode):
    # Calculate the direction vector from the previous position to the current position
    if mode == 'x':
        return df[center_coordinate_columns['x']] - df['prev_time_step_center_x']
    elif mode == 'y':
        return df[center_coordinate_columns['y']] - df['prev_time_step_center_y']

    return df


def calculate_trajectory_direction_angle_all(df, center_coordinate_columns, col_target=''):
    # Calculate the direction vector from the previous position to the current position
    trajectory_direction = \
        (df[[center_coordinate_columns['x'] + col_target, center_coordinate_columns['y'] + col_target]].values -
         df[['prev_time_step_center_x' + col_target, 'prev_time_step_center_y' + col_target]].values)

    # Calculate the angle of the direction vector
    angle = np.arctan2(trajectory_direction[:, 1], trajectory_direction[:, 0])

    return angle


def calculate_trajectory_direction_daughters_mother(df, center_coordinate_columns):
    # Calculate the direction vector from the previous position to the current position
    trajectory_direction = \
        (df[[center_coordinate_columns['x'] + '_2', center_coordinate_columns['y'] + '_2']].values -
         df[[center_coordinate_columns['x'] + '_1', center_coordinate_columns['y'] + '_1']].values)

    # Calculate the angle of the direction vector
    angle = np.arctan2(trajectory_direction[:, 1], trajectory_direction[:, 0])

    return angle


def calculate_slope_intercept(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    else:
        slope = np.nan
        intercept = np.nan
    return slope, intercept


def calculate_slope_intercept_batch(df):
    df.loc[df['endpoint1_X'] == df['endpoint2_X'], 'bacteria_slope'] = np.nan

    df.loc[df['endpoint1_X'] != df['endpoint2_X'], 'bacteria_slope'] = (
            (df['endpoint2_Y'] - df['endpoint1_Y']) / (df['endpoint2_X'] - df['endpoint1_X']))

    return df


# find vertices of an ellipse (bacteria):
# references:
# https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate
# https://math.stackexchange.com/questions/2645689/what-is-the-parametric-equation-of-a-rotated-ellipse-given-the-angle-of-rotatio
def find_vertex(center, major, angle_rotation, angle_tolerance=1e-6):
    if np.abs(angle_rotation - np.pi / 2) < angle_tolerance:  # Bacteria parallel to the vertical axis
        vertex_1_x = center[0]
        vertex_1_y = center[1] + major
        vertex_2_x = center[0]
        vertex_2_y = center[1] - major

    elif np.abs(angle_rotation) < angle_tolerance:  # Bacteria parallel to the horizontal axis
        vertex_1_x = center[0] + major
        vertex_1_y = center[1]
        vertex_2_x = center[0] - major
        vertex_2_y = center[1]

    else:
        # (x- center_x) * np.sin(angle_rotation) - (y-center_y) * np.cos(angle_rotation) = 0
        # np.power((x - center_x) * np.cos(angle_rotation) + (y - center_y) * np.sin(angle_rotation), 2) =
        # np.power(major, 2)
        # vertex_1_x = semi_major / (np.cos(angle_rotation) + np.tan(angle_rotation) * np.sin(angle_rotation)) +
        #              center[0]
        # vertex_2_x = -semi_major / (np.cos(angle_rotation) + np.tan(angle_rotation) * np.sin(angle_rotation)) +
        #               center[0]

        semi_major = major / 2
        temp_vertex_1_x = float((semi_major * np.cos(angle_rotation)) + center[0])
        temp_vertex_1_y = float((temp_vertex_1_x - center[0]) * np.tan(angle_rotation) + center[1])
        temp_vertex_2_x = float((- semi_major * np.cos(angle_rotation)) + center[0])
        temp_vertex_2_y = float((temp_vertex_2_x - center[0]) * np.tan(angle_rotation) + center[1])

        if temp_vertex_1_x > center[0]:
            vertex_1_x = temp_vertex_1_x
            vertex_1_y = temp_vertex_1_y
            vertex_2_x = temp_vertex_2_x
            vertex_2_y = temp_vertex_2_y
        else:
            vertex_1_x = temp_vertex_2_x
            vertex_1_y = temp_vertex_2_y
            vertex_2_x = temp_vertex_1_x
            vertex_2_y = temp_vertex_1_y

    return [[vertex_1_x, vertex_1_y], [vertex_2_x, vertex_2_y]]


def find_vertex_batch(df, center_coordinate_columns, angle_tolerance=1e-6):
    # condition 1
    # Bacteria parallel to the vertical axis
    condition1 = np.abs(df["AreaShape_Orientation"] - np.pi / 2) < angle_tolerance
    df.loc[condition1, 'endpoint1_X'] = df[center_coordinate_columns['x']]
    df.loc[condition1, 'endpoint1_Y'] = df[center_coordinate_columns['y']] + df["AreaShape_MajorAxisLength"]
    df.loc[condition1, 'endpoint2_X'] = df[center_coordinate_columns['x']]
    df.loc[condition1, 'endpoint2_Y'] = df[center_coordinate_columns['y']] - df["AreaShape_MajorAxisLength"]

    # Bacteria parallel to the horizontal axis
    condition2 = np.abs(df["AreaShape_Orientation"]) < angle_tolerance
    condition3 = ~condition1 & condition2
    df.loc[condition3, 'endpoint1_X'] = df[center_coordinate_columns['x']] + df["AreaShape_MajorAxisLength"]
    df.loc[condition3, 'endpoint1_Y'] = df[center_coordinate_columns['y']]
    df.loc[condition3, 'endpoint2_X'] = df[center_coordinate_columns['x']] - df["AreaShape_MajorAxisLength"]
    df.loc[condition3, 'endpoint2_Y'] = df[center_coordinate_columns['y']]

    # (x- center_x) * np.sin(angle_rotation) - (y-center_y) * np.cos(angle_rotation) = 0
    # np.power((x - center_x) * np.cos(angle_rotation) + (y - center_y) * np.sin(angle_rotation), 2) =
    # np.power(major, 2)
    condition4 = ~condition1 & ~condition2
    other_bac_df = df.loc[condition4][["AreaShape_MajorAxisLength", "AreaShape_Orientation",
                                       center_coordinate_columns['x'], center_coordinate_columns['y']]].copy()

    other_bac_df['semi_major'] = df["AreaShape_MajorAxisLength"] / 2
    other_bac_df['temp_vertex_1_x'] = \
        (other_bac_df['semi_major'] * np.cos(other_bac_df["AreaShape_Orientation"]) +
         other_bac_df[center_coordinate_columns['x']])

    other_bac_df['temp_vertex_1_y'] = \
        ((other_bac_df['temp_vertex_1_x'] - other_bac_df[center_coordinate_columns['x']]) *
         np.tan(other_bac_df["AreaShape_Orientation"]) + other_bac_df[center_coordinate_columns['y']])

    other_bac_df['temp_vertex_2_x'] = \
        (-other_bac_df['semi_major'] * np.cos(other_bac_df["AreaShape_Orientation"]) +
         other_bac_df[center_coordinate_columns['x']])

    other_bac_df['temp_vertex_2_y'] = \
        ((other_bac_df['temp_vertex_2_x'] - other_bac_df[center_coordinate_columns['x']]) *
         np.tan(other_bac_df["AreaShape_Orientation"]) + other_bac_df[center_coordinate_columns['y']])

    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] > other_bac_df[center_coordinate_columns['x']], 'vertex_1_x'] = \
        other_bac_df['temp_vertex_1_x']
    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] > other_bac_df[center_coordinate_columns['x']], 'vertex_1_y'] = \
        other_bac_df['temp_vertex_1_y']
    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] > other_bac_df[center_coordinate_columns['x']], 'vertex_2_x'] = \
        other_bac_df['temp_vertex_2_x']
    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] > other_bac_df[center_coordinate_columns['x']], 'vertex_2_y'] = \
        other_bac_df['temp_vertex_2_y']

    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] < other_bac_df[center_coordinate_columns['x']], 'vertex_1_x'] = \
        other_bac_df['temp_vertex_2_x']
    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] < other_bac_df[center_coordinate_columns['x']], 'vertex_1_y'] = \
        other_bac_df['temp_vertex_2_y']
    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] < other_bac_df[center_coordinate_columns['x']], 'vertex_2_x'] = \
        other_bac_df['temp_vertex_1_x']
    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] < other_bac_df[center_coordinate_columns['x']], 'vertex_2_y'] = \
        other_bac_df['temp_vertex_1_y']

    df.loc[condition4, 'endpoint1_X'] = other_bac_df['vertex_1_x']
    df.loc[condition4, 'endpoint1_Y'] = other_bac_df['vertex_1_y']
    df.loc[condition4, 'endpoint2_X'] = other_bac_df['vertex_2_x']
    df.loc[condition4, 'endpoint2_Y'] = other_bac_df['vertex_2_y']

    return df


def angle_convert_to_radian(df):
    # modification of bacterium orientation
    # -(angle + 90) * np.pi / 180
    df["AreaShape_Orientation"] = -(df["AreaShape_Orientation"] + 90) * np.pi / 180

    return df


def bacteria_features(df, center_coordinate_columns):
    """
    output: length, radius, orientation, center coordinate
    """
    major = df['AreaShape_MajorAxisLength']
    minor = df['AreaShape_MinorAxisLength']
    radius = df['AreaShape_MinorAxisLength'] / 2
    orientation = df['AreaShape_Orientation']
    center_x = df[center_coordinate_columns['x']]
    center_y = df[center_coordinate_columns['y']]

    features = {'major': major, 'minor': minor, 'radius': radius, 'orientation': orientation, 'center_x': center_x,
                'center_y': center_y}
    return features


def k_nearest_neighbors(target_bacterium, other_bacteria, center_coordinate_columns, k=1, distance_threshold=None,
                        distance_check=True):
    """
    goal: find k nearest neighbors to target bacterium
    @param target_bacterium  series value of features of bacterium that we want to find its neighbors
    @param other_bacteria dataframe
    @param distance_check bool Is the distance threshold checked?
    @return nearest_neighbors list index of the nearest bacteria
    """
    # calculate distance matrix
    distance_df = calc_distance_matrix(target_bacterium, other_bacteria, center_coordinate_columns['x'],
                                       center_coordinate_columns['y'])

    distance_df = distance_df.reset_index(drop=True).sort_values(by=0, ascending=True, axis=1)
    distance_val = list(zip(distance_df.columns.values, distance_df.iloc[0].values))

    if distance_check:
        nearest_neighbors_index = [elem[0] for elem in distance_val if elem[1] <= distance_threshold][
                                  :min(k, len(distance_val))]
    else:
        nearest_neighbors_index = [elem[0] for elem in distance_val][:min(k, len(distance_val))]

    return nearest_neighbors_index


def calc_normalized_angle_between_motion_for_df(df, this_link_motion_x_col, this_link_motion_y_col,
                                                neighbors_avg_dir_motion_x, neighbors_avg_dir_motion_y):
    df[this_link_motion_x_col] = df[this_link_motion_x_col].astype(float)
    df[this_link_motion_y_col] = df[this_link_motion_y_col].astype(float)
    df[neighbors_avg_dir_motion_x] = df[neighbors_avg_dir_motion_x].astype(float)
    df[neighbors_avg_dir_motion_y] = df[neighbors_avg_dir_motion_y].astype(float)

    # Calculate the dot product
    df['dot_product'] = (df[this_link_motion_x_col] * df[neighbors_avg_dir_motion_x] +
                         df[this_link_motion_y_col] * df[neighbors_avg_dir_motion_y])

    # Calculate the magnitudes of the vectors

    df['magnitude_a'] = np.sqrt(np.power(df[this_link_motion_x_col], 2) + np.power(df[this_link_motion_y_col], 2))
    df['magnitude_b'] = \
        np.sqrt(np.power(df[neighbors_avg_dir_motion_x], 2) + np.power(df[neighbors_avg_dir_motion_y], 2))

    # Calculate the cosine of the angle
    condition1 = (df['magnitude_a'] == 0) | (df['magnitude_b'] == 0)
    df.loc[condition1, 'cos_angle'] = np.nan  # Angle is undefined
    condition2 = (df['magnitude_a'] != 0) & (df['magnitude_b'] != 0)
    df.loc[condition2, 'cos_angle'] = df.loc[condition2, 'dot_product'] / (
            df.loc[condition2, 'magnitude_a'] * df.loc[condition2, 'magnitude_b'])

    # Calculate the angle in radians and then convert to degrees
    df['angle_radians'] = np.arccos(df['cos_angle'])
    df['angle_degrees'] = np.degrees(df['angle_radians'])
    df['angle_degrees'] = df['angle_degrees'] / 180

    return df


def calc_normalized_angle_between_motion(motion1, motion2):
    # Calculate the dot product
    dot_product = motion1[0] * motion2[0] + motion1[1] * motion2[1]

    # Calculate the magnitudes of the vectors
    magnitude_a = np.hypot(motion1[0], motion1[1])
    magnitude_b = np.hypot(motion2[0], motion2[1])

    # Calculate the cosine of the angle
    if magnitude_a == 0 or magnitude_b == 0:
        cos_angle = np.nan  # Angle is undefined
    else:
        cos_angle = dot_product / (magnitude_a * magnitude_b)

    # Calculate the angle in radians and then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees / 180


def calc_normalized_angle_between_motion_array(neighbors_dir_motion, direction_of_motions):
    # Dot product
    dot_product = np.sum(neighbors_dir_motion * direction_of_motions, axis=1)

    # Magnitudes
    magnitude_neighbor = np.linalg.norm(neighbors_dir_motion)

    magnitude_direction = np.linalg.norm(direction_of_motions, axis=1)

    # Safe division (avoid division by zero)
    valid_indices = (magnitude_neighbor != 0) & (magnitude_direction != 0)
    cos_angle = np.full(dot_product.shape, np.nan)  # Initialize with NaNs

    cos_angle[valid_indices] = dot_product[valid_indices] / (magnitude_neighbor * magnitude_direction[valid_indices])

    # Calculate the angle in radians and then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)

    angle_between_motions = angle_degrees / 180  # Normalize by 180

    return angle_between_motions


def calc_normalized_angle_between_motion_all(neighbors_dir_motion, direction_of_motions):
    # Dot product
    dot_product = np.sum(neighbors_dir_motion * direction_of_motions, axis=1)

    # Magnitudes
    magnitude_neighbors = np.linalg.norm(neighbors_dir_motion, axis=1)

    magnitude_direction = np.linalg.norm(direction_of_motions, axis=1)

    # Safe division (avoid division by zero)
    valid_indices = (magnitude_neighbors != 0) & (magnitude_direction != 0)
    cos_angle = np.full(dot_product.shape, np.nan)  # Initialize with NaNs

    cos_angle[valid_indices] = dot_product[valid_indices] / (
            magnitude_neighbors[valid_indices] * magnitude_direction[valid_indices])

    # Calculate the angle in radians and then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)

    angle_between_motions = angle_degrees / 180  # Normalize by 180

    return angle_between_motions


def calculate_orientation_angle(slope1, slope2):
    # Calculate the angle in radians between the lines
    angle_radians = np.arctan(abs((slope1 - slope2) / (1 + slope1 * slope2)))
    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)

    if str(angle_degrees) == 'nan':
        angle_degrees = 90
    return angle_degrees


def calculate_orientation_angle_batch(slope1, slope2):
    # Calculate the angle in radians between the lines
    angle_radians = np.arctan(abs((slope1 - slope2) / (1 + slope1 * slope2)))
    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)

    # Replace NaN values with 90
    angle_degrees = np.nan_to_num(angle_degrees, nan=90)

    return angle_degrees


def distance_normalization(df, distance_df):
    # Extract 'bacteria_movement' values, drop NaNs
    bacteria_movement = df['bacteria_movement'].dropna()

    # Calculate min and max
    min_val = bacteria_movement.min()
    max_val = bacteria_movement.max()

    # Normalize the entire DataFrame at once
    normalized_df = (distance_df - min_val) / (max_val - min_val)

    return normalized_df


def slope_normalization(df, slope_df):
    # Extract 'bacteria_slope_changes' values, drop NaNs
    bacteria_slope_changes = df['slope_bac_bac'].dropna()

    # Calculate min and max
    min_val = bacteria_slope_changes.min()
    max_val = bacteria_slope_changes.max()

    # Normalize the entire DataFrame at once
    slope_df = (slope_df - min_val) / (max_val - min_val)

    return slope_df


def min_max_normalize_row(row):
    min_val = row.min()
    max_val = row.max()
    return (row - min_val) / (max_val - min_val)


def calc_distance_matrix(target_bacteria, other_bacteria, col1, col2, col3=None, col4=None, normalization=False):
    """
    goal: this function is useful to create adjacency matrix (distance matrix)
    I want to find distance of one dataframe to another
    example1: distance of unexpected_beginning bacteria from all bacteria in previous time step
    example1: distance of unexpected-end bacterium from all bacteria in same time step

    @param target_bacteria dataframe or series The value of the features of the bacteria that we want
    to find its distance from other bacteria
    @param other_bacteria dataframe or series
    @param col1 str distance of bacteria will be calculated depending on `col1` value
    @param col2 str distance of bacteria will be calculated depending on `col2` value

    """
    # create distance matrix (rows: next time step sudden bacteria, columns: another time step bacteria)
    if col3 is None and col4 is None:
        distance_df = pd.DataFrame(distance_matrix(target_bacteria[[col1, col2]].values,
                                                   other_bacteria[[col1, col2]].values),
                                   index=target_bacteria.index, columns=other_bacteria.index)
    else:
        distance_df = pd.DataFrame(distance_matrix(target_bacteria[[col1, col2]].values,
                                                   other_bacteria[[col3, col4]].values),
                                   index=target_bacteria.index, columns=other_bacteria.index)

    if normalization:
        # Apply Min-Max normalization to each row
        distance_df = distance_df.apply(min_max_normalize_row, axis=1)

    return distance_df


def bacteria_in_specific_time_step(dataframe, t):
    """
    goal: find bacteria in specific time step with ['drop'] = False
    @param dataframe dataframe bacteria information dataframe
    @param t int timestep
    """
    # & (dataframe["drop"] == False)
    correspond_bacteria = dataframe.loc[(dataframe["ImageNumber"] == t)]

    return correspond_bacteria


def remove_rows(df, col, true_value):
    """
        goal: remove bacteria that have no parents
        @param df    dataframe   bacteria dataframe
        @param col str column name
        @param true_value bool
        output: df   dataframe   modified dataframe

    """

    df = df.loc[df[col] == true_value].reset_index(drop=True)

    return df


def find_related_bacteria(df, target_bacterium, target_bacterium_index, parent_image_number_col,
                          parent_object_number_col, bacteria_index_list=None):
    """
    goal:  From the bacteria dataframe, find the row index of bacteria related (same or daughter) to
    a specific bacterium
    @param df dataframe features value of bacteria in each time step
    @param target_bacterium series target bacterium features value
    @param target_bacterium_index   int   The row index of the corresponding bacterium
    @param bacteria_index_list list row index of related bacteria
    @return  bacteria_index_list   list   row index of related bacteria
    """

    if bacteria_index_list is None:
        bacteria_index_list = [target_bacterium_index]

    # related bacteria in next time steps
    if target_bacterium['ImageNumber'] <= df.iloc[-1]['ImageNumber']:
        bac_in_next_timestep = df.loc[(df[parent_image_number_col] == target_bacterium['ImageNumber']) &
                                      (df[parent_object_number_col] == target_bacterium['ObjectNumber'])]

        if bac_in_next_timestep.shape[0] > 0:
            bacteria_index_list.extend(bac_in_next_timestep.index.tolist())
            for bac_index, next_bacterium in bac_in_next_timestep.iterrows():
                bacteria_index_list = find_related_bacteria(df, next_bacterium, bac_index, parent_image_number_col,
                                                            parent_object_number_col, bacteria_index_list)

    return bacteria_index_list


def convert_ends_to_pixel(ends, um_per_pixel=0.144):
    ends = np.array(ends) / um_per_pixel
    return ends


def convert_to_pixel(length, radius, ends, pos, um_per_pixel=0.144):
    # Convert distances to pixel (0.144 um/pixel on 63X objective)
    length = length / um_per_pixel
    radius = radius / um_per_pixel
    ends = np.array(ends) / um_per_pixel
    pos = np.array(pos) / um_per_pixel

    return length, radius, ends, pos


def convert_to_um(data_frame, um_per_pixel, all_center_coordinate_columns):
    # Convert distances to um (0.144 um/pixel on 63X objective)

    data_frame[all_center_coordinate_columns['x']] *= um_per_pixel

    data_frame[all_center_coordinate_columns['y']] *= um_per_pixel

    data_frame['AreaShape_MajorAxisLength'] *= um_per_pixel
    data_frame['AreaShape_MinorAxisLength'] *= um_per_pixel

    return data_frame


def checking_columns(data_frame):
    center_coordinate_columns = []
    all_center_coordinate_columns = []

    parent_image_number_col = [col for col in data_frame.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in data_frame.columns if 'TrackObjects_ParentObjectNumber_' in col][0]
    # column name
    label_col = [col for col in data_frame.columns if 'TrackObjects_Label_' in col][0]

    if 'Location_Center_X' in data_frame.columns and 'Location_Center_Y' in data_frame.columns:
        center_coordinate_columns = {'x': 'Location_Center_X', 'y': 'Location_Center_Y'}
        all_center_coordinate_columns = {'x': ['Location_Center_X'], 'y': ['Location_Center_Y']}

        if 'AreaShape_Center_X' in data_frame.columns and 'AreaShape_Center_Y' in data_frame.columns:
            all_center_coordinate_columns = {'x': ['Location_Center_X', 'AreaShape_Center_X'],
                                             'y': ['Location_Center_Y', 'AreaShape_Center_Y']}

    elif 'AreaShape_Center_X' in data_frame.columns and 'AreaShape_Center_Y' in data_frame.columns:
        center_coordinate_columns = {'x': 'AreaShape_Center_X', 'y': 'AreaShape_Center_Y'}
        all_center_coordinate_columns = {'x': ['AreaShape_Center_X'], 'y': ['AreaShape_Center_Y']}

    else:
        print('There was no column corresponding to the center of bacteria.')
        breakpoint()

    return (center_coordinate_columns, all_center_coordinate_columns, parent_image_number_col, parent_object_number_col,
            label_col)


def calc_new_features_after_oad(df, neighbor_df, center_coordinate_columns, parent_image_number_col,
                                parent_object_number_col, label_col):
    """
    goal: assign new features like: `id`, `divideFlag`, `daughters_index`, `bad_division_flag`,
    `unexpected_end`, `division_time`, `unexpected_beginning`, `LifeHistory`, `parent_id` to bacteria and find errors

    @param df dataframe bacteria features value
    """

    df["direction_of_motion"] = np.nan

    # check division
    # _2: bac2, _1: bac1 (source bac)
    merged_df = df.merge(df, left_on=[parent_image_number_col, parent_object_number_col],
                         right_on=['ImageNumber', 'ObjectNumber'], how='inner', suffixes=('_2', '_1'))

    division = \
        merged_df[merged_df.duplicated(subset='index_1', keep=False)][['ImageNumber_1', 'ObjectNumber_1',
                                                                       'index_1', 'index_2',
                                                                       'ImageNumber_2', 'ObjectNumber_2',
                                                                       'AreaShape_MajorAxisLength_1',
                                                                       'AreaShape_MajorAxisLength_2',
                                                                       center_coordinate_columns['x'] + '_1',
                                                                       center_coordinate_columns['y'] + '_1',
                                                                       center_coordinate_columns['x'] + '_2',
                                                                       center_coordinate_columns['y'] + '_2',
                                                                       parent_object_number_col + '_2',
                                                                       parent_image_number_col + '_2']].copy()

    division['daughters_index'] = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['index_2'].transform(lambda x: ', '.join(x.astype(str)))

    daughter_to_daughter = division.merge(division,
                                          on=[parent_image_number_col + '_2', parent_object_number_col + '_2'],
                                          suffixes=('_daughter1', '_daughter2'))

    daughter_to_daughter = daughter_to_daughter.loc[daughter_to_daughter['index_2_daughter1'] !=
                                                    daughter_to_daughter['index_2_daughter2']]

    mothers_df_last_time_step = division.drop_duplicates(subset='index_1', keep='first')

    division['daughters_TrajectoryX'] = \
        division[center_coordinate_columns['x'] + '_2'] - division[center_coordinate_columns['x'] + '_1']

    division['daughters_TrajectoryY'] = \
        division[center_coordinate_columns['y'] + '_2'] - division[center_coordinate_columns['y'] + '_1']

    direction_of_motion = calculate_trajectory_direction_daughters_mother(division, center_coordinate_columns)

    division["direction_of_motion"] = direction_of_motion

    sum_daughters_len = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['AreaShape_MajorAxisLength_2'].sum()

    max_daughters_len = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['AreaShape_MajorAxisLength_2'].max()

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

    df = df.merge(temp_df_sum_daughters, on=['ImageNumber', 'ObjectNumber'], how='outer')
    df = df.merge(temp_df_max_daughters, on=['ImageNumber', 'ObjectNumber'], how='outer')
    df = df.merge(temp_df_avg_daughters_trajectory_x, on=['ImageNumber', 'ObjectNumber'], how='outer')
    df = df.merge(temp_df_avg_daughters_trajectory_y, on=['ImageNumber', 'ObjectNumber'], how='outer')

    df.loc[division['index_1'].unique(), 'daughters_index'] = mothers_df_last_time_step['daughters_index'].values

    df.loc[division['index_2'].values, "direction_of_motion"] = division["direction_of_motion"].values

    df.loc[division['index_2'].values, "TrajectoryX"] = division['daughters_TrajectoryX'].values

    df.loc[division['index_2'].values, "TrajectoryY"] = division["daughters_TrajectoryY"].values
    df.loc[division['index_2'].values, "parent_index"] = division["index_1"].values

    df.loc[daughter_to_daughter['index_2_daughter1'].values, "other_daughter_index"] = \
        daughter_to_daughter['index_2_daughter2'].values

    df['daughters_index'] = df.groupby('id')['daughters_index'].transform(lambda x: x.ffill().bfill())

    df = adding_features_to_continues_life_history_after_oad(df, neighbor_df, division, center_coordinate_columns,
                                                             parent_image_number_col, parent_object_number_col)

    # average length of a bacteria in its life history
    df['AverageLengthChangeRatio'] = df.groupby('id')['LengthChangeRatio'].transform('mean')

    df['checked'] = True
    df['daughters_index'] = df['daughters_index'].fillna('')

    return df


def update_values_using_at(df, indices, cols, values):
    for j, col in enumerate(cols):
        for i, idx in enumerate(indices):
            df.at[idx, col] = values[j][i]

    return df


def update_values_using_at_same_val_for_col(df, indices, cols, values):
    for j, col in enumerate(cols):
        for i, idx in enumerate(indices):
            df.at[idx, col] = values[j]

    return df
