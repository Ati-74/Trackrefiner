import numpy as np
import pandas as pd
import time
from Trackrefiner.strain.correction.action.helperFunctions import remove_rows, convert_to_um, find_vertex_batch, \
    angle_convert_to_radian, calculate_slope_intercept_batch, print_progress_bar, \
    adding_features_to_continues_life_history, calculate_trajectory_direction_daughters_mother, \
    calculate_orientation_angle_batch
from Trackrefiner.strain.correction.action.fluorescenceIntensity import assign_cell_type
from Trackrefiner.strain.correction.nanLabel import modify_nan_labels
from Trackrefiner.strain.correction.unexpectedBeginning import correction_unexpected_beginning
from Trackrefiner.strain.correction.overAssignedDaughters import remove_over_assigned_daughters_link
from Trackrefiner.strain.correction.redundantParentLink import (detect_redundant_parent_link,
                                                                detect_and_remove_redundant_parent_link)
from Trackrefiner.strain.correction.noiseRemover import noise_remover
from Trackrefiner.strain.correction.neighborChecking import neighbor_checking
from Trackrefiner.strain.correction.missingConnectivityLink import missing_connectivity_link, \
    detect_missing_connectivity_link
from Trackrefiner.strain.correction.unExpectedEnd import unexpected_end_bacteria
from Trackrefiner.strain.correction.action.multiRegionsDetection import multi_region_detection
from Trackrefiner.strain.correction.action.finalMatching import final_matching
from Trackrefiner.strain.correction.action.Modeling.trainingModels import training_models
from Trackrefiner.strain.correction.action.wallCorrection import find_wall_objects
from Trackrefiner.strain.correction.action.generateLogFile import generate_log_file
from scipy.sparse import lil_matrix
import warnings
import logging
import traceback

# Configure logging to capture warnings
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Custom warning handler to include traceback
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    log_message = f"{message} (File: {filename}, Line: {lineno})"
    # Capture the current stack trace
    stack_trace = traceback.format_stack()
    # Log the warning with the stack trace
    logging.warning(f"{log_message}\nStack Trace:\n{''.join(stack_trace)}")


# Apply the custom warning handler
warnings.showwarning = custom_warning_handler


def assign_feature_find_errors(dataframe, intensity_threshold, check_cell_type, neighbor_df, center_coordinate_columns,
                               parent_image_number_col, parent_object_number_col, without_tracking_correction):
    """
    goal: assign new features like: `id`, `divideFlag`, `daughters_index`, `bad_division_flag`,
    `unexpected_end`, `division_time`, `unexpected_beginning`, `LifeHistory`, `parent_id` to bacteria and find errors

    @param dataframe dataframe bacteria features value
    @param intensity_threshold float min intensity value of channel
    """
    dataframe['checked'] = False

    dataframe["divideFlag"] = False
    dataframe['daughters_index'] = ''
    dataframe['bad_division_flag'] = False
    dataframe['ovd_flag'] = False
    dataframe['bad_daughters_flag'] = False

    dataframe['unexpected_end'] = False
    dataframe['division_time'] = np.nan
    dataframe['unexpected_beginning'] = False
    dataframe["parent_id"] = np.nan
    dataframe["bacteria_movement"] = np.nan

    # find vertex
    dataframe = find_vertex_batch(dataframe, center_coordinate_columns)
    dataframe = calculate_slope_intercept_batch(dataframe)

    # unexpected_beginning_bacteria
    cond1 = dataframe[parent_image_number_col] == 0
    cond2 = dataframe['ImageNumber'] > 1
    cond3 = dataframe['ImageNumber'] == 1

    dataframe.loc[cond1 & cond2, ['checked', 'unexpected_beginning', 'parent_id', "bacteria_movement"]] = \
        [
            True,
            True,
            0,
            np.nan,
        ]

    dataframe.loc[cond1 & cond3, ['checked', 'unexpected_beginning', 'parent_id', "bacteria_movement"]] = \
        [
            True,
            False,
            0,
            np.nan
        ]

    dataframe.loc[cond1, 'id'] = dataframe.loc[cond1, 'index'].values + 1
    # dataframe.loc[cond1, label_col] = dataframe.loc[cond1, 'index'].values + 1

    # check division
    # _2: bac2, _1: bac1 (source bac)
    merged_df = dataframe.merge(dataframe, left_on=[parent_image_number_col, parent_object_number_col],
                                right_on=['ImageNumber', 'ObjectNumber'], how='inner', suffixes=('_2', '_1'))

    division = merged_df[merged_df.duplicated(subset='index_1', keep=False)][['ImageNumber_1', 'ObjectNumber_1',
                                                                              'index_1', 'id_1', 'index_2',
                                                                              'ImageNumber_2', 'ObjectNumber_2',
                                                                              'AreaShape_MajorAxisLength_1',
                                                                              'AreaShape_MajorAxisLength_2',
                                                                              'bacteria_slope_1', 'bacteria_slope_2',
                                                                              center_coordinate_columns['x'] + '_1',
                                                                              center_coordinate_columns['y'] + '_1',
                                                                              center_coordinate_columns['x'] + '_2',
                                                                              center_coordinate_columns['y'] + '_2',
                                                                              parent_object_number_col + '_2',
                                                                              parent_image_number_col + '_2']].copy()

    division['daughters_index'] = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['index_2'].transform(lambda x: ', '.join(x.astype(str)))

    division['daughters_TrajectoryX'] = \
        (division[center_coordinate_columns['x'] + '_2'] - division[center_coordinate_columns['x'] + '_1'])

    division['daughters_TrajectoryY'] = \
        (division[center_coordinate_columns['y'] + '_2'] - division[center_coordinate_columns['y'] + '_1'])

    direction_of_motion = calculate_trajectory_direction_daughters_mother(division, center_coordinate_columns)

    division["direction_of_motion"] = direction_of_motion

    dataframe.loc[division['index_2'].values, "direction_of_motion"] = division["direction_of_motion"].values

    dataframe.loc[division['index_2'].values, "TrajectoryX"] = division['daughters_TrajectoryX'].values

    dataframe.loc[division['index_2'].values, "TrajectoryY"] = division["daughters_TrajectoryY"].values
    dataframe.loc[division['index_2'].values, "parent_index"] = division["index_1"].values

    mothers_df_last_time_step = division.drop_duplicates(subset='index_1', keep='first')

    mothers_with_more_than_two_daughters = \
        division[division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['ImageNumber_1'].transform('count') > 2]

    mothers_with_two_daughters = \
        division[division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['ImageNumber_1'].transform('count') == 2].copy()

    dataframe.loc[division['index_1'].unique(), 'daughters_index'] = mothers_df_last_time_step['daughters_index'].values
    dataframe.loc[division['index_1'].unique(), 'division_time'] = mothers_df_last_time_step['ImageNumber_2'].values
    dataframe.loc[division['index_1'].unique(), "divideFlag"] = True

    # bad divisions
    dataframe.loc[mothers_with_more_than_two_daughters['index_1'].unique(), "bad_division_flag"] = True
    dataframe.loc[mothers_with_more_than_two_daughters['index_1'].unique(), "ovd_flag"] = True
    dataframe.loc[mothers_with_more_than_two_daughters['index_2'].unique(), "ovd_flag"] = True
    dataframe.loc[mothers_with_more_than_two_daughters['index_2'].unique(), "bad_daughters_flag"] = True

    ##################################################################################################################
    mothers_with_two_daughters['daughter_mother_LengthChangeRatio'] = \
        (mothers_with_two_daughters['AreaShape_MajorAxisLength_2'] /
         mothers_with_two_daughters['AreaShape_MajorAxisLength_1'])

    mothers_with_two_daughters['daughter_mother_slope'] = \
        calculate_orientation_angle_batch(mothers_with_two_daughters['bacteria_slope_2'].values,
                                          mothers_with_two_daughters['bacteria_slope_1'].values)

    dataframe.loc[mothers_with_two_daughters['index_2'].values, 'daughter_mother_LengthChangeRatio'] = \
        mothers_with_two_daughters['daughter_mother_LengthChangeRatio'].values

    # shuld calc for all bacteria
    dataframe.loc[mothers_with_two_daughters['index_2'].values, 'slope_bac_bac'] = \
        mothers_with_two_daughters['daughter_mother_slope'].values

    dataframe.loc[mothers_with_two_daughters['index_2'].values, 'prev_bacteria_slope'] = \
        mothers_with_two_daughters['bacteria_slope_1'].values

    # correct divisions
    daughter_to_daughter = mothers_with_two_daughters.merge(mothers_with_two_daughters,
                                                            on=[parent_image_number_col + '_2',
                                                                parent_object_number_col + '_2'],
                                                            suffixes=('_daughter1', '_daughter2'))

    daughter_to_daughter = daughter_to_daughter.loc[daughter_to_daughter['index_2_daughter1'] !=
                                                    daughter_to_daughter['index_2_daughter2']]

    sum_daughters_len = \
        mothers_with_two_daughters.groupby(['ImageNumber_1', 'ObjectNumber_1'])['AreaShape_MajorAxisLength_2'].sum()

    max_daughters_len = \
        mothers_with_two_daughters.groupby(['ImageNumber_1', 'ObjectNumber_1'])['AreaShape_MajorAxisLength_2'].max()

    sum_daughters_len_to_mother = \
        sum_daughters_len / mothers_with_two_daughters.groupby(['ImageNumber_1',
                                                                'ObjectNumber_1'])['AreaShape_MajorAxisLength_1'].mean()

    max_daughters_len_to_mother = \
        max_daughters_len / mothers_with_two_daughters.groupby(['ImageNumber_1',
                                                                'ObjectNumber_1'])['AreaShape_MajorAxisLength_1'].mean()

    avg_daughters_trajectory_x = \
        mothers_with_two_daughters.groupby(['ImageNumber_1', 'ObjectNumber_1'])['daughters_TrajectoryX'].mean()

    avg_daughters_trajectory_y = \
        mothers_with_two_daughters.groupby(['ImageNumber_1', 'ObjectNumber_1'])['daughters_TrajectoryY'].mean()

    # Create a temporary DataFrame from sum_daughters_len_to_mother for easier merging
    temp_df_sum_daughters = sum_daughters_len_to_mother.reset_index()
    temp_df_sum_daughters.columns = ['ImageNumber', 'ObjectNumber', 'daughter_length_to_mother']

    temp_df_max_daughters = max_daughters_len_to_mother.reset_index()
    temp_df_max_daughters.columns = ['ImageNumber', 'ObjectNumber', 'max_daughter_len_to_mother']

    temp_df_avg_daughters_trajectory_x = avg_daughters_trajectory_x.reset_index()
    temp_df_avg_daughters_trajectory_x.columns = ['ImageNumber', 'ObjectNumber', 'avg_daughters_TrajectoryX']

    temp_df_avg_daughters_trajectory_y = avg_daughters_trajectory_y.reset_index()
    temp_df_avg_daughters_trajectory_y.columns = ['ImageNumber', 'ObjectNumber', 'avg_daughters_TrajectoryY']

    dataframe = dataframe.merge(temp_df_sum_daughters, on=['ImageNumber', 'ObjectNumber'], how='outer')
    dataframe = dataframe.merge(temp_df_max_daughters, on=['ImageNumber', 'ObjectNumber'], how='outer')
    dataframe = dataframe.merge(temp_df_avg_daughters_trajectory_x, on=['ImageNumber', 'ObjectNumber'], how='outer')
    dataframe = dataframe.merge(temp_df_avg_daughters_trajectory_y, on=['ImageNumber', 'ObjectNumber'], how='outer')

    dataframe.loc[daughter_to_daughter['index_2_daughter1'].values, "other_daughter_index"] = \
        daughter_to_daughter['index_2_daughter2'].values

    # other bacteria
    other_bac_df = dataframe.loc[~ dataframe['checked']]

    temp_df = dataframe.copy()
    temp_df.index = (temp_df['ImageNumber'].astype(str) + '_' + temp_df['ObjectNumber'].astype(str))

    bac_index_dict = temp_df['index'].to_dict()

    id_list = []
    parent_id_list = []
    # label_list = []

    same_bac_dict = {}

    last_bac_id = dataframe['id'].max() + 1

    for row_index, row in other_bac_df.iterrows():

        image_number, object_number, parent_img_num, parent_obj_num = \
            row[['ImageNumber', 'ObjectNumber', parent_image_number_col, parent_object_number_col]]

        if str(int(parent_img_num)) + '_' + str(parent_obj_num) not in same_bac_dict.keys():
            source_link = dataframe.iloc[bac_index_dict[str(int(parent_img_num)) + '_' + str(parent_obj_num)]]

            # life history continues
            parent_id = source_link['parent_id']
            source_bac_id = source_link['id']
            # this_bac_label = source_link[label_col]
            division_stat = source_link['divideFlag']

        else:
            # this_bac_label
            parent_id, source_bac_id, division_stat = \
                same_bac_dict[str(int(parent_img_num)) + '_' + str(parent_obj_num)]

        if division_stat:
            # division occurs
            new_bac_id = last_bac_id
            last_bac_id += 1
            # this_bac_label
            same_bac_dict[str(int(image_number)) + '_' + str(object_number)] = \
                [source_bac_id, new_bac_id, row['divideFlag']]

            parent_id_list.append(source_bac_id)
            id_list.append(new_bac_id)
            # label_list.append(this_bac_label)

        else:
            parent_id_list.append(parent_id)
            id_list.append(source_bac_id)
            # label_list.append(this_bac_label)

            # same bacteria
            # this_bac_label
            same_bac_dict[str(int(image_number)) + '_' + str(object_number)] = \
                [parent_id, source_bac_id, row['divideFlag']]

    dataframe.loc[other_bac_df.index, 'id'] = id_list
    dataframe.loc[other_bac_df.index, 'parent_id'] = parent_id_list
    # dataframe.loc[other_bac_df.index, label_col] = label_list

    dataframe['LifeHistory'] = dataframe.groupby('id')['id'].transform('size')

    mothers_with_more_than_two_daughters = \
        mothers_with_more_than_two_daughters.merge(dataframe, left_on=['ImageNumber_1', 'ObjectNumber_1'],
                                                   right_on=['ImageNumber', 'ObjectNumber'], how='inner')
    division = \
        division.merge(dataframe, left_on=['ImageNumber_1', 'ObjectNumber_1'],
                       right_on=['ImageNumber', 'ObjectNumber'], how='inner')

    mothers_id = division['id'].unique()
    bad_mothers_id = mothers_with_more_than_two_daughters['id'].unique()
    dataframe.loc[dataframe['id'].isin(mothers_id), "divideFlag"] = True
    # dataframe.loc[dataframe['id'].isin(bad_mothers_id), "bad_division_flag"] = True

    last_time_step_df = dataframe.loc[dataframe['ImageNumber'] == dataframe['ImageNumber'].max()]
    bac_without_division = dataframe.loc[~ dataframe["divideFlag"]]
    bac_without_division = bac_without_division.loc[~ bac_without_division['id'].isin(last_time_step_df['id'].values)]
    bac_without_division_last_time_step = bac_without_division.drop_duplicates(subset='id', keep='last')

    dataframe.loc[bac_without_division_last_time_step.index, 'unexpected_end'] = True

    dataframe['division_time'] = dataframe.groupby('id')['division_time'].transform(lambda x: x.ffill().bfill())
    dataframe['daughters_index'] = dataframe.groupby('id')['daughters_index'].transform(lambda x: x.ffill().bfill())

    # bacteria movement
    dataframe = adding_features_to_continues_life_history(dataframe, neighbor_df, division, center_coordinate_columns,
                                                          parent_image_number_col, parent_object_number_col,
                                                          calc_all=True)

    dataframe['checked'] = True

    # assign cell type
    if check_cell_type:
        cell_type_array = assign_cell_type(dataframe, intensity_threshold)
    else:
        cell_type_array = np.array([])

    dataframe['division_time'] = dataframe['division_time'].fillna(0)
    dataframe['daughters_index'] = dataframe['daughters_index'].fillna('')

    # set age
    dataframe['age'] = dataframe.groupby('id').cumcount() + 1

    dataframe['AverageLengthChangeRatio'] = dataframe.groupby('id')['LengthChangeRatio'].transform('mean')

    # now we should check bacteria prev slope
    bac_idx_not_needed_to_update = dataframe.loc[(~ dataframe['prev_bacteria_slope'].isna())]['index'].values
    temporal_df = dataframe.loc[bac_idx_not_needed_to_update]

    dataframe['prev_bacteria_slope'] = dataframe.groupby('id')['bacteria_slope'].shift(1)
    dataframe.loc[bac_idx_not_needed_to_update, 'prev_bacteria_slope'] = temporal_df['prev_bacteria_slope'].values

    # now we should cal slope bac to bac
    bac_need_to_cal_dir_motion = dataframe.loc[(dataframe['slope_bac_bac'].isna()) &
                                               (dataframe['unexpected_beginning'] == False) &
                                               (dataframe['ImageNumber'] != 1) &
                                               (dataframe['bad_daughters_flag'] == False)].copy()

    dataframe.loc[bac_need_to_cal_dir_motion['index'].values, 'slope_bac_bac'] = \
        calculate_orientation_angle_batch(bac_need_to_cal_dir_motion['bacteria_slope'].values,
                                          bac_need_to_cal_dir_motion['prev_bacteria_slope'])

    # now add neighbor index list
    # adding measured features of bacteria to neighbors_df
    neighbor_df = neighbor_df.merge(dataframe[['ImageNumber', 'ObjectNumber', 'index']],
                                    left_on=['Second Image Number', 'Second Object Number'],
                                    right_on=['ImageNumber', 'ObjectNumber'], how='inner')

    df_bac_with_neighbors = \
        dataframe[['ImageNumber', 'ObjectNumber', 'index']].merge(neighbor_df, left_on=['ImageNumber', 'ObjectNumber'],
                                                                  right_on=['First Image Number',
                                                                            'First Object Number'], how='left',
                                                                  suffixes=('', '_neighbor'))

    neighbor_list_array = lil_matrix((df_bac_with_neighbors['index'].max().astype('int64') + 1,
                                      df_bac_with_neighbors['index_neighbor'].max().astype('int64') + 1), dtype=bool)

    # values shows:
    # ImageNumber, ObjectNumber, index,	First Image Number,	First Object Number, Second Image Number,
    # Second Object Number,	ImageNumber_neighbor, ObjectNumber_neighbor,	index_neighbor
    df_bac_with_neighbors = df_bac_with_neighbors.fillna(-1)
    df_bac_with_neighbors_values = df_bac_with_neighbors.to_numpy(dtype='int64')

    for row in df_bac_with_neighbors_values:

        bac_idx = row[2]
        neighbor_idx = row[-1]

        if neighbor_idx != -1:
            neighbor_list_array[bac_idx, neighbor_idx] = True

    dataframe['checked'] = True

    # dataframe.drop(labels='checked', axis=1, inplace=True)
    return dataframe, neighbor_list_array, cell_type_array


def data_modification(dataframe, intensity_threshold, check_cell_type, neighbors_df, center_coordinate_columns,
                      parent_image_number_col, parent_object_number_col, label_col, without_tracking_correction):
    # for detecting each bacterium
    dataframe['index'] = dataframe.index

    dataframe, neighbor_list_array, cell_type_array = assign_feature_find_errors(dataframe, intensity_threshold,
                                                                                 check_cell_type,
                                                                                 neighbors_df,
                                                                                 center_coordinate_columns,
                                                                                 parent_image_number_col,
                                                                                 parent_object_number_col,
                                                                                 without_tracking_correction)

    return dataframe, neighbor_list_array, cell_type_array


def redefine_ids(df, label_col):
    df['prev_id'] = df['id']
    df['prev_parent_id'] = df['parent_id']
    df['prev_label'] = df[label_col]

    # Extract unique sorted values
    unique_id_sorted_values = pd.Series(df['id'].unique()).reset_index(drop=True)
    unique_label_sorted_values = pd.Series(df[label_col].unique()).reset_index(drop=True)

    # Create a mapping from each value to its rank
    id_value_to_rank = pd.Series(data=range(1, len(unique_id_sorted_values) + 1), index=unique_id_sorted_values)
    id_value_to_rank.loc[0] = 0

    label_value_to_rank = pd.Series(data=range(1, len(unique_label_sorted_values) + 1),
                                    index=unique_label_sorted_values)
    label_value_to_rank.loc[0] = 0

    # Map the original column to the new ranks
    df['id'] = df['prev_id'].map(id_value_to_rank)
    df['parent_id'] = df['prev_parent_id'].map(id_value_to_rank)
    df[label_col] = df['prev_label'].map(label_value_to_rank)

    return df


def label_correction(df, parent_image_number_col, parent_object_number_col, label_col):
    df['checked'] = False
    df[label_col] = np.nan
    cond1 = df[parent_image_number_col] == 0

    df.loc[cond1, label_col] = df.loc[cond1, 'index'].values + 1
    df.loc[cond1, 'checked'] = True

    # other bacteria
    other_bac_df = df.loc[~ df['checked']]

    temp_df = df.copy()
    temp_df.index = (temp_df['ImageNumber'].astype(str) + '_' + temp_df['ObjectNumber'].astype(str))

    bac_index_dict = temp_df['index'].to_dict()

    label_list = []

    same_bac_dict = {}

    for row_index, row in other_bac_df.iterrows():

        image_number, object_number, parent_img_num, parent_obj_num = \
            row[['ImageNumber', 'ObjectNumber', parent_image_number_col, parent_object_number_col]]

        if str(int(parent_img_num)) + '_' + str(parent_obj_num) not in same_bac_dict.keys():
            source_link = df.iloc[bac_index_dict[str(int(parent_img_num)) + '_' + str(parent_obj_num)]]

            this_bac_label = source_link[label_col]

        else:

            this_bac_label = same_bac_dict[str(int(parent_img_num)) + '_' + str(parent_obj_num)]

        label_list.append(this_bac_label)

        # same bacteria
        same_bac_dict[str(int(image_number)) + '_' + str(object_number)] = this_bac_label

    df.loc[other_bac_df.index, label_col] = label_list

    return df


def data_conversion(dataframe, um_per_pixel, all_center_coordinate_columns):

    dataframe['noise_bac'] = False
    dataframe['mother_rpl'] = False
    dataframe['daughter_rpl'] = False
    dataframe['source_mcl'] = False
    dataframe['target_mcl'] = False

    dataframe = convert_to_um(dataframe, um_per_pixel, all_center_coordinate_columns)
    dataframe = angle_convert_to_radian(dataframe)

    return dataframe


def find_fix_errors(dataframe, sorted_npy_files_list, neighbors_df, center_coordinate_columns,
                    all_center_coordinate_columns, parent_image_number_col, parent_object_number_col, label_col,
                    um_per_pixel=0.144, intensity_threshold=0.1, check_cell_type=True,
                    interval_time=1, min_life_history_of_bacteria=20, warn=True, without_tracking_correction=False,
                    output_directory=None, clf=None, n_cpu=-1, boundary_limits=None,
                    boundary_limits_per_time_step=None):

    logs_list = []
    logs_df = pd.DataFrame(columns=dataframe.columns)

    df = data_conversion(dataframe, um_per_pixel, all_center_coordinate_columns)

    # useful for only final comparison between original & modified dataframe
    df['prev_index'] = df.index

    raw_df = df.copy()

    if not without_tracking_correction:
        # correction of multi regions
        df, coordinate_array, color_array = multi_region_detection(df, sorted_npy_files_list, um_per_pixel,
                                                                   center_coordinate_columns,
                                                                   all_center_coordinate_columns,
                                                                   parent_image_number_col, parent_object_number_col,
                                                                   warn)

    if boundary_limits is not None or boundary_limits_per_time_step is not None:
        df = \
            find_wall_objects(boundary_limits, boundary_limits_per_time_step, df, center_coordinate_columns,
                              um_per_pixel)

    print_progress_bar(1, prefix='Progress:', suffix='Complete', length=50)

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))

    msg = " At " + end_tracking_errors_correction_time_str

    print(msg)
    msg = '10.0% Complete' + msg
    logs_list.append(msg)

    df.to_csv(output_directory + '/10.percent.csv', index=False)

    # remove noise objects
    df, neighbors_df = noise_remover(df, neighbors_df, parent_image_number_col, parent_object_number_col)

    print_progress_bar(2, prefix='Progress:', suffix='Complete', length=50)

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))

    msg = " At " + end_tracking_errors_correction_time_str

    print(msg)
    msg = '20.0% Complete' + msg
    logs_list.append(msg)

    df.to_csv(output_directory + '/20.percent.csv', index=False)

    df, neighbor_list_array, cell_type_array = data_modification(df, intensity_threshold, check_cell_type, neighbors_df,
                                                                 center_coordinate_columns, parent_image_number_col,
                                                                 parent_object_number_col,
                                                                 label_col, without_tracking_correction)

    print_progress_bar(3, prefix='Progress:', suffix='Complete', length=50)

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))

    msg = " At " + end_tracking_errors_correction_time_str

    print(msg)
    msg = '30.0% Complete' + msg
    logs_list.append(msg)

    df.to_csv(output_directory + '/30.percent.csv', index=False)

    data_preparation_time = time.time()
    data_preparation_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data_preparation_time))
    data_preparation_log = "At " + data_preparation_time_str + ', data preparation was completed.'

    print(data_preparation_log)
    logs_list.append(data_preparation_log)

    if not without_tracking_correction:
        # df = merged_bacteria(df, check_cell_type)

        # check neighbors
        df = neighbor_checking(df, neighbor_list_array, parent_image_number_col, parent_object_number_col)

        print_progress_bar(4, prefix='Progress:', suffix='Complete', length=50)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        msg = " At " + end_tracking_errors_correction_time_str

        print(msg)
        msg = '40.0% Complete' + msg
        logs_list.append(msg)

        df.to_csv(output_directory + '/40.percent.csv', index=False)

        # redundant links
        # df = detect_redundant_parent_link(df, parent_image_number_col,
        #                                  parent_object_number_col, label_col, center_coordinate_columns)

        df = detect_missing_connectivity_link(df, parent_image_number_col, parent_object_number_col)

        comparing_divided_non_divided_model, non_divided_bac_model, divided_bac_model = \
            training_models(df, neighbors_df, neighbor_list_array, center_coordinate_columns, parent_image_number_col,
                            parent_object_number_col, output_directory, clf, n_cpu, coordinate_array)

        # more than two daughters
        df = remove_over_assigned_daughters_link(df, parent_image_number_col,
                                                 parent_object_number_col, label_col, center_coordinate_columns,
                                                 divided_bac_model, coordinate_array)

        df_before_more_detection_and_removing = df.copy()

        print_progress_bar(4.5, prefix='Progress:', suffix='Complete', length=50)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        msg = " At " + end_tracking_errors_correction_time_str

        print(msg)
        msg = '45.0% Complete' + msg
        logs_list.append(msg)

        df.to_csv(output_directory + '/45.percent.csv', index=False)

        # remove redundant links
        df = detect_and_remove_redundant_parent_link(df, neighbors_df, neighbor_list_array, parent_image_number_col,
                                                     parent_object_number_col, label_col, center_coordinate_columns,
                                                     non_divided_bac_model, coordinate_array)
        print_progress_bar(5, prefix='Progress:', suffix='Complete', length=50)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        msg = " At " + end_tracking_errors_correction_time_str

        print(msg)
        msg = '50.0% Complete' + msg
        logs_list.append(msg)

        df.to_csv(output_directory + '/50.percent.csv', index=False)

        df = missing_connectivity_link(df, neighbors_df, neighbor_list_array, min_life_history_of_bacteria,
                                       interval_time, parent_image_number_col, parent_object_number_col, label_col,
                                       center_coordinate_columns, comparing_divided_non_divided_model,
                                       non_divided_bac_model, divided_bac_model, coordinate_array)

        print_progress_bar(6, prefix='Progress:', suffix='Complete', length=50)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        msg = " At " + end_tracking_errors_correction_time_str

        print(msg)
        msg = '60.0% Complete' + msg
        logs_list.append(msg)

        df.to_csv(output_directory + '/60.percent.csv', index=False)

        # try to assign new link
        df = correction_unexpected_beginning(df, neighbors_df, neighbor_list_array, check_cell_type,
                                             interval_time, min_life_history_of_bacteria, parent_image_number_col,
                                             parent_object_number_col, label_col, center_coordinate_columns,
                                             comparing_divided_non_divided_model, non_divided_bac_model,
                                             divided_bac_model, color_array, coordinate_array)

        print_progress_bar(7, prefix='Progress:', suffix='Complete', length=50)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        msg = " At " + end_tracking_errors_correction_time_str

        print(msg)
        msg = '70.0% Complete' + msg
        logs_list.append(msg)

        df.to_csv(output_directory + '/70.percent.csv', index=False)

        df = unexpected_end_bacteria(df, neighbors_df, neighbor_list_array, min_life_history_of_bacteria, interval_time,
                                     parent_image_number_col, parent_object_number_col, label_col,
                                     center_coordinate_columns, comparing_divided_non_divided_model,
                                     non_divided_bac_model, divided_bac_model, color_array, coordinate_array)

        print_progress_bar(8, prefix='Progress:', suffix='Complete', length=50)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        msg = " At " + end_tracking_errors_correction_time_str

        print(msg)
        msg = '80.0% Complete' + msg
        logs_list.append(msg)

        df.to_csv(output_directory + '/80.percent.csv', index=False)

        df = final_matching(df, neighbors_df, neighbor_list_array, min_life_history_of_bacteria, interval_time,
                            parent_image_number_col, parent_object_number_col, label_col, center_coordinate_columns,
                            df_before_more_detection_and_removing, non_divided_bac_model, divided_bac_model,
                            coordinate_array)

        df = remove_rows(df, 'noise_bac', False)

        print_progress_bar(9, prefix='Progress:', suffix='Complete', length=50)

        # label correction
        df = label_correction(df, parent_image_number_col, parent_object_number_col, label_col)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        end_tracking_errors_correction_log = " At " + end_tracking_errors_correction_time_str + \
                                             ", the corrections to the tracking errors were completed."

        print(end_tracking_errors_correction_log)
        end_tracking_errors_correction_log = '90.0% Complete' + end_tracking_errors_correction_log

        df.to_csv(output_directory + '/90.percent.csv', index=False)
        logs_list.append(end_tracking_errors_correction_log)

        df = redefine_ids(df, label_col)

        # now I want to generate log file
        logs_df, identified_tracking_errors_df, fixed_errors, remaining_errors_df, logs_list = \
            generate_log_file(raw_df, df, logs_list, parent_image_number_col, parent_object_number_col,
                              center_coordinate_columns)

    else:
        identified_tracking_errors_df = pd.DataFrame()
        fixed_errors = pd.DataFrame()
        remaining_errors_df = pd.DataFrame()

    return (df, logs_list, logs_df, identified_tracking_errors_df, fixed_errors, remaining_errors_df, neighbors_df,
            cell_type_array)
