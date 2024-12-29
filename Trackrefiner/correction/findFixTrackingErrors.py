import numpy as np
import pandas as pd
import time
from Trackrefiner.correction.action.helper import convert_pixel_to_um, calculate_all_bac_endpoints, \
    convert_angle_to_radian, calculate_all_bac_slopes, print_progress_bar, \
    calculate_bacterial_life_history_features, calculate_trajectory_angles, calculate_angles_between_slopes
from Trackrefiner.correction.action.featuresCalculation.fluorescenceIntensity import assign_cell_types
from Trackrefiner.correction.trackingErrors.unexpectedBeginning import handle_unexpected_beginning_bacteria
from Trackrefiner.correction.trackingErrors.overAssignedDaughters import resolve_over_assigned_daughters_link
from Trackrefiner.correction.trackingErrors.redundantParentLink import detect_and_resolve_redundant_parent_link
from Trackrefiner.correction.segmentationErrors.noiseObjects import detect_and_remove_noise_bacteria
from Trackrefiner.correction.action.neighborAnalysis import compare_neighbor_sets
from Trackrefiner.correction.trackingErrors.missingConnectivityLink import detect_and_resolve_missing_connectivity_link
from Trackrefiner.correction.trackingErrors.unExpectedEnd import handle_unexpected_end_bacteria
from Trackrefiner.correction.segmentationErrors.multiRegionsDetection import map_and_detect_multi_regions
from Trackrefiner.correction.trackingErrors.restoringTrackingLinks import restore_tracking_links
from Trackrefiner.correction.modelTraning.bacterialBehaviorModelTraining import train_bacterial_behavior_models
from Trackrefiner.correction.action.regionOfInterestFilter import find_inside_boundary_objects
from Trackrefiner.correction.action.generateLogFile import generate_log_file
from Trackrefiner.correction.action.propagateBacteriaLabels import propagate_bacteria_labels
from scipy.sparse import lil_matrix
import warnings
import logging
import traceback

# Configure logging to capture warnings
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Custom warning handler to include traceback
def custom_warning_handler(message, filename, lineno):
    log_message = f"{message} (File: {filename}, Line: {lineno})"
    # Capture the current stack trace
    stack_trace = traceback.format_stack()
    # Log the warning with the stack trace
    logging.warning(f"{log_message}\nStack Trace:\n{''.join(stack_trace)}")


# Apply the custom warning handler
warnings.showwarning = custom_warning_handler


def calculate_bacteria_features_and_assign_flags(dataframe, intensity_threshold, assigning_cell_type, neighbor_df,
                                                 center_coord_cols, parent_image_number_col, parent_object_number_col):
    """
    Calculate bacterial features and assign initial flags for further analysis.

    This function computes features related to bacterial behavior, such as movement, division, life history,
    and spatial relationships. Additionally, it assigns flags and classifications (e.g., division flags,
    unexpected events, and daughters assignment) for downstream analysis.

    :param pd.DataFrame dataframe:
        Input dataframe containing bacterial measured bacterial features.
    :param float intensity_threshold:
        Threshold value used for intensity-based cell type assignment.
    :param bool assigning_cell_type:
        Whether to assign cell types based on intensity values.
    :param pd.DataFrame neighbor_df:
        Dataframe containing information about neighboring bacteria.
    :param dict center_coord_cols:
        Dictionary specifying the column names for x and y coordinates of bacterial centers.
        Example: {'x': 'Center_X', 'y': 'Center_Y'}
    :param str parent_image_number_col:
        Column name for the parent image number in the dataframe.
    :param str parent_object_number_col:
        Column name for the parent object number in the dataframe.

    :return:
        tuple:

        - **dataframe** (*pd.DataFrame*): Updated dataframe with calculated features and assigned flags.
        - **neighbor_list_array** (*lil_matrix*): Sparse matrix indicating neighboring relationships between bacteria.
        - **cell_type_array** (*np.array*): Array containing assigned cell types if `assigning_cell_type` is True;
          otherwise, an empty array.
    """

    dataframe['index'] = dataframe.index

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
    dataframe = calculate_all_bac_endpoints(dataframe, center_coord_cols)
    dataframe = calculate_all_bac_slopes(dataframe)

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
                                                                              center_coord_cols['x'] + '_1',
                                                                              center_coord_cols['y'] + '_1',
                                                                              center_coord_cols['x'] + '_2',
                                                                              center_coord_cols['y'] + '_2',
                                                                              parent_object_number_col + '_2',
                                                                              parent_image_number_col + '_2']].copy()

    division['daughters_index'] = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['index_2'].transform(lambda x: ', '.join(x.astype(str)))

    division['daughters_TrajectoryX'] = \
        (division[center_coord_cols['x'] + '_2'] - division[center_coord_cols['x'] + '_1'])

    division['daughters_TrajectoryY'] = \
        (division[center_coord_cols['y'] + '_2'] - division[center_coord_cols['y'] + '_1'])

    direction_of_motion = calculate_trajectory_angles(division, center_coord_cols, suffix1='_2', suffix2='_1')

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

    mothers_with_two_daughters['daughter_mother_LengthChangeRatio'] = \
        (mothers_with_two_daughters['AreaShape_MajorAxisLength_2'] /
         mothers_with_two_daughters['AreaShape_MajorAxisLength_1'])

    mothers_with_two_daughters['daughter_mother_slope'] = \
        calculate_angles_between_slopes(mothers_with_two_daughters['bacteria_slope_2'].values,
                                        mothers_with_two_daughters['bacteria_slope_1'].values)

    dataframe.loc[mothers_with_two_daughters['index_2'].values, 'daughter_mother_LengthChangeRatio'] = \
        mothers_with_two_daughters['daughter_mother_LengthChangeRatio'].values

    # should calc for all bacteria
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
            division_stat = source_link['divideFlag']

        else:
            # this_bac_label
            parent_id, source_bac_id, division_stat = same_bac_dict[f"{int(parent_img_num)}_{int(parent_obj_num)}"]

        if division_stat:
            # division occurs
            new_bac_id = last_bac_id
            last_bac_id += 1
            # this_bac_label
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

    dataframe.loc[other_bac_df.index, 'id'] = id_list
    dataframe.loc[other_bac_df.index, 'parent_id'] = parent_id_list

    dataframe['LifeHistory'] = dataframe.groupby('id')['id'].transform('size')

    division = \
        division.merge(dataframe, left_on=['ImageNumber_1', 'ObjectNumber_1'],
                       right_on=['ImageNumber', 'ObjectNumber'], how='inner')

    mothers_id = division['id'].unique()
    dataframe.loc[dataframe['id'].isin(mothers_id), "divideFlag"] = True

    last_time_step_df = dataframe.loc[dataframe['ImageNumber'] == dataframe['ImageNumber'].max()]
    bac_without_division = dataframe.loc[~ dataframe["divideFlag"]]
    bac_without_division = bac_without_division.loc[~ bac_without_division['id'].isin(last_time_step_df['id'].values)]
    bac_without_division_last_time_step = bac_without_division.drop_duplicates(subset='id', keep='last')

    dataframe.loc[bac_without_division_last_time_step.index, 'unexpected_end'] = True

    dataframe['division_time'] = dataframe.groupby('id')['division_time'].transform(lambda x: x.ffill().bfill())
    dataframe['daughters_index'] = dataframe.groupby('id')['daughters_index'].transform(lambda x: x.ffill().bfill())

    # bacteria movement
    dataframe = calculate_bacterial_life_history_features(dataframe, calc_all_features=True, neighbor_df=neighbor_df,
                                                          division_df=division, center_coord_cols=center_coord_cols,
                                                          use_selected_rows=False)

    dataframe['checked'] = True

    # assign cell type
    if assigning_cell_type:
        cell_type_array = assign_cell_types(dataframe, intensity_threshold)
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
        calculate_angles_between_slopes(bac_need_to_cal_dir_motion['bacteria_slope'].values,
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

    return dataframe, neighbor_list_array, cell_type_array


def reassign_sequential_ids_labels(df, label_col):
    """
    Reassign sequential IDs and labels for improved readability.

    This function modifies the `id` and `parent_id` columns to have sequential, compact values,
    starting from 1. It also reassigns sequential values to the specified label column.
    Original values are preserved in the `prev_id`, `prev_parent_id`, and `prev_label` columns
    for reference.

    This is particularly useful when the original IDs are non-sequential or excessively large,
    making them difficult to interpret or visualize (e.g., IDs starting at 10^6 or with large gaps).

    :param pd.DataFrame df:
        Input dataframe containing the original `id`, `parent_id`, and label data to be reassigned.
    :param str label_col:
        Name of the label column to be reassigned sequential values.

    :return:
        pd.DataFrame

        Updated dataframe where `id`, `parent_id`, and the specified label column have been
        reassigned sequential values. Original values are stored in backup columns (`prev_id`,
        `prev_parent_id`, and `prev_label`).
    """

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


def convert_tracking_data_units(dataframe, pixel_per_micron, all_rel_center_coord_cols):
    """
    Convert tracking data units from pixels to microns and angles to radians.

    This function updates a dataframe by performing unit conversions. It also initializes new columns for tracking noise
    and relationship flags, which can be used for further analysis.

    :param dataframe: pd.DataFrame
        Input dataframe containing bacterial tracking data.
    :param pixel_per_micron: float
        Conversion factor from pixels to microns.
    :param all_rel_center_coord_cols: list of str
        List of columns representing relative center coordinates to be converted to microns.
    :return: pd.DataFrame
        Updated dataframe with measurements converted to microns and angles to radians.
        Additional columns (`noise_bac`, `mother_rpl`, `daughter_rpl`, `source_mcl`,
        `target_mcl`) are added and initialized to `False`.
    """

    dataframe['noise_bac'] = False
    dataframe['mother_rpl'] = False
    dataframe['daughter_rpl'] = False
    dataframe['source_mcl'] = False
    dataframe['target_mcl'] = False

    dataframe = convert_pixel_to_um(dataframe, pixel_per_micron, all_rel_center_coord_cols)
    dataframe = convert_angle_to_radian(dataframe)

    return dataframe


def find_fix_tracking_errors(cp_output_df, sorted_seg_npy_files_list, neighbors_df, center_coord_cols,
                             all_rel_center_coord_cols, parent_image_number_col, parent_object_number_col, label_col,
                             interval_time, doubling_time, pixel_per_micron=0.144, intensity_threshold=0.1,
                             assigning_cell_type=True, disable_tracking_correction=False, clf=None, n_cpu=-1,
                             image_boundaries=None, dynamic_boundaries=None, out_dir=None, verbose=True):

    """
    This function detects and corrects tracking errors in bacterial tracking data. It applies a comprehensive pipeline
    that includes filter out objects outside the image area, noise removal, feature calculation,
    training machine learning models for behavioral analysis, resolving tracking errors, and generating logs.

    **Key Steps**:
        1. **Unit Conversion**:
            Converts tracking data units (e.g., pixel to micron) for consistency.
        2. **Boundary Checking**:
            Ensures bacteria remain within defined image or dynamic boundaries.
        3. **Noise Removal**:
            Removes small or low-intensity objects that may represent noise.
        4. **Feature Calculation**:
            Computes features like length, direction, and motion alignment for all bacteria.
        5. **Model Training**:
            Trains machine learning models for bacterial behavior:
            - **Division vs. Non-Division**: Identifies dividing bacteria.
            - **Continuity**: Tracks the life history of bacteria across frames.
        6. **Error Resolution**:
            - Resolves over assigned daughters, redundant parent links and missing connectivity.
            - Handles unexpected beginnings and ends in tracking.
            - Restores tracking links removed due to errors in previous steps.
        7. **Label Propagation**:
            Reassigns sequential IDs and propagates labels for corrected data.
        8. **Log Generation**:
            Creates a detailed log of all corrections, errors, and progress.

    :param pandas.DataFrame cp_output_df:
        Input DataFrame from the CellProfiler pipeline containing tracking information and measured bacterial features.
    :param list sorted_seg_npy_files_list:
        List of sorted `.npy` files containing segmentation masks for each time step.
    :param pandas.DataFrame neighbors_df:
        DataFrame representing neighbor relationships between bacteria.
    :param dict center_coord_cols:
        Dictionary specifying the x and y coordinate column names, e.g., `{'x': 'Center_X', 'y': 'Center_Y'}`.
    :param dict all_rel_center_coord_cols:
        Relative center coordinate columns for mapping bacteria in segmentation data.
    :param str parent_image_number_col:
        Column name for the parent bacterium's image number.
    :param str parent_object_number_col:
        Column name for the parent bacterium's object number.
    :param str label_col:
        Column name for bacteria labels.
    :param float interval_time:
        Time interval (in minutes) between consecutive frames.
    :param float doubling_time:
        Expected doubling time (in minutes) for bacterial division.
    :param float pixel_per_micron:
        Conversion factor from pixels to microns (default: `0.144`).
    :param float intensity_threshold:
        Threshold to classify objects into specific cell types based on intensity values. (default: `0.1`).
    :param bool assigning_cell_type:
        If True, assigns cell types to objects based on intensity thresholds.
    :param bool disable_tracking_correction:
        Flag to disable tracking error correction (default: `False`).
    :param str clf:
        Classifier to use for machine learning models (default: `None`).
    :param int n_cpu:
        Number of CPUs to use for parallel processing (default: `-1`, use all available CPUs).
    :param str image_boundaries:
        Boundary limits defined for all time steps to filter out objects outside the image area.
    :param str dynamic_boundaries:
        Path to a CSV file specifying boundary limits for each time step. The file should contain columns:
        `Time Step`, `Lower X Limit`, `Upper X Limit`, `Lower Y Limit`, `Upper Y Limit`.
    :param str out_dir:
        Directory to save output files and logs.
    :param bool verbose:
        If True, displays warnings and additional details during processing.

    **Returns**:
        tuple:

        - **pandas.DataFrame df**: Corrected tracking data.
        - **list logs_list**: List of progress and log messages.
        - **pandas.DataFrame logs_df**: Detailed log of tracking corrections.
        - **pandas.DataFrame identified_tracking_errors_df**: Identified tracking errors before correction.
        - **pandas.DataFrame fixed_errors**: Corrected tracking errors.
        - **pandas.DataFrame remaining_errors_df**: Remaining errors after corrections.
        - **pandas.DataFrame neighbors_df**: Updated neighbors DataFrame.
        - **numpy.ndarray cell_type_array**: Array of assigned cell types.
    """

    logs_list = []

    logs_df = pd.DataFrame(columns=cp_output_df.columns)

    df = convert_tracking_data_units(cp_output_df, pixel_per_micron, all_rel_center_coord_cols)

    # useful only for the final comparison between the original and modified dataframes
    df['prev_index'] = df.index

    raw_df = df.copy()

    step = 1

    if not disable_tracking_correction:
        # correction of multi regions
        df, coordinate_array, color_array = \
            map_and_detect_multi_regions(df, sorted_seg_npy_files_list, pixel_per_micron, center_coord_cols,
                                         all_rel_center_coord_cols, parent_image_number_col, parent_object_number_col,
                                         verbose)

        if image_boundaries is not None or dynamic_boundaries is not None:
            df = \
                find_inside_boundary_objects(image_boundaries, dynamic_boundaries, df, center_coord_cols,
                                             pixel_per_micron)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        print_progress_bar(step, prefix='Progress:', suffix=f"Completed At {end_tracking_errors_correction_time_str}",
                           length=50)

        msg = f"{step * 10}% Completed At {end_tracking_errors_correction_time_str}"
        logs_list.append(msg)
        step += 1

        # remove noise objects
        df, neighbors_df = detect_and_remove_noise_bacteria(df, neighbors_df, parent_image_number_col,
                                                            parent_object_number_col)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        print_progress_bar(step, prefix='Progress:', suffix=f"Completed At {end_tracking_errors_correction_time_str}",
                           length=50)

        msg = f"{step * 10}% Completed At {end_tracking_errors_correction_time_str}"
        logs_list.append(msg)
        step += 1

    df, neighbor_matrix, cell_type_array = \
        calculate_bacteria_features_and_assign_flags(df, intensity_threshold, assigning_cell_type, neighbors_df,
                                                     center_coord_cols, parent_image_number_col,
                                                     parent_object_number_col)

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))

    print_progress_bar(step, prefix='Progress:', suffix=f"Completed At {end_tracking_errors_correction_time_str}",
                       length=50)

    msg = f"{step * 10}% Completed At {end_tracking_errors_correction_time_str}"
    logs_list.append(msg)
    step += 1

    # now set `ub` & `ue` flag to raw_df
    merge_raw_modified_df = raw_df[['prev_index']].merge(
            df[['prev_index', 'unexpected_end', 'unexpected_beginning']], on=['prev_index'])
    raw_df.loc[merge_raw_modified_df['prev_index'].values, 'unexpected_end'] = \
        merge_raw_modified_df['unexpected_end'].values

    raw_df.loc[merge_raw_modified_df['prev_index'].values, 'unexpected_beginning'] = \
        merge_raw_modified_df['unexpected_beginning'].values

    # check neighbors
    df = compare_neighbor_sets(df, neighbor_matrix, parent_image_number_col, parent_object_number_col)

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))

    print_progress_bar(step, prefix='Progress:', suffix=f"Completed At {end_tracking_errors_correction_time_str}",
                       length=50)

    msg = f"{step * 10}% Completed At {end_tracking_errors_correction_time_str}"
    logs_list.append(msg)
    step += 0.5

    if not disable_tracking_correction:

        divided_vs_non_divided_model, non_divided_model, division_model = \
            train_bacterial_behavior_models(df, neighbors_df, neighbor_matrix, center_coord_cols,
                                            parent_image_number_col, parent_object_number_col, out_dir, clf, n_cpu,
                                            coordinate_array)

        # more than two daughters
        df = resolve_over_assigned_daughters_link(df, parent_image_number_col, parent_object_number_col,
                                                  center_coord_cols, division_model, coordinate_array)

        df_raw_before_rpl_errors = df.copy()

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        print_progress_bar(step, prefix='Progress:', suffix=f"Completed At {end_tracking_errors_correction_time_str}",
                           length=50)

        msg = f"{step * 10}% Completed At {end_tracking_errors_correction_time_str}"
        logs_list.append(msg)
        step += 0.5

        # remove redundant links
        df = detect_and_resolve_redundant_parent_link(df, neighbors_df, neighbor_matrix, parent_image_number_col,
                                                      parent_object_number_col, center_coord_cols, non_divided_model,
                                                      coordinate_array)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        print_progress_bar(step, prefix='Progress:', suffix=f"Completed At {end_tracking_errors_correction_time_str}",
                           length=50)

        msg = f"{step * 10}% Completed At {end_tracking_errors_correction_time_str}"
        logs_list.append(msg)
        step += 1

        df = detect_and_resolve_missing_connectivity_link(df, neighbors_df, neighbor_matrix, doubling_time,
                                                          interval_time, parent_image_number_col,
                                                          parent_object_number_col, center_coord_cols,
                                                          divided_vs_non_divided_model, non_divided_model,
                                                          division_model, coordinate_array)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        print_progress_bar(step, prefix='Progress:', suffix=f"Completed At {end_tracking_errors_correction_time_str}",
                           length=50)

        msg = f"{step * 10}% Completed At {end_tracking_errors_correction_time_str}"
        logs_list.append(msg)
        step += 1

        # try to assign new link
        df = handle_unexpected_beginning_bacteria(df, neighbors_df, neighbor_matrix,
                                                  interval_time, doubling_time, parent_image_number_col,
                                                  parent_object_number_col, center_coord_cols,
                                                  divided_vs_non_divided_model, non_divided_model,
                                                  division_model, color_array, coordinate_array)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        print_progress_bar(step, prefix='Progress:', suffix=f"Completed At {end_tracking_errors_correction_time_str}",
                           length=50)

        msg = f"{step * 10}% Completed At {end_tracking_errors_correction_time_str}"
        logs_list.append(msg)
        step += 1

        df = handle_unexpected_end_bacteria(df, neighbors_df, neighbor_matrix, interval_time, doubling_time,
                                            parent_image_number_col, parent_object_number_col, center_coord_cols,
                                            divided_vs_non_divided_model, non_divided_model, division_model,
                                            color_array, coordinate_array)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        print_progress_bar(step, prefix='Progress:', suffix=f"Completed At {end_tracking_errors_correction_time_str}",
                           length=50)

        msg = f"{step * 10}% Completed At {end_tracking_errors_correction_time_str}"
        logs_list.append(msg)
        step += 1

        df = restore_tracking_links(df, neighbors_df, neighbor_matrix, parent_image_number_col,
                                    parent_object_number_col, center_coord_cols, df_raw_before_rpl_errors,
                                    non_divided_model, division_model, coordinate_array)

        # remove noise objects
        df = df.loc[df['noise_bac'] == False].reset_index(drop=True)

        # label correction
        df = propagate_bacteria_labels(df, parent_image_number_col, parent_object_number_col, label_col)

        end_tracking_errors_correction_time = time.time()
        end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(end_tracking_errors_correction_time))

        print_progress_bar(step, prefix='Progress:', suffix=f"Completed At {end_tracking_errors_correction_time_str}",
                           length=50)

        end_tracking_errors_correction_log = (f"{step * 10}% Completed At {end_tracking_errors_correction_time_str}, "
                                              f"Corrections of the tracking errors have been completed.")

        logs_list.append(end_tracking_errors_correction_log)

        df = reassign_sequential_ids_labels(df, label_col)

        # now I want to generate log file
        logs_df, logs_list = \
            generate_log_file(raw_df, df, logs_list, parent_image_number_col, parent_object_number_col,
                              center_coord_cols)

    return df, logs_list, logs_df, neighbors_df, cell_type_array
