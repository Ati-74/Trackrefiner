import pandas as pd
import numpy as np


def compute_intersections_and_unions(df, col1, col2, coordinate_array):
    """
    Computes intersections and unions between two sets of objects based on their spatial coordinates,
    and calculates the Intersection over Union (IoU) for each pair.

    :param pandas.DataFrame df:
        Input DataFrame containing object indices in columns `col1` and `col2`.
    :param str col1:
        Column names representing indices of two sets of objects.
    :param str col2:
        Column names representing indices of two sets of objects.
    :param csr_matrix coordinate_array:
        Sparse boolean array where each row corresponds to the coordinates of an object.

    :returns:
        pandas.DataFrame: Updated DataFrame with a new column `iou` containing the IoU values for each pair of objects.
    """

    # col1 , col2 refer to index

    idx1_vals = df[col1].to_numpy(dtype='int64')
    idx2_vals = df[col2].to_numpy(dtype='int64')

    # Perform element-wise concatenation
    intersection_size_element_wise_concatenation = \
        coordinate_array[idx1_vals].multiply(coordinate_array[idx2_vals])

    union_size_element_wise_concatenation = \
        coordinate_array[idx1_vals] + coordinate_array[idx2_vals]

    # Count the number of True values in `AND` and `OR` results
    intersection_lens = intersection_size_element_wise_concatenation.getnnz(axis=1)
    union_lens = union_size_element_wise_concatenation.getnnz(axis=1)

    # Preallocate the iou column in df
    iou_values = (intersection_lens / union_lens).astype('float32')

    # Create a new DataFrame with 'iou' as the column
    df_iou = pd.DataFrame(index=df.index, data={'iou': iou_values})

    df = pd.concat([df, df_iou], axis=1)

    return df


def compute_intersections_and_unions_division(df, col1, col2, coordinate_array):
    """
    Computes intersections and unique areas for a pair of objects, typically for a parent-daughter relationship,
    and calculates Intersection over Union (IoU) with unique area consideration.

    :param pandas.DataFrame df:
        Input DataFrame containing object indices in columns `col1` and `col2`.
    :param str col1:
        Column names representing indices of two sets of objects.
    :param str col2:
        Column names representing indices of two sets of objects.
    :param csr_matrix coordinate_array:
        Sparse boolean array where each row corresponds to the coordinates of an object.

    :returns:
        pandas.DataFrame: Updated DataFrame with a new column `iou` containing the IoU values.
    """

    idx1_vals = df[col1].to_numpy(dtype='int64')
    idx2_vals = df[col2].to_numpy(dtype='int64')

    # Perform element-wise concatenation
    intersection_size_element_wise_concatenation = \
        coordinate_array[idx1_vals].multiply(coordinate_array[idx2_vals])

    # Count the number of True values in `AND` and `OR` results
    intersection_lens = intersection_size_element_wise_concatenation.getnnz(axis=1)

    # Calculate unique areas for mask 2 (daughter)
    masks2_lens = coordinate_array[idx2_vals].getnnz(axis=1)

    unique_masks2_lens = masks2_lens - intersection_lens

    # Calculate IoU based on daughter_flag
    daughter_union = (intersection_lens + unique_masks2_lens)
    # Correcting the sum operation
    iou = intersection_lens / daughter_union

    # Create a new DataFrame with 'iou' as the column
    df_iou = pd.DataFrame(index=df.index, data={'iou': iou})

    df = pd.concat([df, df_iou], axis=1)

    return df


def find_overlap_oad_mother_daughters(mother_bad_daughters_df, coordinate_array):

    """
    Identifies overlaps between mother objects of OAD (mothers with over assign daughters status)
    types and their associated daughter objects by calculating IoU.

    :param pandas.DataFrame mother_bad_daughters_df:
        Input DataFrame containing indices of mother and daughter objects.
    :param csr_matrix coordinate_array:
        Sparse boolean array where each row corresponds to the coordinates of an object.

    :returns:
        pandas.DataFrame: Pivoted DataFrame showing the IoU values for each mother-daughter pair.
    """

    # Calculate intersection by merging on coordinates
    mother_bad_daughters_df = \
        compute_intersections_and_unions(mother_bad_daughters_df, 'prev_index_mother',
                                         'prev_index_daughter', coordinate_array)

    # Pivot this DataFrame to get the desired structure
    overlap_df = \
        mother_bad_daughters_df[['index_mother', 'index_daughter', 'iou']].pivot(index='index_mother',
                                                                                 columns='index_daughter',
                                                                                 values='iou')
    overlap_df.columns.name = None
    overlap_df.index.name = None
    overlap_df = overlap_df.astype(float)

    return overlap_df


def track_object_overlaps_to_next_frame(current_df, selected_objects, next_df, selected_objects_in_next_time_step,
                                        center_coordinate_columns, color_array,
                                        daughter_flag=False, coordinate_array=None):

    """
    Tracks overlaps between objects in the current frame and candidate objects in the next frame using IoU,
    with optional division-specific calculations.

    :param pandas.DataFrame current_df:
        DataFrame containing current frame objects.
    :param pandas.DataFrame selected_objects:
        Subset of objects selected from the current frame.
    :param pandas.DataFrame next_df:
        DataFrame containing next frame objects.
    :param pandas.DataFrame selected_objects_in_next_time_step:
        Subset of objects selected from the next frame.
    :param dict center_coordinate_columns:
        Dictionary containing column names for x and y center coordinates.
    :param numpy.ndarray color_array:
        Array storing colors for each object.
    :param bool daughter_flag:
        Whether to calculate unique areas for division analysis.
    :param csr_matrix coordinate_array:
        Sparse boolean array where each row corresponds to the coordinates of an object.

    :returns:
        tuple: Pivoted DataFrame showing IoU values for overlaps and a product DataFrame with detailed pairwise data.
    """

    current_df_color_array = color_array[current_df['prev_index'].values]
    next_df_color_array = color_array[next_df['prev_index'].values]

    if np.unique(current_df_color_array, axis=0).shape[0] != current_df_color_array.shape[0]:
        # Find unique rows and their indices
        unique_rows, indices, counts = np.unique(current_df_color_array, axis=0, return_index=True, return_counts=True)

        # Get the duplicate rows (those that appear more than once)
        duplicate_rows = unique_rows[counts > 1]
        raise ValueError(
            f"Duplicate rows detected in the current frame (ImageNumber: {current_df['ImageNumber'].values[0]}). "
            f"Duplicate entries: {duplicate_rows}"
        )

    if np.unique(next_df_color_array, axis=0).shape[0] != next_df_color_array.shape[0]:
        # Find unique rows and their indices
        unique_rows, indices, counts = np.unique(next_df_color_array, axis=0, return_index=True, return_counts=True)

        # Get the duplicate rows (those that appear more than once)
        duplicate_rows = unique_rows[counts > 1]
        raise ValueError(
            f"Duplicate rows detected in the next frame (df: {next_df['df'].values[0]}). "
            f"Duplicate entries: {duplicate_rows}"
        )

    product_df = pd.merge(
        selected_objects.reset_index(),
        selected_objects_in_next_time_step.reset_index(),
        how='cross',
        suffixes=('_current', '_next')
    )
    product_df[['index_current', 'index_next']] = product_df[['index_current', 'index_next']].astype('int64')
    product_df[[f"{center_coordinate_columns['x']}_current", f"{center_coordinate_columns['y']}_current",
                f"{center_coordinate_columns['x']}_next", f"{center_coordinate_columns['y']}_next",
                'endpoint1_X_current', 'endpoint1_Y_current', 'endpoint2_X_current', 'endpoint2_Y_current',
                'endpoint1_X_next', 'endpoint1_Y_next', 'endpoint2_X_next', 'endpoint2_Y_next']] = (
        product_df[[f"{center_coordinate_columns['x']}_current", f"{center_coordinate_columns['y']}_current",
                    f"{center_coordinate_columns['x']}_next", f"{center_coordinate_columns['y']}_next",
                    'endpoint1_X_current', 'endpoint1_Y_current', 'endpoint2_X_current', 'endpoint2_Y_current',
                    'endpoint1_X_next', 'endpoint1_Y_next', 'endpoint2_X_next', 'endpoint2_Y_next'
                    ]].astype('float64'))

    if daughter_flag:
        # Calculate intersection by merging on coordinates
        product_df = \
            compute_intersections_and_unions_division(product_df, 'prev_index_current',
                                                      'prev_index_next', coordinate_array)
    else:
        # Calculate intersection by merging on coordinates
        product_df = compute_intersections_and_unions(product_df, 'prev_index_current',
                                                      'prev_index_next', coordinate_array)

    # Pivot this DataFrame to get the desired structure
    overlap_df = \
        product_df[['index_current', 'index_next', 'iou']].pivot(index='index_current', columns='index_next',
                                                                 values='iou')

    overlap_df.columns.name = None
    overlap_df.index.name = None
    overlap_df = overlap_df.astype(float)

    return overlap_df, product_df


def calculate_frame_existence_links(division_df, continuity_df, coordinate_array):

    """
    Calculates overlaps source & target object of existence links in consecutive frames, distinguishing between
    objects involved in divisions and those maintaining continuity.

    :param pandas.DataFrame division_df:
        DataFrame containing objects involved in division, with parent and daughter indices for overlap computation.
    :param pandas.DataFrame continuity_df:
        DataFrame containing objects maintaining continuity between frames, with indices for overlap computation.
    :param csr_matrix coordinate_array:
        Sparse boolean array where each row corresponds to the coordinates of an object.

    :returns:
        pandas.DataFrame

        A combined pivoted DataFrame containing IoU values:
        - For objects involved in division, it shows overlaps between parent and daughter objects.
        - For continuous objects, it shows overlaps between objects in the two frames.

    """

    division_overlap = pd.DataFrame()
    continuity_df_overlap = pd.DataFrame()

    if division_df.shape[0] > 0:
        division_df = compute_intersections_and_unions_division(division_df, 'prev_index_parent',
                                                                'prev_index_daughter', coordinate_array)

        # Pivot this DataFrame to get the desired structure
        division_overlap = division_df[['index_parent', 'index_daughter', 'iou']].pivot(index='index_parent',
                                                                                        columns='index_daughter',
                                                                                        values='iou')

    if continuity_df.shape[0] > 0:
        continuity_df = compute_intersections_and_unions(continuity_df, 'prev_index_1', 'prev_index_2',
                                                         coordinate_array)
        continuity_df_overlap = continuity_df[['index_1', 'index_2', 'iou']].pivot(index='index_1', columns='index_2',
                                                                                   values='iou')

    if division_df.shape[0] > 0 and continuity_df.shape[0] > 0:
        overlap_df = pd.concat([division_overlap, continuity_df_overlap], axis=0)
    elif division_df.shape[0] > 0:
        overlap_df = division_overlap
    elif continuity_df.shape[0] > 0:
        overlap_df = continuity_df_overlap
    else:
        raise ValueError("Both `division_df` and `continuity_df` are empty. Cannot calculate overlaps.")

    overlap_df.columns.name = None
    overlap_df.index.name = None
    overlap_df = overlap_df.astype(float)

    return overlap_df
