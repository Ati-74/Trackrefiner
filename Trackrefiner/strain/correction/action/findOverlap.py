import pandas as pd
import numpy as np


def calculate_intersections_and_unions(df, col1, col2, coordinate_array):

    # col1 , col2 refer to index

    idx1_vals = df[col1].to_numpy(dtype='int64')
    idx2_vals = df[col2].to_numpy(dtype='int64')

    # Perform element-wise concatenation
    intersection_size_element_wise_concatenation = \
        coordinate_array[idx1_vals].multiply(coordinate_array[idx2_vals])

    union_size_element_wise_concatenation = \
        coordinate_array[idx1_vals] + coordinate_array[idx2_vals]

    # Count the number of True values in AND and OR results
    intersection_lens = intersection_size_element_wise_concatenation.getnnz(axis=1)
    union_lens = union_size_element_wise_concatenation.getnnz(axis=1)

    # Preallocate the iou column in df
    iou_values = (intersection_lens / union_lens).astype('float32')

    # Create a new DataFrame with 'iou' as the column
    df_iou = pd.DataFrame(index=df.index, data={'iou': iou_values})

    df = pd.concat([df, df_iou], axis=1)

    return df


def calculate_intersections_and_unions_division(df, col1, col2, coordinate_array):

    idx1_vals = df[col1].to_numpy(dtype='int64')
    idx2_vals = df[col2].to_numpy(dtype='int64')

    # Perform element-wise concatenation
    intersection_size_element_wise_concatenation = \
        coordinate_array[idx1_vals].multiply(coordinate_array[idx2_vals])

    # Count the number of True values in AND and OR results
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


def find_overlap_mother_bad_daughters(mother_bad_daughters_df, coordinate_array):

    # Calculate intersection by merging on coordinates
    mother_bad_daughters_df = \
        calculate_intersections_and_unions(mother_bad_daughters_df, 'prev_index_mother',
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


def find_overlap_object_for_division_chance(source_with_candidate_neighbors, center_coordinate_columns, col1,
                                            col2, daughter_flag=False, coordinate_array=None):
    if daughter_flag:
        # Calculate intersection by merging on coordinates
        source_with_candidate_neighbors = \
            calculate_intersections_and_unions_division(source_with_candidate_neighbors,
                                                        'prev_index' + col1, 'prev_index' + col2, coordinate_array)
    else:
        # Calculate intersection by merging on coordinates
        source_with_candidate_neighbors = \
            calculate_intersections_and_unions(source_with_candidate_neighbors,
                                               'prev_index' + col1, 'prev_index' + col2, coordinate_array)

    # Pivot this DataFrame to get the desired structure
    overlap_df = \
        source_with_candidate_neighbors[['index' + col1, 'index' + col2,
                                         'iou']].pivot(index='index' + col1, columns='index' + col2, values='iou')

    overlap_df.columns.name = None
    overlap_df.index.name = None
    overlap_df = overlap_df.astype(float)

    return overlap_df


def find_overlap_object_to_next_frame(current_df, selected_objects, next_df, selected_objects_in_next_time_step,
                                      center_coordinate_columns, color_array,
                                      daughter_flag=False, coordinate_array=None):

    current_df_color_array = color_array[current_df['prev_index'].values]
    next_df_color_array = color_array[next_df['prev_index'].values]

    if np.unique(current_df_color_array, axis=0).shape[0] != current_df_color_array.shape[0]:
        # Find unique rows and their indices
        unique_rows, indices, counts = np.unique(current_df_color_array, axis=0, return_index=True, return_counts=True)

        # Get the duplicate rows (those that appear more than once)
        duplicate_rows = unique_rows[counts > 1]
        print(current_df['ImageNumber'].values[0])
        print(duplicate_rows)
        breakpoint()

    if np.unique(next_df_color_array, axis=0).shape[0] != next_df_color_array.shape[0]:
        # Find unique rows and their indices
        unique_rows, indices, counts = np.unique(next_df_color_array, axis=0, return_index=True, return_counts=True)

        # Get the duplicate rows (those that appear more than once)
        duplicate_rows = unique_rows[counts > 1]
        print(next_df['df'].values[0])
        print(duplicate_rows)
        breakpoint()

    product_df = pd.merge(
        selected_objects.reset_index(),
        selected_objects_in_next_time_step.reset_index(),
        how='cross',
        suffixes=('_current', '_next')
    )
    product_df[['index_current', 'index_next']] = product_df[['index_current', 'index_next']].astype('int64')
    product_df[[center_coordinate_columns['x'] + '_current',
                center_coordinate_columns['y'] + '_current',
                center_coordinate_columns['x'] + '_next',
                center_coordinate_columns['y'] + '_next',
                'endpoint1_X_current', 'endpoint1_Y_current',
                'endpoint2_X_current', 'endpoint2_Y_current',
                'endpoint1_X_next', 'endpoint1_Y_next',
                'endpoint2_X_next', 'endpoint2_Y_next']] = (
        product_df[[center_coordinate_columns['x'] + '_current', center_coordinate_columns['y'] + '_current',
                    center_coordinate_columns['x'] + '_next', center_coordinate_columns['y'] + '_next',
                    'endpoint1_X_current', 'endpoint1_Y_current',
                    'endpoint2_X_current', 'endpoint2_Y_current',
                    'endpoint1_X_next', 'endpoint1_Y_next',
                    'endpoint2_X_next', 'endpoint2_Y_next'
                    ]].astype('float64'))

    if daughter_flag:
        # Calculate intersection by merging on coordinates
        product_df = \
            calculate_intersections_and_unions_division(product_df, 'prev_index_current',
                                                        'prev_index_next', coordinate_array)
    else:
        # Calculate intersection by merging on coordinates
        product_df = calculate_intersections_and_unions(product_df, 'prev_index_current',
                                                        'prev_index_next', coordinate_array)

    # Pivot this DataFrame to get the desired structure
    overlap_df = \
        product_df[['index_current', 'index_next', 'iou']].pivot(index='index_current', columns='index_next',
                                                                 values='iou')

    overlap_df.columns.name = None
    overlap_df.index.name = None
    overlap_df = overlap_df.astype(float)

    return overlap_df, product_df


def find_overlap_object_to_next_frame_maintain(division_df, same_df, coordinate_array):

    if division_df.shape[0] > 0:
        division_df = calculate_intersections_and_unions_division(division_df, 'prev_index_parent',
                                                                  'prev_index_daughter', coordinate_array)

        # Pivot this DataFrame to get the desired structure
        division_overlap = division_df[['index_parent', 'index_daughter', 'iou']].pivot(index='index_parent',
                                                                                        columns='index_daughter',
                                                                                        values='iou')

    if same_df.shape[0] > 0:
        same_df = calculate_intersections_and_unions(same_df, 'prev_index_1', 'prev_index_2',
                                                     coordinate_array)
        same_df_overlap = same_df[['index_1', 'index_2', 'iou']].pivot(index='index_1', columns='index_2', values='iou')

    if division_df.shape[0] > 0 and same_df.shape[0] > 0:
        overlap_df = pd.concat([division_overlap, same_df_overlap], axis=0)
    elif division_df.shape[0] > 0:
        overlap_df = division_overlap
    elif same_df.shape[0] > 0:
        overlap_df = same_df_overlap
    else:
        breakpoint()

    overlap_df.columns.name = None
    overlap_df.index.name = None
    overlap_df = overlap_df.astype(float)

    return overlap_df


def find_overlap_object_to_next_frame_unexpected(final_candidate_bac, coordinate_array):

    if final_candidate_bac.shape[0] > 0:

        final_candidate_bac = calculate_intersections_and_unions(final_candidate_bac, 'prev_index',
                                                                 'prev_index_candidate', coordinate_array)

    overlap_df = final_candidate_bac[['index', 'index_candidate', 'iou']].pivot(
        index='index', columns='index_candidate', values='iou')
    overlap_df.columns.name = None
    overlap_df.index.name = None
    overlap_df = overlap_df.astype(float)

    return overlap_df
