import pandas as pd
import numpy as np


def calculate_intersections_and_unions(df, col1, col2, stat='same', df_view=None,
                                       coordinate_array=None, both=True):
    # col1 , col2 refer to index

    if df_view is not None:
        idx1_vals = df_view[col1].to_numpy(dtype='int64')
        idx2_vals = df_view[col2].to_numpy(dtype='int64')
    else:
        idx1_vals = df[col1].to_numpy(dtype='int64')
        idx2_vals = df[col2].to_numpy(dtype='int64')

    if stat == 'same':

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

    elif stat == 'div':

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

        if both:
            union_size_element_wise_concatenation = \
                coordinate_array[idx1_vals] + coordinate_array[idx2_vals]

            union_lens = union_size_element_wise_concatenation.getnnz(axis=1)

            iou_same = (intersection_lens / union_lens)
            df_iou_same = pd.DataFrame(index=df.index, data={'iou_same': iou_same})
            df = pd.concat([df, df_iou_same], axis=1)

    return df


def iou_calc(df, stat, col_source='prev_index_prev', col_target='prev_index', df_view=None,
             coordinate_array=None, both=True):

    df = calculate_intersections_and_unions(df, col1=col_source, col2=col_target, stat=stat,
                                            df_view=df_view, coordinate_array=coordinate_array, both=both)

    df['iou'] = 1 - df['iou']

    if stat == 'div' and both:
        df['iou_same'] = 1 - df['iou_same']

    return df
