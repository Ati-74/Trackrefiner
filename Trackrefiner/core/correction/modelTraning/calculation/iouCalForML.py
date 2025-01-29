import pandas as pd


def compute_intersection_union(df, col1, col2, link_type='continuity', df_view=None, coordinate_array=None, both=True):

    """
    Calculates intersections and unions for sparse matrices represented bacteria coordinates in `coordinate_array`
    and derives Intersection over Union (IoU) metrics for tracking links.

    :param pandas.DataFrame df:
        DataFrame containing the indices (`col1` and `col2`) used to query the binary matrices.
    :param str col1:
        Column name in `df` representing the source indices.
    :param str col2:
        Column name in `df` representing the target indices.
    :param str link_type:
        The type of link to evaluate. Options:
        - `'continuity'`: Evaluates IoU using standard intersection and union metrics.
        - `'div'`: Includes a modified IoU calculation for division links, where unique daughter areas are considered.
    :param pandas.DataFrame df_view:
        Optional subset of `df` to use for calculations. Defaults to `None`.
    :param scipy.sparse.csr_matrix coordinate_array:
        Sparse matrix representing the coordinate of bacteria.
    :param bool both:
        If `True`, computes additional IoU metrics for continuity in division links.

    **Returns**:
        pandas.DataFrame:

        The updated DataFrame `df` with added columns:
            - `'iou'`: Calculated IoU values.
            - `'iou_continuity'` (optional): IoU for continuity when `link_type='div'` and `both=True`.
    """

    # col1 , col2 refer to index

    if df_view is not None:
        idx1_vals = df_view[col1].to_numpy(dtype='int64')
        idx2_vals = df_view[col2].to_numpy(dtype='int64')
    else:
        idx1_vals = df[col1].to_numpy(dtype='int64')
        idx2_vals = df[col2].to_numpy(dtype='int64')

    if link_type == 'continuity':

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

    elif link_type == 'div':

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

        if both:
            union_size_element_wise_concatenation = \
                coordinate_array[idx1_vals] + coordinate_array[idx2_vals]

            union_lens = union_size_element_wise_concatenation.getnnz(axis=1)

            iou_continuity = (intersection_lens / union_lens)
            df_iou_continuity = pd.DataFrame(index=df.index, data={'iou_continuity': iou_continuity})
            df = pd.concat([df, df_iou_continuity], axis=1)

    return df


def calculate_iou_ml(df, link_type, col_source='prev_index_prev', col_target='prev_index', df_view=None,
                     coordinate_array=None, both=True):

    """
    Wrapper function for calculating Intersection over Union (IoU) values using `compute_intersection_union` function.

    :param pandas.DataFrame df:
        DataFrame containing the source (`col_source`) and target (`col_target`) indices.
    :param str link_type:
        The type of link to evaluate, passed to `calculate_intersections_and_unions`. Options:
        - `'continuity'`
        - `'div'`
    :param str col_source:
        Column name in `df` for source indices. Default is `'prev_index_prev'`.
    :param str col_target:
        Column name in `df` for target indices. Default is `'prev_index'`.
    :param pandas.DataFrame df_view:
        Optional subset of `df` to use for efficient computation. Defaults to `None`.
    :param scipy.sparse.csr_matrix coordinate_array:
        Sparse matrix representing the coordinate of bacteria.
    :param bool both:
        If `True`, computes additional IoU metrics for continuity in division links.

    **Returns**:
        pandas.DataFrame:

        The updated DataFrame `df` with added columns:
            - `'iou'`: Inverted IoU values (1 - IoU).
            - `'iou_continuity'` (optional): Inverted IoU for continuity when `link_type='div'` and `both=True`.
    """

    df = compute_intersection_union(df, col1=col_source, col2=col_target, link_type=link_type,
                                    df_view=df_view, coordinate_array=coordinate_array, both=both)

    df['iou'] = 1 - df['iou']

    if link_type == 'div' and both:
        df['iou_continuity'] = 1 - df['iou_continuity']

    return df
