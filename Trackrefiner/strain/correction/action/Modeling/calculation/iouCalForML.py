import pandas as pd


def calculate_intersections_and_unions(raw_df, df, col1, col2, stat='same', chunk_size=5000):

    # col1 , col2 refer to index

    dataframe_len = df.shape[0]
    intersection_list = []
    union_list = []
    unique_masks2_list = []

    for i in range(0, dataframe_len, chunk_size):

        # Define the bounds for the current chunk
        start_idx = i
        end_idx = min(i + chunk_size, dataframe_len)

        idx1_vals = df[col1].values[start_idx:end_idx]
        idx2_vals = df[col2].values[start_idx:end_idx]

        if stat == 'same':
            intersection_list.extend(
                raw_df.loc[idx1_vals]['coordinate'].values & raw_df.loc[idx2_vals]['coordinate'].values)

            union_list.extend(
                raw_df.loc[idx1_vals]['coordinate'].values | raw_df.loc[idx2_vals]['coordinate'].values)

        elif stat == 'div':

            intersection_list.extend(
                raw_df.loc[idx1_vals]['coordinate'].values & raw_df.loc[idx2_vals]['coordinate'].values)

            union_list.extend(
                raw_df.loc[idx1_vals]['coordinate'].values | raw_df.loc[idx2_vals]['coordinate'].values)

            # Calculate unique areas for mask 2 (daughter)
            unique_masks2_list.extend(
                raw_df.loc[idx2_vals]['coordinate'].values - raw_df.loc[idx1_vals]['coordinate'].values)

    if stat == 'same':

        df['intersection'] = intersection_list

        df['union'] = union_list

        df['intersection'] = df['intersection'].map(len)
        df['union'] = df['union'].map(len)

        df['intersection'] = df['intersection'].astype('int64')
        df['union'] = df['union'].astype('int64')

        df['iou'] = df['intersection'] / df['union']

    elif stat == 'div':

        df['intersection'] = intersection_list

        df['union'] = union_list
        df['unique_masks2'] = unique_masks2_list

        df['intersection'] = df['intersection'].map(len)

        df['union'] = df['union'].map(len)

        # Calculate unique areas for mask 2 (daughter)
        df['unique_masks2'] = df['unique_masks2'].map(len)

        df['intersection'] = df['intersection'].astype('int64')
        df['unique_masks2'] = df['unique_masks2'].astype('int64')
        df['union'] = df['union'].astype('int64')

        # Calculate IoU based on daughter_flag
        df['iou'] = df['intersection'] / (df['intersection'] + df['unique_masks2'])
        df['iou_same'] = df['intersection'] / df['union']

    return df


def iou_calc(raw_df, df, stat, col_source='prev_index_prev', col_target='prev_index'):

    df = calculate_intersections_and_unions(raw_df, df, col1=col_source, col2=col_target, stat=stat)

    df['iou'] = 1 - df['iou']

    if stat == 'div':
        df['iou_same'] = 1 - df['iou_same']

    return df
