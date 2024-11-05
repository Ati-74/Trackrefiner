import pandas as pd
import numpy as np
from Trackrefiner.strain.correction.action.findOutlier import find_bac_len_boundary, find_lower_bound
from Trackrefiner.strain.correction.action.helperFunctions import remove_rows


def remove_noise_bac_info(df, noise_objects_df, neighbors_df, parent_image_number_col, parent_object_number_col):

    df.loc[noise_objects_df['prev_index'].values, 'noise_bac'] = True

    bac_with_noise_objects_as_parent = \
        noise_objects_df.merge(df, left_on=['ImageNumber', 'ObjectNumber'],
                               right_on=[parent_image_number_col, parent_object_number_col],
                               suffixes=('', '_target_obj'), how='inner')

    if bac_with_noise_objects_as_parent.shape[0] > 0:

        df.loc[bac_with_noise_objects_as_parent['prev_index_target_obj'], [
            parent_image_number_col, parent_object_number_col]] = 0

    neighbors_df_incorrect_source = neighbors_df.merge(noise_objects_df,
                                                       left_on=['First Image Number', 'First Object Number'],
                                                       right_on=['ImageNumber', 'ObjectNumber'], how='inner')

    neighbors_df_incorrect_target = neighbors_df.merge(noise_objects_df,
                                                       left_on=['Second Image Number', 'Second Object Number'],
                                                       right_on=['ImageNumber', 'ObjectNumber'], how='inner')

    incorrect_ndx = np.unique(np.concatenate((neighbors_df_incorrect_source['index_neighborhood'].values,
                                              neighbors_df_incorrect_target['index_neighborhood'].values)))

    neighbors_df = neighbors_df.loc[~ neighbors_df['index_neighborhood'].isin(incorrect_ndx)]

    return df, neighbors_df


def noise_remover(df, neighbors_df, parent_image_number_col, parent_object_number_col):

    num_noise_obj = None

    neighbors_df['index_neighborhood'] = neighbors_df.index

    detected_noise_bac_in_prev_time_step = df.loc[df['noise_bac']]

    if detected_noise_bac_in_prev_time_step.shape[0] > 0:
        df, neighbors_df = \
            remove_noise_bac_info(df, detected_noise_bac_in_prev_time_step, neighbors_df, parent_image_number_col,
                                  parent_object_number_col)

    while num_noise_obj != 0:

        bac_len_boundary = find_bac_len_boundary(df)

        bac_len_lower_bound = find_lower_bound(bac_len_boundary)

        noise_objects_df = df.loc[(df['AreaShape_MajorAxisLength'] < bac_len_lower_bound) & (~ df['noise_bac'])]

        num_noise_obj = noise_objects_df.shape[0]

        if noise_objects_df.shape[0] > 0:
            df, neighbors_df = \
                remove_noise_bac_info(df, noise_objects_df, neighbors_df, parent_image_number_col,
                                      parent_object_number_col)

    df = remove_rows(df, 'noise_bac', False)

    return df, neighbors_df
