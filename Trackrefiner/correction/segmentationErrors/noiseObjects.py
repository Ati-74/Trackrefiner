import numpy as np
from Trackrefiner.correction.action.findOutlier import calculate_bacterial_length_boundary, \
    calculate_lower_statistical_bound


def update_tracking_after_removing_noise(df, noise_objects_df, neighbors_df, parent_image_number_col,
                                         parent_object_number_col):

    """
    Update tracking and neighbor relationships after removing noise bacteria.

    This function updates the tracking data and neighbor relationships when noise bacteria
    are identified and flagged. It:
    - Removes relationships where noise bacteria are parents.
    - Cleans up incorrect neighbor relationships involving noise bacteria.

    :param pd.DataFrame df:
        Dataframe containing bacterial tracking data, including IDs, features, and noise flags.
    :param pd.DataFrame noise_objects_df:
        Subset of the dataframe containing the detected noise bacteria to be removed.
    :param pd.DataFrame neighbors_df:
        Dataframe containing neighbor relationship information for bacteria.
    :param str parent_image_number_col:
        Column name for the parent image number in the dataframe.
    :param str parent_object_number_col:
        Column name for the parent object number in the dataframe.

    :return:
        tuple:

        - **df** (*pd.DataFrame*): Updated dataframe with noise bacteria flagged and their parent relationships reset.
        - **neighbors_df** (*pd.DataFrame*): Updated neighbor relationships with noise bacteria removed.

    """

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


def detect_and_remove_noise_bacteria(df, neighbors_df, parent_image_number_col, parent_object_number_col):

    """
    Detect and remove noise bacteria from tracking data.

    This function identifies noise objects (bacteria) based on major length and removes
    them from the dataframe. It iteratively detects noise objects, removes them, and updates
    neighbor relationships.

    :param pd.DataFrame df:
        Dataframe containing bacterial tracking data, including features such as object major length
    :param pd.DataFrame neighbors_df:
        Dataframe containing information about neighboring bacteria relationships.
    :param str parent_image_number_col:
        Column name for the parent image number in the dataframe.
    :param str parent_object_number_col:
        Column name for the parent object number in the dataframe.

    :return:
        tuple:

        - **df** (*pd.DataFrame*): Updated dataframe with noise bacteria removed.
        - **neighbors_df** (*pd.DataFrame*): Updated neighbor relationships after removing noise bacteria.
    """

    num_noise_obj = None

    neighbors_df['index_neighborhood'] = neighbors_df.index

    detected_noise_bac_in_prev_time_step = df.loc[df['noise_bac']]

    if detected_noise_bac_in_prev_time_step.shape[0] > 0:
        df, neighbors_df = \
            update_tracking_after_removing_noise(df, detected_noise_bac_in_prev_time_step, neighbors_df,
                                                 parent_image_number_col, parent_object_number_col)

    while num_noise_obj != 0:

        bac_len_boundary = calculate_bacterial_length_boundary(df)

        bac_len_lower_bound = calculate_lower_statistical_bound(bac_len_boundary)

        noise_objects_df = df.loc[(df['AreaShape_MajorAxisLength'] < bac_len_lower_bound) & (~ df['noise_bac'])]

        num_noise_obj = noise_objects_df.shape[0]

        if noise_objects_df.shape[0] > 0:
            df, neighbors_df = \
                update_tracking_after_removing_noise(df, noise_objects_df, neighbors_df, parent_image_number_col,
                                                     parent_object_number_col)

    df = df.loc[df['noise_bac'] == False].reset_index(drop=True)

    return df, neighbors_df
