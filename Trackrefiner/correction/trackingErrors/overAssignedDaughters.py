import numpy as np
from Trackrefiner.correction.action.helper import calculate_angles_between_slopes
from Trackrefiner.correction.modelTraning.calculation.iouCalForML import calculate_iou_ml
from Trackrefiner.correction.modelTraning.calculation.calcDistanceForML import calc_min_distance_ml


def resolve_over_assigned_daughters_link(df, parent_image_number_col, parent_object_number_col, center_coord_cols,
                                         division_links_model, coordinate_array):

    """
    This function identifies parent bacteria with over-assigned daughter links (OAD), evaluates
    the validity of these links using a machine learning model, and resolves the issue by:
    - Removing invalid daughter links.
    - Updating features for parent bacteria with corrected daughters.
    - Propagating corrected features to the daughter bacteria.

    :param pandas.DataFrame df:
        DataFrame containing tracking data for bacteria, including division-related features.
    :param str parent_image_number_col:
        Column name for the parent image numbers.
    :param str parent_object_number_col:
        Column name for the parent object numbers.
    :param dict center_coord_cols:
        Dictionary specifying column names for bacterial centroid coordinates
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param sklearn.Model division_links_model:
        Machine learning model used to evaluate the probability of valid division links.
    :param csr_matrix coordinate_array:
        Array of spatial coordinates used for evaluating spatial relationships.

    **Returns**:
    :returns pandas.DataFrame: Updated DataFrame with corrected division links and propagated features.
    """

    # OAD = over assigned daughters
    mothers_with_oad = df.loc[df['bad_division_flag'] == True]

    bad_daughters_list = []

    if mothers_with_oad.shape[0] > 0:

        mothers_and_oads = \
            mothers_with_oad.merge(df, left_on=['ImageNumber', 'ObjectNumber'],
                                   right_on=[parent_image_number_col, parent_object_number_col], how='inner',
                                   suffixes=('_parent', '_daughter'))

        # now we should apply ml model

        # IOU
        mothers_and_oads = \
            calculate_iou_ml(mothers_and_oads, col_source='prev_index_parent', col_target='prev_index_daughter',
                             link_type='div', coordinate_array=coordinate_array, both=False)

        # distance
        mothers_and_oads = calc_min_distance_ml(mothers_and_oads, center_coord_cols, '_daughter',
                                                '_parent', link_type='div')

        mothers_and_oads['LengthChangeRatio'] = (mothers_and_oads['AreaShape_MajorAxisLength_daughter'] /
                                                 mothers_and_oads['AreaShape_MajorAxisLength_parent'])

        mothers_and_oads['angle_mother_daughter'] = \
            calculate_angles_between_slopes(mothers_and_oads['bacteria_slope_parent'].values,
                                            mothers_and_oads['bacteria_slope_daughter'].values)

        mothers_and_oads['adjusted_common_neighbors_daughter'] = np.where(
            mothers_and_oads['common_neighbors_daughter'] == 0,
            mothers_and_oads['common_neighbors_daughter'] + 1,
            mothers_and_oads['common_neighbors_daughter']
        )

        mothers_and_oads['neighbor_ratio_daughter'] = \
            (mothers_and_oads['difference_neighbors_daughter'] / (
                mothers_and_oads['adjusted_common_neighbors_daughter']))

        # rename columns
        mothers_and_oads = mothers_and_oads.rename(
            {'difference_neighbors_daughter': 'difference_neighbors',
             'neighbor_ratio_daughter': 'neighbor_ratio',
             'direction_of_motion_daughter': 'direction_of_motion',
             'MotionAlignmentAngle_daughter': 'MotionAlignmentAngle'}, axis=1)

        # we ignored the effect of each daughter to another daughters in motion alignment
        # (because we didn't calc daughter_length_to_mother for bad mothers)

        if mothers_and_oads.shape[0] > 0:

            # difference_neighbors
            feature_list = ['iou', 'min_distance', 'neighbor_ratio', 'angle_mother_daughter']

            y_prob_compare_division = division_links_model.predict_proba(mothers_and_oads[feature_list])[:, 1]

            mothers_and_oads['prob_compare'] = y_prob_compare_division

            # Pivot this DataFrame to get the desired structure
            cost_df = \
                mothers_and_oads[['index_parent', 'index_daughter', 'prob_compare']].pivot(
                    index='index_parent', columns='index_daughter', values='prob_compare')
            cost_df.columns.name = None
            cost_df.index.name = None

            cost_df = 1 - cost_df

            mother_with_bad_daughters = cost_df[cost_df.notna().sum(axis=1) > 2]

            while mother_with_bad_daughters.shape[0] > 0:
                bad_daughters = mother_with_bad_daughters.idxmax(axis=1)
                cost_df[bad_daughters.values] = np.nan

                bad_daughters_list.extend(bad_daughters.values.tolist())

                mother_with_bad_daughters = cost_df[cost_df.notna().sum(axis=1) > 2]

    # remove bad daughters link
    if len(bad_daughters_list) > 0:

        df.loc[bad_daughters_list, [
            parent_image_number_col, parent_object_number_col, 'parent_id', 'unexpected_beginning', 'LengthChangeRatio',
            'TrajectoryX', 'TrajectoryY', 'direction_of_motion', 'MotionAlignmentAngle',
            'prev_time_step_index', 'difference_neighbors', 'common_neighbors']] = \
            [0, 0, 0, True, np.nan, np.nan, np.nan, np.nan, np.nan, -1, 0, 0]

        for bad_daughter_idx in bad_daughters_list:
            df.loc[df['id'] == df.at[bad_daughter_idx, 'id'], 'parent_id'] = 0

        # change the value of bad division flag
        df['bad_division_flag'] = False

    # now mothers with correct daughters
    division = \
        mothers_with_oad.merge(df, left_on=['ImageNumber', 'ObjectNumber'],
                               right_on=[parent_image_number_col, parent_object_number_col], how='inner',
                               suffixes=('_1', '_2')).copy()

    division['daughters_index'] = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['index_2'].transform(lambda x: ', '.join(x.astype(str)))

    division['temp_sum_daughters_len_to_mother'] = division.groupby(['ImageNumber_1', 'ObjectNumber_1'])[
                                                       'AreaShape_MajorAxisLength_2'].transform('sum') / division[
                                                       'AreaShape_MajorAxisLength_1']

    division['temp_max_daughters_len_to_mother'] = division.groupby(['ImageNumber_1', 'ObjectNumber_1'])[
                                                       'AreaShape_MajorAxisLength_2'].transform('max') / division[
                                                       'AreaShape_MajorAxisLength_1']

    division['temp_avg_daughters_trajectory_x'] = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['TrajectoryX_2'].transform('mean')

    division['temp_avg_daughters_trajectory_y'] = \
        division.groupby(['ImageNumber_1', 'ObjectNumber_1'])['TrajectoryY_2'].transform('mean')

    mothers_df_last_time_step = division.drop_duplicates(subset='index_1', keep='first')

    mothers_idx = division['index_1'].unique()

    df.loc[mothers_idx, 'daughters_index'] = mothers_df_last_time_step['daughters_index'].values

    df.loc[mothers_idx, 'daughter_length_to_mother'] = \
        mothers_df_last_time_step['temp_sum_daughters_len_to_mother'].values
    df.loc[mothers_idx, 'max_daughter_len_to_mother'] = \
        mothers_df_last_time_step['temp_max_daughters_len_to_mother'].values
    df.loc[mothers_idx, 'avg_daughters_TrajectoryX'] = \
        mothers_df_last_time_step['temp_avg_daughters_trajectory_x'].values
    df.loc[mothers_idx, 'avg_daughters_TrajectoryY'] = \
        mothers_df_last_time_step['temp_avg_daughters_trajectory_y'].values

    division['daughter_mother_LengthChangeRatio'] = \
        (division['AreaShape_MajorAxisLength_2'] / division['AreaShape_MajorAxisLength_1'])

    division['daughter_mother_slope'] = \
        calculate_angles_between_slopes(division['bacteria_slope_2'].values,
                                        division['bacteria_slope_1'].values)

    daughters_idx = division['index_2'].values

    df.loc[daughters_idx, 'daughter_mother_LengthChangeRatio'] = \
        division['daughter_mother_LengthChangeRatio'].values

    # should calc for all bacteria
    df.loc[daughters_idx, 'slope_bac_bac'] = division['daughter_mother_slope'].values

    df.loc[daughters_idx, 'prev_bacteria_slope'] = division['bacteria_slope_1'].values

    df.loc[daughters_idx, 'prev_time_step_index'] = division['index_1'].values

    # correct divisions
    daughter_to_daughter = division.merge(division, on=[parent_image_number_col + '_2',
                                                        parent_object_number_col + '_2'],
                                          suffixes=('_daughter1', '_daughter2'))

    daughter_to_daughter = daughter_to_daughter.loc[daughter_to_daughter['index_2_daughter1'] !=
                                                    daughter_to_daughter['index_2_daughter2']]

    df.loc[daughter_to_daughter['index_2_daughter1'].values, "other_daughter_index"] = \
        daughter_to_daughter['index_2_daughter2'].values

    df['bad_division_flag'] = False
    df['ovd_flag'] = False
    df['bad_daughters_flag'] = False

    return df
