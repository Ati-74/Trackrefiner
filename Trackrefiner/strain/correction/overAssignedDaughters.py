import numpy as np
from Trackrefiner.strain.correction.action.helperFunctions import calculate_orientation_angle_batch
from Trackrefiner.strain.correction.action.Modeling.calculation.iouCalForML import iou_calc
from Trackrefiner.strain.correction.action.Modeling.calculation.calcDistanceForML import calc_distance


def remove_over_assigned_daughters_link(raw_df, df, neighbor_df, parent_image_number_col, parent_object_number_col,
                                        label_col, center_coordinate_columns, divided_bac_model):
    """
        goal: modification of bad daughters (try to assign bad daughters to new parent)
        @param df    dataframe   bacteria dataframe
        in last time step of its life history before unexpected_beginning bacterium to length of candidate
        parent bacterium in investigated time step
        output: df   dataframe   modified dataframe (without bad daughters)
    """

    # OAD = over assigned daughters
    # The reason I attempted to remove the duplicate is that when I detect that a bacterium is dividing,
    # I change the value of the division flag for the bacterium's life history, not just for the last time step before
    # division.
    # .drop_duplicates(subset=['daughters_index'], keep='last')
    # it means bad mothers
    mothers_with_oad = df.loc[df['bad_division_flag'] == True]

    bad_daughters_list = []

    if mothers_with_oad.shape[0] > 0:

        mothers_and_oads = \
            mothers_with_oad.merge(df, left_on=['ImageNumber', 'ObjectNumber'],
                                   right_on=[parent_image_number_col, parent_object_number_col], how='inner',
                                   suffixes=('_parent', '_daughter'))

        # now we should apply ml model

        # IOU
        mothers_and_oads = iou_calc(raw_df, mothers_and_oads,
                                    col_source='prev_index_parent', col_target='prev_index_daughter', stat='div')

        # distance
        mothers_and_oads = calc_distance(mothers_and_oads, center_coordinate_columns, '_daughter',
                                         '_parent', stat='div')

        mothers_and_oads['LengthChangeRatio'] = (mothers_and_oads['AreaShape_MajorAxisLength_daughter'] /
                                                 mothers_and_oads['AreaShape_MajorAxisLength_parent'])

        mothers_and_oads['angle_mother_daughter'] = \
            calculate_orientation_angle_batch(mothers_and_oads['bacteria_slope_parent'].values,
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

            y_prob_compare_division = divided_bac_model.predict_proba(mothers_and_oads[feature_list])[:, 1]

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
            'prev_time_step_NeighborIndexList', 'difference_neighbors', 'common_neighbors']] = \
            [0, 0, 0, True, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0]

        for bad_daughter_idx in bad_daughters_list:
            df.loc[df['id'] == df.at[bad_daughter_idx, 'id'], 'parent_id'] = 0

        # change the value of bad division flag
        df['bad_division_flag'] = False

    # now mothers with correct daughters
    division = \
        mothers_with_oad.merge(df, left_on=['ImageNumber', 'ObjectNumber'],
                               right_on=[parent_image_number_col, parent_object_number_col], how='inner',
                               suffixes=('_1', '_2')).copy()

    # ====================================================================================================================
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

    df.loc[division['index_1'].unique(), 'daughters_index'] = mothers_df_last_time_step['daughters_index'].values

    df.loc[division['index_1'].unique(), 'daughter_length_to_mother'] = \
        mothers_df_last_time_step['temp_sum_daughters_len_to_mother'].values
    df.loc[division['index_1'].unique(), 'max_daughter_len_to_mother'] = \
        mothers_df_last_time_step['temp_max_daughters_len_to_mother'].values
    df.loc[division['index_1'].unique(), 'avg_daughters_TrajectoryX'] = \
        mothers_df_last_time_step['temp_avg_daughters_trajectory_x'].values
    df.loc[division['index_1'].unique(), 'avg_daughters_TrajectoryY'] = \
        mothers_df_last_time_step['temp_avg_daughters_trajectory_y'].values

    ##################################################################################################################
    division['daughter_mother_LengthChangeRatio'] = \
        (division['AreaShape_MajorAxisLength_2'] / division['AreaShape_MajorAxisLength_1'])

    division['daughter_mother_slope'] = \
        calculate_orientation_angle_batch(division['bacteria_slope_2'].values,
                                          division['bacteria_slope_1'].values)

    df.loc[division['index_2'].values, 'daughter_mother_LengthChangeRatio'] = \
        division['daughter_mother_LengthChangeRatio'].values

    # should calc for all bacteria
    df.loc[division['index_2'].values, 'slope_bac_bac'] = division['daughter_mother_slope'].values

    df.loc[division['index_2'].values, 'prev_bacteria_slope'] = division['bacteria_slope_1'].values

    df.loc[division['index_2'].values, 'prev_time_step_NeighborIndexList'] = division['NeighborIndexList_1'].values

    # correct divisions
    daughter_to_daughter = division.merge(division, on=[parent_image_number_col + '_2',
                                                        parent_object_number_col + '_2'],
                                          suffixes=('_daughter1', '_daughter2'))

    daughter_to_daughter = daughter_to_daughter.loc[daughter_to_daughter['index_2_daughter1'] !=
                                                    daughter_to_daughter['index_2_daughter2']]

    df.loc[daughter_to_daughter['index_2_daughter1'].values, "other_daughter_index"] = \
        daughter_to_daughter['index_2_daughter2'].values

    # df = calc_new_features_after_oad(df, neighbor_df, center_coordinate_columns, parent_image_number_col,
    #                                 parent_object_number_col, label_col)

    df['bad_division_flag'] = False
    df['ovd_flag'] = False
    df['bad_daughters_flag'] = False

    return df
