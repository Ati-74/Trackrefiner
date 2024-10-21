import numpy as np
from Trackrefiner.strain.correction.action.findOutlier import find_daughter_len_to_mother_ratio_outliers, \
    find_bac_len_to_bac_ratio_boundary, find_lower_bound
from Trackrefiner.strain.correction.action.Modeling.calculation.iouCalForML import iou_calc
from Trackrefiner.strain.correction.action.Modeling.calculation.calcDistanceForML import calc_distance
from Trackrefiner.strain.correction.action.Modeling.calculation.lengthRatio import check_len_ratio
from Trackrefiner.strain.correction.action.Modeling.calculation.calMotionAlignmentAngle import calc_MotionAlignmentAngle
from Trackrefiner.strain.correction.neighborChecking import neighbor_checking
from Trackrefiner.strain.correction.action.helperFunctions import calc_neighbors_dir_motion_all


def calc_movement(bac2, bac1, center_coordinate_columns):
    center_movement = \
        np.linalg.norm(bac2[[center_coordinate_columns['x'], center_coordinate_columns['y']]].values -
                       bac1[[center_coordinate_columns['x'], center_coordinate_columns['y']]].values)

    endpoint1_1_movement = \
        np.linalg.norm(bac2[['endpoint1_X', 'endpoint1_Y']].values -
                       bac1[['endpoint1_X', 'endpoint1_Y']].values)

    endpoint1_endpoint2_movement = \
        np.linalg.norm(bac2[['endpoint1_X', 'endpoint1_Y']].values -
                       bac1[['endpoint1_X', 'endpoint1_Y']].values)

    endpoint2_2_movement = \
        np.linalg.norm(bac2[['endpoint2_X', 'endpoint2_Y']].values -
                       bac1[['endpoint2_X', 'endpoint2_Y']].values)

    endpoint2_endpoint1_movement = \
        np.linalg.norm(bac2[['endpoint2_X', 'endpoint2_Y']].values -
                       bac1[['endpoint2_X', 'endpoint2_Y']].values)

    movement = min(center_movement, endpoint1_1_movement, endpoint2_2_movement, endpoint1_endpoint2_movement,
                   endpoint2_endpoint1_movement)

    return movement


def detect_redundant_parent_link(df, parent_image_number_col, parent_object_number_col, label_col,
                                 center_coordinate_columns):
    num_redundant_links = None

    while num_redundant_links != 0 and len(df["daughter_length_to_mother"].dropna().values) > 0:
        df_check_outliers = df.loc[df['mother_rpl'] == False]

        # note: it's only for division (mother - daughters relation)
        # check daughter length (sum daughters length or max daughter length) to mother length
        bacteria_with_redundant_parent_link_error = find_daughter_len_to_mother_ratio_outliers(df_check_outliers)
        num_redundant_links = bacteria_with_redundant_parent_link_error.shape[0]

        bacteria_with_redundant_parent_link_error_and_daughters = \
            bacteria_with_redundant_parent_link_error.merge(df, left_on=['ImageNumber', 'ObjectNumber'],
                                                            right_on=[parent_image_number_col,
                                                                      parent_object_number_col], how='inner',
                                                            suffixes=('_mother', '_daughter'))

        df.loc[
            bacteria_with_redundant_parent_link_error_and_daughters['index_mother'].unique(), 'mother_rpl'] = True
        df.loc[
            bacteria_with_redundant_parent_link_error_and_daughters['index_daughter'].unique(), 'daughter_rpl'] = True

    # note: by removing rpl, neighbor set doesn't change (different or common)
    # df = calc_new_features_after_rpl(df, neighbor_df, center_coordinate_columns, parent_image_number_col,
    #                                 parent_object_number_col, label_col)

    return df


def detect_and_remove_redundant_parent_link(df, neighbor_df, neighbor_list_array, parent_image_number_col,
                                            parent_object_number_col,
                                            label_col, center_coordinate_columns, non_divided_bac_model,
                                            coordinate_array):
    num_redundant_links = None

    while num_redundant_links != 0 and len(df["daughter_length_to_mother"].dropna().values) > 0:

        wrong_daughters_index_list = []
        mother_with_rpl_list = []
        unexpected_end_mothers = []

        incorrect_link_target_list = []

        # note: it's only for division (mother - daughters relation)
        # check daughter length (sum daughters length or max daughter length) to mother length
        bacteria_with_redundant_parent_link_error = find_daughter_len_to_mother_ratio_outliers(df)
        num_redundant_links = bacteria_with_redundant_parent_link_error.shape[0]

        if num_redundant_links != 0:

            bac_len_to_bac_ratio_boundary = find_bac_len_to_bac_ratio_boundary(df)

            lower_bound_threshold = find_lower_bound(bac_len_to_bac_ratio_boundary)

            # rpl mother with daughters
            source_bac_with_rpl = \
                bacteria_with_redundant_parent_link_error.merge(df, left_on=['ImageNumber', 'ObjectNumber'],
                                                                right_on=[parent_image_number_col,
                                                                          parent_object_number_col], how='inner',
                                                                suffixes=('_bac1', ''))

            col_target = ''
            col_source = '_bac1'

            # index correction
            source_bac_with_rpl['index_prev' + col_target] = \
                source_bac_with_rpl['index' + col_target]
            source_bac_with_rpl['index2' + col_target] = source_bac_with_rpl.index.values

            # check incorrect links
            incorrect_links = \
                source_bac_with_rpl.loc[source_bac_with_rpl['LengthChangeRatio' + col_target] < lower_bound_threshold]

            # also this
            incorrect_link_target_list.extend(incorrect_links['index_prev' + col_target].values.tolist())

            # calculate features & apply model

            source_bac_with_rpl = iou_calc(source_bac_with_rpl, col_source='prev_index' + col_source,
                                           col_target='prev_index' + col_target, stat='same',
                                           coordinate_array=coordinate_array)

            source_bac_with_rpl = calc_distance(source_bac_with_rpl, center_coordinate_columns,
                                                postfix_target=col_target, postfix_source=col_source, stat=None)

            source_bac_with_rpl['difference_neighbors' + col_target] = np.nan
            source_bac_with_rpl['other_daughter_index' + col_target] = np.nan
            source_bac_with_rpl['parent_id' + col_target] = source_bac_with_rpl['id' + col_source]
            source_bac_with_rpl['prev_time_step_index' + col_target] = \
                source_bac_with_rpl['prev_time_step_index' + col_target].astype('int32')

            source_bac_with_rpl = \
                neighbor_checking(source_bac_with_rpl, neighbor_list_array,
                                  parent_image_number_col, parent_object_number_col,
                                  selected_rows_df=source_bac_with_rpl, selected_time_step_df=df,
                                  return_common_elements=True, col_target=col_target)

            source_bac_with_rpl = check_len_ratio(df, source_bac_with_rpl, col_target=col_target,
                                                  col_source=col_source)

            # motion alignment
            # calculated for original df and we should calc for new df
            source_bac_with_rpl["MotionAlignmentAngle" + col_target] = np.nan
            source_bac_with_rpl = \
                calc_MotionAlignmentAngle(df, neighbor_df, center_coordinate_columns,
                                          selected_rows=source_bac_with_rpl, col_target=col_target,
                                          col_source=col_source)

            source_bac_with_rpl['adjusted_common_neighbors' + col_target] = np.where(
                source_bac_with_rpl['common_neighbors' + col_target] == 0,
                source_bac_with_rpl['common_neighbors' + col_target] + 1,
                source_bac_with_rpl['common_neighbors' + col_target]
            )

            source_bac_with_rpl['neighbor_ratio' + col_target] = \
                (source_bac_with_rpl['difference_neighbors' + col_target] / (
                    source_bac_with_rpl['adjusted_common_neighbors' + col_target]))

            raw_feature_list = ['iou', 'min_distance', 'difference_neighbors' + col_target,
                                'common_neighbors' + col_target,
                                'length_dynamic' + col_target,
                                'MotionAlignmentAngle' + col_target,
                                'neighbor_ratio' + col_target,
                                'index' + col_source, 'index_prev' + col_target]

            source_bac_with_rpl = source_bac_with_rpl[raw_feature_list].copy()
            source_bac_with_rpl = source_bac_with_rpl.rename(
                {
                    'common_neighbors' + col_target: 'common_neighbors',
                    'neighbor_ratio' + col_target: 'neighbor_ratio',
                    'difference_neighbors' + col_target: 'difference_neighbors',
                    'length_dynamic' + col_target: 'length_dynamic',
                    'MotionAlignmentAngle' + col_target: 'MotionAlignmentAngle',
                    'index_prev' + col_target: 'index_prev',
                }, axis=1)

            # difference_neighbors
            feature_list_for_non_divided_bac_model = \
                ['iou', 'min_distance', 'neighbor_ratio', 'length_dynamic', 'MotionAlignmentAngle']

            y_prob_non_divided_bac_model = \
                non_divided_bac_model.predict_proba(
                    source_bac_with_rpl[feature_list_for_non_divided_bac_model])[:, 1]

            source_bac_with_rpl['prob_non_divided_bac_model'] = y_prob_non_divided_bac_model

            # incorrect_links_based_on_neighbors = \
            #    source_bac_with_rpl.loc[(source_bac_with_rpl['difference_neighbors'] >
            #                                    source_bac_with_rpl['common_neighbors'])]

            # incorrect_link_target_list.extend(incorrect_links_based_on_neighbors['index_prev'].values.tolist())

            # Pivot this DataFrame to get the desired structure
            same_link_cost_df = \
                source_bac_with_rpl[['index' + col_source, 'index_prev', 'prob_non_divided_bac_model']].pivot(
                    index='index' + col_source, columns='index_prev', values='prob_non_divided_bac_model')

            same_link_cost_df.columns.name = None
            same_link_cost_df.index.name = None

            same_link_cost_df = 1 - same_link_cost_df

            for parent_with_redundant_link_idx, this_rpl_cost in same_link_cost_df.iterrows():

                daughters_idx_of_this_mother = this_rpl_cost[this_rpl_cost.notna()].index.tolist()
                parent_with_redundant_link = df.loc[parent_with_redundant_link_idx]

                incorrect_daughters_detected_in_prev_conditions = [v for v in daughters_idx_of_this_mother if v in
                                                                   incorrect_link_target_list]

                if len(incorrect_daughters_detected_in_prev_conditions) == 1:

                    # it means that one daughter is incorrect and the other is correct
                    correct_daughter = [v for v in daughters_idx_of_this_mother if v not in
                                        incorrect_daughters_detected_in_prev_conditions][0]

                    df.loc[df['id'] == df.at[incorrect_daughters_detected_in_prev_conditions[0], 'id'], 'parent_id'] = 0
                    wrong_daughters_index_list.append(incorrect_daughters_detected_in_prev_conditions[0])

                    correct_daughter_info = df.loc[correct_daughter]

                    # change feature of prev parent
                    df.loc[df['id'] == parent_with_redundant_link['id'], 'divideFlag'] = \
                        df.at[correct_daughter, 'divideFlag']

                    df.loc[df['id'] == parent_with_redundant_link['id'], 'division_time'] = \
                        df.at[correct_daughter, 'division_time']

                    df.at[correct_daughter, 'LengthChangeRatio'] = \
                        df.at[correct_daughter, 'daughter_mother_LengthChangeRatio']

                    bac_movement = \
                        calc_movement(correct_daughter_info, parent_with_redundant_link, center_coordinate_columns)

                    df.at[correct_daughter, 'bacteria_movement'] = bac_movement

                    df.at[correct_daughter, 'other_daughter_index'] = np.nan
                    df.at[correct_daughter, 'parent_index'] = np.nan
                    df.at[correct_daughter, 'daughter_mother_LengthChangeRatio'] = np.nan

                    df.at[correct_daughter, 'prev_time_step_MajorAxisLength'] = \
                        parent_with_redundant_link['AreaShape_MajorAxisLength']

                    df.at[correct_daughter, 'prev_time_step_center_x'] = \
                        parent_with_redundant_link[center_coordinate_columns['x']]
                    df.at[correct_daughter, 'prev_time_step_center_y'] = \
                        parent_with_redundant_link[center_coordinate_columns['y']]

                    df.at[correct_daughter, 'prev_time_step_endpoint1_X'] = parent_with_redundant_link['endpoint1_X']
                    df.at[correct_daughter, 'prev_time_step_endpoint1_Y'] = parent_with_redundant_link['endpoint1_Y']
                    df.at[correct_daughter, 'prev_time_step_endpoint2_X'] = parent_with_redundant_link['endpoint2_X']
                    df.at[correct_daughter, 'prev_time_step_endpoint2_Y'] = parent_with_redundant_link['endpoint2_Y']

                    df.loc[df['parent_id'] == correct_daughter_info['id'], 'parent_id'] = \
                        parent_with_redundant_link['id']
                    df.loc[df['id'] == correct_daughter_info['id'], 'parent_id'] = \
                        parent_with_redundant_link['parent_id']
                    df.loc[df['id'] == correct_daughter_info['id'], 'id'] = parent_with_redundant_link['id']

                    # now we should change features of prev parent bacterium
                    df.loc[parent_with_redundant_link_idx, ['daughters_index', 'daughter_length_to_mother',
                                                            'max_daughter_len_to_mother', 'avg_daughters_TrajectoryX',
                                                            'avg_daughters_TrajectoryY']] = \
                        [np.nan, np.nan, np.nan, np.nan, np.nan, ]

                elif len(incorrect_daughters_detected_in_prev_conditions) == 0:

                    # it means both daughters pass the prev conditions
                    # cost is equal to 1 - probability
                    incorrect_daughter = this_rpl_cost.idxmax()
                    # incorrect_daughter_prob = this_rpl_cost.max()

                    correct_daughter = this_rpl_cost.idxmin()
                    correct_daughter_prob = this_rpl_cost.min()

                    if incorrect_daughter == correct_daughter:
                        incorrect_daughter = [v for v in this_rpl_cost[this_rpl_cost.notna()].index.values if
                                              v != correct_daughter][0]

                    if correct_daughter_prob <= 0.5:

                        # it means one daughter is incorrect and one daughter is correct
                        df.loc[df['id'] == df.at[incorrect_daughter, 'id'], 'parent_id'] = 0
                        wrong_daughters_index_list.append(incorrect_daughter)

                        # now we should calculate features value of correct bac
                        correct_daughter_info = df.loc[correct_daughter]

                        # change feature of prev parent
                        df.loc[df['id'] == parent_with_redundant_link['id'], 'divideFlag'] = \
                            df.at[correct_daughter, 'divideFlag']

                        df.loc[df['id'] == parent_with_redundant_link['id'], 'division_time'] = \
                            df.at[correct_daughter, 'division_time']

                        df.at[correct_daughter, 'LengthChangeRatio'] = \
                            df.at[correct_daughter, 'daughter_mother_LengthChangeRatio']

                        bac_movement = \
                            calc_movement(correct_daughter_info, parent_with_redundant_link, center_coordinate_columns)
                        df.at[correct_daughter, 'bacteria_movement'] = bac_movement

                        df.at[correct_daughter, 'other_daughter_index'] = np.nan
                        df.at[correct_daughter, 'parent_index'] = np.nan
                        df.at[correct_daughter, 'daughter_mother_LengthChangeRatio'] = np.nan

                        df.at[correct_daughter, 'prev_time_step_MajorAxisLength'] = \
                            parent_with_redundant_link['AreaShape_MajorAxisLength']

                        df.at[correct_daughter, 'prev_time_step_center_x'] = \
                            parent_with_redundant_link[center_coordinate_columns['x']]
                        df.at[correct_daughter, 'prev_time_step_center_y'] = \
                            parent_with_redundant_link[center_coordinate_columns['y']]

                        df.at[correct_daughter, 'prev_time_step_endpoint1_X'] = parent_with_redundant_link[
                            'endpoint1_X']
                        df.at[correct_daughter, 'prev_time_step_endpoint1_Y'] = parent_with_redundant_link[
                            'endpoint1_Y']
                        df.at[correct_daughter, 'prev_time_step_endpoint2_X'] = parent_with_redundant_link[
                            'endpoint2_X']
                        df.at[correct_daughter, 'prev_time_step_endpoint2_Y'] = parent_with_redundant_link[
                            'endpoint2_Y']

                        df.loc[df['parent_id'] == correct_daughter_info['id'], 'parent_id'] = \
                            parent_with_redundant_link['id']
                        df.loc[df['id'] == correct_daughter_info['id'], 'parent_id'] = parent_with_redundant_link[
                            'parent_id']
                        df.loc[df['id'] == correct_daughter_info['id'], 'id'] = parent_with_redundant_link['id']

                        # now we should change features of prev parent bacterium
                        df.loc[parent_with_redundant_link_idx, ['daughters_index', 'daughter_length_to_mother',
                                                                'max_daughter_len_to_mother',
                                                                'avg_daughters_TrajectoryX',
                                                                'avg_daughters_TrajectoryY']] = \
                            [np.nan, np.nan, np.nan, np.nan, np.nan, ]

                    else:
                        # it means the probability of both daughters are low
                        df.loc[df['id'] == df.at[incorrect_daughter, 'id'], 'parent_id'] = 0
                        df.loc[df['id'] == df.at[correct_daughter, 'id'], 'parent_id'] = 0

                        unexpected_end_mothers.append(parent_with_redundant_link_idx)
                        wrong_daughters_index_list.extend([incorrect_daughter, correct_daughter])

                elif len(incorrect_daughters_detected_in_prev_conditions) == 2:

                    daughter_1_idx = incorrect_daughters_detected_in_prev_conditions[0]
                    daughter_2_idx = incorrect_daughters_detected_in_prev_conditions[1]

                    # it's challenging and means both daughters are incorrect based on conditions
                    df.loc[df['id'] == df.at[daughter_1_idx, 'id'], 'parent_id'] = 0
                    df.loc[df['id'] == df.at[daughter_2_idx, 'id'], 'parent_id'] = 0

                    unexpected_end_mothers.append(parent_with_redundant_link_idx)
                    wrong_daughters_index_list.extend([daughter_1_idx, daughter_2_idx])

            if len(wrong_daughters_index_list) > 0:
                df.loc[wrong_daughters_index_list, [parent_image_number_col, parent_object_number_col,
                                                    'unexpected_beginning', 'direction_of_motion', "TrajectoryX",
                                                    "TrajectoryY", 'prev_time_step_index',
                                                    'other_daughter_index', 'prev_bacteria_slope', 'slope_bac_bac',
                                                    'parent_index', 'daughter_mother_LengthChangeRatio']] = \
                    [0, 0, True, np.nan, np.nan, np.nan, -1, np.nan, np.nan, np.nan, np.nan, np.nan]

                df.loc[mother_with_rpl_list, ["daughter_length_to_mother", "max_daughter_len_to_mother",
                                              'avg_daughters_TrajectoryX', 'avg_daughters_TrajectoryY']] = \
                    [np.nan, np.nan, np.nan, np.nan]

            if len(unexpected_end_mothers) > 0:
                df.loc[unexpected_end_mothers, ['daughters_index', 'daughter_length_to_mother',
                                                'max_daughter_len_to_mother', 'avg_daughters_TrajectoryX',
                                                'avg_daughters_TrajectoryY', 'unexpected_end']] = \
                    [np.nan, np.nan, np.nan, np.nan, np.nan, True]

    # note: by removing rpl, neighbor set doesn't change (different or common)
    # set age
    df['age'] = df.groupby('id').cumcount() + 1
    df['LifeHistory'] = df.groupby('id')['id'].transform('size')
    # average length of a bacteria in its life history
    df['AverageLengthChangeRatio'] = df.groupby('id')['LengthChangeRatio'].transform('mean')

    # check division
    # _2: bac2, _1: bac1 (source bac)
    merged_df = df.merge(df, left_on=[parent_image_number_col, parent_object_number_col],
                         right_on=['ImageNumber', 'ObjectNumber'], how='inner', suffixes=('_2', '_1'))

    division_df = merged_df[merged_df.duplicated(subset='index_1', keep=False)][['ImageNumber_1', 'ObjectNumber_1',
                                                                                 'index_1', 'id_1', 'index_2',
                                                                                 'ImageNumber_2', 'ObjectNumber_2',
                                                                                 'AreaShape_MajorAxisLength_1',
                                                                                 'AreaShape_MajorAxisLength_2',
                                                                                 'bacteria_slope_1', 'bacteria_slope_2',
                                                                                 center_coordinate_columns['x'] + '_1',
                                                                                 center_coordinate_columns['y'] + '_1',
                                                                                 center_coordinate_columns['x'] + '_2',
                                                                                 center_coordinate_columns['y'] + '_2',
                                                                                 parent_object_number_col + '_2',
                                                                                 parent_image_number_col + '_2']].copy()
    # motion alignment
    df = calc_neighbors_dir_motion_all(df, neighbor_df, division_df, parent_image_number_col,
                                       parent_object_number_col)

    # check neighbors
    df = neighbor_checking(df, neighbor_list_array, parent_image_number_col, parent_object_number_col)

    df['ovd_flag'] = False
    df['bad_daughters_flag'] = False
    df['bad_division_flag'] = False
    df['daughter_rpl'] = False
    df['mother_rpl'] = False
    # df = calc_new_features_after_rpl(df, neighbor_df, center_coordinate_columns, parent_image_number_col,
    #                                 parent_object_number_col, label_col)

    return df
