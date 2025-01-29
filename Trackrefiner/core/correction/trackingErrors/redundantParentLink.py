import numpy as np
from Trackrefiner.core.correction.action.findOutlier import detect_daughter_to_mother_length_outliers, \
    calculate_length_change_ratio_boundary, calculate_lower_statistical_bound
from Trackrefiner.core.correction.modelTraning.calculation.iouCalForML import calculate_iou_ml
from Trackrefiner.core.correction.modelTraning.calculation.calcDistanceForML import calc_min_distance_ml
from Trackrefiner.core.correction.modelTraning.calculation.lengthRatio import check_len_ratio
from Trackrefiner.core.correction.modelTraning.calculation.calMotionAlignmentAngle import calc_motion_alignment_angle_ml
from Trackrefiner.core.correction.action.neighborAnalysis import compare_neighbor_sets
from Trackrefiner.core.correction.action.helper import calc_neighbors_dir_motion_all, calc_movement


def detect_and_resolve_redundant_parent_link(df, neighbor_df, neighbor_list_array, parent_image_number_col,
                                             parent_object_number_col, center_coord_cols, continuity_links_model,
                                             coordinate_array):

    """
    Detects and resolves redundant parent links (RPL) in bacterial lineage tracking by evaluating mother-daughter
    relationships based on geometric, motion, and neighbor-based features.
    This function performs iterative checks to detect RPLs.

    :param pandas.DataFrame df:
        DataFrame containing tracking data for bacteria.
    :param pandas.DataFrame neighbor_df:
        DataFrame containing neighbor relationships between bacteria.
    :param lil_matrix neighbor_list_array:
        Matrix representing neighbor connections between bacteria.
    :param str parent_image_number_col:
        Column name in `df` for the parent bacterium's image number.
    :param str parent_object_number_col:
        Column name in `df` for the parent bacterium's object number.
    :param dict center_coord_cols:
        Dictionary specifying column names for bacterial centroid coordinates
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param sklearn.Model continuity_links_model:
        Machine learning model used to evaluate which target bacterium the source bacterium should link to in order to
         continue its life history.
    :param csr_matrix coordinate_array:
        Matrix of spatial coordinates used for geometric and spatial calculations.

    **Returns**:
        pandas.DataFrame:

        Updated DataFrame with resolved RPLs and recalculated features.
    """

    num_redundant_links = None

    while num_redundant_links != 0 and len(df["Total_Daughter_Mother_Length_Ratio"].dropna().values) > 0:

        wrong_daughters_index_list = []
        mother_with_rpl_list = []
        unexpected_end_mothers = []

        incorrect_link_target_list = []

        # note: it's only for division (mother - daughters relation)
        # check daughter length (sum daughters length or max daughter length) to mother length
        bacteria_with_redundant_parent_link_error = detect_daughter_to_mother_length_outliers(df)
        num_redundant_links = bacteria_with_redundant_parent_link_error.shape[0]

        if num_redundant_links != 0:

            bac_len_to_bac_ratio_boundary = calculate_length_change_ratio_boundary(df)

            lower_bound_threshold = calculate_lower_statistical_bound(bac_len_to_bac_ratio_boundary)

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
                source_bac_with_rpl.loc[source_bac_with_rpl['Length_Change_Ratio' + col_target] < lower_bound_threshold]

            # also this
            incorrect_link_target_list.extend(incorrect_links['index_prev' + col_target].values.tolist())

            # calculate features & apply model

            source_bac_with_rpl = calculate_iou_ml(source_bac_with_rpl, col_source='prev_index' + col_source,
                                                   col_target='prev_index' + col_target, link_type='continuity',
                                                   coordinate_array=coordinate_array)

            source_bac_with_rpl = \
                calc_min_distance_ml(source_bac_with_rpl, center_coord_cols, postfix_target=col_target,
                                     postfix_source=col_source, link_type=None)

            source_bac_with_rpl['Neighbor_Difference_Count' + col_target] = np.nan
            source_bac_with_rpl['other_daughter_index' + col_target] = np.nan
            source_bac_with_rpl['parent_id' + col_target] = source_bac_with_rpl['id' + col_source]
            source_bac_with_rpl['prev_time_step_index' + col_target] = \
                source_bac_with_rpl['prev_time_step_index' + col_target].astype('int32')

            source_bac_with_rpl = \
                compare_neighbor_sets(source_bac_with_rpl, neighbor_list_array,
                                      parent_image_number_col, parent_object_number_col,
                                      selected_rows_df=source_bac_with_rpl, selected_time_step_df=df,
                                      return_common_elements=True, col_target=col_target)

            source_bac_with_rpl = check_len_ratio(df, source_bac_with_rpl, col_target=col_target,
                                                  col_source=col_source)

            # motion alignment
            # calculated for original df and we should calc for new df
            source_bac_with_rpl["Motion_Alignment_Angle" + col_target] = np.nan
            source_bac_with_rpl = \
                calc_motion_alignment_angle_ml(df, neighbor_df, center_coord_cols,
                                               selected_rows=source_bac_with_rpl, col_target=col_target,
                                               col_source=col_source)

            source_bac_with_rpl['adjusted_Neighbor_Shared_Count' + col_target] = np.where(
                source_bac_with_rpl['Neighbor_Shared_Count' + col_target] == 0,
                source_bac_with_rpl['Neighbor_Shared_Count' + col_target] + 1,
                source_bac_with_rpl['Neighbor_Shared_Count' + col_target]
            )

            source_bac_with_rpl['neighbor_ratio' + col_target] = \
                (source_bac_with_rpl['Neighbor_Difference_Count' + col_target] / (
                    source_bac_with_rpl['adjusted_Neighbor_Shared_Count' + col_target]))

            raw_feature_list = ['iou', 'min_distance', 'Neighbor_Difference_Count' + col_target,
                                'Neighbor_Shared_Count' + col_target,
                                'length_dynamic' + col_target,
                                'Motion_Alignment_Angle' + col_target,
                                'neighbor_ratio' + col_target,
                                'index' + col_source, 'index_prev' + col_target]

            source_bac_with_rpl = source_bac_with_rpl[raw_feature_list].copy()
            source_bac_with_rpl = source_bac_with_rpl.rename(
                {
                    'Neighbor_Shared_Count' + col_target: 'Neighbor_Shared_Count',
                    'neighbor_ratio' + col_target: 'neighbor_ratio',
                    'Neighbor_Difference_Count' + col_target: 'Neighbor_Difference_Count',
                    'length_dynamic' + col_target: 'length_dynamic',
                    'Motion_Alignment_Angle' + col_target: 'Motion_Alignment_Angle',
                    'index_prev' + col_target: 'index_prev',
                }, axis=1)

            # difference_neighbors
            feature_list_for_non_divided_bac_model = \
                ['iou', 'min_distance', 'neighbor_ratio', 'length_dynamic', 'Motion_Alignment_Angle']

            y_prob_non_divided_bac_model = \
                continuity_links_model.predict_proba(
                    source_bac_with_rpl[feature_list_for_non_divided_bac_model])[:, 1]

            source_bac_with_rpl['prob_non_divided_bac_model'] = y_prob_non_divided_bac_model

            # Pivot this DataFrame to get the desired structure
            continuity_link_cost_df = \
                source_bac_with_rpl[['index' + col_source, 'index_prev', 'prob_non_divided_bac_model']].pivot(
                    index='index' + col_source, columns='index_prev', values='prob_non_divided_bac_model')

            continuity_link_cost_df.columns.name = None
            continuity_link_cost_df.index.name = None

            continuity_link_cost_df = 1 - continuity_link_cost_df

            for parent_with_redundant_link_idx, this_rpl_cost in continuity_link_cost_df.iterrows():

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

                    df.loc[df['id'] == parent_with_redundant_link['id'], 'Division_TimeStep'] = \
                        df.at[correct_daughter, 'Division_TimeStep']

                    df.at[correct_daughter, 'Length_Change_Ratio'] = \
                        df.at[correct_daughter, 'Daughter_Mother_Length_Ratio']

                    bac_movement = calc_movement(correct_daughter_info, parent_with_redundant_link, center_coord_cols)

                    df.at[correct_daughter, 'Bacterium_Movement'] = bac_movement

                    df.at[correct_daughter, 'other_daughter_index'] = np.nan
                    df.at[correct_daughter, 'parent_index'] = np.nan
                    df.at[correct_daughter, 'Daughter_Mother_Length_Ratio'] = np.nan

                    df.at[correct_daughter, 'Prev_MajorAxisLength'] = \
                        parent_with_redundant_link['AreaShape_MajorAxisLength']

                    df.at[correct_daughter, 'Prev_Center_X'] = \
                        parent_with_redundant_link[center_coord_cols['x']]
                    df.at[correct_daughter, 'Prev_Center_Y'] = \
                        parent_with_redundant_link[center_coord_cols['y']]

                    df.at[correct_daughter, 'Prev_Endpoint1_X'] = parent_with_redundant_link['Endpoint1_X']
                    df.at[correct_daughter, 'Prev_Endpoint1_Y'] = parent_with_redundant_link['Endpoint1_Y']
                    df.at[correct_daughter, 'Prev_Endpoint2_X'] = parent_with_redundant_link['Endpoint2_X']
                    df.at[correct_daughter, 'Prev_Endpoint2_Y'] = parent_with_redundant_link['Endpoint2_Y']

                    df.loc[df['parent_id'] == correct_daughter_info['id'], 'parent_id'] = \
                        parent_with_redundant_link['id']
                    df.loc[df['id'] == correct_daughter_info['id'], 'parent_id'] = \
                        parent_with_redundant_link['parent_id']
                    df.loc[df['id'] == correct_daughter_info['id'], 'id'] = parent_with_redundant_link['id']

                    # now we should change features of prev parent bacterium
                    df.loc[parent_with_redundant_link_idx, ['daughters_index', 'Total_Daughter_Mother_Length_Ratio',
                                                            'Max_Daughter_Mother_Length_Ratio', 'Daughter_Avg_TrajectoryX',
                                                            'Daughter_Avg_TrajectoryY']] = \
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

                        df.loc[df['id'] == parent_with_redundant_link['id'], 'Division_TimeStep'] = \
                            df.at[correct_daughter, 'Division_TimeStep']

                        df.at[correct_daughter, 'Length_Change_Ratio'] = \
                            df.at[correct_daughter, 'Daughter_Mother_Length_Ratio']

                        bac_movement = \
                            calc_movement(correct_daughter_info, parent_with_redundant_link, center_coord_cols)
                        df.at[correct_daughter, 'Bacterium_Movement'] = bac_movement

                        df.at[correct_daughter, 'other_daughter_index'] = np.nan
                        df.at[correct_daughter, 'parent_index'] = np.nan
                        df.at[correct_daughter, 'Daughter_Mother_Length_Ratio'] = np.nan

                        df.at[correct_daughter, 'Prev_MajorAxisLength'] = \
                            parent_with_redundant_link['AreaShape_MajorAxisLength']

                        df.at[correct_daughter, 'Prev_Center_X'] = \
                            parent_with_redundant_link[center_coord_cols['x']]
                        df.at[correct_daughter, 'Prev_Center_Y'] = \
                            parent_with_redundant_link[center_coord_cols['y']]

                        df.at[correct_daughter, 'Prev_Endpoint1_X'] = parent_with_redundant_link[
                            'Endpoint1_X']
                        df.at[correct_daughter, 'Prev_Endpoint1_Y'] = parent_with_redundant_link[
                            'Endpoint1_Y']
                        df.at[correct_daughter, 'Prev_Endpoint2_X'] = parent_with_redundant_link[
                            'Endpoint2_X']
                        df.at[correct_daughter, 'Prev_Endpoint2_Y'] = parent_with_redundant_link[
                            'Endpoint2_Y']

                        df.loc[df['parent_id'] == correct_daughter_info['id'], 'parent_id'] = \
                            parent_with_redundant_link['id']
                        df.loc[df['id'] == correct_daughter_info['id'], 'parent_id'] = parent_with_redundant_link[
                            'parent_id']
                        df.loc[df['id'] == correct_daughter_info['id'], 'id'] = parent_with_redundant_link['id']

                        # now we should change features of prev parent bacterium
                        df.loc[parent_with_redundant_link_idx, ['daughters_index', 'Total_Daughter_Mother_Length_Ratio',
                                                                'Max_Daughter_Mother_Length_Ratio',
                                                                'Daughter_Avg_TrajectoryX',
                                                                'Daughter_Avg_TrajectoryY']] = \
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
                                                    'Unexpected_Beginning', 'Direction_of_Motion', "TrajectoryX",
                                                    "TrajectoryY", 'prev_time_step_index',
                                                    'other_daughter_index', 'Prev_Bacterium_Slope', 'Orientation_Angle_Between_Slopes',
                                                    'parent_index', 'Daughter_Mother_Length_Ratio']] = \
                    [0, 0, True, np.nan, np.nan, np.nan, -1, np.nan, np.nan, np.nan, np.nan, np.nan]

                df.loc[mother_with_rpl_list, ["Total_Daughter_Mother_Length_Ratio", "Max_Daughter_Mother_Length_Ratio",
                                              'Daughter_Avg_TrajectoryX', 'Daughter_Avg_TrajectoryY']] = \
                    [np.nan, np.nan, np.nan, np.nan]

            if len(unexpected_end_mothers) > 0:
                df.loc[unexpected_end_mothers, ['daughters_index', 'Total_Daughter_Mother_Length_Ratio',
                                                'Max_Daughter_Mother_Length_Ratio', 'Daughter_Avg_TrajectoryX',
                                                'Daughter_Avg_TrajectoryY', 'Unexpected_End']] = \
                    [np.nan, np.nan, np.nan, np.nan, np.nan, True]

    # note: by removing rpl, neighbor set doesn't change (different or common)
    # set age
    df['age'] = df.groupby('id').cumcount() + 1
    df['LifeHistory'] = df.groupby('id')['id'].transform('size')
    # average length of a bacteria in its life history
    df['Avg_Length_Change_Ratio'] = df.groupby('id')['Length_Change_Ratio'].transform('mean')

    # check division
    # _2: bac2, _1: bac1 (source bac)
    merged_df = df.merge(df, left_on=[parent_image_number_col, parent_object_number_col],
                         right_on=['ImageNumber', 'ObjectNumber'], how='inner', suffixes=('_2', '_1'))

    division_df = merged_df[merged_df.duplicated(subset='index_1', keep=False)][['ImageNumber_1', 'ObjectNumber_1',
                                                                                 'index_1', 'id_1', 'index_2',
                                                                                 'ImageNumber_2', 'ObjectNumber_2',
                                                                                 'AreaShape_MajorAxisLength_1',
                                                                                 'AreaShape_MajorAxisLength_2',
                                                                                 'Bacterium_Slope_1', 'Bacterium_Slope_2',
                                                                                 center_coord_cols['x'] + '_1',
                                                                                 center_coord_cols['y'] + '_1',
                                                                                 center_coord_cols['x'] + '_2',
                                                                                 center_coord_cols['y'] + '_2',
                                                                                 parent_object_number_col + '_2',
                                                                                 parent_image_number_col + '_2']].copy()
    # motion alignment
    df = calc_neighbors_dir_motion_all(df, neighbor_df, division_df)

    # check neighbors
    df = compare_neighbor_sets(df, neighbor_list_array, parent_image_number_col, parent_object_number_col)

    df['ovd_flag'] = False
    df['bad_daughters_flag'] = False
    df['bad_division_flag'] = False
    df['daughter_rpl'] = False
    df['mother_rpl'] = False

    return df
