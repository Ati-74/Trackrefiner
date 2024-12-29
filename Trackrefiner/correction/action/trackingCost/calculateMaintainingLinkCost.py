import pandas as pd
import numpy as np
from Trackrefiner.correction.modelTraning.calculation.iouCalForML import calculate_iou_ml
from Trackrefiner.correction.modelTraning.calculation.calcDistanceForML import calc_min_distance_ml


def calc_maintain_exist_link_cost(sel_source_bacteria_info, sel_target_bacteria_info, center_coord_cols,
                                  divided_vs_non_divided_model, coordinate_array):

    """
    Calculates the cost of maintaining existing links between bacteria in consecutive time steps.
    This function calculates features such as IOU, distance, neighbor relationships,
    and motion alignment for each source-target pair. These features are used as inputs to a predictive
    model to calculate the cost of maintaining each link.

    :param pandas.DataFrame sel_source_bacteria_info:
        Subset of bacteria in the source time step selected for analysis.
    :param pandas.DataFrame sel_target_bacteria_info:
        Subset of bacteria in the target time step selected for analysis.
    :param dict center_coord_cols:
        Dictionary specifying column names for bacterial centroid coordinates
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param sklearn.Model divided_vs_non_divided_model:
        Machine learning model used to evaluate the probability of division vs. continuity links.
    :param csr_matrix coordinate_array:
        Array of spatial coordinates used for evaluating spatial relationships between bacteria.

    :returns:
        pandas.DataFrame:

        A cost matrix for maintaining existing links between bacteria.
        Rows correspond to source bacteria, columns correspond to target bacteria, and
        values represent the cost of maintaining each link.
    """

    if sel_target_bacteria_info.shape[0] > 0 and sel_source_bacteria_info.shape[0] > 0:

        # difference_neighbors,
        feature_list = ['iou', 'min_distance', 'neighbor_ratio', 'direction_of_motion',
                        'MotionAlignmentAngle', 'LengthChangeRatio']

        important_cols_division = ['id', 'parent_id', 'prev_index',
                                   center_coord_cols['x'], center_coord_cols['y'],
                                   'endpoint1_X', 'endpoint1_Y', 'endpoint2_X', 'endpoint2_Y',
                                   'AreaShape_MajorAxisLength', 'common_neighbors',
                                   'difference_neighbors', 'direction_of_motion', 'MotionAlignmentAngle', 'index']

        important_cols_continuity = ['id', 'parent_id', 'prev_index',
                                     center_coord_cols['x'], center_coord_cols['y'],
                                     'endpoint1_X', 'endpoint1_Y', 'endpoint2_X', 'endpoint2_Y',
                                     'AreaShape_MajorAxisLength', 'common_neighbors',
                                     'difference_neighbors', 'direction_of_motion', 'MotionAlignmentAngle', 'index',
                                     'LengthChangeRatio']

        division_merged_df = \
            sel_source_bacteria_info[important_cols_division].merge(
                sel_target_bacteria_info[important_cols_division],
                left_on='id', right_on='parent_id', how='inner', suffixes=('_parent', '_daughter'))

        continuity_bac_merged_df = \
            sel_source_bacteria_info[important_cols_continuity].merge(
                sel_target_bacteria_info[important_cols_continuity], left_on='id', right_on='id', how='inner',
                suffixes=('_bac1', '_bac2'))

        # IOU
        division_merged_df = calculate_iou_ml(division_merged_df, col_source='prev_index_parent',
                                              col_target='prev_index_daughter',  link_type='div',
                                              coordinate_array=coordinate_array, both=False)

        continuity_bac_merged_df = calculate_iou_ml(continuity_bac_merged_df, col_source='prev_index_bac1',
                                                    col_target='prev_index_bac2', link_type='continuity',
                                                    coordinate_array=coordinate_array)

        # distance
        division_merged_df = calc_min_distance_ml(division_merged_df, center_coord_cols, '_daughter',
                                                  '_parent', link_type='div')

        continuity_bac_merged_df = calc_min_distance_ml(continuity_bac_merged_df, center_coord_cols,
                                                        '_bac2', '_bac1')

        division_merged_df['LengthChangeRatio'] = (division_merged_df['AreaShape_MajorAxisLength_daughter'] /
                                                   division_merged_df['AreaShape_MajorAxisLength_parent'])

        division_merged_df['adjusted_common_neighbors_daughter'] = np.where(
            division_merged_df['common_neighbors_daughter'] == 0,
            division_merged_df['common_neighbors_daughter'] + 1,
            division_merged_df['common_neighbors_daughter']
        )

        division_merged_df['neighbor_ratio_daughter'] = \
            (division_merged_df['difference_neighbors_daughter'] / (
                division_merged_df['adjusted_common_neighbors_daughter']))

        continuity_bac_merged_df['adjusted_common_neighbors_bac2'] = np.where(
            continuity_bac_merged_df['common_neighbors_bac2'] == 0,
            continuity_bac_merged_df['common_neighbors_bac2'] + 1,
            continuity_bac_merged_df['common_neighbors_bac2']
        )

        continuity_bac_merged_df['neighbor_ratio_bac2'] = \
            (continuity_bac_merged_df['difference_neighbors_bac2'] / (
                continuity_bac_merged_df['adjusted_common_neighbors_bac2']))

        # rename columns
        division_merged_df = division_merged_df.rename(
            {'difference_neighbors_daughter': 'difference_neighbors',
             'neighbor_ratio_daughter': 'neighbor_ratio',
             'direction_of_motion_daughter': 'direction_of_motion',
             'MotionAlignmentAngle_daughter': 'MotionAlignmentAngle'}, axis=1)

        continuity_bac_merged_df = continuity_bac_merged_df.rename(
            {'difference_neighbors_bac2': 'difference_neighbors',
             'neighbor_ratio_bac2': 'neighbor_ratio',
             'direction_of_motion_bac2': 'direction_of_motion',
             'MotionAlignmentAngle_bac2': 'MotionAlignmentAngle',
             'LengthChangeRatio_bac2': 'LengthChangeRatio'}, axis=1)

        if division_merged_df.shape[0] > 0:
            y_prob_compare_division = \
                divided_vs_non_divided_model.predict_proba(division_merged_df[feature_list])[:, 0]

            division_merged_df['prob_compare'] = y_prob_compare_division

            # Pivot this DataFrame to get the desired structure
            cost_df_division = \
                division_merged_df[['index_parent', 'index_daughter', 'prob_compare']].pivot(
                    index='index_parent', columns='index_daughter', values='prob_compare')
            cost_df_division.columns.name = None
            cost_df_division.index.name = None

        if continuity_bac_merged_df.shape[0] > 0:
            y_prob_compare_continuity_bac = \
                divided_vs_non_divided_model.predict_proba(continuity_bac_merged_df[feature_list])[:, 1]

            continuity_bac_merged_df['prob_compare'] = y_prob_compare_continuity_bac

            cost_df_continuity = \
                continuity_bac_merged_df[['index_bac1', 'index_bac2', 'prob_compare']].pivot(
                    index='index_bac1', columns='index_bac2', values='prob_compare')
            cost_df_continuity.columns.name = None
            cost_df_continuity.index.name = None

        if continuity_bac_merged_df.shape[0] > 0 and division_merged_df.shape[0] > 0:
            final_maintenance_cost = pd.concat([cost_df_division, cost_df_continuity], axis=1, sort=False)
        elif continuity_bac_merged_df.shape[0] > 0:
            final_maintenance_cost = cost_df_continuity
        elif division_merged_df.shape[0] > 0:
            final_maintenance_cost = cost_df_division

    else:
        final_maintenance_cost = pd.DataFrame()

    return final_maintenance_cost
