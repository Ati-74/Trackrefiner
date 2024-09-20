from Trackrefiner.strain.correction.action.Modeling.calculation.iouCalForML import iou_calc
from Trackrefiner.strain.correction.action.Modeling.calculation.calcDistanceForML import calc_distance
from Trackrefiner.strain.correction.action.Modeling.runML import run_ml_model
import pandas as pd
import numpy as np


def comparing_divided_non_divided_bacteria(raw_df, df, connected_bac, neighbors_df, center_coordinate_columns,
                                           parent_image_number_col, parent_object_number_col, output_directory,
                                           clf, n_cpu):

    non_divided_bac = \
        connected_bac.drop_duplicates(subset=[parent_image_number_col, parent_object_number_col],
                                      keep=False).copy()

    divided_bac = \
        connected_bac[
            connected_bac.duplicated(subset=[parent_image_number_col, parent_object_number_col],
                                     keep=False)].copy()

    divided_bac['LengthChangeRatio'] = divided_bac['daughter_mother_LengthChangeRatio']

    # now we should calculate features
    # IOU
    non_divided_bac = iou_calc(raw_df, non_divided_bac, col_source='prev_index_prev', col_target='prev_index',
                               stat='same')
    # stat='div'
    divided_bac = iou_calc(raw_df, divided_bac, col_source='prev_index_prev', col_target='prev_index', stat='div')

    # distance
    non_divided_bac = calc_distance(non_divided_bac, center_coordinate_columns, postfix_target='',
                                    postfix_source='_prev')
    # stat='div'
    divided_bac = calc_distance(divided_bac, center_coordinate_columns, postfix_target='', postfix_source='_prev',
                                stat='div')

    # neighbor_ratio
    non_divided_bac['adjusted_common_neighbors'] = np.where(
        non_divided_bac['common_neighbors'] == 0,
        non_divided_bac['common_neighbors'] + 1,
        non_divided_bac['common_neighbors']
    )

    non_divided_bac['neighbor_ratio'] = \
        (non_divided_bac['difference_neighbors'] / (non_divided_bac['adjusted_common_neighbors']))

    divided_bac['adjusted_common_neighbors'] = np.where(
        divided_bac['common_neighbors'] == 0,
        divided_bac['common_neighbors'] + 1,
        divided_bac['common_neighbors']
    )

    divided_bac['neighbor_ratio'] = (divided_bac['difference_neighbors'] / (divided_bac['adjusted_common_neighbors']))

    # (Length Ratio(source-target)- Length Dynamics source )
    # non_divided_bac = check_len_ratio(df, non_divided_bac, '', '_prev')

    # now we should define label
    non_divided_bac['label'] = 'positive'
    divided_bac['label'] = 'negative'

    # difference_neighbors
    feature_list = ['iou', 'min_distance', 'neighbor_ratio', 'direction_of_motion',
                    'MotionAlignmentAngle', 'LengthChangeRatio', 'label']

    # merge dfs
    merged_df = pd.concat([non_divided_bac[feature_list], divided_bac[feature_list]], ignore_index=True)

    comparing_divided_non_divided_model = \
        run_ml_model(merged_df, feature_list=feature_list[:-1], columns_to_scale=feature_list[1:-1], stat='compare',
                     output_directory=output_directory, clf=clf, n_cpu=n_cpu)

    return comparing_divided_non_divided_model
