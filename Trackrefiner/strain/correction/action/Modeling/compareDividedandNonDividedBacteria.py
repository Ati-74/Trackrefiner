from Trackrefiner.strain.correction.action.Modeling.calculation.iouCalForML import iou_calc
from Trackrefiner.strain.correction.action.Modeling.calculation.calcDistanceForML import calc_distance
from Trackrefiner.strain.correction.action.Modeling.runML import run_ml_model
import pandas as pd
import numpy as np


def comparing_divided_non_divided_bacteria(connected_bac, center_coordinate_columns,
                                           parent_image_number_col, parent_object_number_col, output_directory,
                                           clf, n_cpu, coordinate_array):
    # non_divided_bac = \
    #    connected_bac.drop_duplicates(subset=[parent_image_number_col, parent_object_number_col],
    #                                  keep=False).copy()

    is_duplicated = connected_bac.duplicated(subset=[parent_image_number_col, parent_object_number_col], keep=False)
    rep_one_values = ~ is_duplicated
    # num_rep = connected_bac.groupby([parent_image_number_col, parent_object_number_col]).transform('size')

    # non_divided_bac_view = connected_bac[num_rep == 1]
    non_divided_bac_view = \
        connected_bac[['LengthChangeRatio', 'prev_index_prev', 'prev_index',
                       'common_neighbors', 'difference_neighbors',
                       'direction_of_motion', 'MotionAlignmentAngle',
                       center_coordinate_columns['x'],
                       center_coordinate_columns['y'],
                       center_coordinate_columns['x'] + '_prev',
                       center_coordinate_columns['y'] + '_prev',
                       'endpoint1_X', 'endpoint1_Y',
                       'endpoint2_X', 'endpoint2_Y',
                       'endpoint1_X_prev', 'endpoint1_Y_prev',
                       'endpoint2_X_prev', 'endpoint2_Y_prev']][rep_one_values]

    non_divided_bac = pd.DataFrame(index=non_divided_bac_view.index)

    # divided_bac_view = connected_bac[num_rep > 1]
    divided_bac_view = connected_bac[['daughter_mother_LengthChangeRatio', 'prev_index_prev', 'prev_index',
                                      'common_neighbors', 'difference_neighbors',
                                      'direction_of_motion', 'MotionAlignmentAngle',
                                      center_coordinate_columns['x'],
                                      center_coordinate_columns['y'],
                                      center_coordinate_columns['x'] + '_prev',
                                      center_coordinate_columns['y'] + '_prev',
                                      'endpoint1_X', 'endpoint1_Y',
                                      'endpoint2_X', 'endpoint2_Y',
                                      'endpoint1_X_prev', 'endpoint1_Y_prev',
                                      'endpoint2_X_prev', 'endpoint2_Y_prev']][is_duplicated]
    divided_bac = pd.DataFrame(index=divided_bac_view.index)

    non_divided_bac['LengthChangeRatio'] = non_divided_bac_view['LengthChangeRatio']
    divided_bac['LengthChangeRatio'] = divided_bac_view['daughter_mother_LengthChangeRatio']

    # now we should calculate features
    # IOU
    non_divided_bac = iou_calc(non_divided_bac, col_source='prev_index_prev', col_target='prev_index',
                               stat='same', df_view=non_divided_bac_view, coordinate_array=coordinate_array)

    # stat='div'
    divided_bac = iou_calc(divided_bac, col_source='prev_index_prev', col_target='prev_index', stat='div',
                           df_view=divided_bac_view, coordinate_array=coordinate_array, both=False)

    # distance
    non_divided_bac = calc_distance(non_divided_bac, center_coordinate_columns, postfix_target='',
                                    postfix_source='_prev', df_view=non_divided_bac_view)
    # stat='div'
    divided_bac = calc_distance(divided_bac, center_coordinate_columns, postfix_target='', postfix_source='_prev',
                                stat='div', df_view=divided_bac_view)

    # neighbor_ratio
    # non_divided_bac['adjusted_common_neighbors'] = np.where(
    #    non_divided_bac_view['common_neighbors'] == 0,
    #    non_divided_bac_view['common_neighbors'] + 1,
    #    non_divided_bac_view['common_neighbors']
    # )
    non_divided_bac['adjusted_common_neighbors'] = non_divided_bac_view['common_neighbors'].replace(0, 1)

    non_divided_bac['neighbor_ratio'] = \
        (non_divided_bac_view['difference_neighbors'] / (non_divided_bac['adjusted_common_neighbors']))

    # divided_bac['adjusted_common_neighbors'] = np.where(
    #    divided_bac_view['common_neighbors'] == 0,
    #    divided_bac_view['common_neighbors'] + 1,
    #    divided_bac_view['common_neighbors']
    # )
    divided_bac['adjusted_common_neighbors'] = divided_bac_view['common_neighbors'].replace(0, 1)

    divided_bac['neighbor_ratio'] = (divided_bac_view['difference_neighbors'] /
                                     (divided_bac['adjusted_common_neighbors']))

    # (Length Ratio(source-target)- Length Dynamics source )
    # non_divided_bac = check_len_ratio(df, non_divided_bac, '', '_prev')

    # now we should define label
    non_divided_bac['direction_of_motion'] = non_divided_bac_view['direction_of_motion']
    non_divided_bac['MotionAlignmentAngle'] = non_divided_bac_view['MotionAlignmentAngle']

    divided_bac['direction_of_motion'] = divided_bac_view['direction_of_motion']
    divided_bac['MotionAlignmentAngle'] = divided_bac_view['MotionAlignmentAngle']

    non_divided_bac['label'] = 'positive'
    divided_bac['label'] = 'negative'

    # difference_neighbors
    feature_list = ['iou', 'min_distance', 'neighbor_ratio', 'direction_of_motion', 'MotionAlignmentAngle',
                    'LengthChangeRatio', 'label']

    # merge dfs
    merged_df = pd.concat([non_divided_bac[feature_list], divided_bac[feature_list]], ignore_index=True,
                          copy=False)

    comparing_divided_non_divided_model = \
        run_ml_model(merged_df, feature_list=feature_list[:-1], columns_to_scale=feature_list[1:-1], stat='compare',
                     output_directory=output_directory, clf=clf, n_cpu=n_cpu)

    return comparing_divided_non_divided_model
