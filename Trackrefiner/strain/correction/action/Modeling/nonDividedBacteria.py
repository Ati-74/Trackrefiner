from Trackrefiner.strain.correction.action.findOutlier import find_bac_len_to_bac_ratio_boundary
from Trackrefiner.strain.correction.action.Modeling.calculation.iouCalForML import iou_calc
from Trackrefiner.strain.correction.action.Modeling.calculation.calcDistanceForML import calc_distance
from Trackrefiner.strain.correction.action.Modeling.calculation.lengthRatio import check_len_ratio
from Trackrefiner.strain.correction.action.Modeling.runML import run_ml_model
from Trackrefiner.strain.correction.neighborChecking import neighbor_checking
from Trackrefiner.strain.correction.action.Modeling.calculation.calMotionAlignmentAngle import calc_MotionAlignmentAngle
import numpy as np
import pandas as pd
import time


def make_ml_model_for_non_divided_bacteria(df, connected_bac_high_chance_to_be_correct_with_neighbors_info,
                                           neighbors_df, center_coordinate_columns,
                                           parent_image_number_col, parent_object_number_col, output_directory,
                                           clf, n_cpu):

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))
    print(end_tracking_errors_correction_time_str)

    non_divided_bac = \
        connected_bac_high_chance_to_be_correct_with_neighbors_info.loc[
            connected_bac_high_chance_to_be_correct_with_neighbors_info['daughter_length_to_mother'].isna()].copy()

    col_names = df.columns.tolist().copy()
    org_col_names = df.columns.tolist().copy()
    col_names.extend([v + '_prev_neighbor' for v in df.columns.tolist()])
    org_col_names.extend([v + '_prev' for v in df.columns.tolist()])

    non_divided_bac_with_neighbor_of_source = non_divided_bac[col_names]
    non_divided_bac = non_divided_bac[org_col_names]
    non_divided_bac = non_divided_bac.drop_duplicates(subset=['ImageNumber', 'ObjectNumber'], keep='last').copy()

    non_divided_bac_with_neighbor_of_source = \
        non_divided_bac_with_neighbor_of_source.loc[
            ~ non_divided_bac_with_neighbor_of_source['index_prev_neighbor'].isna()]

    rename_dict = {}
    for col in df.columns.tolist():
        rename_dict[col + '_prev_neighbor'] = col + '_prev'

    # rename columns
    non_divided_bac_with_neighbor_of_source = non_divided_bac_with_neighbor_of_source.rename(rename_dict, axis=1)
    non_divided_bac_with_neighbor_of_source['index'] = non_divided_bac_with_neighbor_of_source.index.values

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))
    print(end_tracking_errors_correction_time_str)

    non_divided_bac_with_neighbor_of_source['LengthChangeRatio'] = \
        (non_divided_bac_with_neighbor_of_source['AreaShape_MajorAxisLength'] /
         non_divided_bac_with_neighbor_of_source['AreaShape_MajorAxisLength_prev'])

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))
    print(end_tracking_errors_correction_time_str)

    # now we should filter links between target bec & source link
    # bac_bac_length_ratio_boundary = find_bac_len_to_bac_ratio_boundary(non_divided_bac)
    # non_divided_bac_with_neighbor_of_source = \
    #    non_divided_bac_with_neighbor_of_source.loc[non_divided_bac_with_neighbor_of_source['LengthChangeRatio'] >=
    #                                                (bac_bac_length_ratio_boundary['avg'] -
    #                                                 1.96 * bac_bac_length_ratio_boundary['std'])]

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))
    print(end_tracking_errors_correction_time_str)

    # now we should calculate features
    # IOU
    non_divided_bac = iou_calc(non_divided_bac, col_source='coordinate_prev', col_target='coordinate', stat='same')
    non_divided_bac_with_neighbor_of_source = iou_calc(non_divided_bac_with_neighbor_of_source,
                                                       col_source='coordinate_prev', col_target='coordinate',
                                                       stat='same')

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))
    print(end_tracking_errors_correction_time_str)

    # distance
    non_divided_bac = calc_distance(non_divided_bac, center_coordinate_columns, postfix_target='',
                                    postfix_source='_prev', stat=None)
    non_divided_bac_with_neighbor_of_source = \
        calc_distance(non_divided_bac_with_neighbor_of_source, center_coordinate_columns,
                      postfix_source='', postfix_target='_prev', stat=None)

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))
    print(end_tracking_errors_correction_time_str)

    # changes in neighbors
    # now we should check neighbors
    # we have already this feature for the links of dataset but we should calcukate it for
    # target - neighbor of source links
    non_divided_bac_with_neighbor_of_source['prev_time_step_NeighborIndexList'] = \
        non_divided_bac_with_neighbor_of_source['NeighborIndexList_prev']

    non_divided_bac_with_neighbor_of_source['difference_neighbors'] = np.nan
    non_divided_bac_with_neighbor_of_source['other_daughter_index'] = np.nan
    non_divided_bac_with_neighbor_of_source['parent_id'] = non_divided_bac_with_neighbor_of_source['id_prev']

    non_divided_bac_with_neighbor_of_source = \
        neighbor_checking(non_divided_bac_with_neighbor_of_source, neighbors_df,
                          parent_image_number_col, parent_object_number_col,
                          selected_rows_df=non_divided_bac_with_neighbor_of_source, selected_time_step_df=df,
                          return_common_elements=True, col_target='')

    non_divided_bac_with_neighbor_of_source['adjusted_common_neighbors'] = np.where(
        non_divided_bac_with_neighbor_of_source['common_neighbors'] == 0,
        non_divided_bac_with_neighbor_of_source['common_neighbors'] + 1,
        non_divided_bac_with_neighbor_of_source['common_neighbors']
    )
    non_divided_bac['adjusted_common_neighbors'] = np.where(
        non_divided_bac['common_neighbors'] == 0,
        non_divided_bac['common_neighbors'] + 1,
        non_divided_bac['common_neighbors']
    )

    non_divided_bac_with_neighbor_of_source['neighbor_ratio'] = \
        (non_divided_bac_with_neighbor_of_source['difference_neighbors'] /
         (non_divided_bac_with_neighbor_of_source['adjusted_common_neighbors']))

    non_divided_bac['neighbor_ratio'] = \
        (non_divided_bac['difference_neighbors'] / (non_divided_bac['adjusted_common_neighbors']))

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))
    print(end_tracking_errors_correction_time_str)

    # (Length Ratio(source-target)- Length Dynamics source )
    non_divided_bac = check_len_ratio(df, non_divided_bac, col_target='', col_source='_prev')
    non_divided_bac_with_neighbor_of_source = check_len_ratio(df, non_divided_bac_with_neighbor_of_source,
                                                              col_target='', col_source='_prev')

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))
    print(end_tracking_errors_correction_time_str)

    # motion alignment
    # calculated for original df and we should calc for new df
    non_divided_bac_with_neighbor_of_source["MotionAlignmentAngle"] = np.nan
    non_divided_bac_with_neighbor_of_source = \
        calc_MotionAlignmentAngle(df, neighbors_df, center_coordinate_columns,
                                  selected_rows=non_divided_bac_with_neighbor_of_source, col_target='',
                                  col_source='_prev')

    end_tracking_errors_correction_time = time.time()
    end_tracking_errors_correction_time_str = time.strftime('%Y-%m-%d %H:%M:%S',
                                                            time.localtime(end_tracking_errors_correction_time))
    print(end_tracking_errors_correction_time_str)

    # breakpoint()

    # now we should define label
    non_divided_bac['label'] = 'positive'
    non_divided_bac_with_neighbor_of_source['label'] = 'negative'

    # 'difference_neighbors'
    feature_list = ['iou', 'min_distance', 'neighbor_ratio', 'length_dynamic', 'MotionAlignmentAngle', 'label']

    # merge dfs
    merged_df = pd.concat([non_divided_bac[feature_list], non_divided_bac_with_neighbor_of_source[feature_list]],
                          ignore_index=True)

    non_divided_bac_model = \
        run_ml_model(merged_df, feature_list=feature_list[:-1], columns_to_scale=feature_list[1:-1], stat='nonDivides',
                     output_directory=output_directory, clf=clf, n_cpu=n_cpu)

    return non_divided_bac_model
