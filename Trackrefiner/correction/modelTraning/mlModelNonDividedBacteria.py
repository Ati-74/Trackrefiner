from Trackrefiner.correction.modelTraning.calculation.iouCalForML import calculate_iou_ml
from Trackrefiner.correction.modelTraning.calculation.calcDistanceForML import calc_min_distance_ml
from Trackrefiner.correction.modelTraning.calculation.lengthRatio import check_len_ratio
from Trackrefiner.correction.modelTraning.machineLearningPipeline import train_model
from Trackrefiner.correction.action.neighborAnalysis import compare_neighbor_sets
from Trackrefiner.correction.modelTraning.calculation.calMotionAlignmentAngle import calc_motion_alignment_angle_ml
import numpy as np
import pandas as pd


def train_non_divided_bacteria_model(df, connected_bac_high_chance_to_be_correct_with_neighbors_info,
                                     neighbors_df, neighbor_list_array, center_coord_cols, parent_image_number_col,
                                     parent_object_number_col, output_directory, clf, n_cpu, coordinate_array):
    """
    Constructs a machine learning model to predict whether a bacterium will continue its life history.

    **Workflow**:
        - Extracts features for bacteria and their neighbors, including:
          - Size relationships (e.g., `LengthChangeRatio`).
          - Spatial relationships (e.g., `IoU`, distances, neighbor differences).
          - Motion alignment.
        - Trains the specified classifier on the engineered features.
        - Outputs the trained model.

    :param pandas.DataFrame df:
        The complete dataset containing bacterial tracking information across time steps.
    :param pandas.DataFrame connected_bac_high_chance_to_be_correct_with_neighbors_info:
        Subset of bacteria with high probabilities of being correctly linked, including neighbor-related information.
    :param pandas.DataFrame neighbors_df:
        DataFrame containing details about bacterial neighbors.
    :param csr_matrix neighbor_list_array:
        Sparse matrix representation of bacterial neighbor relationships.
    :param dict center_coord_cols:
        Dictionary specifying the column names for x and y coordinates of bacterial centroids
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param str parent_image_number_col:
        Column name representing the image number of parent bacteria.
    :param str parent_object_number_col:
        Column name representing the object number of parent bacteria.
    :param str output_directory:
        Directory where model outputs (e.g., performance logs, trained models) will be saved.
    :param str clf:
        Classifier type (e.g., `'LogisticRegression'`, `'GaussianProcessClassifier'`, `'SVC'`).
    :param int n_cpu:
        Number of CPUs to use for parallel processing during training.
    :param scipy.sparse.csr_matrix coordinate_array:
        Sparse matrix of spatial coordinates for evaluating bacterial links.

    **Returns**:
        sklearn.Model:

        Trained machine learning model to predict non-divided bacteria.
    """

    daughter_length_to_mother_values = \
        connected_bac_high_chance_to_be_correct_with_neighbors_info['daughter_length_to_mother'].to_numpy()

    index_prev_neighbor_values = connected_bac_high_chance_to_be_correct_with_neighbors_info['index_prev_neighbor']
    stat_ue_neighbor = connected_bac_high_chance_to_be_correct_with_neighbors_info['unexpected_end_prev_neighbor']

    life_continues_signal = np.isnan(daughter_length_to_mother_values)
    valid_neighbor = (~ np.isnan(index_prev_neighbor_values)) & (stat_ue_neighbor == False)

    col_names = ['ImageNumber', 'ObjectNumber', 'index', 'prev_index', 'AreaShape_MajorAxisLength',
                 'difference_neighbors', 'common_neighbors', 'other_daughter_index', 'id', 'parent_id',
                 parent_image_number_col, parent_object_number_col, center_coord_cols['x'], center_coord_cols['y'],
                 'endpoint1_X', 'endpoint1_Y', 'endpoint2_X', 'endpoint2_Y',
                 'MotionAlignmentAngle', 'daughter_length_to_mother', 'unexpected_beginning', 'unexpected_end', 'age'
                 ]

    neighbor_cols = [col + '_prev_neighbor' for col in col_names]
    source_cols = [col + '_prev' for col in col_names]

    neighbor_full_col_names = col_names.copy()
    neighbor_full_col_names.extend(neighbor_cols)

    continues_source_target_cols = col_names.copy()
    continues_source_target_cols.extend(source_cols)

    non_divided_bac_with_neighbor_of_source = \
        connected_bac_high_chance_to_be_correct_with_neighbors_info[neighbor_full_col_names]
    non_divided_bac_with_neighbor_of_source = \
        non_divided_bac_with_neighbor_of_source[life_continues_signal & valid_neighbor]

    continues_life_history_bac = connected_bac_high_chance_to_be_correct_with_neighbors_info[
        continues_source_target_cols]
    continues_life_history_bac_idx = \
        connected_bac_high_chance_to_be_correct_with_neighbors_info[
            ['ImageNumber', 'ObjectNumber']][life_continues_signal].groupby(
            ['ImageNumber', 'ObjectNumber']).tail(1).index

    continues_life_history_bac = continues_life_history_bac.loc[continues_life_history_bac_idx]

    rename_dict = {}
    for col in col_names:
        rename_dict[col + '_prev_neighbor'] = col + '_prev'

    # rename columns
    non_divided_bac_with_neighbor_of_source = non_divided_bac_with_neighbor_of_source.rename(rename_dict, axis=1)
    # just for adding neighbors info into dataframe (correct position of rows in dataframe)
    non_divided_bac_with_neighbor_of_source['index2'] = non_divided_bac_with_neighbor_of_source.index.values

    non_divided_bac_with_neighbor_of_source['LengthChangeRatio'] = \
        (non_divided_bac_with_neighbor_of_source['AreaShape_MajorAxisLength'] /
         non_divided_bac_with_neighbor_of_source['AreaShape_MajorAxisLength_prev'])

    # now we should filter links between target bec & source link
    # bac_bac_length_ratio_boundary = find_bac_len_to_bac_ratio_boundary(non_divided_bac)
    # non_divided_bac_with_neighbor_of_source = \
    #    non_divided_bac_with_neighbor_of_source.loc[non_divided_bac_with_neighbor_of_source['LengthChangeRatio'] >=
    #                                                (bac_bac_length_ratio_boundary['avg'] -
    #                                                 1.96 * bac_bac_length_ratio_boundary['std'])]

    # now we should calculate features
    # IOU
    continues_life_history_bac = calculate_iou_ml(continues_life_history_bac, col_source='prev_index_prev',
                                                  col_target='prev_index',
                                                  link_type='continuity', coordinate_array=coordinate_array)

    non_divided_bac_with_neighbor_of_source = \
        calculate_iou_ml(non_divided_bac_with_neighbor_of_source, col_source='prev_index_prev', col_target='prev_index',
                         link_type='continuity', coordinate_array=coordinate_array)

    # distance
    continues_life_history_bac = calc_min_distance_ml(continues_life_history_bac, center_coord_cols, postfix_target='',
                                                      postfix_source='_prev', link_type=None)
    non_divided_bac_with_neighbor_of_source = \
        calc_min_distance_ml(non_divided_bac_with_neighbor_of_source, center_coord_cols, postfix_source='',
                             postfix_target='_prev', link_type=None)

    # changes in neighbors
    # now we should check neighbors
    # we have already this feature for the links of dataset, but we should calculate it for
    # target - neighbor of source links
    # non_divided_bac_with_neighbor_of_source['prev_time_step_NeighborIndexList'] = \
    #         non_divided_bac_with_neighbor_of_source['NeighborIndexList_prev']
    non_divided_bac_with_neighbor_of_source['prev_time_step_index'] = \
        non_divided_bac_with_neighbor_of_source['index_prev'].astype('int32')

    non_divided_bac_with_neighbor_of_source['difference_neighbors'] = np.nan
    non_divided_bac_with_neighbor_of_source['common_neighbors'] = np.nan
    non_divided_bac_with_neighbor_of_source['other_daughter_index'] = np.nan
    non_divided_bac_with_neighbor_of_source['parent_id'] = non_divided_bac_with_neighbor_of_source['id_prev']

    non_divided_bac_with_neighbor_of_source = \
        compare_neighbor_sets(non_divided_bac_with_neighbor_of_source, neighbor_list_array=neighbor_list_array,
                              parent_image_number_col=parent_image_number_col,
                              parent_object_number_col=parent_object_number_col,
                              selected_rows_df=non_divided_bac_with_neighbor_of_source, selected_time_step_df=df,
                              return_common_elements=True, col_target='')

    non_divided_bac_with_neighbor_of_source['adjusted_common_neighbors'] = \
        non_divided_bac_with_neighbor_of_source['common_neighbors'].replace(0, 1)

    continues_life_history_bac['adjusted_common_neighbors'] = continues_life_history_bac['common_neighbors'].replace(0,
                                                                                                                     1)

    non_divided_bac_with_neighbor_of_source['neighbor_ratio'] = \
        (non_divided_bac_with_neighbor_of_source['difference_neighbors'] /
         (non_divided_bac_with_neighbor_of_source['adjusted_common_neighbors']))

    continues_life_history_bac['neighbor_ratio'] = \
        (continues_life_history_bac['difference_neighbors'] / (continues_life_history_bac['adjusted_common_neighbors']))

    # (Length Ratio(source-target)- Length Dynamics source )
    continues_life_history_bac = check_len_ratio(df, continues_life_history_bac, col_target='', col_source='_prev')
    non_divided_bac_with_neighbor_of_source = check_len_ratio(df, non_divided_bac_with_neighbor_of_source,
                                                              col_target='', col_source='_prev')

    # motion alignment
    # calculated for original df and we should calc for new df
    non_divided_bac_with_neighbor_of_source["MotionAlignmentAngle"] = np.nan
    non_divided_bac_with_neighbor_of_source = \
        calc_motion_alignment_angle_ml(df, neighbors_df, center_coord_cols,
                                       selected_rows=non_divided_bac_with_neighbor_of_source, col_target='',
                                       col_source='_prev')

    # now we should define label
    continues_life_history_bac['label'] = 'positive'
    non_divided_bac_with_neighbor_of_source['label'] = 'negative'

    # 'difference_neighbors'
    feature_list = ['iou', 'min_distance', 'neighbor_ratio', 'length_dynamic', 'MotionAlignmentAngle', 'label']

    # merge dfs
    merged_df = pd.concat(
        [continues_life_history_bac[feature_list], non_divided_bac_with_neighbor_of_source[feature_list]],
        ignore_index=True, copy=False)

    non_divided_bac_model = \
        train_model(merged_df, feature_list=feature_list[:-1], columns_to_scale=feature_list[1:-1],
                    model_type='non_divided', output_directory=output_directory, clf=clf, n_cpu=n_cpu)

    return non_divided_bac_model
