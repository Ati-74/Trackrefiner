import pandas as pd
import numpy as np
from Trackrefiner.correction.action.helper import calculate_angles_between_slopes
from Trackrefiner.correction.modelTraning.machineLearningPipeline import train_model
from Trackrefiner.correction.modelTraning.calculation.iouCalForML import calculate_iou_ml
from Trackrefiner.correction.modelTraning.calculation.calcDistanceForML import calc_min_distance_ml
from Trackrefiner.correction.action.neighborAnalysis import compare_neighbor_sets


def train_divided_bacteria_model(df, connected_bac_high_chance_to_be_correct_with_neighbors_info, neighbor_list_array,
                                 center_coord_cols, parent_image_number_col, parent_object_number_col,
                                 output_directory, clf, n_cpu, coordinate_array):

    """
    Constructs a machine learning model to predict whether a bacterium will divide and be born daughters.

    **Workflow**:
        - Extracts features for bacteria pairs with division signals (`mother-daughter` relationships).
        - Filters outliers based on size and neighbor characteristics.
        - Computes key features such as:
          - Intersection over Union (IoU).
          - Spatial distances.
          - Neighbor changes (difference and common neighbors).
          - Angle between mother and daughter bacteria.
        - Assigns positive labels for actual divisions and negative labels for unrelated bacteria pairs
          (link between daughter and source's neighbor).
        - Trains the specified machine learning classifier using the derived features.

    :param pandas.DataFrame df:
        The complete dataset containing bacterial tracking information across time steps.
    :param pandas.DataFrame connected_bac_high_chance_to_be_correct_with_neighbors_info:
        Subset of bacteria with high probabilities of being correctly linked, including neighbor-related information.
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
        sparse matrix of spatial coordinates for evaluating bacterial links.

    **Returns**:
        sklearn.Model:

        Trained machine learning model to classify divided bacteria relationships.
    """

    daughter_length_to_mother_values = \
        connected_bac_high_chance_to_be_correct_with_neighbors_info['daughter_length_to_mother'].to_numpy()

    index_prev_neighbor_values = connected_bac_high_chance_to_be_correct_with_neighbors_info['index_prev_neighbor']
    stat_ue_neighbor = connected_bac_high_chance_to_be_correct_with_neighbors_info['unexpected_end_prev_neighbor']

    division_signal = ~ np.isnan(daughter_length_to_mother_values)
    valid_neighbor = (~ np.isnan(index_prev_neighbor_values)) & (stat_ue_neighbor == False)

    col_names = ['ImageNumber', 'ObjectNumber', 'index', 'prev_index', 'AreaShape_MajorAxisLength',
                 'difference_neighbors', 'common_neighbors', 'other_daughter_index', 'id', 'parent_id',
                 parent_image_number_col, parent_object_number_col,
                 center_coord_cols['x'],
                 center_coord_cols['y'],
                 'endpoint1_X', 'endpoint1_Y',
                 'endpoint2_X', 'endpoint2_Y',
                 'MotionAlignmentAngle', 'daughter_length_to_mother', 'unexpected_beginning', 'unexpected_end', 'age',
                 'bacteria_slope'
                 ]

    neighbor_cols = [col + '_prev_neighbor' for col in col_names]
    source_cols = [col + '_prev' for col in col_names]

    neighbor_full_col_names = col_names.copy()
    neighbor_full_col_names.extend(neighbor_cols)

    division_source_target_cols = col_names.copy()
    division_source_target_cols.extend(source_cols)

    divided_bac_with_neighbor_of_source = \
        connected_bac_high_chance_to_be_correct_with_neighbors_info[neighbor_full_col_names]

    divided_bac_with_neighbor_of_source = \
        divided_bac_with_neighbor_of_source[division_signal & valid_neighbor]

    divided_bac = connected_bac_high_chance_to_be_correct_with_neighbors_info[division_source_target_cols]
    divided_bac_idx = \
        connected_bac_high_chance_to_be_correct_with_neighbors_info[
            ['ImageNumber', 'ObjectNumber']][division_signal].groupby(
            ['ImageNumber', 'ObjectNumber']).tail(1).index

    divided_bac = divided_bac.loc[divided_bac_idx]

    # divided_bac = divided_bac.drop_duplicates(subset=['ImageNumber', 'ObjectNumber'], keep='last').copy()

    rename_dict = {}
    for col in col_names:
        rename_dict[col + '_prev_neighbor'] = col + '_prev'

    divided_bac_with_neighbor_of_source = divided_bac_with_neighbor_of_source.rename(rename_dict, axis=1)
    # solve the problem of index
    # just for adding neighbors info into dataframe (correct position of rows in dataframe)
    divided_bac_with_neighbor_of_source['index2'] = divided_bac_with_neighbor_of_source.index.values

    divided_bac_with_neighbor_of_source['max_daughter_len_to_mother_prev'] = \
        divided_bac_with_neighbor_of_source['AreaShape_MajorAxisLength'] / \
        divided_bac_with_neighbor_of_source['AreaShape_MajorAxisLength_prev']

    # now we should remove outliers
    divided_bac_with_neighbor_of_source = \
        divided_bac_with_neighbor_of_source.loc[
            divided_bac_with_neighbor_of_source['max_daughter_len_to_mother_prev'] < 1]

    # max_daughter_length_ratio_boundary = find_max_daughter_len_to_mother_ratio_boundary(divided_bac)
    # divided_bac_with_neighbor_of_source = \
    #    divided_bac_with_neighbor_of_source.loc[
    #        divided_bac_with_neighbor_of_source['max_daughter_len_to_mother_prev'] <=
    #        (max_daughter_length_ratio_boundary['avg'] +
    #         1.96 * max_daughter_length_ratio_boundary['std'])]

    # now we should calculate features
    # IOU
    divided_bac = calculate_iou_ml(divided_bac, col_source='prev_index_prev', col_target='prev_index', link_type='div',
                                   coordinate_array=coordinate_array, both=False)
    divided_bac_with_neighbor_of_source = \
        calculate_iou_ml(divided_bac_with_neighbor_of_source, col_source='prev_index_prev', col_target='prev_index',
                         link_type='div', coordinate_array=coordinate_array, both=False)

    # distance
    divided_bac = calc_min_distance_ml(divided_bac, center_coord_cols, postfix_target='', postfix_source='_prev',
                                       link_type='div')
    divided_bac_with_neighbor_of_source = \
        calc_min_distance_ml(divided_bac_with_neighbor_of_source, center_coord_cols,
                             postfix_target='', postfix_source='_prev', link_type='div')

    # changes in neighbors
    # now we should check neighbors
    # we have already this feature for the links of dataset, but we should calculate it for
    # target - neighbor of source links
    # divided_bac_with_neighbor_of_source['prev_time_step_NeighborIndexList'] = \
    #         divided_bac_with_neighbor_of_source['NeighborIndexList_prev']
    divided_bac_with_neighbor_of_source['prev_time_step_index'] = \
        divided_bac_with_neighbor_of_source['index_prev'].astype('int32')

    divided_bac_with_neighbor_of_source['difference_neighbors'] = np.nan
    divided_bac_with_neighbor_of_source['common_neighbors'] = np.nan
    divided_bac_with_neighbor_of_source['other_daughter_index'] = np.nan
    divided_bac_with_neighbor_of_source['parent_id'] = divided_bac_with_neighbor_of_source['id_prev']

    divided_bac_with_neighbor_of_source = \
        compare_neighbor_sets(divided_bac_with_neighbor_of_source, neighbor_list_array=neighbor_list_array,
                              parent_image_number_col=parent_image_number_col,
                              parent_object_number_col=parent_object_number_col,
                              selected_rows_df=divided_bac_with_neighbor_of_source, selected_time_step_df=df,
                              return_common_elements=True, col_target='')

    divided_bac_with_neighbor_of_source['adjusted_common_neighbors'] = \
        divided_bac_with_neighbor_of_source['common_neighbors'].replace(0, 1)

    divided_bac['adjusted_common_neighbors'] = divided_bac['common_neighbors'].replace(0, 1)

    divided_bac_with_neighbor_of_source['neighbor_ratio'] = \
        (divided_bac_with_neighbor_of_source['difference_neighbors'] /
         (divided_bac_with_neighbor_of_source['adjusted_common_neighbors']))

    divided_bac['neighbor_ratio'] = (divided_bac['difference_neighbors'] / (divided_bac['adjusted_common_neighbors']))

    # angle between source and target (mother & daughter)
    divided_bac['angle_mother_daughter'] = \
        calculate_angles_between_slopes(divided_bac['bacteria_slope_prev'].values,
                                        divided_bac['bacteria_slope'].values)

    divided_bac_with_neighbor_of_source['angle_mother_daughter'] = \
        calculate_angles_between_slopes(divided_bac_with_neighbor_of_source['bacteria_slope_prev'].values,
                                        divided_bac_with_neighbor_of_source['bacteria_slope'].values)

    # now we should define label
    divided_bac['label'] = 'positive'
    divided_bac_with_neighbor_of_source['label'] = 'negative'

    # merge dfs
    # 'difference_neighbors', 'neighbor_ratio'
    feature_list = ['iou', 'min_distance', 'neighbor_ratio', 'angle_mother_daughter', 'label']
    merged_df = pd.concat([divided_bac[feature_list], divided_bac_with_neighbor_of_source[feature_list]],
                          ignore_index=True)

    divided_bac_model = \
        train_model(merged_df, feature_list=feature_list[:-1], columns_to_scale=feature_list[1:-1],
                    model_type='divided', output_directory=output_directory, clf=clf, n_cpu=n_cpu)

    return divided_bac_model
