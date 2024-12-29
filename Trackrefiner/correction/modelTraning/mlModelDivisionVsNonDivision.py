from Trackrefiner.correction.modelTraning.calculation.iouCalForML import calculate_iou_ml
from Trackrefiner.correction.modelTraning.calculation.calcDistanceForML import calc_min_distance_ml
from Trackrefiner.correction.modelTraning.machineLearningPipeline import train_model
import pandas as pd


def train_division_vs_non_division_model(connected_bac, center_coord_cols, parent_image_number_col,
                                         parent_object_number_col, output_directory, clf, n_cpu, coordinate_array):

    """
    Constructs a machine learning model to compare divided and non-divided bacteria.
    The model uses features such as movement, spatial relationships, neighbor changes,
    and bacterial morphology to classify each instance.

    **Workflow**:
        - Splits the dataset into "divided" and "non-divided" bacteria based on duplication in parent relationships.
        - Extracts and calculates features for each group, including:
          - Intersection over Union (IoU).
          - Spatial distances.
          - Neighbor differences and ratios.
          - Direction of motion and motion alignment angles.
          - Length change ratios.
        - Labels the "divided" group as negative and the "non-divided" group as positive.
        - Combines the features and trains the specified classifier.

    :param pandas.DataFrame connected_bac:
        DataFrame containing connected bacteria data, including spatial, temporal, and neighbor information.
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
    :param np.ndarray coordinate_array:
        Array of spatial coordinates for evaluating bacterial links.

    **Returns**:
        sklearn.Model:

        Trained machine learning model to classify divided and non-divided bacteria.
    """

    is_duplicated = connected_bac.duplicated(subset=[parent_image_number_col, parent_object_number_col], keep=False)
    rep_one_values = ~ is_duplicated

    non_divided_bac_view = \
        connected_bac[['LengthChangeRatio', 'prev_index_prev', 'prev_index',
                       'common_neighbors', 'difference_neighbors',
                       'direction_of_motion', 'MotionAlignmentAngle',
                       center_coord_cols['x'],
                       center_coord_cols['y'],
                       center_coord_cols['x'] + '_prev',
                       center_coord_cols['y'] + '_prev',
                       'endpoint1_X', 'endpoint1_Y',
                       'endpoint2_X', 'endpoint2_Y',
                       'endpoint1_X_prev', 'endpoint1_Y_prev',
                       'endpoint2_X_prev', 'endpoint2_Y_prev']][rep_one_values]

    non_divided_bac = pd.DataFrame(index=non_divided_bac_view.index)

    divided_bac_view = connected_bac[['daughter_mother_LengthChangeRatio', 'prev_index_prev', 'prev_index',
                                      'common_neighbors', 'difference_neighbors',
                                      'direction_of_motion', 'MotionAlignmentAngle',
                                      center_coord_cols['x'],
                                      center_coord_cols['y'],
                                      center_coord_cols['x'] + '_prev',
                                      center_coord_cols['y'] + '_prev',
                                      'endpoint1_X', 'endpoint1_Y',
                                      'endpoint2_X', 'endpoint2_Y',
                                      'endpoint1_X_prev', 'endpoint1_Y_prev',
                                      'endpoint2_X_prev', 'endpoint2_Y_prev']][is_duplicated]
    divided_bac = pd.DataFrame(index=divided_bac_view.index)

    non_divided_bac['LengthChangeRatio'] = non_divided_bac_view['LengthChangeRatio']
    divided_bac['LengthChangeRatio'] = divided_bac_view['daughter_mother_LengthChangeRatio']

    # now we should calculate features
    # IOU
    non_divided_bac = calculate_iou_ml(non_divided_bac, col_source='prev_index_prev', col_target='prev_index',
                                       link_type='continuity', df_view=non_divided_bac_view,
                                       coordinate_array=coordinate_array)

    # stat='div'
    divided_bac = calculate_iou_ml(divided_bac, col_source='prev_index_prev', col_target='prev_index', link_type='div',
                                   df_view=divided_bac_view, coordinate_array=coordinate_array, both=False)

    # distance
    non_divided_bac = calc_min_distance_ml(non_divided_bac, center_coord_cols, postfix_target='',
                                           postfix_source='_prev', df_view=non_divided_bac_view)
    # stat='div'
    divided_bac = calc_min_distance_ml(divided_bac, center_coord_cols, postfix_target='', postfix_source='_prev',
                                       link_type='div', df_view=divided_bac_view)

    non_divided_bac['adjusted_common_neighbors'] = non_divided_bac_view['common_neighbors'].replace(0, 1)

    non_divided_bac['neighbor_ratio'] = \
        (non_divided_bac_view['difference_neighbors'] / (non_divided_bac['adjusted_common_neighbors']))

    divided_bac['adjusted_common_neighbors'] = divided_bac_view['common_neighbors'].replace(0, 1)

    divided_bac['neighbor_ratio'] = (divided_bac_view['difference_neighbors'] /
                                     (divided_bac['adjusted_common_neighbors']))

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
        train_model(merged_df, feature_list=feature_list[:-1], columns_to_scale=feature_list[1:-1],
                    model_type='divided_vs_non_divided', output_directory=output_directory, clf=clf, n_cpu=n_cpu)

    return comparing_divided_non_divided_model
