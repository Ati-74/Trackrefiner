from Trackrefiner.correction.modelTraning.mlModelDivisionVsContinuity import train_division_vs_continuity_model
from Trackrefiner.correction.modelTraning.mlModelContinuity import train_continuity_links_model
from Trackrefiner.correction.modelTraning.mlModelDivision import train_division_links_model


def train_bacterial_behavior_models(df, neighbors_df, neighbor_list_array, center_coord_cols, parent_image_number_col,
                                    parent_object_number_col, output_directory, clf, n_cpu, coordinate_array):

    """
    Trains machine learning models to predict bacterial behaviors, including division, non-division,
    and distinguishing between divided and non-divided bacteria.

    :param pandas.DataFrame df:
        DataFrame containing bacterial tracking data with spatial, temporal, and neighbor information.
    :param pandas.DataFrame neighbors_df:
        DataFrame containing bacterial neighbor relationships for all time steps.
    :param csr_matrix neighbor_list_array:
        Sparse matrix representing neighbor relationships for efficient lookup.
    :param dict center_coord_cols:
        Dictionary specifying the column names for x and y coordinates of bacterial centroids
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param str parent_image_number_col:
        Column name representing the image number of parent bacteria.
    :param str parent_object_number_col:
        Column name representing the object number of parent bacteria.
    :param str output_directory:
        Directory to save the trained models and performance logs.
    :param str clf:
        Classifier type (e.g., `'LogisticRegression'`, `'GaussianProcessClassifier'`, `'SVC'`).
    :param int n_cpu:
        Number of CPUs to use for parallel processing during model training.
    :param np.ndarray coordinate_array:
        Array of spatial coordinates for evaluating bacterial connections.

    **Returns**:
        tuple: Three trained machine learning models:
            - **division_vs_continuity_model**: Model to distinguish between division and continuity links.
            - **continuity_links_model**: Model to predict the behavior of continuity links.
            - **division_links_model**: Model to analyze the characteristics of division links.
    """

    # first of all we should fine continues life history
    # inner: for removing unexpected beginning bacteria
    # also unexpected end bacteria removed

    important_cols = ['ImageNumber', 'ObjectNumber', 'unexpected_end', 'unexpected_beginning', 'bad_daughters_flag',
                      parent_image_number_col, parent_object_number_col,
                      'LengthChangeRatio', 'index', 'prev_index',
                      'common_neighbors', 'difference_neighbors',
                      'direction_of_motion', 'MotionAlignmentAngle',
                      center_coord_cols['x'], center_coord_cols['y'],
                      'endpoint1_X', 'endpoint1_Y', 'endpoint2_X', 'endpoint2_Y',
                      'daughter_length_to_mother', 'daughter_mother_LengthChangeRatio',
                      'AreaShape_MajorAxisLength', 'other_daughter_index', 'id', 'parent_id', 'age', 'bacteria_slope']

    connected_bac = df[important_cols].merge(
        df[important_cols], left_on=[parent_image_number_col, parent_object_number_col],
        right_on=['ImageNumber', 'ObjectNumber'], how='inner', suffixes=('', '_prev'))

    target_bac_with_neighbors = connected_bac.merge(neighbors_df, left_on=['ImageNumber', 'ObjectNumber'],
                                                    right_on=['First Image Number', 'First Object Number'], how='left')

    target_bac_with_neighbors_info = \
        target_bac_with_neighbors.merge(df[['ImageNumber', 'ObjectNumber', 'unexpected_beginning']],
                                        left_on=['Second Image Number', 'Second Object Number'],
                                        right_on=['ImageNumber', 'ObjectNumber'], how='left',
                                        suffixes=('', '_neighbor_target'))

    bad_daughters_target = \
        target_bac_with_neighbors_info.loc[(target_bac_with_neighbors_info['bad_daughters_flag'] == True)]

    connected_bac_high_chance_to_be_correct = \
        connected_bac.loc[(~ connected_bac['index'].isin(bad_daughters_target['index'].values))]

    division_vs_continuity_model = \
        train_division_vs_continuity_model(connected_bac_high_chance_to_be_correct,
                                           center_coord_cols, parent_image_number_col,
                                           parent_object_number_col, output_directory, clf, n_cpu, coordinate_array)

    connected_bac_high_chance_to_be_correct_with_neighbors = \
        connected_bac_high_chance_to_be_correct.merge(neighbors_df, left_on=['ImageNumber_prev', 'ObjectNumber_prev'],
                                                      right_on=['First Image Number', 'First Object Number'],
                                                      how='left')

    connected_bac_high_chance_to_be_correct_with_neighbors_info = \
        connected_bac_high_chance_to_be_correct_with_neighbors.merge(
            df[important_cols], left_on=['Second Image Number', 'Second Object Number'],
            right_on=['ImageNumber', 'ObjectNumber'], how='left', suffixes=('', '_prev_neighbor'))

    continuity_links_model = \
        train_continuity_links_model(df, connected_bac_high_chance_to_be_correct_with_neighbors_info,
                                     neighbors_df, neighbor_list_array, center_coord_cols,
                                     parent_image_number_col, parent_object_number_col, output_directory,
                                     clf, n_cpu, coordinate_array)

    division_links_model = \
        train_division_links_model(df, connected_bac_high_chance_to_be_correct_with_neighbors_info,
                                   neighbor_list_array, center_coord_cols,
                                   parent_image_number_col, parent_object_number_col, output_directory,
                                   clf, n_cpu, coordinate_array)

    return division_vs_continuity_model, continuity_links_model, division_links_model
