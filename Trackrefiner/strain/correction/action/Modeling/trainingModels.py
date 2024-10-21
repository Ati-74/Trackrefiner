from Trackrefiner.strain.correction.action.Modeling.compareDividedandNonDividedBacteria import \
    comparing_divided_non_divided_bacteria
from Trackrefiner.strain.correction.action.Modeling.nonDividedBacteria import make_ml_model_for_non_divided_bacteria
from Trackrefiner.strain.correction.action.Modeling.DividedBacteria import make_ml_model_for_divided_bacteria


def training_models(df, neighbors_df, neighbor_list_array, center_coordinate_columns, parent_image_number_col,
                    parent_object_number_col, output_directory, clf, n_cpu, coordinate_array):
    # first of all we should fine continues life history
    # inner: for removing unexpected beginning bacteria
    # also unexpected end bacteria removed

    important_cols = ['ImageNumber', 'ObjectNumber', 'unexpected_end', 'unexpected_beginning', 'bad_daughters_flag',
                      parent_image_number_col, parent_object_number_col,
                      'LengthChangeRatio', 'index', 'prev_index',
                      'common_neighbors', 'difference_neighbors',
                      'direction_of_motion', 'MotionAlignmentAngle',
                      center_coordinate_columns['x'], center_coordinate_columns['y'],
                      'endpoint1_X', 'endpoint1_Y', 'endpoint2_X', 'endpoint2_Y',
                      'daughter_length_to_mother', 'daughter_mother_LengthChangeRatio',
                      'AreaShape_MajorAxisLength', 'other_daughter_index', 'id', 'parent_id', 'age', 'bacteria_slope']

    connected_bac = df[important_cols].merge(
        df[important_cols], left_on=[parent_image_number_col, parent_object_number_col],
        right_on=['ImageNumber', 'ObjectNumber'], how='inner', suffixes=('', '_prev'))

    source_bac_with_neighbors = connected_bac.merge(neighbors_df, left_on=['ImageNumber_prev', 'ObjectNumber_prev'],
                                                    right_on=['First Image Number', 'First Object Number'], how='left')

    target_bac_with_neighbors = connected_bac.merge(neighbors_df, left_on=['ImageNumber', 'ObjectNumber'],
                                                    right_on=['First Image Number', 'First Object Number'], how='left')

    source_bac_with_neighbors_info = \
        source_bac_with_neighbors.merge(df[['ImageNumber', 'ObjectNumber', 'unexpected_end']],
                                        left_on=['Second Image Number', 'Second Object Number'],
                                        right_on=['ImageNumber', 'ObjectNumber'], how='left',
                                        suffixes=('', '_neighbor_source'))

    target_bac_with_neighbors_info = \
        target_bac_with_neighbors.merge(df[['ImageNumber', 'ObjectNumber', 'unexpected_beginning']],
                                        left_on=['Second Image Number', 'Second Object Number'],
                                        right_on=['ImageNumber', 'ObjectNumber'], how='left',
                                        suffixes=('', '_neighbor_target'))

    # (source_bac_with_neighbors_info['bad_division_flag_neighbor_source']) |
    #                                            (source_bac_with_neighbors_info['mother_rpl_neighbor_source']) |
    #                                            (source_bac_with_neighbors_info['source_mcl_neighbor_source'])
    source_bac_near_to_unexpected_end = \
        source_bac_with_neighbors_info.loc[(source_bac_with_neighbors_info['unexpected_end_neighbor_source'] == True)]

    # (target_bac_with_neighbors_info['target_mcl_neighbor_target']) |
    #                                            (target_bac_with_neighbors_info['daughter_rpl_neighbor_target']) |
    #                                            (target_bac_with_neighbors_info['bad_daughters_flag_neighbor_target'])

    # (target_bac_with_neighbors_info['daughter_rpl']) |
    #                                            (target_bac_with_neighbors_info['target_mcl'])
    target_bac_near_to_unexpected_beginning = \
        target_bac_with_neighbors_info.loc[
            (target_bac_with_neighbors_info['unexpected_beginning_neighbor_target'] == True) |
            (target_bac_with_neighbors_info['bad_daughters_flag'] == True)]

    # in this situation, we ignore target bacteria : 1. daughter of bad division 2. daughter of rpl
    # 3. near to unexpected_beginning bac and also bacteria --> source of bacteria is near to unexpected end
    connected_bac_high_chance_to_be_correct = \
        connected_bac.loc[(~ connected_bac['index'].isin(target_bac_near_to_unexpected_beginning['index'].values)) &
                          (~ connected_bac['index_prev'].isin(source_bac_near_to_unexpected_end['index_prev'].values))]

    # now we should check daughters and remove division with one daughter (due to neighboring to unexpected end and
    # unexpected beginning)
    division = \
        connected_bac_high_chance_to_be_correct.loc[
            ~ connected_bac_high_chance_to_be_correct['daughter_length_to_mother_prev'].isna()]

    # sometimes it can be possible one daughter filtered in prev steps and the other daughter is existing
    # I want to remove another daughter
    mother_with_two_daughters = division.duplicated(subset=[parent_image_number_col, parent_object_number_col],
                                                    keep=False)
    mother_with_one_daughter = ~ mother_with_two_daughters
    division_with_one_daughter = division[mother_with_one_daughter]

    connected_bac_high_chance_to_be_correct = \
        connected_bac_high_chance_to_be_correct.loc[~ connected_bac_high_chance_to_be_correct['index'].isin(
            division_with_one_daughter['index'].values)]

    comparing_divided_non_divided_model = \
        comparing_divided_non_divided_bacteria(connected_bac_high_chance_to_be_correct,
                                               center_coordinate_columns, parent_image_number_col,
                                               parent_object_number_col, output_directory, clf, n_cpu, coordinate_array)

    connected_bac_high_chance_to_be_correct_with_neighbors = \
        connected_bac_high_chance_to_be_correct.merge(neighbors_df, left_on=['ImageNumber_prev', 'ObjectNumber_prev'],
                                                      right_on=['First Image Number', 'First Object Number'],
                                                      how='left')

    connected_bac_high_chance_to_be_correct_with_neighbors_info = \
        connected_bac_high_chance_to_be_correct_with_neighbors.merge(
            df[important_cols], left_on=['Second Image Number', 'Second Object Number'],
            right_on=['ImageNumber', 'ObjectNumber'],
            how='left',
            suffixes=('', '_prev_neighbor'))

    non_divided_bac_model = \
        make_ml_model_for_non_divided_bacteria(df, connected_bac_high_chance_to_be_correct_with_neighbors_info,
                                               neighbors_df, neighbor_list_array, center_coordinate_columns,
                                               parent_image_number_col, parent_object_number_col, output_directory,
                                               clf, n_cpu, coordinate_array)

    divided_bac_model = \
        make_ml_model_for_divided_bacteria(df, connected_bac_high_chance_to_be_correct_with_neighbors_info,
                                           neighbor_list_array, center_coordinate_columns,
                                           parent_image_number_col, parent_object_number_col, output_directory,
                                           clf, n_cpu, coordinate_array)

    return comparing_divided_non_divided_model, non_divided_bac_model, divided_bac_model
