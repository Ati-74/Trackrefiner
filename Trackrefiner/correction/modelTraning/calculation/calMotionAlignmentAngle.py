from Trackrefiner.correction.action.helper import calc_normalized_angle_between_motion_for_df


def calc_motion_alignment_angle_ml(df, neighbor_df, center_coord_cols, selected_rows=None, col_target=None,
                                   col_source=None):

    """
    Calculates the motion alignment angle between the movement direction of a link (source-to-target)
    and the average motion direction of neighboring bacteria. The alignment is computed in degrees,
    representing the angular difference between the link's motion and its neighbors' average motion.

    :param pandas.DataFrame df:
        The main DataFrame containing bacterial data.
    :param pandas.DataFrame neighbor_df:
        A DataFrame containing neighbor relationships (e.g., First and Second Image/Object Numbers).
    :param dict center_coord_cols:
        A dictionary specifying the column names for the x and y coordinates of bacterial centroids
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param pandas.DataFrame selected_rows:
        The subset of rows for which motion alignment angles need to be computed.
    :param str col_target:
        Suffix for target-related columns in `selected_rows`.
    :param str col_source:
        Suffix for source-related columns in `selected_rows`.

    **Returns**:
        pandas.DataFrame:

        The updated `selected_rows` DataFrame with a new column:
            - `"MotionAlignmentAngle" + col_target`: The calculated motion alignment angle (degrees).
    """

    temp_df = df[['ImageNumber', 'ObjectNumber']].copy()

    temp_df['source_target_TrajectoryX'] = df.groupby('id')["TrajectoryX"].shift(-1)
    temp_df['source_target_TrajectoryY'] = df.groupby('id')["TrajectoryY"].shift(-1)

    daughters_idx = ~ df['daughter_length_to_mother'].isna()

    temp_df.loc[daughters_idx, ['source_target_TrajectoryX', 'source_target_TrajectoryY']] = \
        df.loc[daughters_idx, ['avg_daughters_TrajectoryX', 'avg_daughters_TrajectoryY']].values

    selected_rows['this_link_dir_x'] = (selected_rows[center_coord_cols['x'] + col_target] -
                                        selected_rows[center_coord_cols['x'] + col_source])

    selected_rows['this_link_dir_y'] = (selected_rows[center_coord_cols['y'] + col_target] -
                                        selected_rows[center_coord_cols['y'] + col_source])

    source_bac_neighbors = \
        selected_rows[['ImageNumber' + col_source, 'ObjectNumber' + col_source,
                       'ImageNumber' + col_target, 'ObjectNumber' + col_target,
                       'this_link_dir_x', 'this_link_dir_y']].merge(
            neighbor_df, left_on=['ImageNumber' + col_source, 'ObjectNumber' + col_source],
            right_on=['First Image Number', 'First Object Number'],
            how='left')[['ImageNumber' + col_source, 'ObjectNumber' + col_source,
                         'ImageNumber' + col_target, 'ObjectNumber' + col_target,
                         'this_link_dir_x', 'this_link_dir_y', 'Second Image Number', 'Second Object Number']]

    # be careful: 'source_target_TrajectoryX', 'source_target_TrajectoryY' is only for neighbors
    source_bac_neighbors_info = \
        source_bac_neighbors.merge(
            temp_df, left_on=['Second Image Number', 'Second Object Number'],
            right_on=['ImageNumber', 'ObjectNumber'], how='left',
            suffixes=('', '_neighbor2'))[['ImageNumber' + col_source, 'ObjectNumber' + col_source,
                                          'ImageNumber' + col_target, 'ObjectNumber' + col_target,
                                          'this_link_dir_x', 'this_link_dir_y',
                                          'source_target_TrajectoryX', 'source_target_TrajectoryY']]

    source_bac_neighbors_info['neighbors_avg_dir_motion_x'] = \
        source_bac_neighbors_info.groupby(['ImageNumber' + col_target,
                                           'ObjectNumber' + col_target,
                                           'ImageNumber' + col_source,
                                           'ObjectNumber' + col_source])['source_target_TrajectoryX'].transform('mean')

    source_bac_neighbors_info['neighbors_avg_dir_motion_y'] = \
        source_bac_neighbors_info.groupby(['ImageNumber' + col_target,
                                           'ObjectNumber' + col_target,
                                           'ImageNumber' + col_source,
                                           'ObjectNumber' + col_source])['source_target_TrajectoryY'].transform('mean')

    source_bac_neighbors_info = source_bac_neighbors_info.groupby(['ImageNumber' + col_target,
                                                                   'ObjectNumber' + col_target,
                                                                   'ImageNumber' + col_source,
                                                                   'ObjectNumber' + col_source
                                                                   ]).head(1)

    source_bac_neighbors_info = \
        calc_normalized_angle_between_motion_for_df(
            source_bac_neighbors_info, 'this_link_dir_x', 'this_link_dir_y',
            'neighbors_avg_dir_motion_x', 'neighbors_avg_dir_motion_y')

    selected_rows = \
        selected_rows.merge(source_bac_neighbors_info[
                                ['ImageNumber' + col_target, 'ObjectNumber' + col_target,
                                 'ImageNumber' + col_source, 'ObjectNumber' + col_source, 'angle_degrees']],
                            on=['ImageNumber' + col_target, 'ObjectNumber' + col_target,
                                'ImageNumber' + col_source, 'ObjectNumber' + col_source], how='inner')

    selected_rows["MotionAlignmentAngle" + col_target] = selected_rows['angle_degrees']
    selected_rows["MotionAlignmentAngle" + col_target] = selected_rows["MotionAlignmentAngle" + col_target].fillna(0)
    selected_rows.drop(columns=['angle_degrees'], inplace=True)

    return selected_rows
