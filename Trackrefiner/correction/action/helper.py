import pandas as pd
import numpy as np
import sys
from scipy.spatial import distance_matrix


def print_progress_bar(iteration, total=10, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

    if iteration == total:
        print('\n')


def calc_movement(bac2, bac1, center_coordinate_columns):
    """
    This function computes the displacement of a bacterium (`bac1` (source) to `bac2` (target))
    using multiple spatial points.

    **The computed movements include**:
        - Centroid movement.
        - Endpoint movements (first and second endpoints).
        - Cross-movements between endpoints.

    The minimum movement among all these calculations is returned to represent the most plausible movement distance.

    :param pandas.Series bac2:
        Data for the bacterium in the next time step.
    :param pandas.Series bac1:
        Data for the bacterium in the current time step.
    :param dict center_coordinate_columns:
        A dictionary specifying the column names for the x and y coordinates of the bacterial centroids
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).

    **Returns**:
        float:

        The minimum movement distance between `bac1` and `bac2`.
    """

    center_movement = \
        np.linalg.norm(bac2[[center_coordinate_columns['x'], center_coordinate_columns['y']]].values -
                       bac1[[center_coordinate_columns['x'], center_coordinate_columns['y']]].values)

    endpoint1_1_movement = \
        np.linalg.norm(bac2[['endpoint1_X', 'endpoint1_Y']].values -
                       bac1[['endpoint1_X', 'endpoint1_Y']].values)

    endpoint1_endpoint2_movement = \
        np.linalg.norm(bac2[['endpoint1_X', 'endpoint1_Y']].values -
                       bac1[['endpoint1_X', 'endpoint1_Y']].values)

    endpoint2_2_movement = \
        np.linalg.norm(bac2[['endpoint2_X', 'endpoint2_Y']].values -
                       bac1[['endpoint2_X', 'endpoint2_Y']].values)

    endpoint2_endpoint1_movement = \
        np.linalg.norm(bac2[['endpoint2_X', 'endpoint2_Y']].values -
                       bac1[['endpoint2_X', 'endpoint2_Y']].values)

    movement = min(center_movement, endpoint1_1_movement, endpoint2_2_movement, endpoint1_endpoint2_movement,
                   endpoint2_endpoint1_movement)

    return movement


def calc_neighbors_dir_motion_all(df, neighbor_df, division_df, selected_rows=None):

    """
    Calculates motion alignment features for bacteria based on neighbor trajectories.

    :param pandas.DataFrame df:
        Main DataFrame containing bacterial motion and trajectory data.
    :param pandas.DataFrame neighbor_df:
        DataFrame containing neighbor information, including image and object numbers for neighbors.
    :param pandas.DataFrame division_df:
        DataFrame containing division-related indices for bacteria.
    :param pandas.DataFrame selected_rows:
        Optional DataFrame for updating specific selected rows. Default is None.

    :returns:
        pandas.DataFrame: The updated DataFrame with motion alignment features added.
    """

    temp_df = df[['ImageNumber', 'ObjectNumber', 'id', "TrajectoryX", "TrajectoryY",
                  "daughter_length_to_mother", 'avg_daughters_TrajectoryX', 'avg_daughters_TrajectoryY']].copy()

    temp_df['source_target_TrajectoryX'] = df.groupby('id')["TrajectoryX"].shift(-1)
    temp_df['source_target_TrajectoryY'] = df.groupby('id')["TrajectoryY"].shift(-1)

    temp_df.loc[~ temp_df['daughter_length_to_mother'].isna(), 'source_target_TrajectoryX'] = \
        temp_df.loc[~ temp_df['daughter_length_to_mother'].isna(), 'avg_daughters_TrajectoryX']

    temp_df.loc[~ temp_df['daughter_length_to_mother'].isna(), 'source_target_TrajectoryY'] = \
        temp_df.loc[~ temp_df['daughter_length_to_mother'].isna(), 'avg_daughters_TrajectoryY']

    if selected_rows is None:

        all_bac_neighbors = df.merge(neighbor_df, left_on=['ImageNumber', 'ObjectNumber'],
                                     right_on=['First Image Number', 'First Object Number'], how='left')

    else:
        all_bac_neighbors = selected_rows.merge(neighbor_df, left_on=['ImageNumber', 'ObjectNumber'],
                                                right_on=['First Image Number', 'First Object Number'], how='left')

    # be careful: 'source_target_TrajectoryX', 'source_target_TrajectoryY' is only for neighbors
    all_bac_neighbors_info = all_bac_neighbors.merge(temp_df, left_on=['Second Image Number', 'Second Object Number'],
                                                     right_on=['ImageNumber', 'ObjectNumber'], how='left',
                                                     suffixes=('_bac', '_neighbor'))

    all_bac_neighbors_info['source_neighbors_TrajectoryX'] = \
        all_bac_neighbors_info.groupby(['ImageNumber_bac',
                                        'ObjectNumber_bac'])['source_target_TrajectoryX'].transform('mean')

    all_bac_neighbors_info['source_neighbors_TrajectoryY'] = \
        all_bac_neighbors_info.groupby(['ImageNumber_bac',
                                        'ObjectNumber_bac'])['source_target_TrajectoryY'].transform('mean')

    all_bac_neighbors_info = all_bac_neighbors_info.drop_duplicates(subset=['ImageNumber_bac', 'ObjectNumber_bac'],
                                                                    keep='first')

    all_bac_neighbors_info.index = all_bac_neighbors_info['index'].values

    if len(np.unique(all_bac_neighbors_info['index'].values)) != len(all_bac_neighbors_info['index'].values):
        raise ValueError("Indices in `all_bac_neighbors_info` are not unique. This may indicate a data issue.")

    # update dataframe
    df.loc[all_bac_neighbors_info['index'].values, 'bac_source_neighbors_TrajectoryX'] = \
        all_bac_neighbors_info['source_neighbors_TrajectoryX'].values

    df.loc[all_bac_neighbors_info['index'].values, 'bac_source_neighbors_TrajectoryY'] = \
        all_bac_neighbors_info['source_neighbors_TrajectoryY'].values

    # now we should focus on target bac
    all_bac_neighbors_info['source_neighbors_TrajectoryX_for_target'] = \
        all_bac_neighbors_info.groupby('id_bac')["source_neighbors_TrajectoryX"].shift(1)

    all_bac_neighbors_info['source_neighbors_TrajectoryY_for_target'] = \
        all_bac_neighbors_info.groupby('id_bac')["source_neighbors_TrajectoryY"].shift(1)

    # daughters at the division time
    all_bac_neighbors_info.loc[division_df['index_2'].values, "source_neighbors_TrajectoryX_for_target"] = \
        df.loc[division_df['index_1'].values, "bac_source_neighbors_TrajectoryX"].values

    all_bac_neighbors_info.loc[division_df['index_2'].values, "source_neighbors_TrajectoryY_for_target"] = \
        df.loc[division_df['index_1'].values, "bac_source_neighbors_TrajectoryY"].values

    if selected_rows is None:
        df["MotionAlignmentAngle"] = \
            calculate_normalized_angle_between_motion_vectors(
                all_bac_neighbors_info[['source_neighbors_TrajectoryX_for_target',
                                        'source_neighbors_TrajectoryY_for_target']].values,
                all_bac_neighbors_info[['TrajectoryX_bac', 'TrajectoryY_bac']].values)
    else:

        # bac with nan
        bac_with_nan = all_bac_neighbors_info.loc[
            (all_bac_neighbors_info["source_neighbors_TrajectoryX_for_target"].isna()) & (
                    all_bac_neighbors_info['unexpected_beginning'] == False)]

        prev_motion_alignment_angle = df.loc[bac_with_nan['index'].values, "MotionAlignmentAngle"].values

        df.loc[all_bac_neighbors_info['index'].values, "MotionAlignmentAngle"] = \
            calculate_normalized_angle_between_motion_vectors(
                all_bac_neighbors_info[['source_neighbors_TrajectoryX_for_target',
                                        'source_neighbors_TrajectoryY_for_target']].values,
                all_bac_neighbors_info[['TrajectoryX_bac', 'TrajectoryY_bac']].values)

        df.loc[bac_with_nan['index'].values, "MotionAlignmentAngle"] = prev_motion_alignment_angle

    df["MotionAlignmentAngle"] = df["MotionAlignmentAngle"].fillna(0)

    return df


def update_dataframe_with_selected_rows(df, selected_rows):
    """
    Updates specific rows in a DataFrame based on another DataFrame of selected rows.

    This function takes an input DataFrame and updates specific rows (identified by the indices of the `selected_rows`
    DataFrame) with the values from the corresponding columns in `selected_rows`.

    :param pandas.DataFrame df:
        The main DataFrame to be updated.
    :param pandas.DataFrame selected_rows:
        A DataFrame containing the selected rows and their updated values. The indices of `selected_rows`
        must match the indices in `df` that are to be updated.

    :returns:
        pandas.DataFrame: The updated DataFrame with the following columns modified or added:

        - **`prev_time_step_MajorAxisLength`**:
          Major axis length of the bacterium from the previous time step.

        - **`LengthChangeRatio`**:
          The ratio of the major axis length between the current and previous time steps.

        - **Center Coordinates**:
          - `prev_time_step_center_x`: X-coordinate of the bacterium's center from the previous time step.
          - `prev_time_step_center_y`: Y-coordinate of the bacterium's center from the previous time step.

        - **Endpoint Coordinates**:
          - `prev_time_step_endpoint1_X`: X-coordinate of the first endpoint from the previous time step.
          - `prev_time_step_endpoint1_Y`: Y-coordinate of the first endpoint from the previous time step.
          - `prev_time_step_endpoint2_X`: X-coordinate of the second endpoint from the previous time step.
          - `prev_time_step_endpoint2_Y`: Y-coordinate of the second endpoint from the previous time step.

        - **`bacteria_movement`**:
          The minimum movement of the bacterium across time steps, calculated as the smallest
          displacement among the center and endpoints.

        - **Trajectory Features**:
          - `direction_of_motion`: The angle (in radians) representing the trajectory direction.
          - `TrajectoryX`: X-component of the trajectory displacement.
          - `TrajectoryY`: Y-component of the trajectory displacement.
    """

    df.loc[selected_rows.index, 'prev_time_step_MajorAxisLength'] = \
        selected_rows['prev_time_step_MajorAxisLength'].values

    df.loc[selected_rows.index, 'LengthChangeRatio'] = selected_rows['LengthChangeRatio'].values

    df.loc[selected_rows.index, 'prev_time_step_center_x'] = selected_rows['prev_time_step_center_x'].values
    df.loc[selected_rows.index, 'prev_time_step_center_y'] = selected_rows['prev_time_step_center_y'].values
    df.loc[selected_rows.index, 'prev_time_step_endpoint1_X'] = \
        selected_rows['prev_time_step_endpoint1_X'].values
    df.loc[selected_rows.index, 'prev_time_step_endpoint1_Y'] = \
        selected_rows['prev_time_step_endpoint1_Y'].values
    df.loc[selected_rows.index, 'prev_time_step_endpoint2_X'] = \
        selected_rows['prev_time_step_endpoint2_X'].values
    df.loc[selected_rows.index, 'prev_time_step_endpoint2_Y'] = \
        selected_rows['prev_time_step_endpoint2_Y'].values

    df.loc[selected_rows.index, "bacteria_movement"] = selected_rows["bacteria_movement"].values

    df.loc[selected_rows.index, "direction_of_motion"] = selected_rows["direction_of_motion"].values
    df.loc[selected_rows.index, "TrajectoryX"] = selected_rows["TrajectoryX"].values
    df.loc[selected_rows.index, "TrajectoryY"] = selected_rows["TrajectoryY"].values

    return df


def calculate_bacterial_life_history_features(dataframe, calc_all_features, neighbor_df, division_df,
                                              center_coord_cols, use_selected_rows, original_df=None):

    """
    Calculates features related to the continuous life history of bacteria.

    This function computes various features for bacteria across time steps, such as changes in length,
    movement patterns, and directional motion.

    :param pandas.DataFrame dataframe:
        A DataFrame containing bacterial features across time steps.
    :param bool calc_all_features:
        If True, computes additional features such as length change ratio, trajectory direction,
        and updates neighbor interactions.
    :param pandas.DataFrame neighbor_df:
        A DataFrame containing information about neighboring bacteria.
    :param pandas.DataFrame division_df:
        A DataFrame containing division-related information for bacteria.
    :param dict center_coord_cols:
        A dictionary specifying the column names for x and y center coordinates.
        Expected keys:
        - `'x'`: Base column name for x-coordinate.
        - `'y'`: Base column name for y-coordinate.
    :param bool use_selected_rows:
        If True, updates features based on selected rows from the original DataFrame.
    :param pandas.DataFrame original_df:
        The original DataFrame (optional) for updating selected rows when `flag_selected_rows` is True.

    :returns:
        pandas.DataFrame: The updated DataFrame with the following newly computed features:

        - **Length Change Ratio** (`LengthChangeRatio`):
          The ratio of the major axis length between the current and previous time steps.

        - **Movement** (`bacteria_movement`):
          The minimum movement of the bacterium across time steps, calculated as the smallest
          displacement among the center and endpoints.

        - **Direction of Motion**:
          - `direction_of_motion`: The angle (in radians) representing the trajectory direction.
          - `TrajectoryX` and `TrajectoryY`: The x and y components of the trajectory displacement.

        - **Neighbor Interactions**:
          Updates direction of motion (`direction_of_motion`) based on neighboring bacteria's
          trajectories and interactions.
    """

    if calc_all_features:
        dataframe['prev_time_step_MajorAxisLength'] = dataframe.groupby('id')["AreaShape_MajorAxisLength"].shift(1)
        dataframe['LengthChangeRatio'] = (dataframe["AreaShape_MajorAxisLength"] /
                                          dataframe['prev_time_step_MajorAxisLength'])

    dataframe['prev_time_step_center_x'] = dataframe.groupby('id')[center_coord_cols['x']].shift(1)
    dataframe['prev_time_step_center_y'] = dataframe.groupby('id')[center_coord_cols['y']].shift(1)
    dataframe['prev_time_step_endpoint1_X'] = dataframe.groupby('id')['endpoint1_X'].shift(1)
    dataframe['prev_time_step_endpoint1_Y'] = dataframe.groupby('id')["endpoint1_Y"].shift(1)
    dataframe['prev_time_step_endpoint2_X'] = dataframe.groupby('id')["endpoint2_X"].shift(1)
    dataframe['prev_time_step_endpoint2_Y'] = dataframe.groupby('id')["endpoint2_Y"].shift(1)

    center_movement = \
        np.linalg.norm(dataframe[[center_coord_cols['x'], center_coord_cols['y']]].values -
                       dataframe[['prev_time_step_center_x', 'prev_time_step_center_y']].values, axis=1)

    endpoint1_1_movement = \
        np.linalg.norm(dataframe[['endpoint1_X', 'endpoint1_Y']].values -
                       dataframe[['prev_time_step_endpoint1_X', 'prev_time_step_endpoint1_Y']].values, axis=1)

    endpoint1_endpoint2_movement = \
        np.linalg.norm(dataframe[['endpoint1_X', 'endpoint1_Y']].values -
                       dataframe[['prev_time_step_endpoint2_X', 'prev_time_step_endpoint2_Y']].values, axis=1)

    endpoint2_2_movement = \
        np.linalg.norm(dataframe[['endpoint2_X', 'endpoint2_Y']].values -
                       dataframe[['prev_time_step_endpoint2_X', 'prev_time_step_endpoint2_Y']].values, axis=1)

    endpoint2_endpoint1_movement = \
        np.linalg.norm(dataframe[['endpoint2_X', 'endpoint2_Y']].values -
                       dataframe[['prev_time_step_endpoint1_X', 'prev_time_step_endpoint1_Y']].values, axis=1)

    dataframe["bacteria_movement"] = \
        np.minimum.reduce([center_movement, endpoint1_1_movement, endpoint2_2_movement,
                           endpoint1_endpoint2_movement, endpoint2_endpoint1_movement])

    if calc_all_features:

        bac_need_to_cal_dir_motion_condition = dataframe["direction_of_motion"].isna()

        direction_of_motion = \
            calculate_trajectory_angles(dataframe[bac_need_to_cal_dir_motion_condition],
                                        center_coord_cols)

        dataframe.loc[bac_need_to_cal_dir_motion_condition, "direction_of_motion"] = direction_of_motion

        calculated_trajectory_x = calculate_trajectory_displacement(
            dataframe.loc[bac_need_to_cal_dir_motion_condition],
            center_coord_cols, axis='x')

        calculated_trajectory_y = calculate_trajectory_displacement(
            dataframe.loc[bac_need_to_cal_dir_motion_condition],
            center_coord_cols, axis='y')

        dataframe.loc[bac_need_to_cal_dir_motion_condition, 'TrajectoryX'] = calculated_trajectory_x
        dataframe.loc[bac_need_to_cal_dir_motion_condition, 'TrajectoryY'] = calculated_trajectory_y

        if use_selected_rows:

            original_df = update_dataframe_with_selected_rows(original_df, dataframe)
            selected_rows = original_df.loc[(original_df['ImageNumber'] >= dataframe['ImageNumber'].min()) &
                                            (original_df['ImageNumber'] <= dataframe['ImageNumber'].max())]

            dataframe = calc_neighbors_dir_motion_all(original_df, neighbor_df, division_df, selected_rows)

        else:
            dataframe = calc_neighbors_dir_motion_all(dataframe, neighbor_df, division_df)

    return dataframe


def calculate_trajectory_displacement(df, center_coord_cols, axis):

    """
    Calculates the displacement of objects along a specified axis between consecutive positions.

    This function computes the difference (displacement) between the current and previous positions
    of objects along a specified axis (x or y). It is useful for analyzing movement patterns or
    trajectories over time.

    :param pandas.DataFrame df:
        A DataFrame containing the current and previous positions of objects.
    :param dict center_coord_cols:
        A dictionary specifying the column names for the x and y coordinates of the current position.
        Expected keys:
        - `'x'`: Base column name for the x-coordinate.
        - `'y'`: Base column name for the y-coordinate.
    :param str axis:
        The axis along which to calculate the displacement. Must be one of:
        - `'x'`: Calculates displacement along the x-axis.
        - `'y'`: Calculates displacement along the y-axis.

    :returns:
        pandas.Series: A Series containing the displacements along the specified axis.

    """

    # Calculate the direction vector from the previous position to the current position
    if axis == 'x':
        return df[center_coord_cols['x']] - df['prev_time_step_center_x']
    elif axis == 'y':
        return df[center_coord_cols['y']] - df['prev_time_step_center_y']

    return df


def calculate_trajectory_angles(df, center_coord_cols, suffix1='', suffix2=None):

    """
    Calculates the angles of trajectory directions between two objects.

    This function computes the direction vector between two bacteria and calculates the angle of that direction in
    radians. The angles are measured counterclockwise from the positive x-axis.

    :param pandas.DataFrame df:
        A DataFrame containing the x and y coordinates for two positions of objects.
        These positions are used to calculate the trajectory direction.
    :param dict center_coord_cols:
        A dictionary specifying the column names for the x and y coordinates of object centers.
    :param str suffix1:
        A suffix added to the base column names in `center_coord_cols` to identify the first set of positions.
        Default is an empty string.
    :param str suffix2:
        An optional suffix added to the base column names in `center_coord_cols` to identify the second set of
        positions. If not provided, the function assumes the second set of positions corresponds to columns
        prefixed with `prev_time_step`.

    :returns:
        np.ndarray: A 1D array of angles (in radians) corresponding to the direction of trajectory for
        each object.
    """

    # Calculate the direction vector from the previous position to the current position
    if suffix2 is None:
        trajectory_direction = \
            (df[[f"{center_coord_cols['x']}{suffix1}", f"{center_coord_cols['y']}{suffix1}"]].values -
             df[[f'prev_time_step_center_x{suffix1}', f'prev_time_step_center_y{suffix1}']].values)
    else:
        trajectory_direction = \
            (df[[f"{center_coord_cols['x']}{suffix1}", f"{center_coord_cols['y']}{suffix1}"]].values -
             df[[f"{center_coord_cols['x']}{suffix2}", f"{center_coord_cols['y']}{suffix2}"]].values)

    # Calculate the angle of the direction vector
    angle = np.arctan2(trajectory_direction[:, 1], trajectory_direction[:, 0])

    return angle


def calculate_all_bac_slopes(df):

    """
    Calculates the slopes of lines formed by bacterial endpoints in the input DataFrame.

    This function determines the slope for each bacterium based on its two endpoints
    (`endpoint1_X`, `endpoint1_Y`, `endpoint2_X`, `endpoint2_Y`) provided in the input DataFrame.
    The slope is calculated for all bacteria, with special handling for vertical lines
    (where the slope is undefined).

    :param pandas.DataFrame df:
        A DataFrame containing columns for bacterial endpoints:
        - `endpoint1_X`, `endpoint1_Y`: Coordinates of the first endpoint.
        - `endpoint2_X`, `endpoint2_Y`: Coordinates of the second endpoint.

    :returns:
        pandas.DataFrame: The input DataFrame with an additional column:

        - **bacteria_slope**: The slope of the line formed by the bacterial endpoints.
          NaN is assigned for vertical lines.

    """

    df.loc[df['endpoint1_X'] == df['endpoint2_X'], 'bacteria_slope'] = np.nan

    df.loc[df['endpoint1_X'] != df['endpoint2_X'], 'bacteria_slope'] = (
            (df['endpoint2_Y'] - df['endpoint1_Y']) / (df['endpoint2_X'] - df['endpoint1_X']))

    return df


# find vertices of an ellipse (bacteria):
# references:
# https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate
# https://math.stackexchange.com/questions/2645689/what-is-the-parametric-equation-of-a-rotated-ellipse-given-the-angle-of-rotatio
def calculate_bac_endpoints(center, major, angle_rotation, angle_tolerance=1e-6):

    """
    Calculates the endpoints of the major axis of a bacterium based on its center, length, and orientation.

    This function computes the two endpoints of the major axis for a bacterium given its center coordinates,
    major axis length, and orientation angle. It handles three cases: bacteria aligned with the vertical axis,
    bacteria aligned with the horizontal axis, and bacteria with arbitrary orientations.

    **Mathematical Explanation**:

    - **Condition 1**: Bacteria parallel to the vertical axis :math:`\pi / 2`.
      For bacteria with orientation close to :math:`\pm \pi / 2` (absolute difference less than :math:`10^{-6}`),
      the endpoints are calculated as:

      .. math::

          (x_{\t{end1}}, y_{\t{end1}}) = (x_{\t{center}}, y_{\t{center}} + L)

          (x_{\t{end2}}, y_{\t{end2}}) = (x_{\t{center}}, y_{\t{center}} - L)

      where :math:`L` is the major axis length.

    - **Condition 2**: Bacteria parallel to the horizontal axis.
      For bacteria with orientation close to 0 (absolute difference less than :math:`10^{-6}`):

      .. math::

          (x_{\t{end1}}, y_{\t{end1}}) = (x_{\t{center}} + L, y_{\t{center}})

          (x_{\t{end2}}, y_{\t{end2}}) = (x_{\t{center}} - L, y_{\t{center}})


    - **Condition 3**: Non-axis-aligned bacteria:
      "For bacteria with arbitrary orientation :math:`\\theta`, the endpoints are computed using the
      following formulas:"

      .. math::

          x_{\t{end1}} = x_{\t{center}} + \\frac{L}{2} \\cdot \\cos(\\theta)

          y_{\t{end1}} = y_{\t{center}} + (x_{\t{end1}} - x_{\t{center}}) \\cdot \\tan(\\theta)

          x_{\t{end2}} = x_{\t{center}} - \\frac{L}{2} \\cdot \\cos(\\theta)

          y_{\t{end2}} = y_{\t{center}} + (x_{\t{end2}} - x_{\t{center}}) \\cdot \\tan(\\theta)

    - Endpoint selection ensures correct ordering along the major axis.

    :param list center:
        A list containing the x and y coordinates of the bacterium's center (e.g., `(x_center, y_center)`).
    :param float major:
        The length of the major axis of the bacterium.
    :param float angle_rotation:
        The orientation angle of the bacterium in radians.
    :param float angle_tolerance:
        The tolerance for detecting vertical or horizontal orientations (default: 1e-6).

    :returns:
        list: A list containing two sub lists for the endpoints:
        - `[x_end1, y_end1]`: Coordinates of the first endpoint.
        - `[x_end2, y_end2]`: Coordinates of the second endpoint.
    """

    # Bacteria parallel to the vertical axis
    if np.abs(angle_rotation - np.pi / 2) < angle_tolerance or np.abs(angle_rotation + np.pi / 2) < angle_tolerance:
        vertex_1_x = center[0]
        vertex_1_y = center[1] + major
        vertex_2_x = center[0]
        vertex_2_y = center[1] - major

    elif np.abs(angle_rotation) < angle_tolerance:  # Bacteria parallel to the horizontal axis
        vertex_1_x = center[0] + major
        vertex_1_y = center[1]
        vertex_2_x = center[0] - major
        vertex_2_y = center[1]

    else:
        # (x- center_x) * np.sin(angle_rotation) - (y-center_y) * np.cos(angle_rotation) = 0
        # np.power((x - center_x) * np.cos(angle_rotation) + (y - center_y) * np.sin(angle_rotation), 2) =
        # np.power(major, 2)
        # vertex_1_x = semi_major / (np.cos(angle_rotation) + np.tan(angle_rotation) * np.sin(angle_rotation)) +
        #              center[0]
        # vertex_2_x = -semi_major / (np.cos(angle_rotation) + np.tan(angle_rotation) * np.sin(angle_rotation)) +
        #               center[0]

        semi_major = major / 2
        temp_vertex_1_x = float((semi_major * np.cos(angle_rotation)) + center[0])
        temp_vertex_1_y = float((temp_vertex_1_x - center[0]) * np.tan(angle_rotation) + center[1])
        temp_vertex_2_x = float((- semi_major * np.cos(angle_rotation)) + center[0])
        temp_vertex_2_y = float((temp_vertex_2_x - center[0]) * np.tan(angle_rotation) + center[1])

        if temp_vertex_1_x > center[0]:
            vertex_1_x = temp_vertex_1_x
            vertex_1_y = temp_vertex_1_y
            vertex_2_x = temp_vertex_2_x
            vertex_2_y = temp_vertex_2_y
        else:
            vertex_1_x = temp_vertex_2_x
            vertex_1_y = temp_vertex_2_y
            vertex_2_x = temp_vertex_1_x
            vertex_2_y = temp_vertex_1_y

    return [[vertex_1_x, vertex_1_y], [vertex_2_x, vertex_2_y]]


def calculate_all_bac_endpoints(df, center_coord_cols, angle_tolerance=1e-6):
    """
    Calculates the endpoints of bacterial major axes based on orientation, length, and position.

    This function determines the two endpoints of a bacterium's major axis by considering its orientation
    (parallel to vertical, horizontal, or otherwise) and center coordinates.

    **Mathematical Explanation**:

    - **Condition 1**: Bacteria parallel to the vertical axis :math:`\pi / 2`.
      For bacteria with orientation close to :math:`\pm \pi / 2` (absolute difference less than :math:`10^{-6}`),
      the endpoints are calculated as:

      .. math::

          (x_{\t{end1}}, y_{\t{end1}}) = (x_{\t{center}}, y_{\t{center}} + L)

          (x_{\t{end2}}, y_{\t{end2}}) = (x_{\t{center}}, y_{\t{center}} - L)

      where :math:`L` is the major axis length.

    - **Condition 2**: Bacteria parallel to the horizontal axis.
      For bacteria with orientation close to 0 (absolute difference less than :math:`10^{-6}`):

      .. math::

          (x_{\t{end1}}, y_{\t{end1}}) = (x_{\t{center}} + L, y_{\t{center}})

          (x_{\t{end2}}, y_{\t{end2}}) = (x_{\t{center}} - L, y_{\t{center}})


    - **Condition 3**: Non-axis-aligned bacteria:
      "For bacteria with arbitrary orientation :math:`\\theta`, the endpoints are computed using the
      following formulas:"

      .. math::

          x_{\t{end1}} = x_{\t{center}} + \\frac{L}{2} \\cdot \\cos(\\theta)

          y_{\t{end1}} = y_{\t{center}} + (x_{\t{end1}} - x_{\t{center}}) \\cdot \\tan(\\theta)

          x_{\t{end2}} = x_{\t{center}} - \\frac{L}{2} \\cdot \\cos(\\theta)

          y_{\t{end2}} = y_{\t{center}} + (x_{\t{end2}} - x_{\t{center}}) \\cdot \\tan(\\theta)

    - Endpoint selection ensures correct ordering along the major axis.

    :param pandas.DataFrame df:
        A DataFrame containing bacterial data, including orientation, center coordinates,
        and major axis length.
    :param dict center_coord_cols:
        A dictionary with keys 'x' and 'y', specifying the column names for center coordinates.
    :param float angle_tolerance:
        The tolerance for detecting vertical or horizontal orientations (default: 1e-6).

    :returns:
        pandas.DataFrame: The modified DataFrame with additional columns for endpoints:
        - `endpoint1_X`, `endpoint1_Y`: Coordinates of the first endpoint.
        - `endpoint2_X`, `endpoint2_Y`: Coordinates of the second endpoint.
    """

    # condition 1
    # Bacteria parallel to the vertical axis
    condition1 = (np.abs(df["AreaShape_Orientation"] - np.pi / 2) < angle_tolerance) | \
                 (np.abs(df["AreaShape_Orientation"] + np.pi / 2) < angle_tolerance)

    df.loc[condition1, 'endpoint1_X'] = df[center_coord_cols['x']]
    df.loc[condition1, 'endpoint1_Y'] = df[center_coord_cols['y']] + df["AreaShape_MajorAxisLength"]
    df.loc[condition1, 'endpoint2_X'] = df[center_coord_cols['x']]
    df.loc[condition1, 'endpoint2_Y'] = df[center_coord_cols['y']] - df["AreaShape_MajorAxisLength"]

    # Bacteria parallel to the horizontal axis
    condition2 = np.abs(df["AreaShape_Orientation"]) < angle_tolerance
    condition3 = ~condition1 & condition2
    df.loc[condition3, 'endpoint1_X'] = df[center_coord_cols['x']] + df["AreaShape_MajorAxisLength"]
    df.loc[condition3, 'endpoint1_Y'] = df[center_coord_cols['y']]
    df.loc[condition3, 'endpoint2_X'] = df[center_coord_cols['x']] - df["AreaShape_MajorAxisLength"]
    df.loc[condition3, 'endpoint2_Y'] = df[center_coord_cols['y']]

    # (x- center_x) * np.sin(angle_rotation) - (y-center_y) * np.cos(angle_rotation) = 0
    # np.power((x - center_x) * np.cos(angle_rotation) + (y - center_y) * np.sin(angle_rotation), 2) =
    # np.power(major, 2)
    condition4 = ~condition1 & ~condition2
    other_bac_df = df.loc[condition4][["AreaShape_MajorAxisLength", "AreaShape_Orientation",
                                       center_coord_cols['x'], center_coord_cols['y']]].copy()

    other_bac_df['semi_major'] = df["AreaShape_MajorAxisLength"] / 2
    other_bac_df['temp_vertex_1_x'] = \
        (other_bac_df['semi_major'] * np.cos(other_bac_df["AreaShape_Orientation"]) +
         other_bac_df[center_coord_cols['x']])

    other_bac_df['temp_vertex_1_y'] = \
        ((other_bac_df['temp_vertex_1_x'] - other_bac_df[center_coord_cols['x']]) *
         np.tan(other_bac_df["AreaShape_Orientation"]) + other_bac_df[center_coord_cols['y']])

    other_bac_df['temp_vertex_2_x'] = \
        (-other_bac_df['semi_major'] * np.cos(other_bac_df["AreaShape_Orientation"]) +
         other_bac_df[center_coord_cols['x']])

    other_bac_df['temp_vertex_2_y'] = \
        ((other_bac_df['temp_vertex_2_x'] - other_bac_df[center_coord_cols['x']]) *
         np.tan(other_bac_df["AreaShape_Orientation"]) + other_bac_df[center_coord_cols['y']])

    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] > other_bac_df[center_coord_cols['x']], 'vertex_1_x'] = \
        other_bac_df['temp_vertex_1_x']
    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] > other_bac_df[center_coord_cols['x']], 'vertex_1_y'] = \
        other_bac_df['temp_vertex_1_y']
    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] > other_bac_df[center_coord_cols['x']], 'vertex_2_x'] = \
        other_bac_df['temp_vertex_2_x']
    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] > other_bac_df[center_coord_cols['x']], 'vertex_2_y'] = \
        other_bac_df['temp_vertex_2_y']

    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] < other_bac_df[center_coord_cols['x']], 'vertex_1_x'] = \
        other_bac_df['temp_vertex_2_x']
    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] < other_bac_df[center_coord_cols['x']], 'vertex_1_y'] = \
        other_bac_df['temp_vertex_2_y']
    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] < other_bac_df[center_coord_cols['x']], 'vertex_2_x'] = \
        other_bac_df['temp_vertex_1_x']
    other_bac_df.loc[other_bac_df['temp_vertex_1_x'] < other_bac_df[center_coord_cols['x']], 'vertex_2_y'] = \
        other_bac_df['temp_vertex_1_y']

    df.loc[condition4, 'endpoint1_X'] = other_bac_df['vertex_1_x']
    df.loc[condition4, 'endpoint1_Y'] = other_bac_df['vertex_1_y']
    df.loc[condition4, 'endpoint2_X'] = other_bac_df['vertex_2_x']
    df.loc[condition4, 'endpoint2_Y'] = other_bac_df['vertex_2_y']

    return df


def convert_angle_to_radian(df):
    """
    Converts bacterial orientation angles from degrees to radians and modifies the values.

    The orientation is adjusted using the formula:

    .. math::
        Orientation (radian) = - \\frac{(\\text{Orientation (degrees)} + 90) \\cdot \\pi}{180}

    This ensures the angles are transformed to a range suitable for downstream computations.

    :param pandas.DataFrame df:
        A DataFrame containing a column named "AreaShape_Orientation" with orientation angles in degrees.

    :returns:
        pandas.DataFrame: The modified DataFrame with the "AreaShape_Orientation" column converted to radians.
    """

    # modification of bacterium orientation
    # -(angle + 90) * np.pi / 180
    df["AreaShape_Orientation"] = -(df["AreaShape_Orientation"] + 90) * np.pi / 180

    return df


def extract_bacteria_features(df, center_coord_cols):
    """
    Extracts key geometric and spatial features of bacteria from the input DataFrame.

    This function retrieves the major axis length, minor axis length, radius, orientation,
    and center coordinates of bacteria, organizing them into a dictionary for further use.

    :param pandas.DataFrame df:
        A DataFrame containing bacterial data with required columns for shape and position.
    :param dict center_coord_cols:
        A dictionary with keys 'x' and 'y', specifying the column names for center coordinates.

    :returns:
        dict: A dictionary containing the extracted features:
        - 'major': Major axis length of bacteria.
        - 'minor': Minor axis length of bacteria.
        - 'radius': Radius, calculated as half of the minor axis length.
        - 'orientation': Orientation of the bacteria (e.g., angle or direction).
        - 'center_x': X-coordinate of the bacteria's center.
        - 'center_y': Y-coordinate of the bacteria's center.
    """

    major = df['AreaShape_MajorAxisLength']
    minor = df['AreaShape_MinorAxisLength']
    radius = df['AreaShape_MinorAxisLength'] / 2
    orientation = df['AreaShape_Orientation']
    center_x = df[center_coord_cols['x']]
    center_y = df[center_coord_cols['y']]

    features = {'major': major, 'minor': minor, 'radius': radius, 'orientation': orientation, 'center_x': center_x,
                'center_y': center_y}
    return features


def calc_normalized_angle_between_motion_for_df(df, target_bac_motion_x_col, target_bac_motion_y_col,
                                                neighbors_avg_dir_motion_x, neighbors_avg_dir_motion_y):
    """
    Calculates the normalized angles between motion vectors for each row in a DataFrame.

    This function computes the angle between a motion vector and its neighbors' average motion vector
    for each row in the DataFrame. The angles are normalized by dividing by 180 degrees.

    **Mathematical Explanation**:

    - **Dot Product**:
      The dot product of two vectors is a measure of how aligned they are. For two vectors:

      .. math::

          \\mathbf{v_1} = (x_1, y_1), \\quad \\mathbf{v_2} = (x_2, y_2)

      The dot product is given by:

      .. math::

          \\mathbf{v_1} \\cdot \\mathbf{v_2} = x_1 \\cdot x_2 + y_1 \\cdot y_2

    - **Magnitude of a Vector**:
      The magnitude (length) of a vector is calculated as:

      .. math::

          \\| \\mathbf{v} \\| = \\sqrt{x^2 + y^2}

    - **Cosine of the Angle**:
      The cosine of the angle \\( \\theta \\) between two vectors is computed as:

      .. math::

          \\cos(\\theta) = \\frac{\\mathbf{v_1} \\cdot \\mathbf{v_2}}{\\| \\mathbf{v_1} \\| \\| \\mathbf{v_2} \\|}

      If either vector has zero magnitude, the cosine is undefined.

    - **Angle in Radians**:
      The angle \\( \\theta \\) in radians is obtained using the arccosine function:

      .. math::

          \\theta = \\arccos(\\cos(\\theta))

      To handle numerical precision, the cosine value is clipped to the range \\([-1, 1]\\).

    - **Convert Radians to Degrees**:
      The angle is converted from radians to degrees using the formula:

      .. math::

          \\text{Angle (degrees)} = \\frac{\\theta \\cdot 180}{\\pi}

    - **Normalized Angle**:
      Finally, the angle is normalized by dividing the degrees by 180:

      .. math::

          \\text{Normalized Angle} = \\frac{\\text{Angle (degrees)}}{180}

    :param pandas.DataFrame df:
        A DataFrame containing direction of motion of objects and average direction of motion of neighbors.
    :param str target_bac_motion_x_col:
        The column name for the x-component of the object's motion vector.
    :param str target_bac_motion_y_col:
        The column name for the y-component of the object's motion vector.
    :param str neighbors_avg_dir_motion_x:
        The column name for the x-component of the neighbors' average motion vector.
    :param str neighbors_avg_dir_motion_y:
        The column name for the y-component of the neighbors' average motion vector.

    :returns:
        np.ndarray: A 1D array of normalized angles (0 to 1) for each row.
    """

    df[target_bac_motion_x_col] = df[target_bac_motion_x_col].astype(float)
    df[target_bac_motion_y_col] = df[target_bac_motion_y_col].astype(float)
    df[neighbors_avg_dir_motion_x] = df[neighbors_avg_dir_motion_x].astype(float)
    df[neighbors_avg_dir_motion_y] = df[neighbors_avg_dir_motion_y].astype(float)

    # Calculate the dot product
    dot_product = (df[target_bac_motion_x_col] * df[neighbors_avg_dir_motion_x] +
                   df[target_bac_motion_y_col] * df[neighbors_avg_dir_motion_y])

    # Calculate the magnitudes of the vectors
    magnitude_a = np.sqrt(np.power(df[target_bac_motion_x_col], 2) + np.power(df[target_bac_motion_y_col], 2))
    magnitude_b = np.sqrt(np.power(df[neighbors_avg_dir_motion_x], 2) + np.power(df[neighbors_avg_dir_motion_y], 2))

    # Calculate the cosine of the angle, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_angle = np.where((magnitude_a > 0) & (magnitude_b > 0),
                             dot_product / (magnitude_a * magnitude_b),
                             np.nan)

    # Calculate the angle in radians and then convert to degrees
    angle_radians = np.arccos(np.clip(cos_angle, -1, 1))
    angle_degrees = np.degrees(angle_radians)
    df['angle_degrees'] = angle_degrees / 180

    return df


def calculate_normalized_angle_between_motion_vectors(neighbor_motion_vectors, target_motion_vectors):
    """
    Calculates the normalized angle (0 to 1) between two sets of direction of motion vectors.

    This function computes the angles between corresponding direction of motion vectors in two arrays,
    normalizes the angles by dividing by 180, and returns the result.

    **Mathematical Explanation**:

    - **Dot Product**:
      The dot product of two vectors is a measure of how aligned they are. For two vectors:

      .. math::

          \\mathbf{v_1} = (x_1, y_1), \\quad \\mathbf{v_2} = (x_2, y_2)

      The dot product is given by:

      .. math::

          \\mathbf{v_1} \\cdot \\mathbf{v_2} = x_1 \\cdot x_2 + y_1 \\cdot y_2

    - **Magnitude of a Vector**:
      The magnitude (length) of a vector is calculated as:

      .. math::

          \\| \\mathbf{v} \\| = \\sqrt{x^2 + y^2}

    - **Cosine of the Angle**:
      The cosine of the angle \\( \\theta \\) between two vectors is computed as:

      .. math::

          \\cos(\\theta) = \\frac{\\mathbf{v_1} \\cdot \\mathbf{v_2}}{\\| \\mathbf{v_1} \\| \\| \\mathbf{v_2} \\|}

      If either vector has zero magnitude, the cosine is undefined.

    - **Angle in Radians**:
      The angle \\( \\theta \\) in radians is obtained using the arccosine function:

      .. math::

          \\theta = \\arccos(\\cos(\\theta))

      To handle numerical precision, the cosine value is clipped to the range \\([-1, 1]\\).

    - **Convert Radians to Degrees**:
      The angle is converted from radians to degrees using the formula:

      .. math::

          \\text{Angle (degrees)} = \\frac{\\theta \\cdot 180}{\\pi}

    - **Normalized Angle**:
      Finally, the angle is normalized by dividing the degrees by 180:

      .. math::

          \\text{Normalized Angle} = \\frac{\\text{Angle (degrees)}}{180}

    :param np.ndarray neighbor_motion_vectors:
        A 2D array where each row represents a direction of motion vector for neighbors.
    :param np.ndarray target_motion_vectors:
        A 2D array where each row represents a vector for target motions.

    :returns:
        A 1D array of normalized angles (0 to 1) between the vectors.
    """

    # Dot product
    dot_product = np.sum(neighbor_motion_vectors * target_motion_vectors, axis=1)

    # Magnitudes
    magnitude_neighbors = np.linalg.norm(neighbor_motion_vectors, axis=1)

    magnitude_direction = np.linalg.norm(target_motion_vectors, axis=1)

    # Safe division (avoid division by zero)
    valid_indices = (magnitude_neighbors != 0) & (magnitude_direction != 0)
    cos_angle = np.full(dot_product.shape, np.nan)  # Initialize with NaNs

    cos_angle[valid_indices] = dot_product[valid_indices] / (
            magnitude_neighbors[valid_indices] * magnitude_direction[valid_indices])

    # Calculate the angle in radians and then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)

    angle_between_motions = angle_degrees / 180  # Normalize by 180

    return angle_between_motions


def calculate_angles_between_slopes(slope_a, slope_b):
    """
    Calculates the orientation angles between two slopes in degrees.

    This function computes the angle in degrees between two lines defined by their slopes.
    NaN values are replaced with 90 degrees to handle undefined cases (e.g., perpendicular lines).



    **Mathematical Explanation**:

    - **Angle Between Two Slopes**:
      The angle :math:`\\theta` between two lines with slopes :math:`m_1` and :math:`m_2` is given by:

      .. math::

          \\theta = \\arctan\\left( \\left| \\frac{m_1 - m_2}{1 + m_1 m_2} \\right| \\right)

      where:
        :math:`m_1` is the slope of the first line.
        :math:`m_2` is the slope of the second line.

    - **Convert Radians to Degrees**:
      The angle in degrees is calculated using:

      .. math::

          \\text{Angle (degrees)} = \\frac{\\theta \\cdot 180}{\\pi}

    - **Handling Undefined Cases**:
      When :math:`1 + m_1 m_2 = 0`, the angle is undefined (lines are perpendicular). In such cases,
      NaN values are replaced with 90 degrees.

    :param np.ndarray slope_a:
        An array of slope values for the first set of lines.
    :param np.ndarray slope_b:
        An array of slope values for the second set of lines.

    :returns:
        An array of angles in degrees between the lines.
    """

    # Calculate the angle in radians between the lines
    angle_radians = np.arctan(abs((slope_a - slope_b) / (1 + slope_a * slope_b)))
    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)

    # Replace NaN values with 90
    angle_degrees = np.nan_to_num(angle_degrees, nan=90)

    return angle_degrees


def calc_distance_matrix(target_bac_coords_df, reference_bac_coords_df, x_col, y_col):
    """
    Calculates a distance matrix between two sets of bacteria coordinates.

    This function computes the pairwise Euclidean distances between the coordinates of
    target bacteria and reference bacteria

    :param pandas.DataFrame target_bac_coords_df:
        A DataFrame containing the coordinates of the target bacteria.
    :param pandas.DataFrame reference_bac_coords_df:
        A DataFrame containing the coordinates of the reference bacteria.
    :param str x_col:
        The column name for the x-coordinate in both DataFrames.
    :param str y_col:
        The column name for the y-coordinate in both DataFrames.

    :returns:
        A pandas DataFrame representing the distance matrix, where rows correspond to
        `target_bac_coords_df` indices and columns correspond to `reference_bac_coords_df` indices.
    """

    distance_df = pd.DataFrame(distance_matrix(target_bac_coords_df[[x_col, y_col]].values,
                                               reference_bac_coords_df[[x_col, y_col]].values),
                               index=target_bac_coords_df.index, columns=reference_bac_coords_df.index)

    return distance_df


def convert_end_points_um_to_pixel(end_points, um_per_pixel=0.144):
    """
    Converts endpoint coordinates from micrometers (um) to pixel units based on a specified conversion factor.

    :param list or np.array end_points:
        A list or array of endpoint coordinates in micrometers to be converted.
    :param float um_per_pixel:
        The conversion factor representing micrometers per pixel (default: 0.144).

    :returns:
        A numpy array of endpoint coordinates converted to pixel units.
    """

    end_points = np.array(end_points) / um_per_pixel

    return end_points


def convert_um_to_pixel(major_len, radius, endpoints, center_pos, pixel_per_micron=0.144):
    """
    Converts physical measurements in pixel to micrometers (um) based on a specified conversion factor.

    :param float major_len:
        The length measurement in um to be converted.
    :param float radius:
        The radius measurement in um to be converted.
    :param list or np.array endpoints:
        A list or array of endpoint coordinates in um.
    :param list or np.array center_pos:
        A list or array of center coordinate in um.
    :param float pixel_per_micron:
        The conversion factor representing pixel per micrometers(default: 0.144).

    :returns:
        A tuple containing the converted major_len, radius, endpoints, center_pos in pixel units.
    """

    major_len = major_len / pixel_per_micron
    radius = radius / pixel_per_micron
    endpoints = np.array(endpoints) / pixel_per_micron
    center_pos = np.array(center_pos) / pixel_per_micron

    return major_len, radius, endpoints, center_pos


def convert_pixel_to_um(df, pixel_per_micron, all_rel_center_coord_cols):
    """
        Converts distance measurements from pixel units to micrometers (um)
        within a DataFrame based on a specified conversion factor.

        :param pandas.DataFrame df:
            The input DataFrame containing spatial and shape measurements in pixel units.
        :param float pixel_per_micron:
            The conversion factor representing micrometers per pixel.
        :param dict all_rel_center_coord_cols:
            A dictionary containing the column names for x and y center coordinates,
            e.g., {'x': ['Location_Center_X', 'AreaShape_Center_X'], 'y': ['Location_Center_Y', 'AreaShape_Center_Y']}.

        :returns:
            The modified DataFrame with measurements converted to micrometers.
        """

    # Convert distances to um

    df[all_rel_center_coord_cols['x']] *= pixel_per_micron

    df[all_rel_center_coord_cols['y']] *= pixel_per_micron

    df['AreaShape_MajorAxisLength'] *= pixel_per_micron
    df['AreaShape_MinorAxisLength'] *= pixel_per_micron

    return df


def identify_important_columns(df):
    """
    Identifies key columns related to center coordinates, labels, and parent objects
    from the provided DataFrame.

    :param pandas.DataFrame df:
        Input DataFrame containing columns with bacterial feature data, such as tracking information
        and spatial coordinates.

    :returns:
        tuple: A tuple containing:

        - **center_coord_cols** (*dict*): Primary center coordinate columns
          (e.g., {'x': 'Location_Center_X', 'y': 'Location_Center_Y'}).
        - **all_rel_center_coord_cols** (*dict*): All available center coordinate columns
          (e.g., {'x': ['Location_Center_X', 'AreaShape_Center_X'], 'y': ['Location_Center_Y', 'AreaShape_Center_Y']}).
        - **parent_image_number_col** (*str*): Name of the column for parent image number.
        - **parent_object_number_col** (*str*): Name of the column for parent object number.
        - **label_col** (*str*): Name of the column for object labels.


    """

    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    # label column name
    label_col = [col for col in df.columns if 'TrackObjects_Label_' in col][0]

    if 'Location_Center_X' in df.columns and 'Location_Center_Y' in df.columns:
        center_coord_cols = {'x': 'Location_Center_X', 'y': 'Location_Center_Y'}
        all_rel_center_coord_cols = {'x': ['Location_Center_X'], 'y': ['Location_Center_Y']}

        if 'AreaShape_Center_X' in df.columns and 'AreaShape_Center_Y' in df.columns:
            all_rel_center_coord_cols = {'x': ['Location_Center_X', 'AreaShape_Center_X'],
                                         'y': ['Location_Center_Y', 'AreaShape_Center_Y']}

    elif 'AreaShape_Center_X' in df.columns and 'AreaShape_Center_Y' in df.columns:
        center_coord_cols = {'x': 'AreaShape_Center_X', 'y': 'AreaShape_Center_Y'}
        all_rel_center_coord_cols = {'x': ['AreaShape_Center_X'], 'y': ['AreaShape_Center_Y']}

    else:
        raise ValueError('No column corresponding to the center of bacteria.')

    return (center_coord_cols, all_rel_center_coord_cols, parent_image_number_col, parent_object_number_col,
            label_col)
