import numpy as np


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
    condition1 = (np.abs(df["Orientation"] - np.pi / 2) < angle_tolerance) | \
                 (np.abs(df["Orientation"] + np.pi / 2) < angle_tolerance)

    df.loc[condition1, 'endpoint1_X'] = df[center_coord_cols['x']]
    df.loc[condition1, 'endpoint1_Y'] = df[center_coord_cols['y']] + df["Major_axis"]
    df.loc[condition1, 'endpoint2_X'] = df[center_coord_cols['x']]
    df.loc[condition1, 'endpoint2_Y'] = df[center_coord_cols['y']] - df["Major_axis"]

    # Bacteria parallel to the horizontal axis
    condition2 = np.abs(df["Orientation"]) < angle_tolerance
    condition3 = ~condition1 & condition2
    df.loc[condition3, 'endpoint1_X'] = df[center_coord_cols['x']] + df["Major_axis"]
    df.loc[condition3, 'endpoint1_Y'] = df[center_coord_cols['y']]
    df.loc[condition3, 'endpoint2_X'] = df[center_coord_cols['x']] - df["Major_axis"]
    df.loc[condition3, 'endpoint2_Y'] = df[center_coord_cols['y']]

    # (x- center_x) * np.sin(angle_rotation) - (y-center_y) * np.cos(angle_rotation) = 0
    # np.power((x - center_x) * np.cos(angle_rotation) + (y - center_y) * np.sin(angle_rotation), 2) =
    # np.power(major, 2)
    condition4 = ~condition1 & ~condition2
    other_bac_df = df.loc[condition4][["Major_axis", "Orientation",
                                       center_coord_cols['x'], center_coord_cols['y']]].copy()

    other_bac_df['semi_major'] = df["Major_axis"] / 2
    other_bac_df['temp_vertex_1_x'] = \
        (other_bac_df['semi_major'] * np.cos(other_bac_df["Orientation"]) +
         other_bac_df[center_coord_cols['x']])

    other_bac_df['temp_vertex_1_y'] = \
        ((other_bac_df['temp_vertex_1_x'] - other_bac_df[center_coord_cols['x']]) *
         np.tan(other_bac_df["Orientation"]) + other_bac_df[center_coord_cols['y']])

    other_bac_df['temp_vertex_2_x'] = \
        (-other_bac_df['semi_major'] * np.cos(other_bac_df["Orientation"]) +
         other_bac_df[center_coord_cols['x']])

    other_bac_df['temp_vertex_2_y'] = \
        ((other_bac_df['temp_vertex_2_x'] - other_bac_df[center_coord_cols['x']]) *
         np.tan(other_bac_df["Orientation"]) + other_bac_df[center_coord_cols['y']])

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


def calculate_trajectory_angles(df, col1, col2):
    """
    Calculates the angles of trajectory directions between two objects.

    This function computes the direction vector between two bacteria and calculates the angle of that direction in
    radians. The angles are measured counterclockwise from the positive x-axis.

    :param pandas.DataFrame df:
        A DataFrame containing the x and y coordinates for two positions of objects.
        These positions are used to calculate the trajectory direction.
    :param str col1:
        center x column name
    :param str col2:
        center y column name

    :returns:
        np.ndarray: A 1D array of angles (in radians) corresponding to the direction of trajectory for
        each object.
    """

    # Calculate the direction vector from the previous position to the current position
    trajectory_direction = \
        (df[[f"{col1}", f"{col2}"]].values -
         df[[f'prev_{col1}', f'prev_{col2}']].values)

    # Calculate the angle of the direction vector
    angle = np.arctan2(trajectory_direction[:, 1], trajectory_direction[:, 0])

    return angle
