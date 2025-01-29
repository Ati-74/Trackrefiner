import pandas as pd


def filter_objects_outside_boundary(x_min, x_max, y_min, y_max, df, pixel_per_micron, center_coord_cols):

    """
    Identifies and marks objects outside a user-defined boundary in the image, applying the same boundary
    limits across all time steps.

    :param float x_min:
        The minimum x-coordinate (in micrometers) for the boundary.
    :param float x_max:
        The maximum x-coordinate (in micrometers) for the boundary.
    :param float y_min:
        The minimum y-coordinate (in micrometers) for the boundary.
    :param float y_max:
        The maximum y-coordinate (in micrometers) for the boundary.
    :param pandas.DataFrame df:
        The DataFrame containing tracking data and measured bacterial features for objects.
        This DataFrame is updated in place.
    :param float pixel_per_micron:
        The conversion factor from pixels to micrometers.
    :param dict center_coord_cols:
        A dictionary specifying the column names for the x and y coordinates of object centroids
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).

    :returns:
        pandas.DataFrame: The updated DataFrame with objects outside the boundary marked as noise (`noise_bac=True`).
    """

    x_min = x_min * pixel_per_micron
    x_max = x_max * pixel_per_micron
    y_min = y_min * pixel_per_micron
    y_max = y_max * pixel_per_micron

    wall_objects = df.loc[(df[center_coord_cols['x']] < x_min) |
                          (df[center_coord_cols['x']] > x_max) |
                          (df[center_coord_cols['y']] < y_min) |
                          (df[center_coord_cols['y']] > y_max)
                          ]

    df.loc[wall_objects['prev_index'].values, 'noise_bac'] = True

    return df


def filter_objects_outside_dynamic_boundary(dynamic_boundaries, df, pixel_per_micron, center_coord_cols):

    """
    Identifies and marks objects outside user-defined, time-dependent boundaries in the image.

    :param pandas.DataFrame dynamic_boundaries:
        A DataFrame containing time-dependent boundary limits for each time step. It includes columns specifying
        lower and upper limits for x and y coordinates (`Lower X Limit`, `Upper X Limit`, `Lower Y Limit`,
        `Upper Y Limit`) and the corresponding time step (`Time Step`).
    :param pandas.DataFrame df:
        The main DataFrame containing tracking data and measured bacterial features for all objects.
        This DataFrame is updated in place.
    :param float pixel_per_micron:
        The conversion factor from micrometers to pixels.
    :param dict center_coord_cols:
        A dictionary specifying the column names for the x and y coordinates of object centroids
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).

    :returns:
        pandas.DataFrame: The updated DataFrame with objects outside the boundaries marked as noise (`noise_bac=True`).
    """

    cols_related_to_boundary = ['Lower X Limit', 'Upper X Limit', 'Lower Y Limit', 'Upper Y Limit']
    dynamic_boundaries[cols_related_to_boundary] = \
        dynamic_boundaries[cols_related_to_boundary] * pixel_per_micron

    incorrect_object_idx_list = []

    for row_idx, row in dynamic_boundaries.iterrows():

        sel_time_step = row['Time Step']
        x_min, x_max, y_min, y_max = row[cols_related_to_boundary]

        current_df = df.loc[df['ImageNumber'] == sel_time_step]
        wall_objects = current_df.loc[(current_df[center_coord_cols['x']] < x_min) |
                                      (current_df[center_coord_cols['x']] > x_max) |
                                      (current_df[center_coord_cols['y']] < y_min) |
                                      (current_df[center_coord_cols['y']] > y_max)]

        incorrect_object_idx_list.extend(wall_objects['prev_index'].values)

    df.loc[incorrect_object_idx_list, 'noise_bac'] = True

    return df


def find_inside_boundary_objects(boundary_limits, dynamic_boundaries, df, center_coord_cols, pixel_per_micron):
    """
    Identifies and retains objects within specified boundary limits, either static or time-dependent,
    by filtering out objects outside these boundaries.

    :param str boundary_limits:
        A comma-separated string defining static boundary limits in micrometers as `x_min, x_max, y_min, y_max`.
        If provided, these limits are applied consistently across all time steps.
    :param str dynamic_boundaries:
        Path to a CSV file containing time-dependent boundary limits. The CSV must include columns specifying
        `Lower X Limit`, `Upper X Limit`, `Lower Y Limit`, `Upper Y Limit`, and `Time Step`. This parameter is
        used if `boundary_limits` is `None`.
    :param pandas.DataFrame df:
        The main DataFrame containing tracking data and measured bacterial features for all objects.
        This DataFrame is updated in place.
    :param dict center_coord_cols:
        A dictionary specifying the column names for the x and y coordinates of object centroids
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param float pixel_per_micron:
        The conversion factor from micrometers to pixels.

    :returns:
        pandas.DataFrame: The updated DataFrame with objects outside the boundaries marked as noise (`noise_bac=True`)
    """

    if boundary_limits is not None:
        boundary_limits = [float(v.strip()) for v in boundary_limits.split(',')]
        x_min, x_max, y_min, y_max = boundary_limits

        df = filter_objects_outside_boundary(x_min, x_max, y_min, y_max, df, pixel_per_micron,
                                             center_coord_cols)

    else:
        boundary_limits_per_time_step_df = pd.read_csv(dynamic_boundaries)

        df = filter_objects_outside_dynamic_boundary(boundary_limits_per_time_step_df, df, pixel_per_micron,
                                                     center_coord_cols)

    return df
