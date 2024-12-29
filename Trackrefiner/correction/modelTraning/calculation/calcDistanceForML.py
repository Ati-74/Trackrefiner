import numpy as np


def calc_min_distance_ml(df, center_coord_cols, postfix_source, postfix_target, link_type=None, df_view=None):

    """
    Computes the minimum distance between source and target bacteria based on their centroid
    and endpoint coordinates. Supports multiple distance calculations, including distances for
    division-specific comparisons.

    **Details**:
        - **Centroid Distance**: Euclidean distance between the centroids of source and target bacteria.
        - **Endpoint Distances**:
            - Endpoint1-to-Endpoint1 and Endpoint2-to-Endpoint2.
            - For division-specific calculations, additional distances are considered:
              - Endpoint1-to-Endpoint2 and vice versa.
              - Centroid-to-Endpoints for cross-movement.

    :param pandas.DataFrame df:
        The main DataFrame containing bacterial data.
    :param dict center_coord_cols:
        A dictionary specifying the column names for the x and y coordinates of bacterial centroids
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param str postfix_source:
        Suffix used to identify source-related columns in the DataFrame.
    :param str postfix_target:
        Suffix used to identify target-related columns in the DataFrame.
    :param str link_type:
        Specifies the type of distance calculation:
        - `'div'`: Includes additional cross-movement distances (e.g., endpoint-to-endpoint).
        - `None`: Computes only basic distances (centroid-to-centroid and same-endpoint distances).
    :param pandas.DataFrame df_view:
        A subset of the main DataFrame for efficient computation (optional).

    **Returns**:
        pandas.DataFrame:

        The updated DataFrame with the following additional columns:
            - `'min_distance'`: Minimum distance between source and target considering all point pairs.
            - `'min_distance_continuity'` (if `stat='div'`): Minimum distance excluding cross-movement distances.
    """

    if df_view is not None:
        # Convert the required columns to numpy arrays with float32 to save memory
        x_center_source = df_view[center_coord_cols['x'] + postfix_source].to_numpy()
        y_center_source = df_view[center_coord_cols['y'] + postfix_source].to_numpy()

        x_endpoint1_source = df_view['endpoint1_X' + postfix_source].to_numpy()
        y_endpoint1_source = df_view['endpoint1_Y' + postfix_source].to_numpy()

        x_endpoint2_source = df_view['endpoint2_X' + postfix_source].to_numpy()
        y_endpoint2_source = df_view['endpoint2_Y' + postfix_source].to_numpy()

        x_center_target = df_view[center_coord_cols['x'] + postfix_target].to_numpy()
        y_center_target = df_view[center_coord_cols['y'] + postfix_target].to_numpy()

        x_endpoint1_target = df_view['endpoint1_X' + postfix_target].to_numpy()
        y_endpoint1_target = df_view['endpoint1_Y' + postfix_target].to_numpy()

        x_endpoint2_target = df_view['endpoint2_X' + postfix_target].to_numpy()
        y_endpoint2_target = df_view['endpoint2_Y' + postfix_target].to_numpy()
    else:
        x_center_source = df[center_coord_cols['x'] + postfix_source].to_numpy()
        y_center_source = df[center_coord_cols['y'] + postfix_source].to_numpy()

        x_endpoint1_source = df['endpoint1_X' + postfix_source].to_numpy()
        y_endpoint1_source = df['endpoint1_Y' + postfix_source].to_numpy()

        x_endpoint2_source = df['endpoint2_X' + postfix_source].to_numpy()
        y_endpoint2_source = df['endpoint2_Y' + postfix_source].to_numpy()

        x_center_target = df[center_coord_cols['x'] + postfix_target].to_numpy()
        y_center_target = df[center_coord_cols['y'] + postfix_target].to_numpy()

        x_endpoint1_target = df['endpoint1_X' + postfix_target].to_numpy()
        y_endpoint1_target = df['endpoint1_Y' + postfix_target].to_numpy()

        x_endpoint2_target = df['endpoint2_X' + postfix_target].to_numpy()
        y_endpoint2_target = df['endpoint2_Y' + postfix_target].to_numpy()

    center_distance = np.sqrt((x_center_source - x_center_target) ** 2 +
                              (y_center_source - y_center_target) ** 2)

    endpoint1_1_distance = \
        np.sqrt((x_endpoint1_source - x_endpoint1_target) ** 2 +
                (y_endpoint1_source - y_endpoint1_target) ** 2)

    endpoint2_2_distance = \
        np.sqrt((x_endpoint2_source - x_endpoint2_target) ** 2 +
                (y_endpoint2_source - y_endpoint2_target) ** 2)

    if link_type == 'div':

        endpoint1_2_distance = \
            np.sqrt((x_endpoint1_source - x_endpoint2_target) ** 2 +
                    (y_endpoint1_source - y_endpoint2_target) ** 2)

        endpoint2_1_distance = \
            np.sqrt((x_endpoint2_source - x_endpoint1_target) ** 2 +
                    (y_endpoint2_source - y_endpoint1_target) ** 2)

        center_endpoint1_distance = \
            np.sqrt((x_center_source - x_endpoint1_target) ** 2 +
                    (y_center_source - y_endpoint1_target) ** 2)

        center_endpoint2_distance = \
            np.sqrt((x_center_source - x_endpoint2_target) ** 2 +
                    (y_center_source - y_endpoint2_target) ** 2)

        stacked_arrays = np.stack([center_distance, endpoint1_1_distance, endpoint2_2_distance,
                                   endpoint1_2_distance, endpoint2_1_distance,
                                   center_endpoint1_distance, center_endpoint2_distance])

    else:
        stacked_arrays = np.stack([center_distance, endpoint1_1_distance, endpoint2_2_distance])

    df['min_distance'] = np.min(stacked_arrays, axis=0)

    if link_type == 'div':
        stacked_arrays = np.stack([center_distance, endpoint1_1_distance, endpoint2_2_distance])
        df['min_distance_continuity'] = np.min(stacked_arrays, axis=0)

    return df
