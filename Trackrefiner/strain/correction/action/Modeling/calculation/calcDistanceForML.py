import numpy as np
import pandas as pd


def calc_distance(df, center_coordinate_columns, postfix_source, postfix_target, stat=None, df_view=None, both=True):

    # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)

    if df_view is not None:
        # Convert the required columns to numpy arrays with float32 to save memory
        x_center_source = df_view[center_coordinate_columns['x'] + postfix_source].to_numpy()
        y_center_source = df_view[center_coordinate_columns['y'] + postfix_source].to_numpy()

        x_endpoint1_source = df_view['endpoint1_X' + postfix_source].to_numpy()
        y_endpoint1_source = df_view['endpoint1_Y' + postfix_source].to_numpy()

        x_endpoint2_source = df_view['endpoint2_X' + postfix_source].to_numpy()
        y_endpoint2_source = df_view['endpoint2_Y' + postfix_source].to_numpy()

        x_center_target = df_view[center_coordinate_columns['x'] + postfix_target].to_numpy()
        y_center_target = df_view[center_coordinate_columns['y'] + postfix_target].to_numpy()

        x_endpoint1_target = df_view['endpoint1_X' + postfix_target].to_numpy()
        y_endpoint1_target = df_view['endpoint1_Y' + postfix_target].to_numpy()

        x_endpoint2_target = df_view['endpoint2_X' + postfix_target].to_numpy()
        y_endpoint2_target = df_view['endpoint2_Y' + postfix_target].to_numpy()
    else:
        x_center_source = df[center_coordinate_columns['x'] + postfix_source].to_numpy()
        y_center_source = df[center_coordinate_columns['y'] + postfix_source].to_numpy()

        x_endpoint1_source = df['endpoint1_X' + postfix_source].to_numpy()
        y_endpoint1_source = df['endpoint1_Y' + postfix_source].to_numpy()

        x_endpoint2_source = df['endpoint2_X' + postfix_source].to_numpy()
        y_endpoint2_source = df['endpoint2_Y' + postfix_source].to_numpy()

        x_center_target = df[center_coordinate_columns['x'] + postfix_target].to_numpy()
        y_center_target = df[center_coordinate_columns['y'] + postfix_target].to_numpy()

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

    if stat == 'div':

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

    if stat == 'div':
        stacked_arrays = np.stack([center_distance, endpoint1_1_distance, endpoint2_2_distance])
        df['min_distance_same'] = np.min(stacked_arrays, axis=0)

    return df
