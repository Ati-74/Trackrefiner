import numpy as np


def calc_distance(df, center_coordinate_columns, postfix_source, postfix_target, stat=None):

    dis_cols = ['center_distance', 'endpoint1_1_distance', 'endpoint2_2_distance']
    dis_col_same = dis_cols.copy()

    # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)
    df['center_distance'] = \
        np.linalg.norm(
            df[[center_coordinate_columns['x'] + postfix_source,
                center_coordinate_columns['y'] + postfix_source]].values -
            df[[center_coordinate_columns['x'] + postfix_target,
                center_coordinate_columns['y'] + postfix_target]].values,
            axis=1)

    df['endpoint1_1_distance'] = \
        np.linalg.norm(df[['endpoint1_X' + postfix_source, 'endpoint1_Y' + postfix_source]].values -
                       df[['endpoint1_X' + postfix_target, 'endpoint1_Y' + postfix_target]].values, axis=1)

    df['endpoint2_2_distance'] = \
        np.linalg.norm(df[['endpoint2_X' + postfix_source, 'endpoint2_Y' + postfix_source]].values -
                       df[['endpoint2_X' + postfix_target, 'endpoint2_Y' + postfix_target]].values, axis=1)

    if stat == 'div':

        dis_cols.extend(['endpoint1_2_distance', 'endpoint2_1_distance', 'center_endpoint1_distance',
                         'center_endpoint2_distance'])

        df['endpoint1_2_distance'] = \
            np.linalg.norm(df[['endpoint1_X' + postfix_source, 'endpoint1_Y' + postfix_source]].values -
                           df[['endpoint2_X' + postfix_target, 'endpoint2_Y' + postfix_target]].values,
                           axis=1)

        df['endpoint2_1_distance'] = \
            np.linalg.norm(df[['endpoint2_X' + postfix_source, 'endpoint2_Y' + postfix_source]].values -
                           df[['endpoint1_X' + postfix_target, 'endpoint1_Y' + postfix_target]].values,
                           axis=1)

        df['center_endpoint1_distance'] = \
            np.linalg.norm(df[[center_coordinate_columns['x'] + postfix_source,
                               center_coordinate_columns['y'] + postfix_source]].values -
                           df[['endpoint1_X' + postfix_target, 'endpoint1_Y' + postfix_target]].values,
                           axis=1)

        df['center_endpoint2_distance'] = \
            np.linalg.norm(df[[center_coordinate_columns['x'] + postfix_source,
                               center_coordinate_columns['y'] + postfix_source]].values -
                           df[['endpoint2_X' + postfix_target, 'endpoint2_Y' + postfix_target]].values, axis=1)

    df['min_distance'] = df[dis_cols].min(axis=1)

    if stat == 'div':
        df['min_distance_same'] = df[dis_col_same].min(axis=1)

    return df
