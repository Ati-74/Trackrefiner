import numpy as np
import pandas as pd


def calc_distance(df, center_coordinate_columns, postfix_source, postfix_target, stat=None, df_view=None):

    dis_cols = ['center_distance', 'endpoint1_1_distance', 'endpoint2_2_distance']
    dis_col_same = dis_cols.copy()

    # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)
    if df_view is not None:
        df['center_distance'] = \
            np.linalg.norm(
                df_view[[center_coordinate_columns['x'] + postfix_source,
                    center_coordinate_columns['y'] + postfix_source]].values -
                df_view[[center_coordinate_columns['x'] + postfix_target,
                    center_coordinate_columns['y'] + postfix_target]].values,
                axis=1)

        df['endpoint1_1_distance'] = \
            np.linalg.norm(df_view[['endpoint1_X' + postfix_source, 'endpoint1_Y' + postfix_source]].values -
                           df_view[['endpoint1_X' + postfix_target, 'endpoint1_Y' + postfix_target]].values, axis=1)

        df['endpoint2_2_distance'] = \
            np.linalg.norm(df_view[['endpoint2_X' + postfix_source, 'endpoint2_Y' + postfix_source]].values -
                           df_view[['endpoint2_X' + postfix_target, 'endpoint2_Y' + postfix_target]].values, axis=1)

        if stat == 'div':
            dis_cols.extend(['endpoint1_2_distance', 'endpoint2_1_distance', 'center_endpoint1_distance',
                             'center_endpoint2_distance'])

            df['endpoint1_2_distance'] = \
                np.linalg.norm(df_view[['endpoint1_X' + postfix_source, 'endpoint1_Y' + postfix_source]].values -
                               df_view[['endpoint2_X' + postfix_target, 'endpoint2_Y' + postfix_target]].values,
                               axis=1)

            df['endpoint2_1_distance'] = \
                np.linalg.norm(df_view[['endpoint2_X' + postfix_source, 'endpoint2_Y' + postfix_source]].values -
                               df_view[['endpoint1_X' + postfix_target, 'endpoint1_Y' + postfix_target]].values,
                               axis=1)

            df['center_endpoint1_distance'] = \
                np.linalg.norm(df_view[[center_coordinate_columns['x'] + postfix_source,
                                   center_coordinate_columns['y'] + postfix_source]].values -
                               df_view[['endpoint1_X' + postfix_target, 'endpoint1_Y' + postfix_target]].values,
                               axis=1)

            df['center_endpoint2_distance'] = \
                np.linalg.norm(df_view[[center_coordinate_columns['x'] + postfix_source,
                                   center_coordinate_columns['y'] + postfix_source]].values -
                               df_view[['endpoint2_X' + postfix_target, 'endpoint2_Y' + postfix_target]].values, axis=1)

    else:

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


def new_calc_distance(df, center_coordinate_columns, postfix_source, postfix_target, stat=None):

    distance_df = pd.DataFrame(index=df.index)

    dis_cols = ['center_distance', 'endpoint1_1_distance', 'endpoint2_2_distance']
    dis_col_same = dis_cols.copy()

    # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)
    distance_df['center_distance'] = \
        np.linalg.norm(
            df[[center_coordinate_columns['x'] + postfix_source,
                center_coordinate_columns['y'] + postfix_source]].values -
            df[[center_coordinate_columns['x'] + postfix_target,
                center_coordinate_columns['y'] + postfix_target]].values,
            axis=1)

    distance_df['endpoint1_1_distance'] = \
        np.linalg.norm(df[['endpoint1_X' + postfix_source, 'endpoint1_Y' + postfix_source]].values -
                       df[['endpoint1_X' + postfix_target, 'endpoint1_Y' + postfix_target]].values, axis=1)

    distance_df['endpoint2_2_distance'] = \
        np.linalg.norm(df[['endpoint2_X' + postfix_source, 'endpoint2_Y' + postfix_source]].values -
                       df[['endpoint2_X' + postfix_target, 'endpoint2_Y' + postfix_target]].values, axis=1)

    if stat == 'div':

        dis_cols.extend(['endpoint1_2_distance', 'endpoint2_1_distance', 'center_endpoint1_distance',
                         'center_endpoint2_distance'])

        distance_df['endpoint1_2_distance'] = \
            np.linalg.norm(df[['endpoint1_X' + postfix_source, 'endpoint1_Y' + postfix_source]].values -
                           df[['endpoint2_X' + postfix_target, 'endpoint2_Y' + postfix_target]].values,
                           axis=1)

        distance_df['endpoint2_1_distance'] = \
            np.linalg.norm(df[['endpoint2_X' + postfix_source, 'endpoint2_Y' + postfix_source]].values -
                           df[['endpoint1_X' + postfix_target, 'endpoint1_Y' + postfix_target]].values,
                           axis=1)

        distance_df['center_endpoint1_distance'] = \
            np.linalg.norm(df[[center_coordinate_columns['x'] + postfix_source,
                               center_coordinate_columns['y'] + postfix_source]].values -
                           df[['endpoint1_X' + postfix_target, 'endpoint1_Y' + postfix_target]].values,
                           axis=1)

        distance_df['center_endpoint2_distance'] = \
            np.linalg.norm(df[[center_coordinate_columns['x'] + postfix_source,
                               center_coordinate_columns['y'] + postfix_source]].values -
                           df[['endpoint2_X' + postfix_target, 'endpoint2_Y' + postfix_target]].values, axis=1)

    distance_df['min_distance'] = distance_df[dis_cols].min(axis=1)

    if stat == 'div':
        distance_df['min_distance_same'] = distance_df[dis_col_same].min(axis=1)

    return distance_df
