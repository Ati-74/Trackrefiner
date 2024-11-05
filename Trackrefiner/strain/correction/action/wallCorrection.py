import numpy as np
import pandas as pd


def remove_wall_objects_same_boundary(x_min, x_max, y_min, y_max, df, um_per_pixel, center_coordinate_columns):
    x_min = x_min * um_per_pixel
    x_max = x_max * um_per_pixel
    y_min = y_min * um_per_pixel
    y_max = y_max * um_per_pixel

    wall_objects = df.loc[(df[center_coordinate_columns['x']] < x_min) |
                          (df[center_coordinate_columns['x']] > x_max) |
                          (df[center_coordinate_columns['y']] < y_min) |
                          (df[center_coordinate_columns['y']] > y_max)
                          ]

    df.loc[wall_objects['prev_index'].values, 'noise_bac'] = True

    return df


def remove_wall_objects_different_boundary(boundary_limits_per_time_step_df, df, um_per_pixel,
                                           center_coordinate_columns):

    cols_related_to_boundary = ['Lower X Limit', 'Upper X Limit', 'Lower Y Limit', 'Upper Y Limit']
    boundary_limits_per_time_step_df[cols_related_to_boundary] = \
        boundary_limits_per_time_step_df[cols_related_to_boundary] * um_per_pixel

    incorrect_object_idx_list = []

    for row_idx, row in boundary_limits_per_time_step_df.iterrows():

        sel_time_step = row['Time Step']
        x_min, x_max, y_min, y_max = row[cols_related_to_boundary]

        current_df = df.loc[df['ImageNumber'] == sel_time_step]
        wall_objects = current_df.loc[(current_df[center_coordinate_columns['x']] < x_min) |
                                      (current_df[center_coordinate_columns['x']] > x_max) |
                                      (current_df[center_coordinate_columns['y']] < y_min) |
                                      (current_df[center_coordinate_columns['y']] > y_max)]

        incorrect_object_idx_list.extend(wall_objects['prev_index'].values)

    df.loc[incorrect_object_idx_list, 'noise_bac'] = True

    return df


def find_wall_objects(boundary_limits, boundary_limits_per_time_step, df, center_coordinate_columns, um_per_pixel):

    if boundary_limits is not None:
        boundary_limits = [float(v.strip()) for v in boundary_limits.split(',')]
        x_min, x_max, y_min, y_max = boundary_limits

        df = remove_wall_objects_same_boundary(x_min, x_max, y_min, y_max, df, um_per_pixel, center_coordinate_columns)

    else:
        boundary_limits_per_time_step_df = pd.read_csv(boundary_limits_per_time_step)

        df = remove_wall_objects_different_boundary(boundary_limits_per_time_step_df, df, um_per_pixel,
                                                    center_coordinate_columns)

    return df
