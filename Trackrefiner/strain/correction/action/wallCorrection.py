import numpy as np


def find_wall_objects(boundary_limits, df, neighbors_df, center_coordinate_columns, parent_image_number_col,
                      parent_object_number_col, um_per_pixel):

    boundary_limits = [float(v.strip()) for v in boundary_limits.split(',')]
    x_min, x_max, y_min, y_max = boundary_limits

    x_min = x_min * um_per_pixel
    x_max = x_max * um_per_pixel
    y_min = y_min * um_per_pixel
    y_max = y_max * um_per_pixel

    correct_objects = df.loc[(df[center_coordinate_columns['x']] > x_min) &
                             (df[center_coordinate_columns['x']] < x_max) &
                             (df[center_coordinate_columns['y']] > y_min) &
                             (df[center_coordinate_columns['y']] < y_max)
                             ]

    wall_objects = df.loc[(df[center_coordinate_columns['x']] <= x_min) |
                          (df[center_coordinate_columns['x']] >= x_max) |
                          (df[center_coordinate_columns['y']] <= y_min) |
                          (df[center_coordinate_columns['y']] >= y_max)
                          ]

    correct_bac_with_wall_objects_as_parent = wall_objects.merge(correct_objects,
                                                                 left_on=['ImageNumber', 'ObjectNumber'],
                                                                 right_on=[parent_image_number_col,
                                                                           parent_object_number_col],
                                                                 suffixes=('', '_correct_obj'), how='inner')

    if correct_bac_with_wall_objects_as_parent.shape[0] > 0:
        correct_objects.loc[correct_bac_with_wall_objects_as_parent['index_correct_obj'], [
            parent_image_number_col, parent_object_number_col]] = 0

    # now we should correct relationship

    neighbors_df['index_neighborhood'] = neighbors_df.index

    neighbors_df_incorrect_source = neighbors_df.merge(wall_objects,
                                                       left_on=['First Image Number', 'First Object Number'],
                                                       right_on=['ImageNumber', 'ObjectNumber'], how='inner')

    neighbors_df_incorrect_target = neighbors_df.merge(wall_objects,
                                                       left_on=['Second Image Number', 'Second Object Number'],
                                                       right_on=['ImageNumber', 'ObjectNumber'], how='inner')

    incorrect_ndx = np.unique(np.concatenate((neighbors_df_incorrect_source['index_neighborhood'].values,
                                              neighbors_df_incorrect_target['index_neighborhood'].values)))

    neighbors_df = neighbors_df.loc[~ neighbors_df['index_neighborhood'].isin(incorrect_ndx)]

    return correct_objects, neighbors_df
