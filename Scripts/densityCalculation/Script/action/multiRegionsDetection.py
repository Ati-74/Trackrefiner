import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.spatial import distance_matrix
from Scripts.densityCalculation.Script.action.multiRegionsCorrection import multi_region_correction


def generate_new_color(existing_colors, seed=None):
    """Generate a new color not in existing_colors."""
    np.random.seed(seed)  # Optional: for reproducible colors
    while True:
        new_color = np.random.randint(0, 256, size=3) / 256
        # Use broadcasting and np.all to efficiently compare the new_color against all existing_colors
        if not np.any(np.all(np.isclose(existing_colors, new_color), axis=1)):
            return new_color


def multi_region_detection(df, sorted_seg_npy_files_list, pixel_per_micron, center_coord_cols,
                           all_rel_center_coord_cols, parent_image_number_col, parent_object_number_col,
                           time_step_list_this_dataset):

    """
    Map objects to segmentation files, detect multi-region areas, and prepare pixel data for tacking correction.

    This function maps objects from the tracking dataframe (`df`) to their corresponding regions in segmentation
    files (`npy` format). It detects multi-region areas where disconnected regions share the same color and flags
    them for correction in a separate function. For all objects, their pixels are mapped into a sparse coordinate
    matrix and color array for further analysis.

    :param pd.DataFrame df:
        Dataframe containing bacterial measured bacterial features.
    :param list sorted_seg_npy_files_list:
        sorted List of file paths to the segmentation files in `.npy` format.
    :param float pixel_per_micron:
        Conversion factor from pixels to microns for spatial measurements.
    :param dict center_coord_cols:
        Dictionary specifying the column names for x and y coordinates of object centers.
        Example: {'x': 'Center_X', 'y': 'Center_Y'}
    :param dict all_rel_center_coord_cols:
        Dictionary containing x and y coordinate column names.
    :param str parent_image_number_col:
        Column name for the parent image number in the dataframe.
    :param str parent_object_number_col:
        Column name for the parent object number in the dataframe.
    :param list time_step_list_this_dataset:
        List of time steps the user wants to map to related objects in segmentation files.

    :return:

        - **df** (*pd.DataFrame*): Updated dataframe with mapped object data (pixel locations of objects).
    """

    # coordinate_array = np.zeros(df.shape[0], dtype=object)

    for img_ndx, img_npy_file in enumerate(sorted_seg_npy_files_list):

        current_df = df.loc[df['ImageNumber'] == img_ndx + 1]

        img_number = img_ndx + 1

        if img_number in time_step_list_this_dataset:

            img_array = np.load(img_npy_file)

            # Flatten the image to get a list of RGB values
            rgb_values = img_array.reshape(-1, 3)

            unique_colors, indices = np.unique(rgb_values, axis=0, return_index=True)

            all_colors = unique_colors.copy()

            # Exclude the background color (0,0,0)
            unique_colors = np.delete(unique_colors, 0, axis=0)

            regions_center_raw_objects = []
            regions_center_prev_objects_stat = []
            regions_center_particles = []
            regions_center_particles_stat = []
            regions_features = []
            regions_color = []
            regions_coordinates = []
            multi_region_flag = False

            for color in unique_colors:
                # Create a mask for the current color
                mask = np.all(img_array == color.reshape(1, 1, 3), axis=2)

                # Label the regions
                labeled_mask = label(mask)
                regions = regionprops(labeled_mask)

                if len(regions) > 1:

                    multi_region_flag = True

                    labeled_mask_cp = labeled_mask.copy()

                    # convert to one connected region
                    labeled_mask_cp[labeled_mask_cp > 1] = 1
                    regions_cp = regionprops(labeled_mask_cp)

                    y0_cp, x0_cp = regions_cp[0].centroid

                    regions_center_raw_objects.append((x0_cp * pixel_per_micron, y0_cp * pixel_per_micron))
                    regions_center_prev_objects_stat.append('multi')

                    # Separate regions
                    for region_ndx, region in enumerate(regions):

                        # fetching information
                        y0, x0 = region.centroid
                        orientation = region.orientation
                        orientation = orientation * (180 / np.pi)
                        major_length = region.major_axis_length * pixel_per_micron
                        minor_length = region.minor_axis_length * pixel_per_micron

                        regions_center_particles.append((x0 * pixel_per_micron, y0 * pixel_per_micron))
                        regions_center_particles_stat.append('particle')

                        regions_features.append({'center_x': x0 * pixel_per_micron, 'center_y': y0 * pixel_per_micron,
                                                 'orientation': orientation, 'major': major_length,
                                                 'minor': minor_length})

                        # Generate a new color that's not already in unique_colors
                        new_color = generate_new_color(all_colors)
                        all_colors = np.append(all_colors, [new_color], axis=0)

                        regions_color.append(new_color)
                        regions_coordinates.append(region.coords)

                else:
                    y0, x0 = regions[0].centroid
                    orientation = regions[0].orientation
                    orientation = orientation * (180 / np.pi)
                    major_length = regions[0].major_axis_length * pixel_per_micron
                    minor_length = regions[0].minor_axis_length * pixel_per_micron

                    regions_center_raw_objects.append((x0 * pixel_per_micron, y0 * pixel_per_micron))
                    regions_center_particles.append((x0 * pixel_per_micron, y0 * pixel_per_micron))

                    regions_center_prev_objects_stat.append('natural')
                    regions_center_particles_stat.append('natural')

                    regions_features.append({'center_x': x0 * pixel_per_micron, 'center_y': y0 * pixel_per_micron,
                                             'orientation': orientation, 'major': major_length, 'minor': minor_length})

                    regions_color.append(color)
                    regions_coordinates.append(regions[0].coords)

            # check objects
            df_centers_raw_objects = pd.DataFrame(regions_center_raw_objects, columns=['center_x', 'center_y'])
            df_centers_particles = pd.DataFrame(regions_center_particles, columns=['center_x', 'center_y'])

            # rows: index from df_centers_raw_objects
            # columns: index of cp records
            distance_df_raw_objects = \
                pd.DataFrame(distance_matrix(
                    df_centers_raw_objects[['center_x', 'center_y']].values,
                    current_df[[center_coord_cols['x'], center_coord_cols['y']]].values,
                ),
                    index=df_centers_raw_objects.index, columns=current_df.index)

            distance_df_raw_objects_min_val_idx = distance_df_raw_objects.idxmin()
            distance_df_raw_objects_min_val = distance_df_raw_objects.min()

            # Extracting corresponding values
            regions_center_prev_stat = [regions_center_prev_objects_stat[i] for i in distance_df_raw_objects_min_val_idx]
            regions_center_raw_x = [regions_center_raw_objects[i][0] for i in distance_df_raw_objects_min_val_idx]
            regions_center_raw_y = [regions_center_raw_objects[i][1] for i in distance_df_raw_objects_min_val_idx]

            min_distance_prev_objects_df = pd.DataFrame({
                'cp index': distance_df_raw_objects.columns,
                'row idx prev': distance_df_raw_objects_min_val_idx,
                'Cost prev': distance_df_raw_objects_min_val,
                'stat prev': regions_center_prev_stat,
                'center_x_prev': regions_center_raw_x,
                'center_y_prev': regions_center_raw_y,
            })

            if multi_region_flag:

                distance_df_particles = pd.DataFrame(distance_matrix(
                    df_centers_particles[['center_x', 'center_y']].values,
                    current_df[[center_coord_cols['x'], center_coord_cols['y']]].values,
                ),
                    index=df_centers_particles.index, columns=current_df.index)

                df = \
                    multi_region_correction(df, img_array, distance_df_particles, img_number,
                                            regions_center_particles_stat, regions_center_particles, regions_color,
                                            regions_coordinates, regions_features, all_rel_center_coord_cols,
                                            parent_image_number_col, parent_object_number_col,
                                            min_distance_prev_objects_df)

            else:

                for correct_bac_idx, correct_bac_row in min_distance_prev_objects_df.iterrows():
                    bac_ndx = correct_bac_row['cp index']
                    this_region_color = regions_color[correct_bac_row['row idx prev']]
                    # df.at[bac_ndx, 'color_mask'] = tuple(this_region_color)
                    df.at[bac_ndx, 'coordinate'] = set(map(tuple, regions_coordinates[correct_bac_row['row idx prev']]))

    return df

