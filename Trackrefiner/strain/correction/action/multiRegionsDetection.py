import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.spatial import distance_matrix
from Trackrefiner.strain.correction.action.multiRegionsCorrection import multi_region_correction


def generate_new_color(existing_colors, seed=None):
    """Generate a new color not in existing_colors."""
    np.random.seed(seed)  # Optional: for reproducible colors
    while True:
        new_color = np.random.randint(0, 256, size=3) / 256
        # Use broadcasting and np.all to efficiently compare the new_color against all existing_colors
        if not np.any(np.all(np.isclose(existing_colors, new_color), axis=1)):
            return new_color


def multi_region_detection(df, img_npy_file_list, um_per_pixel, center_coordinate_columns,
                           all_center_coordinate_columns, parent_image_number_col, parent_object_number_col, warn,
                           img=None):

    for img_ndx, img_npy_file in enumerate(img_npy_file_list):

        current_df = df.loc[df['ImageNumber'] == img_ndx + 1]

        img_number = img_ndx + 1

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

                regions_center_raw_objects.append((x0_cp * um_per_pixel, y0_cp * um_per_pixel))
                regions_center_prev_objects_stat.append('multi')

                # Separate regions
                for region_ndx, region in enumerate(regions):

                    # fetching information
                    y0, x0 = region.centroid
                    orientation = region.orientation
                    orientation = orientation * (180 / np.pi)
                    major_length = region.major_axis_length * um_per_pixel
                    minor_length = region.minor_axis_length * um_per_pixel

                    regions_center_particles.append((x0 * um_per_pixel, y0 * um_per_pixel))
                    regions_center_particles_stat.append('particle')

                    regions_features.append({'center_x': x0 * um_per_pixel, 'center_y': y0 * um_per_pixel,
                                             'orientation': orientation, 'major': major_length, 'minor': minor_length})

                    # Generate a new color that's not already in unique_colors
                    new_color = generate_new_color(all_colors)
                    all_colors = np.append(all_colors, [new_color], axis=0)

                    regions_color.append(new_color)
                    regions_coordinates.append(region.coords)

            else:
                y0, x0 = regions[0].centroid
                orientation = regions[0].orientation
                orientation = orientation * (180 / np.pi)
                major_length = regions[0].major_axis_length * um_per_pixel
                minor_length = regions[0].minor_axis_length * um_per_pixel

                regions_center_raw_objects.append((x0 * um_per_pixel, y0 * um_per_pixel))
                regions_center_particles.append((x0 * um_per_pixel, y0 * um_per_pixel))

                regions_center_prev_objects_stat.append('natural')
                regions_center_particles_stat.append('natural')

                regions_features.append({'center_x': x0 * um_per_pixel, 'center_y': y0 * um_per_pixel,
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
                current_df[[center_coordinate_columns['x'], center_coordinate_columns['y']]].values,
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
                current_df[[center_coordinate_columns['x'], center_coordinate_columns['y']]].values,
            ),
                index=df_centers_particles.index, columns=current_df.index)

            df = \
                multi_region_correction(df, img_array, img_npy_file, distance_df_particles, img_number,
                                        regions_center_particles_stat, regions_center_particles, regions_color,
                                        regions_coordinates, regions_features, all_center_coordinate_columns,
                                        parent_image_number_col, parent_object_number_col,
                                        min_distance_prev_objects_df)

        else:

            for correct_bac_idx, correct_bac_row in min_distance_prev_objects_df.iterrows():
                bac_ndx = correct_bac_row['cp index']
                this_region_color = regions_color[correct_bac_row['row idx prev']]
                df.at[bac_ndx, 'color_mask'] = tuple(this_region_color)
                df.at[bac_ndx, 'coordinate'] = set(map(tuple, regions_coordinates[correct_bac_row['row idx prev']]))

    return df
