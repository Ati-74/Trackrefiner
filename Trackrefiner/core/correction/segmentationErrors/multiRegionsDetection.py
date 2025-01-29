import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.spatial import distance_matrix
from scipy.sparse import lil_matrix
from Trackrefiner.core.correction.segmentationErrors.multiRegionsCorrection import multi_region_correction


def generate_new_color(existing_colors, seed=None):
    """Generate a new color not in existing_colors."""
    np.random.seed(seed)  # Optional: for reproducible colors
    while True:
        new_color = np.random.randint(0, 256, size=3) / 256
        # Use broadcasting and np.all to efficiently compare the new_color against all existing_colors
        if not np.any(np.all(np.isclose(existing_colors, new_color), axis=1)):
            return new_color


def map_and_detect_multi_regions(df, sorted_seg_npy_files_list, pixel_per_micron, center_coord_cols,
                                 all_rel_center_coord_cols, parent_image_number_col, parent_object_number_col, verbose):
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
    :param bool verbose:
        Whether to print detailed information about the processing steps for debugging.

    :return:
        A tuple containing:

        - **df** (*pd.DataFrame*): Updated dataframe with mapped object data.
          Objects with incorrect regions in the segmentation file are flagged as noise.
        - **coordinate_array** (*csr_matrix*): Sparse matrix indicating the pixel locations of objects.
        - **color_array** (*np.ndarray*): Array of RGB color values for each object.
    """

    for img_ndx, img_npy_file in enumerate(sorted_seg_npy_files_list):

        current_df = df.loc[df['ImageNumber'] == img_ndx + 1]

        img_number = img_ndx + 1

        img_array = np.load(img_npy_file)

        # Flatten the image to get a list of RGB values
        rgb_values = img_array.reshape(-1, 3)

        if img_ndx == 0:
            h, w, d = img_array.shape
            encoded_number = (w + h) * (w + h + 1) // 2 + w
            coordinate_array = lil_matrix((df.shape[0], encoded_number), dtype=bool)
            color_array = np.zeros((df.shape[0], 3))

        unique_colors, indices = np.unique(rgb_values, axis=0, return_index=True)

        all_colors = unique_colors.copy()

        # Exclude the background color (0,0,0)
        unique_colors = np.delete(unique_colors, 0, axis=0)

        mask_centeroid = []
        mask_stat = []
        connected_region_centroid = []
        connected_region_stat = []
        connected_region_features = []
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

                mask_centeroid.append((x0_cp * pixel_per_micron, y0_cp * pixel_per_micron))
                mask_stat.append('multi')

                # Separate regions
                for region_ndx, region in enumerate(regions):
                    # fetching information
                    y0, x0 = region.centroid
                    orientation = region.orientation
                    orientation = orientation * (180 / np.pi)
                    major_length = region.major_axis_length * pixel_per_micron
                    minor_length = region.minor_axis_length * pixel_per_micron

                    connected_region_centroid.append((x0 * pixel_per_micron, y0 * pixel_per_micron))
                    connected_region_stat.append('particle')

                    connected_region_features.append(
                        {'center_x': x0 * pixel_per_micron, 'center_y': y0 * pixel_per_micron,
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
                major_length = regions[0].major_axis_length * pixel_per_micron
                minor_length = regions[0].minor_axis_length * pixel_per_micron

                mask_centeroid.append((x0 * pixel_per_micron, y0 * pixel_per_micron))
                connected_region_centroid.append((x0 * pixel_per_micron, y0 * pixel_per_micron))

                mask_stat.append('natural')
                connected_region_stat.append('natural')

                connected_region_features.append({'center_x': x0 * pixel_per_micron, 'center_y': y0 * pixel_per_micron,
                                                  'orientation': orientation, 'major': major_length,
                                                  'minor': minor_length})

                regions_color.append(color)
                regions_coordinates.append(regions[0].coords)

        # check objects
        df_centers_masks = pd.DataFrame(mask_centeroid, columns=['center_x', 'center_y'])
        df_centers_connected_regions = pd.DataFrame(connected_region_centroid, columns=['center_x', 'center_y'])

        # rows: index from df_centers_masks
        # columns: index of cp records
        distance_df_cp_masks = \
            pd.DataFrame(distance_matrix(
                df_centers_masks[['center_x', 'center_y']].values,
                current_df[[center_coord_cols['x'], center_coord_cols['y']]].values,
            ),
                index=df_centers_masks.index, columns=current_df.index)

        distance_df_raw_objects_min_val_idx = distance_df_cp_masks.idxmin()
        distance_df_raw_objects_min_val = distance_df_cp_masks.min()

        # Extracting corresponding values
        regions_center_prev_stat = [mask_stat[i] for i in distance_df_raw_objects_min_val_idx]
        regions_center_raw_x = [mask_centeroid[i][0] for i in distance_df_raw_objects_min_val_idx]
        regions_center_raw_y = [mask_centeroid[i][1] for i in distance_df_raw_objects_min_val_idx]

        min_distance_prev_objects_df = pd.DataFrame({
            'cp index': distance_df_cp_masks.columns,
            'row idx prev': distance_df_raw_objects_min_val_idx,
            'Cost prev': distance_df_raw_objects_min_val,
            'stat prev': regions_center_prev_stat,
            'center_x_prev': regions_center_raw_x,
            'center_y_prev': regions_center_raw_y,
        })

        if multi_region_flag:

            distance_df_cp_connected_regions = pd.DataFrame(distance_matrix(
                df_centers_connected_regions[['center_x', 'center_y']].values,
                current_df[[center_coord_cols['x'], center_coord_cols['y']]].values,
            ),
                index=df_centers_connected_regions.index, columns=current_df.index)

            df, coordinate_array, color_array = \
                multi_region_correction(df, img_array, distance_df_cp_connected_regions, img_number,
                                        connected_region_stat, connected_region_centroid, regions_color,
                                        regions_coordinates, connected_region_features, all_rel_center_coord_cols,
                                        parent_image_number_col, parent_object_number_col,
                                        min_distance_prev_objects_df, coordinate_array, color_array)

        else:

            for correct_bac_idx, correct_bac_row in min_distance_prev_objects_df.iterrows():
                bac_ndx = correct_bac_row['cp index']
                this_region_color = regions_color[correct_bac_row['row idx prev']]

                color_array[bac_ndx] = this_region_color
                coordinates_array = regions_coordinates[correct_bac_row['row idx prev']]
                x, y = coordinates_array[:, 0], coordinates_array[:, 1]
                # Cantor Pairing Function
                encoded_numbers = (x + y) * (x + y + 1) // 2 + y
                coordinate_array[bac_ndx, encoded_numbers] = True

    coordinate_array = coordinate_array.tocsr()

    return df, coordinate_array, color_array
