import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_dilation
from scipy.ndimage import binary_dilation
import os


def find_neighbors_direct_expansion(img_array, object_info):
    # Extract unique colors (excluding the background)
    rgb_values = img_array.reshape(-1, 3)
    unique_colors = set(map(tuple, rgb_values))
    unique_colors.discard((0, 0, 0))  # Remove background color

    # Initialize an empty array for the expanded objects
    expanded_img_array = np.zeros_like(img_array)

    # Create a binary mask for each unique color and calculate its distance transform
    distance_maps = {}
    for color in unique_colors:
        mask = np.all(img_array == np.array(color).reshape(1, 1, 3), axis=2)
        distance_maps[color] = distance_transform_edt(~mask)

    # Determine the minimum distance to another object for each pixel
    all_distances = np.stack(list(distance_maps.values()), axis=0)
    min_distance_to_other = np.min(all_distances, axis=0)

    # Expand each object based on the calculated minimum distances
    for color, distance_map in distance_maps.items():
        # Find pixels where the object's distance map is less than or equal to the minimum distance to another object
        # This identifies the maximum expansion for each object without intersecting another object
        mask = np.all(img_array == np.array(color).reshape(1, 1, 3), axis=2)
        expansion_mask = distance_map <= min_distance_to_other

        # Apply the expansion mask to the expanded image array
        expanded_img_array[expansion_mask] = np.array(color).reshape(1, 1, 3)

    # Identify neighbors based on the expanded regions
    neighbors = {object_info[color]: set() for color in unique_colors}
    for color in unique_colors:
        # Create a dilated mask for the current color to detect neighbors
        current_mask = np.all(expanded_img_array == np.array(color).reshape(1, 1, 3), axis=2)
        dilated_mask = binary_dilation(current_mask)

        for other_color in unique_colors:
            if other_color == color:
                continue

            other_mask = np.all(expanded_img_array == np.array(other_color).reshape(1, 1, 3), axis=2)
            if np.any(dilated_mask & other_mask):
                neighbors[object_info[color]].add(object_info[other_color])

    return neighbors


def generate_new_color(existing_colors, seed=None):
    """Generate a new color not in existing_colors."""
    np.random.seed(seed)  # Optional: for reproducible colors
    while True:
        new_color = tuple(np.random.randint(0, 256, size=3) / 256)
        if new_color not in existing_colors:
            return new_color


def multi_region_correction(df, img_npy_file_list, neighbors_df, um_per_pixel, center_coordinate_columns,
                            all_center_coordinate_columns, parent_image_number_col, parent_object_number_col, warn):
    # idea: https://github.com/CellProfiler/CellProfiler/blob/ea6a2e6d001b10983301c10994e319abef41e618/src/frontend/cellprofiler/modules/measureobjectneighbors.py#L91

    for img_ndx, img_npy_file in enumerate(img_npy_file_list):

        object_info = {}

        current_df = df.loc[df['ImageNumber'] == img_ndx + 1]

        neighbor_current_df = neighbors_df.loc[neighbors_df['First Image Number'] == img_ndx + 1]
        modified_flag = False
        npy_change_flag = False

        img_number = img_ndx + 1
        last_obj_number = max(current_df["ObjectNumber"].values.tolist())

        img_array = np.load(img_npy_file)

        # Flatten the image to get a list of RGB values
        rgb_values = img_array.reshape(-1, 3)

        # Convert RGB values to tuples
        tuple_rgb_values = [tuple(row) for row in rgb_values]

        # Convert to set to get unique RGB values
        unique_colors = set(tuple_rgb_values)

        all_colors = unique_colors.copy()

        # Exclude the background color (0,0,0)
        if (0, 0, 0) in unique_colors:
            unique_colors.remove((0, 0, 0))

        for color in unique_colors:

            # Create a mask for the current color
            mask = np.all(img_array == np.array(color).reshape(1, 1, 3), axis=2)

            # Label the regions
            labeled_mask = label(mask)
            regions = regionprops(labeled_mask)

            if len(regions) > 1:

                if warn:
                    print('===========================================================================================')
                    print(img_npy_file)
                    print("WARNING: one mask with two regions! Number of regions: " + str(len(regions)))
                    print('===========================================================================================')

                labeled_mask_cp = labeled_mask.copy()

                for label_val in np.unique(labeled_mask):
                    if label_val not in [0, 1]:
                        labeled_mask_cp[labeled_mask_cp == label_val] = 1
                regions_cp = regionprops(labeled_mask_cp)

                y0_cp, x0_cp = regions_cp[0].centroid

                distances = ((current_df[center_coordinate_columns['x']] - (x0_cp * um_per_pixel)) ** 2 +
                            (current_df[center_coordinate_columns['y']] - (y0_cp * um_per_pixel)) ** 2) ** 0.5

                distance_list = distances.tolist()
                closest_index = np.argmin(distance_list)

                # Separate regions
                for region_ndx, region in enumerate(regions):

                    # fetching information
                    y0, x0 = region.centroid
                    orientation = region.orientation
                    orientation = orientation * (180 / np.pi)
                    major_length = region.major_axis_length
                    minor_length = region.minor_axis_length

                    distances_sub_reg = ((current_df[center_coordinate_columns['x']] - (x0 * um_per_pixel)) ** 2 +
                                        (current_df[center_coordinate_columns['y']] - (y0 * um_per_pixel)) ** 2) ** 0.5

                    distances_sub_reg_list = distances_sub_reg.tolist()

                    if np.min(distances_sub_reg_list) < 1e-3:

                        if warn:
                            print('Warning repeate!!!!!')
                        closest_reg_index = np.argmin(distances_sub_reg_list)
                        bac_reg_ndx = current_df.index.values.tolist()[closest_reg_index]

                        # Generate a new color that's not already in unique_colors
                        new_color = generate_new_color(all_colors)
                        all_colors.add(new_color)

                        # Update the img_array with the new color for this region
                        for coord in region.coords:
                            img_array[coord[0], coord[1]] = new_color

                        df.at[bac_reg_ndx, 'color_mask'] = new_color

                        object_info[new_color] = (current_df['ImageNumber'].values.tolist()[closest_reg_index],
                                              current_df['ObjectNumber'].values.tolist()[closest_reg_index])

                    elif region_ndx == 0:

                        modified_flag = True
                        bac_ndx = current_df.index.values.tolist()[closest_index]

                        # Generate a new color that's not already in unique_colors
                        new_color = generate_new_color(all_colors)
                        all_colors.add(new_color)

                        # Update the img_array with the new color for this region
                        for coord in region.coords:
                            img_array[coord[0], coord[1]] = new_color

                        # update dataframe
                        df.at[bac_ndx, 'color_mask'] = new_color
                        for x_center_col in all_center_coordinate_columns['x']:
                            df.at[bac_ndx, x_center_col] = x0 * um_per_pixel
                        for y_center_col in all_center_coordinate_columns['y']:
                            df.at[bac_ndx, y_center_col] = y0 * um_per_pixel
                        df.at[bac_ndx, "AreaShape_MajorAxisLength"] = major_length * um_per_pixel
                        df.at[bac_ndx, "AreaShape_MinorAxisLength"] = minor_length * um_per_pixel
                        df.at[bac_ndx, "AreaShape_Orientation"] = -(orientation + 90) * np.pi / 180
                        df.at[bac_ndx, parent_image_number_col] = 0
                        df.at[bac_ndx, parent_object_number_col] = 0

                        object_info[new_color] = (df.loc[bac_ndx]["ImageNumber"], df.loc[bac_ndx]["ObjectNumber"])

                    else:
                        modified_flag = True
                        last_obj_number += 1

                        # Generate a new color that's not already in unique_colors
                        new_color = generate_new_color(all_colors)
                        all_colors.add(new_color)

                        # Update the img_array with the new color for this region
                        for coord in region.coords:
                            img_array[coord[0], coord[1]] = new_color

                        # now update dataframe
                        new_row = {"ImageNumber": img_number, "ObjectNumber": last_obj_number,
                                   "AreaShape_MajorAxisLength": major_length * um_per_pixel,
                                   "AreaShape_MinorAxisLength": minor_length * um_per_pixel,
                                   "AreaShape_Orientation": -(orientation + 90) * np.pi / 180,
                                   parent_image_number_col: 0, parent_object_number_col: 0,
                                   'color_mask': new_color}

                        for x_center_col in all_center_coordinate_columns['x']:
                            new_row[x_center_col] = x0 * um_per_pixel
                        for y_center_col in all_center_coordinate_columns['y']:
                            new_row[y_center_col] = y0 * um_per_pixel

                        object_info[new_color] = (img_number, last_obj_number)

                        # Convert the dictionary to a DataFrame before concatenating
                        new_row_df = pd.DataFrame([new_row])

                        # Concatenate the new row DataFrame to the existing DataFrame
                        df = pd.concat([df, new_row_df], ignore_index=True)

            else:
                y0, x0 = regions[0].centroid
                distances = ((current_df[center_coordinate_columns['x']] - x0 * um_per_pixel) ** 2 +
                            (current_df[center_coordinate_columns['y']] - y0 * um_per_pixel) ** 2) ** 0.5

                # Convert the Series to a list
                distance_list = distances.tolist()
                closest_index = np.argmin(distance_list)

                bac_ndx = current_df.index.values.tolist()[closest_index]

                object_info[color] = (current_df['ImageNumber'].values.tolist()[closest_index],
                                      current_df['ObjectNumber'].values.tolist()[closest_index])

                df.at[bac_ndx, 'color_mask'] = color

        # print(img_npy_file)
        # neighbors = find_neighbors_direct_expansion(img_array, object_info)

        # Display the image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(img_array)
        # plt.axis('off')  # Remove axis ticks and labels for clarity
        # plt.show()

        if modified_flag:
            neighbors = find_neighbors_direct_expansion(img_array, object_info)
            neighbors_df = neighbors_df.loc[~neighbors_df.index.isin(neighbor_current_df.index.values.tolist())]

            for key_val in neighbors.keys():
                for real_neighbor in neighbors[key_val]:
                    new_row = {'Module': 'MeasureObjectNeighbors', 'Relationship': 'Neighbors',
                               'First Object Name': 'FilterObjects', 'First Image Number': key_val[0],
                               'First Object Number': key_val[1], 'Second Object Name': 'FilterObjects',
                               'Second Image Number': real_neighbor[0], 'Second Object Number': real_neighbor[1]}

                    new_row_df = pd.DataFrame([new_row])
                    neighbors_df = pd.concat([neighbors_df, new_row_df], ignore_index=True)

            # Save the modified img_array to a new .npy file
            new_file_name = os.path.splitext(img_npy_file)[0] + '_modified.npy'
            np.save(new_file_name, img_array)

        elif npy_change_flag:
            # Save the modified img_array to a new .npy file
            new_file_name = os.path.splitext(img_npy_file)[0] + '_modified.npy'
            np.save(new_file_name, img_array)

    df_sorted = df.sort_values(by=["ImageNumber", "ObjectNumber"])
    df_sorted.reset_index(drop=True, inplace=True)
    neighbors_df_sorted = neighbors_df.sort_values(by=['First Image Number', 'First Object Number',
                                                       'Second Image Number', 'Second Object Number'])
    neighbors_df_sorted.reset_index(drop=True, inplace=True)

    return df_sorted, neighbors_df_sorted
