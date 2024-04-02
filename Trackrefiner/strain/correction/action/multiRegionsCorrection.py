import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_dilation
from scipy.ndimage import binary_dilation
import os


def find_neighbors_direct_expansion(img_array, object_info, neighbors_df, img_number, multi_regions_new_colors,
                                    obj_number_multi_region):

    # idea: https://github.com/CellProfiler/CellProfiler/blob/ea6a2e6d001b10983301c10994e319abef41e618/src/frontend/cellprofiler/modules/measureobjectneighbors.py#L91

    # we should find the neighbors of multi region objects
    neighbors_multi_region_obj = neighbors_df.loc[(neighbors_df['First Image Number'] == img_number) &
                                                  (neighbors_df['First Object Number'].isin(obj_number_multi_region))]

    neighbors_color = \
        [color for color in object_info.keys() if object_info[color][1] in
         neighbors_multi_region_obj['Second Object Number'].values.tolist()]

    # Extract unique colors (excluding the background)
    unique_colors = list(object_info.keys())

    # Initialize an empty array for the expanded objects
    expanded_img_array = np.zeros_like(img_array)

    # Create a binary mask for each unique color and calculate its distance transform
    color_arrays = {color: np.array(color).reshape(1, 1, 3) for color in unique_colors}

    masks = {color: np.all(img_array == color_arrays[color], axis=2) for color in unique_colors}

    # Compute distance maps once
    distance_maps = {color: distance_transform_edt(~masks[color]) for color in masks.keys()}

    # Determine the minimum distance to another object for each pixel
    all_distances = np.stack(list(distance_maps.values()), axis=0)

    min_distance_to_other = np.min(all_distances, axis=0)

    # Expand each object based on the calculated minimum distances
    for color, distance_map in distance_maps.items():
        # Find pixels where the object's distance map is less than or equal to the minimum distance to another object
        # This identifies the maximum expansion for each object without intersecting another object
        # mask = np.all(img_array == np.array(color).reshape(1, 1, 3), axis=2)
        expansion_mask = distance_map <= min_distance_to_other

        # Apply the expansion mask to the expanded image array
        expanded_img_array[expansion_mask] = color_arrays[color]

    expanded_masks = {color: np.all(expanded_img_array == color, axis=2) for color in unique_colors}
    # Create a dilated mask for the current color to detect neighbors
    dilated_masks = {color:  binary_dilation(expanded_masks[color]) for color in unique_colors}

    # Identify neighbors based on the expanded regions
    neighbors_new_bac_multi_regions = {object_info[color]: set() for color in multi_regions_new_colors}
    new_neighbors_multi_regions_neighbors = {object_info[color]: set() for color in neighbors_color}

    for color in multi_regions_new_colors:
        for other_color in neighbors_color:
            if np.any(dilated_masks[color] & expanded_masks[other_color]):
                neighbors_new_bac_multi_regions[object_info[color]].add(object_info[other_color])


    for color in multi_regions_new_colors:
        for other_color in multi_regions_new_colors:
            if color != other_color and np.any(dilated_masks[color] & expanded_masks[other_color]):
                neighbors_new_bac_multi_regions[object_info[color]].add(object_info[other_color])

    for color in neighbors_color:
        for other_color in multi_regions_new_colors:
            if np.any(dilated_masks[color] & expanded_masks[other_color]):
                new_neighbors_multi_regions_neighbors[object_info[color]].add(object_info[other_color])

    return neighbors_new_bac_multi_regions, new_neighbors_multi_regions_neighbors

def generate_new_color(existing_colors, seed=None):
    """Generate a new color not in existing_colors."""
    np.random.seed(seed)  # Optional: for reproducible colors
    while True:
        new_color = np.random.randint(0, 256, size=3) / 256
        # Use broadcasting and np.all to efficiently compare the new_color against all existing_colors
        if not np.any(np.all(np.isclose(existing_colors, new_color), axis=1)):
            return new_color


def multi_region_correction(df, img_npy_file_list, neighbors_df, um_per_pixel, center_coordinate_columns,
                            all_center_coordinate_columns, parent_image_number_col, parent_object_number_col, warn):

    for img_ndx, img_npy_file in enumerate(img_npy_file_list):

        current_time_step_objects_info = {}

        current_df = df.loc[df['ImageNumber'] == img_ndx + 1]

        obj_number_multi_region = []
        multi_regions_new_colors = []

        neighbor_current_df = neighbors_df.loc[neighbors_df['First Image Number'] == img_ndx + 1]
        modified_flag = False
        npy_change_flag = False

        img_number = img_ndx + 1
        last_obj_number = max(current_df["ObjectNumber"].values.tolist())

        img_array = np.load(img_npy_file)

        # Flatten the image to get a list of RGB values
        rgb_values = img_array.reshape(-1, 3)

        unique_colors, indices = np.unique(rgb_values, axis=0, return_index=True)

        all_colors = unique_colors.copy()

        # Exclude the background color (0,0,0)
        unique_colors = np.delete(unique_colors, 0, axis=0)

        for color in unique_colors:
            # Create a mask for the current color
            mask = np.all(img_array == color.reshape(1, 1, 3), axis=2)

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

                labeled_mask_cp[labeled_mask_cp > 1] = 1
                regions_cp = regionprops(labeled_mask_cp)

                y0_cp, x0_cp = regions_cp[0].centroid

                distances = ((current_df[center_coordinate_columns['x']] - (x0_cp * um_per_pixel)) ** 2 +
                             (current_df[center_coordinate_columns['y']] - (y0_cp * um_per_pixel)) ** 2) ** 0.5

                closest_index = np.argmin(distances.values)

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

                    if np.min(distances_sub_reg.values) < 1e-3:

                        if warn:
                            print('Warning repeate!!!!!')
                        closest_reg_index = np.argmin(distances_sub_reg.values)
                        bac_reg_ndx = current_df.index[closest_reg_index]

                        # Generate a new color that's not already in unique_colors
                        new_color = generate_new_color(all_colors)
                        all_colors = np.append(all_colors, [new_color], axis=0)

                        # Update the img_array with the new color for this region
                        img_array[tuple(zip(*region.coords))] = new_color

                        df.at[bac_reg_ndx, 'color_mask'] = tuple(new_color)
                        multi_regions_new_colors.append(tuple(new_color))

                        current_time_step_objects_info[tuple(new_color)] = (img_number,
                                                  current_df['ObjectNumber'].values[closest_reg_index])

                    elif region_ndx == 0:

                        modified_flag = True
                        bac_ndx = current_df.index.values[closest_index]

                        # Generate a new color that's not already in unique_colors
                        new_color = generate_new_color(all_colors)
                        all_colors = np.append(all_colors, [new_color], axis=0)

                        multi_regions_new_colors.append(tuple(new_color))

                        # Update the img_array with the new color for this region
                        img_array[tuple(zip(*region.coords))] = new_color

                        # update dataframe
                        updates = {
                            'color_mask': tuple(new_color),
                            **{x_center_col: x0 * um_per_pixel for x_center_col in all_center_coordinate_columns['x']},
                            **{y_center_col: y0 * um_per_pixel for y_center_col in all_center_coordinate_columns['y']},
                            "AreaShape_MajorAxisLength": major_length * um_per_pixel,
                            "AreaShape_MinorAxisLength": minor_length * um_per_pixel,
                            "AreaShape_Orientation": -(orientation + 90) * np.pi / 180,
                            parent_image_number_col: 0,
                            parent_object_number_col: 0
                        }

                        # Update the DataFrame for the given index (bac_ndx) with all updates
                        df.loc[bac_ndx, updates.keys()] = updates.values()

                        obj_number_multi_region.append(df.loc[bac_ndx]["ObjectNumber"])

                        current_time_step_objects_info[tuple(new_color)] = img_number, df.loc[bac_ndx]["ObjectNumber"]

                    else:
                        modified_flag = True
                        last_obj_number += 1

                        # Generate a new color that's not already in unique_colors
                        new_color = generate_new_color(all_colors)
                        all_colors = np.append(all_colors, [new_color], axis=0)

                        multi_regions_new_colors.append(tuple(new_color))

                        # Update the img_array with the new color for this region
                        img_array[tuple(zip(*region.coords))] = new_color

                        # now update dataframe
                        new_row = {"ImageNumber": img_number, "ObjectNumber": last_obj_number,
                                   **{x_center_col: x0 * um_per_pixel for x_center_col in
                                      all_center_coordinate_columns['x']},
                                   **{y_center_col: y0 * um_per_pixel for y_center_col in
                                      all_center_coordinate_columns['y']},
                                   "AreaShape_MajorAxisLength": major_length * um_per_pixel,
                                   "AreaShape_MinorAxisLength": minor_length * um_per_pixel,
                                   "AreaShape_Orientation": -(orientation + 90) * np.pi / 180,
                                   parent_image_number_col: 0, parent_object_number_col: 0,
                                   'color_mask': tuple(new_color)}

                        current_time_step_objects_info[tuple(new_color)] = (img_number, last_obj_number)

                        # Convert the dictionary to a DataFrame before concatenating
                        new_row_df = pd.DataFrame([new_row])

                        # Concatenate the new row DataFrame to the existing DataFrame
                        df = pd.concat([df, new_row_df], ignore_index=True)

            else:
                y0, x0 = regions[0].centroid
                distances = ((current_df[center_coordinate_columns['x']] - x0 * um_per_pixel) ** 2 +
                             (current_df[center_coordinate_columns['y']] - y0 * um_per_pixel) ** 2) ** 0.5

                closest_index = np.argmin(distances.values)

                bac_ndx = current_df.index.values[closest_index]

                current_time_step_objects_info[tuple(color)] = \
                    (img_number, current_df['ObjectNumber'].values[closest_index])

                df.at[bac_ndx, 'color_mask'] = tuple(color)

            # Display the image
            # plt.figure(figsize=(10, 10))
            # plt.imshow(mask)
            # plt.axis('off')  # Remove axis ticks and labels for clarity
            # plt.show()

        if modified_flag:
            neighbors_new_bac_multi_regions, new_neighbors_multi_regions_neighbors = \
                find_neighbors_direct_expansion(img_array, current_time_step_objects_info, neighbors_df,
                                                img_number, multi_regions_new_colors, obj_number_multi_region)

            neighbor_current_df_delete = \
                neighbor_current_df.loc[(neighbor_current_df['First Object Number'].isin(obj_number_multi_region)) |
                                        (neighbor_current_df['Second Object Number'].isin(obj_number_multi_region))]

            neighbors_df = neighbors_df.loc[~neighbors_df.index.isin(neighbor_current_df_delete.index.values.tolist())]

            for key_val in neighbors_new_bac_multi_regions.keys():
                for real_neighbor in neighbors_new_bac_multi_regions[key_val]:
                    new_row = {'Module': 'MeasureObjectNeighbors', 'Relationship': 'Neighbors',
                               'First Object Name': 'FilterObjects', 'First Image Number': key_val[0],
                               'First Object Number': key_val[1], 'Second Object Name': 'FilterObjects',
                               'Second Image Number': real_neighbor[0], 'Second Object Number': real_neighbor[1]}

                    new_row_df = pd.DataFrame([new_row])
                    neighbors_df = pd.concat([neighbors_df, new_row_df], ignore_index=True)


            for key_val in new_neighbors_multi_regions_neighbors.keys():
                for real_neighbor in new_neighbors_multi_regions_neighbors[key_val]:
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
