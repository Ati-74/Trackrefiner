import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2


def is_outer_contour(contour, mask, unique_colors, current_color, output_image):
    # Check each point of the contour
    for point in contour:
        x, y = point[0]
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # Direct neighbors (4-connectivity)

        for nx, ny in neighbors:
            if 0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0]:
                neighbor_color = tuple(output_image[ny, nx])
                if neighbor_color in unique_colors and neighbor_color != current_color:
                    return True  # If any contour point is adjacent to a different object, it is not fully enclosed
    return False


def find_outer_objects(output_image):
    unique_colors = set(tuple(map(tuple, output_image.reshape(-1, 3))))  # Extract unique colors
    unique_colors.discard((0, 0, 0))  # Exclude the background color

    outer_objects_ids = []

    for color in unique_colors:
        # Create a binary mask for the current object
        mask = np.all(output_image == color, axis=-1).astype(np.uint8)

        # Find contours of the object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check each contour to see if it's enclosed by other objects
        for contour in contours:
            if is_outer_contour(contour, mask, unique_colors, color, output_image):
                outer_objects_ids.append(color[2] - 1)
                break  # If any contour of the object is outer, consider the whole object as outer

    return outer_objects_ids


def calculate_iou_and_draw_overlap(selected_object_image1, selected_object_image2):
    # Convert images to boolean masks (True where the pixel is not black)
    mask1 = np.any(selected_object_image1 != [0, 0, 0], axis=-1)
    mask2 = np.any(selected_object_image2 != [0, 0, 0], axis=-1)

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    # Calculate IOU
    iou = np.sum(intersection) / np.sum(union)

    # Create an image to visualize the overlap
    overlap_image = np.zeros_like(selected_object_image1)
    overlap_image[intersection] = [255, 0, 0]  # Marking overlap with red, for example

    return iou, overlap_image


def set_color_to_objects(img_npy_file, current_df, um_per_pixel):

    img_array = np.load(img_npy_file)

    # Flatten the image to get a list of RGB values
    rgb_values = img_array.reshape(-1, 3)

    # Convert RGB values to tuples
    tuple_rgb_values = [tuple(row) for row in rgb_values]

    # Convert to set to get unique RGB values
    unique_colors = set(tuple_rgb_values)

    # Exclude the background color (0,0,0)
    if (0, 0, 0) in unique_colors:
        unique_colors.remove((0, 0, 0))

        # The number of unique colors is the number of objects
        num_objects = len(unique_colors)

    # Find the center for each object based on color
    object_centers = {}
    for color in unique_colors:
        # Create a mask for the current color
        mask = np.all(img_array == np.array(color).reshape(1, 1, 3), axis=2)
        y, x = np.where(mask)
        center_x, center_y = x.mean(), y.mean()
        object_centers[color] = (center_x * um_per_pixel, center_y * um_per_pixel)

    # Map each object to its closest CSV row based on centers
    object_to_id = {}
    for color, (obj_x, obj_y) in object_centers.items():
        try:
            distances = ((current_df['Location_Center_X'] - obj_x) ** 2 +
                         (current_df['Location_Center_Y'] - obj_y) ** 2) ** 0.5
        except:
            distances = ((current_df['AreaShape_Center_X'] - obj_x) ** 2 +
                         (current_df['AreaShape_Center_Y'] - obj_y) ** 2) ** 0.5

        # Convert the Series to a list
        distance_list = distances.tolist()
        closest_index = np.argmin(distance_list)
        object_to_id[color] = (0, 0, (current_df['id'].values.tolist()[closest_index] + 1))

    output_image = np.zeros_like(img_array)

    if len(set(object_to_id.values())) != len(object_to_id.values()):
        print(img_npy_file)

    for color in unique_colors:
        mask = np.all(img_array == np.array(color).reshape(1, 1, 3), axis=2)
        id_color = object_to_id[color]  # you'd need to define a function or mapping for this
        output_image[mask] = id_color

    return output_image


def find_overlap_object_to_next_frame(current_npy_file, current_df, selected_objects, next_npy_file, next_df,
                                      selected_objects_in_next_time_step, um_per_pixel):

    current_img_array = set_color_to_objects(current_npy_file, current_df, um_per_pixel)
    next_img_array = set_color_to_objects(next_npy_file, next_df, um_per_pixel)

    overlap_dict = {}

    for selected_object_indx, selected_object in selected_objects.iterrows():
        selected_object_mask = np.all(current_img_array ==
                                      np.array((0, 0, selected_object['id'] + 1)).reshape(1, 1, 3), axis=2)

        # Apply the mask to the image
        selected_object_image = np.zeros_like(current_img_array)
        selected_object_image[selected_object_mask] = current_img_array[selected_object_mask]

        overlap_dict[selected_object_indx] = {}

        for obj_next_time_step_indx, obj_next_time_step in selected_objects_in_next_time_step.iterrows():

            # find the related object from image array
            obj_next_time_step_mask = np.all(next_img_array ==
                                             np.array((0, 0, obj_next_time_step['id'] + 1)).reshape(1, 1, 3), axis=2)

            obj_next_time_step_image = np.zeros_like(next_img_array)
            obj_next_time_step_image[obj_next_time_step_mask] = next_img_array[obj_next_time_step_mask]

            # calculate IOU
            iou_val, overlap_image = calculate_iou_and_draw_overlap(selected_object_image, obj_next_time_step_image)

            overlap_dict[selected_object_indx][obj_next_time_step_indx] = iou_val

    overlap_df = pd.DataFrame.from_dict(overlap_dict, orient='index')
    # overlap_df = overlap_df.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
    return overlap_df


def find_overlap(current_bacteria_df, current_img_array, bacteria_in_next_time_step, next_img_array, parent, daughters,
                 um_per_pixel):

    daughters_ids = daughters['id'].values.tolist()
    daughters_masks = []
    iou_dict = {}

    current_time_step_objects = set_color_to_objects(current_img_array, current_bacteria_df, um_per_pixel)
    next_time_step_objects = set_color_to_objects(next_img_array, bacteria_in_next_time_step, um_per_pixel)

    parent_mask = np.all(current_time_step_objects == np.array((0, 0, parent['id'] + 1)).reshape(1, 1, 3), axis=2)

    for daughters_id in daughters_ids:
        daughter_mask = np.all(next_time_step_objects == np.array((0, 0, daughters_id + 1)).reshape(1, 1, 3), axis=2)
        daughters_masks.append(daughter_mask)

    # Apply the mask to the image
    parent_object_image = np.zeros_like(current_time_step_objects)
    parent_object_image[parent_mask] = current_time_step_objects[parent_mask]

    for daughter_indx, daughters_id in enumerate(daughters_ids):

        daughter_object_image = np.zeros_like(next_time_step_objects)
        daughter_object_image[daughters_masks[daughter_indx]] = next_time_step_objects[daughters_masks[daughter_indx]]

        iou, overlap_image = calculate_iou_and_draw_overlap(parent_object_image, daughter_object_image)
        iou_dict[daughters_id] = iou

    return iou_dict, current_time_step_objects, next_time_step_objects


