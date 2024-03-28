import os.path
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def calculate_iou(selected_object_image1, selected_object_image2, daughter_flag):
    # Convert images to boolean masks (True where the pixel is not black)
    mask1 = np.any(selected_object_image1 != [0, 0, 0], axis=-1)
    mask2 = np.any(selected_object_image2 != [0, 0, 0], axis=-1)

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2)
    # Calculate unique areas
    unique_mask1 = np.logical_and(mask1, np.logical_not(mask2))
    unique_mask2 = np.logical_and(mask2, np.logical_not(mask1))

    if daughter_flag:
        # Calculate modified IOU
        # iou = np.sum(intersection) / (np.sum(intersection) + np.sum(unique_mask1) / 2)
        iou = np.sum(intersection) / (np.sum(intersection) + np.sum(unique_mask2))
    else:
        union = np.logical_or(mask1, mask2)
        # Calculate IOU
        # iou = np.sum(intersection) / (np.sum(intersection) + np.sum(unique_mask1))
        iou = np.sum(intersection) / np.sum(union)

    # Create an image to visualize the overlap
    # overlap_image = np.zeros_like(selected_object_image1)
    # overlap_image[intersection] = [255, 0, 0]  # Marking overlap with red, for example

    return iou


def batch_calculate_iou(selected_object_images1, selected_object_images2, daughter_flags):
    # Assuming images are in a batched form where each image is a 2D mask
    # Convert to boolean arrays if they are not already
    masks1 = np.array([np.any(img != [0, 0, 0], axis=-1) for img in selected_object_images1])
    masks2 = np.array([np.any(img != [0, 0, 0], axis=-1) for img in selected_object_images2])

    # Calculate intersections and unions in a vectorized manner
    intersections = np.logical_and(masks1, masks2)
    unions = np.logical_or(masks1, masks2)

    # Calculate unique areas for conditional logic
    unique_masks1 = np.logical_and(masks1, np.logical_not(masks2))
    unique_masks2 = np.logical_and(masks2, np.logical_not(masks1))

    # Initialize IoU array
    ious = np.zeros(len(selected_object_images1))

    # Calculate IoU based on daughter_flag
    for i, daughter_flag in enumerate(daughter_flags):
        if daughter_flag:
            ious[i] = np.sum(intersections[i]) / (np.sum(intersections[i]) + np.sum(unique_masks2[i]))
        else:
            ious[i] = np.sum(intersections[i]) / np.sum(unions[i])

    return ious


def find_overlap_object_to_next_frame(sorted_npy_files_list, current_df, selected_objects, next_df,
                                      selected_objects_in_next_time_step, daughter_flag=False, maintain=False):

    overlap_results = []

    if len(current_df['color_mask'].values.tolist()) != len(set(current_df['color_mask'].values.tolist())):

        print(current_df['ImageNumber'].values[0])
        breakpoint()

    if len(next_df['color_mask'].values.tolist()) != len(set(next_df['color_mask'].values.tolist())):

        print(next_df['ImageNumber'].values[0])
        breakpoint()

    source_img_num = current_df['ImageNumber'].values[0]
    source_img_npy_file = sorted_npy_files_list[source_img_num - 1]
    source_img_npy = np.load(os.path.splitext(source_img_npy_file)[0] + '_modified.npy' if os.path.exists(
        os.path.splitext(source_img_npy_file)[0] + '_modified.npy') else source_img_npy_file)

    target_img_num = next_df['ImageNumber'].values[0]
    target_img_npy_file = sorted_npy_files_list[target_img_num - 1]
    target_img_npy = np.load(os.path.splitext(target_img_npy_file)[0] + '_modified.npy' if os.path.exists(
        os.path.splitext(target_img_npy_file)[0] + '_modified.npy') else target_img_npy_file)

    # Prepare data for batch processing
    selected_object_images1 = []
    selected_object_images2 = []
    daughter_flags = []

    for source_obj_ndx, target_obj_ndx in [(source_obj_ndx, target_obj_ndx) for source_obj_ndx in selected_objects.index
                                           for target_obj_ndx in selected_objects_in_next_time_step.index]:
        selected_object_color = selected_objects.loc[source_obj_ndx]['color_mask']
        selected_object_image = np.all(source_img_npy == np.array(selected_object_color).reshape(1, 1, 3), axis=2)
        obj_next_time_step_color = selected_objects_in_next_time_step.loc[target_obj_ndx]['color_mask']
        obj_next_time_step_image = np.all(target_img_npy == np.array(obj_next_time_step_color).reshape(1, 1, 3), axis=2)

        selected_object_images1.append(selected_object_image)
        selected_object_images2.append(obj_next_time_step_image)

        if maintain and next_df.loc[target_obj_ndx]['parent_id'] == current_df.loc[source_obj_ndx]['id']:
            daughter_flags.append(True)
        else:
            daughter_flags.append(daughter_flag)

    # Calculate IoU values in batch
    iou_values = batch_calculate_iou(selected_object_images1, selected_object_images2, daughter_flags)

    # Populate results
    for i, (source_obj_ndx, target_obj_ndx) in enumerate(
            [(source_obj_ndx, target_obj_ndx) for source_obj_ndx in selected_objects.index for target_obj_ndx in
             selected_objects_in_next_time_step.index]):
        overlap_results.append({'source': source_obj_ndx, 'target': target_obj_ndx, 'IoU': iou_values[i]})

    # Create DataFrame from results
    initial_df = pd.DataFrame(overlap_results)

    # Pivot this DataFrame to get the desired structure
    overlap_df = initial_df.pivot(index='source', columns='target', values='IoU')
    overlap_df.columns.name = None
    overlap_df.index.name = None

    return overlap_df

"""
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
"""
