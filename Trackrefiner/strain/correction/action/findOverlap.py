import os.path
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def func_calculate_iou(coords1, coords2, daughter_flag):

    df_coords1 = pd.DataFrame(coords1, columns=['row', 'col'])
    df_coords2 = pd.DataFrame(coords2, columns=['row', 'col'])

    # Calculate intersections and unions in a vectorized manner
    intersections = pd.merge(df_coords1, df_coords2, how='inner', on=['row', 'col']).shape[0]
    unions = pd.concat([df_coords1, df_coords1]).drop_duplicates().shape[0]

    # Calculate unique areas for conditional logic
    unique_masks1 = df_coords1.shape[0] - intersections
    unique_masks2 = df_coords2.shape[0] - intersections


    # Calculate IoU based on daughter_flag
    if daughter_flag:
            iou_val = intersections / (intersections + unique_masks2)
    else:
            iou_val = intersections / unions

    return iou_val


def find_overlap_object_to_next_frame(current_df, selected_objects, next_df, selected_objects_in_next_time_step,
                                      daughter_flag=False, maintain=False):

    overlap_results = []

    if len(current_df['color_mask'].values.tolist()) != len(set(current_df['color_mask'].values.tolist())):

        print(current_df['ImageNumber'].values[0])
        breakpoint()

    if len(next_df['color_mask'].values.tolist()) != len(set(next_df['color_mask'].values.tolist())):

        print(next_df['ImageNumber'].values[0])
        breakpoint()


    for source_obj_ndx, target_obj_ndx in [(source_obj_ndx, target_obj_ndx) for source_obj_ndx in selected_objects.index
                                           for target_obj_ndx in selected_objects_in_next_time_step.index]:

        selected_object1_coord = selected_objects.loc[source_obj_ndx]['coordinate']
        selected_object2_coord = selected_objects_in_next_time_step.loc[target_obj_ndx]['coordinate']

        if maintain and next_df.loc[target_obj_ndx]['parent_id'] == current_df.loc[source_obj_ndx]['id']:
            final_daughter_flags = True
        else:
            final_daughter_flags = daughter_flag

        # Calculate IoU values in batch
        iou_value = func_calculate_iou(selected_object1_coord, selected_object2_coord, final_daughter_flags)

        overlap_results.append({'source': source_obj_ndx, 'target': target_obj_ndx, 'IoU': iou_value})

    # Create DataFrame from results
    initial_df = pd.DataFrame(overlap_results)

    # Pivot this DataFrame to get the desired structure
    overlap_df = initial_df.pivot(index='source', columns='target', values='IoU')
    overlap_df.columns.name = None
    overlap_df.index.name = None

    return overlap_df
