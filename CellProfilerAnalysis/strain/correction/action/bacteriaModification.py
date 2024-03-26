import numpy as np
import pandas as pd
from CellProfilerAnalysis.strain.correction.action.helperFunctions import find_vertex, calculate_slope_intercept, \
    calculate_orientation_angle, calculate_trajectory_direction_angle, find_related_bacteria, calc_neighbors_dir_motion,\
    calculate_trajectory_direction, calc_normalized_angle_between_motion
from CellProfilerAnalysis.strain.correction.neighborChecking import check_num_neighbors


def new_transition_bacteria(df, fake_daughter_life_history):
    # columns name
    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]
    label_col = [col for col in df.columns if 'TrackObjects_Label_' in col][0]

    fake_daughter_life_history_before_this_time = \
        df.loc[(df['id'] == fake_daughter_life_history['id'].values.tolist()[0]) &
               (df['ImageNumber'] < fake_daughter_life_history['ImageNumber'].values.tolist()[0])]

    fake_daughter_life_history_next_gens_ndx = find_related_bacteria(df, fake_daughter_life_history.iloc[-1],
                                                                     fake_daughter_life_history.index[-1])

    fake_daughter_life_history_next_gens = df.loc[df.index.isin(fake_daughter_life_history_next_gens_ndx[1:])]

    fake_daughter_life_history_next_gen = \
        fake_daughter_life_history_next_gens.loc[fake_daughter_life_history_next_gens['parent_id'] ==
                                                 fake_daughter_life_history['id'].values.tolist()[0]]

    new_label_val = max(df[label_col].values) + 1
    bacterium_id = max(df['id'].values) + 1

    for bac_ndx, bac in fake_daughter_life_history_before_this_time.iterrows():
        df.at[bac_ndx, "LifeHistory"] = fake_daughter_life_history_before_this_time.shape[0]

        if bac_ndx == fake_daughter_life_history_before_this_time.index[-1]:
            df.at[bac_ndx, "unexpected_end"] = True
            df.at[bac_ndx, 'daughter_distance_to_mother'] = ''
            df.at[bac_ndx, "max_daughter_len_to_mother"] = ''
            df.at[bac_ndx, "daughter_orientation"] = ''
            df.at[bac_ndx, "daughter_length_to_mother"] = ''
            df.at[bac_ndx, 'daughters_distance'] = ''
            df.at[bac_ndx, 'mother_last_to_first_bac_length_ratio'] = ''
            df.at[bac_ndx, "next_to_first_bac_length_ratio"] = \
                fake_daughter_life_history_before_this_time.loc[fake_daughter_life_history_before_this_time.index[-1]][
                    'AreaShape_MajorAxisLength'] / \
                fake_daughter_life_history_before_this_time.loc[fake_daughter_life_history_before_this_time.index[0]][
                    'AreaShape_MajorAxisLength']

    for bac_indx, bac in fake_daughter_life_history.iterrows():
        df.at[bac_indx, "parent_id"] = 0
        df.at[bac_indx, "id"] = bacterium_id
        df.at[bac_indx, label_col] = new_label_val
        df.at[bac_indx, 'LifeHistory'] = fake_daughter_life_history.shape[0]

        if bac_indx == fake_daughter_life_history.index[0]:
            df.at[bac_indx, parent_image_number_col] = 0
            df.at[bac_indx, parent_object_number_col] = 0

            df.at[bac_indx, 'bacteria_movement'] = ''
            df.at[bac_indx, "difference_neighbors"] = 0
            df.at[bac_indx, "transition"] = True
            df.at[bac_indx, "next_to_first_bac_length_ratio"] = ''
            df.at[bac_indx, "direction_of_motion"] = 0
            df.at[bac_indx, "angle_between_neighbor_motion_bac_motion"] = 0
            df.at[bac_indx, "bac_length_to_back"] = ''
            df.at[bac_indx, "max_daughter_len_to_mother"] = ''
            df.at[bac_indx, "daughter_length_to_mother"] = ''
            df.at[bac_indx, "daughter_orientation"] = ''
            df.at[bac_indx, "daughters_distance"] = ''
            df.at[bac_indx, "mother_last_to_first_bac_length_ratio"] = ''
            df.at[bac_indx, "bac_length_to_back_orientation_changes"] = ''

        if fake_daughter_life_history_next_gens.shape[0] > 0:
            if bac_indx != fake_daughter_life_history.index[0] and bac_indx != fake_daughter_life_history.index[-1]:
                df.at[bac_indx, "next_to_first_bac_length_ratio"] = df.iloc[bac_indx]["AreaShape_MajorAxisLength"] / \
                                                                    df.iloc[fake_daughter_life_history.index[0]][
                                                                        "AreaShape_MajorAxisLength"]
            elif bac_indx != fake_daughter_life_history.index[0]:
                # it means it's the division time
                df.at[bac_indx, "mother_last_to_first_bac_length_ratio"] = \
                    df.iloc[bac_indx]["AreaShape_MajorAxisLength"] / \
                    df.iloc[fake_daughter_life_history.index[0]]["AreaShape_MajorAxisLength"]

            else:
                df.at[bac_indx, "mother_last_to_first_bac_length_ratio"] = ''
                df.at[bac_indx, "next_to_first_bac_length_ratio"] = ''

        else:
            if bac_indx != fake_daughter_life_history.index[0]:
                df.at[bac_indx, "next_to_first_bac_length_ratio"] = df.iloc[bac_indx]["AreaShape_MajorAxisLength"] / \
                                                                    df.iloc[fake_daughter_life_history.index[0]][
                                                                        "AreaShape_MajorAxisLength"]

    for next_gen_bac_ndx, next_gen_bac in fake_daughter_life_history_next_gen.iterrows():
        df.at[next_gen_bac_ndx, 'parent_id'] = bacterium_id

    # Changing the label of the fake girl's family
    for bac_indx, bac in fake_daughter_life_history_next_gens.iterrows():
        df.at[bac_indx, label_col] = new_label_val

    return df


def new_parent_with_two_daughters(df, parent_life_history_before_current_time_step, old_daughter_current_time_step,
                                  new_daughter_current_time_step):
    for parent_bac_indx, parent_bac in parent_life_history_before_current_time_step.iterrows():
        df.at[parent_bac_indx, "divideFlag"] = True
        df.at[parent_bac_indx, "daughters_index"] = [old_daughter_current_time_step.index.values.tolist()[0],
                                                     new_daughter_current_time_step.index.values.tolist()[0]]
        df.at[parent_bac_indx, "division_time"] = new_daughter_current_time_step['ImageNumber'].values.tolist()[0]
        df.at[parent_bac_indx, "bad_division_flag"] = False
        df.at[parent_bac_indx, "LifeHistory"] = parent_life_history_before_current_time_step.shape[0]

        if parent_bac_indx == parent_life_history_before_current_time_step.index[-1]:
            df.at[parent_bac_indx, "unexpected_end"] = False

            df.at[parent_bac_indx, "daughter_length_to_mother"] = \
                (old_daughter_current_time_step["AreaShape_MajorAxisLength"].values.tolist()[0] +
                 new_daughter_current_time_step["AreaShape_MajorAxisLength"].values.tolist()[0]) / \
                parent_bac["AreaShape_MajorAxisLength"]

            df.at[parent_bac_indx, "max_daughter_len_to_mother"] = \
                max(old_daughter_current_time_step["AreaShape_MajorAxisLength"].values.tolist()[0],
                    new_daughter_current_time_step["AreaShape_MajorAxisLength"].values.tolist()[0]) / \
                parent_bac["AreaShape_MajorAxisLength"]

            # angle between daughters
            daughter1_endpoints = (
                find_vertex([old_daughter_current_time_step[center_str + "Center_X"].values.tolist()[0],
                             old_daughter_current_time_step[center_str + "Center_Y"].values.tolist()[0]],
                            old_daughter_current_time_step["AreaShape_MajorAxisLength"].values.tolist()[0],
                            old_daughter_current_time_step["AreaShape_Orientation"].values.tolist()[0]))

            daughter2_endpoints = (
                find_vertex([new_daughter_current_time_step[center_str + "Center_X"].values.tolist()[0],
                             new_daughter_current_time_step[center_str + "Center_Y"].values.tolist()[0]],
                            new_daughter_current_time_step["AreaShape_MajorAxisLength"].values.tolist()[0],
                            new_daughter_current_time_step["AreaShape_Orientation"].values.tolist()[0]))

            slope_daughter1, intercept_daughter1 = calculate_slope_intercept(daughter1_endpoints[0],
                                                                             daughter1_endpoints[1])

            slope_daughter2, intercept_daughter2 = calculate_slope_intercept(daughter2_endpoints[0],
                                                                             daughter2_endpoints[1])

            # Calculate orientation angle
            daughters_orientation_angle = calculate_orientation_angle(slope_daughter1, slope_daughter2)
            df.at[parent_bac_indx, "daughter_orientation"] = daughters_orientation_angle

            if parent_life_history_before_current_time_step.shape[0] > 1:
                df.at[parent_bac_indx, "next_to_first_bac_length_ratio"] = ''

                df.at[parent_bac_indx, "mother_last_to_first_bac_length_ratio"] = \
                    df.iloc[parent_bac_indx]["AreaShape_MajorAxisLength"] / \
                    df.iloc[parent_life_history_before_current_time_step.index[0]]["AreaShape_MajorAxisLength"]

    return df


def new_daughter_modification(df, parent, daughter_life_history, neighbor_df, assign_new_id=True):
    # columns name
    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]
    label_col = [col for col in df.columns if 'TrackObjects_Label_' in col][0]

    daughter_index_first_time_step_life_history = daughter_life_history.index.values.tolist()[0]
    daughter_first_time_step_life_history = df.iloc[daughter_index_first_time_step_life_history]

    daughter_next_genes_family_ndx = find_related_bacteria(df, daughter_first_time_step_life_history,
                                                           daughter_index_first_time_step_life_history,
                                                           bacteria_index_list=None)

    daughter_next_genes_family = df.loc[df.index.isin(daughter_next_genes_family_ndx)]

    daughter_next_gene = df.loc[df['parent_id'] == daughter_life_history['id'].values.tolist()[0]]

    if assign_new_id:
        bacterium_id = max(df['id'].values) + 1
    else:
        bacterium_id = daughter_life_history['id'].values.tolist()[0]

    for daughter_ndx, daughter_bacterium in daughter_life_history.iterrows():

        df.at[daughter_ndx, "LifeHistory"] = daughter_life_history.shape[0]
        df.at[daughter_ndx, "parent_id"] = parent['id']
        df.at[daughter_ndx, label_col] = parent[label_col]

        if assign_new_id:
            df.at[daughter_ndx, "id"] = bacterium_id

        if daughter_ndx == daughter_life_history.index[0]:
            df.at[daughter_ndx, "transition"] = False
            df.at[daughter_ndx, "difference_neighbors"] = check_num_neighbors(df, neighbor_df, parent,
                                                                              daughter_bacterium)
            df.at[daughter_ndx, "bac_length_to_back"] = ''
            df.at[daughter_ndx, "bacteria_movement"] = ''
            df.at[daughter_ndx, "bac_length_to_back_orientation_changes"] = ''
            df.at[daughter_ndx, "next_to_first_bac_length_ratio"] = ''
            df.at[daughter_ndx, "mother_last_to_first_bac_length_ratio"] = ''

            direction_of_motion = \
                calculate_trajectory_direction_angle(
                    np.array([parent[center_str + "Center_X"], parent[center_str + "Center_Y"]]),
                    np.array([df.iloc[daughter_ndx][center_str + "Center_X"],
                              df.iloc[daughter_ndx][center_str + "Center_Y"]]))

            direction_of_motion_vector = \
                calculate_trajectory_direction(
                    np.array([parent[center_str + "Center_X"], parent[center_str + "Center_Y"]]),
                    np.array([df.iloc[daughter_ndx][center_str + "Center_X"],
                              df.iloc[daughter_ndx][center_str + "Center_Y"]]))

            df.at[daughter_ndx, "direction_of_motion"] = direction_of_motion

            neighbors_dir_motion = calc_neighbors_dir_motion(df, parent, neighbor_df)
            if str(neighbors_dir_motion[0]) != 'nan':
                angle_between_motion = calc_normalized_angle_between_motion(neighbors_dir_motion,
                                                                            direction_of_motion_vector)
            else:
                angle_between_motion = 0

            df.at[daughter_ndx, "angle_between_neighbor_motion_bac_motion"] = angle_between_motion

        if daughter_next_gene.shape[0] > 0:
            if daughter_ndx != daughter_life_history.index[0] and daughter_ndx != daughter_life_history.index[-1] > 0:
                df.at[daughter_ndx, "next_to_first_bac_length_ratio"] = \
                    df.iloc[daughter_ndx]["AreaShape_MajorAxisLength"] / \
                    df.iloc[daughter_life_history.index[0]]["AreaShape_MajorAxisLength"]

            elif daughter_ndx != daughter_life_history.index[0]:
                # it means it's the division time
                df.at[daughter_ndx, "mother_last_to_first_bac_length_ratio"] = \
                    df.iloc[daughter_ndx]["AreaShape_MajorAxisLength"] / \
                    df.iloc[daughter_life_history.index[0]]["AreaShape_MajorAxisLength"]

        else:
            if daughter_ndx != daughter_life_history.index[0]:
                df.at[daughter_ndx, "next_to_first_bac_length_ratio"] = \
                    df.iloc[daughter_ndx]["AreaShape_MajorAxisLength"] / \
                    df.iloc[daughter_life_history.index[0]]["AreaShape_MajorAxisLength"]

        if daughter_ndx == daughter_life_history.index[0]:
            df.at[daughter_ndx, parent_image_number_col] = parent['ImageNumber']
            df.at[daughter_ndx, parent_object_number_col] = parent['ObjectNumber']

    # now we should change parent id information of daughters
    prev_daughters_index = daughter_life_history['daughters_index'].values.tolist()[0]
    if len(prev_daughters_index) > 0:
        prev_daughters_first_time_step_df = df.loc[df.index.isin(prev_daughters_index)]
        prev_daughters_id = prev_daughters_first_time_step_df['id'].unique().tolist()
        prev_daughters_life_history_df = df.loc[df['id'].isin(prev_daughters_id)]

        for prev_daughter_indx, prev_daughter_bac in prev_daughters_life_history_df.iterrows():
            df.at[prev_daughter_indx, 'parent_id'] = bacterium_id
            df.at[prev_daughter_indx, label_col] = parent[label_col]

    for bac_ndx, bac in daughter_next_genes_family.iterrows():
        df.at[bac_ndx, label_col] = parent[label_col]

    return df


def parent_modification(df, parent_life_history, daughters, neighbor_df):

    # columns name
    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    grand_parent = df.loc[(df['ImageNumber'] == parent_life_history[parent_image_number_col].values.tolist()[0]) &
                          (df['ObjectNumber'] == parent_life_history[parent_object_number_col].values.tolist()[0])]
    division_frame_df = daughters.loc[
        daughters['ImageNumber'] == parent_life_history['ImageNumber'].values.tolist()[-1] + 1]

    for parent_indx, parent_bac in parent_life_history.iterrows():
        df.at[parent_indx, "divideFlag"] = True
        df.at[parent_indx, "daughters_index"] = division_frame_df.index.values.tolist()
        df.at[parent_indx, "division_time"] = division_frame_df['ImageNumber'].values.tolist()[0]
        df.at[parent_indx, "bad_division_flag"] = False
        df.at[parent_indx, "LifeHistory"] = parent_life_history.shape[0]

        if parent_indx == parent_life_history.index[0]:
            df.at[parent_indx, "next_to_first_bac_length_ratio"] = ''
            df.at[parent_indx, "mother_last_to_first_bac_length_ratio"] = ''

            if grand_parent.shape[0] == 0:
                df.at[parent_indx, "direction_of_motion"] = 0
                df.at[parent_indx, "angle_between_neighbor_motion_bac_motion"] = 0
            else:
                direction_of_motion = \
                    calculate_trajectory_direction_angle(
                        np.array([df.iloc[grand_parent.index.values.tolist()[0]][center_str + "Center_X"],
                                  df.iloc[grand_parent.index.values.tolist()[0]][center_str + "Center_Y"]]),
                        np.array([df.iloc[parent_indx][center_str + "Center_X"],
                                  df.iloc[parent_indx][center_str + "Center_Y"]]))

                direction_of_motion_vector = \
                    calculate_trajectory_direction(
                        np.array([df.iloc[grand_parent.index.values.tolist()[0]][center_str + "Center_X"],
                                  df.iloc[grand_parent.index.values.tolist()[0]][center_str + "Center_Y"]]),
                        np.array([df.iloc[parent_indx][center_str + "Center_X"],
                                  df.iloc[parent_indx][center_str + "Center_Y"]]))

                neighbors_dir_motion = \
                    calc_neighbors_dir_motion(df, df.iloc[grand_parent.index.values.tolist()[0]], neighbor_df)

                if str(neighbors_dir_motion[0]) != 'nan':
                    angle_between_motion = calc_normalized_angle_between_motion(neighbors_dir_motion,
                                                                                direction_of_motion_vector)
                else:
                    angle_between_motion = 0

                df.at[parent_indx, "direction_of_motion"] = direction_of_motion
                df.at[parent_indx, "angle_between_neighbor_motion_bac_motion"] = angle_between_motion

        if parent_indx != parent_life_history.index[0] and parent_indx != parent_life_history.index[-1]:
            df.at[parent_indx, "next_to_first_bac_length_ratio"] = \
                df.iloc[parent_indx]["AreaShape_MajorAxisLength"] / \
                df.iloc[parent_life_history.index[0]]["AreaShape_MajorAxisLength"]
            df.at[parent_indx, "mother_last_to_first_bac_length_ratio"] = ''

        elif parent_indx != parent_life_history.index[-1]:
            parent_bac["next_to_first_bac_length_ratio"] = ''
            # it means it's the division time
            df.at[parent_indx, "mother_last_to_first_bac_length_ratio"] = \
                df.iloc[parent_indx]["AreaShape_MajorAxisLength"] / \
                df.iloc[parent_life_history.index[0]]["AreaShape_MajorAxisLength"]

        if parent_indx == parent_life_history.index[-1]:
            df.at[parent_indx, "unexpected_end"] = False

            df.at[parent_indx, "daughter_length_to_mother"] = \
                sum(division_frame_df["AreaShape_MajorAxisLength"].values.tolist()) / \
                parent_bac["AreaShape_MajorAxisLength"]

            df.at[parent_indx, "max_daughter_len_to_mother"] = \
                max(division_frame_df["AreaShape_MajorAxisLength"].values.tolist()) / \
                parent_bac["AreaShape_MajorAxisLength"]

            daughter1_endpoints = find_vertex([division_frame_df[center_str + "Center_X"].values.tolist()[0],
                                               division_frame_df[center_str + "Center_Y"].values.tolist()[0]],
                                              division_frame_df["AreaShape_MajorAxisLength"].values.tolist()[0],
                                              division_frame_df["AreaShape_Orientation"].values.tolist()[0])

            daughter2_endpoints = find_vertex([division_frame_df[center_str + "Center_X"].values.tolist()[1],
                                               division_frame_df[center_str + "Center_Y"].values.tolist()[1]],
                                              division_frame_df["AreaShape_MajorAxisLength"].values.tolist()[1],
                                              division_frame_df["AreaShape_Orientation"].values.tolist()[1])

            slope_daughter1, intercept_daughter1 = calculate_slope_intercept(daughter1_endpoints[0],
                                                                             daughter1_endpoints[1])

            slope_daughter2, intercept_daughter2 = calculate_slope_intercept(daughter2_endpoints[0],
                                                                             daughter2_endpoints[1])

            # Calculate orientation angle
            daughters_orientation_angle = calculate_orientation_angle(slope_daughter1, slope_daughter2)
            df.at[parent_indx, "daughter_orientation"] = daughters_orientation_angle

    return df


def same_bacterium_modification(df, bac1_life_history, bac2_life_history, neighbor_df):
    try:
        df['AreaShape_Center_X']
        center_str = 'AreaShape_'
    except:
        center_str = 'Location_'      
    
    # bac2 is after bac1

    # columns name
    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]
    label_col = [col for col in df.columns if 'TrackObjects_Label_' in col][0]

    bac2_next_gen_life_history = df.loc[df['parent_id'] == bac2_life_history['id'].values.tolist()[0]]

    bac2_family_from_next_gen = df.loc[(df[label_col] == bac2_life_history[label_col].values.tolist()[0]) &
                                       (df['ImageNumber'] > bac2_life_history['ImageNumber'].max())]

    for bac1_indx, bacterium1 in bac1_life_history.iterrows():
        df.at[bac1_indx, "divideFlag"] = bac2_life_history["divideFlag"].values.tolist()[0]
        df.at[bac1_indx, "daughters_index"] = bac2_life_history["daughters_index"].values.tolist()[0]
        df.at[bac1_indx, "bad_division_flag"] = bac2_life_history["bad_division_flag"].values.tolist()[0]
        df.at[bac1_indx, "division_time"] = bac2_life_history["division_time"].values.tolist()[0]
        df.at[bac1_indx, "LifeHistory"] = bac1_life_history.shape[0] + bac2_life_history.shape[0]

        if bac1_indx == bac1_life_history.index[-1]:
            df.at[bac1_indx, "unexpected_end"] = False
            df.at[bac1_indx, "daughter_orientation"] = ''
            df.at[bac1_indx, "daughter_length_to_mother"] = ''
            df.at[bac1_indx, "max_daughter_len_to_mother"] = ''
            df.at[bac1_indx, "mother_last_to_first_bac_length_ratio"] = ''
            df.at[bac1_indx, "next_to_first_bac_length_ratio"] = \
                df.iloc[bac1_indx]["AreaShape_MajorAxisLength"] / \
                bac1_life_history["AreaShape_MajorAxisLength"].values.tolist()[0]

    for bac2_ndx, bacterium2 in bac2_life_history.iterrows():
        df.at[bac2_ndx, 'id'] = bac1_life_history['id'].values.tolist()[0]
        df.at[bac2_ndx, 'parent_id'] = bac1_life_history['parent_id'].values.tolist()[0]
        df.at[bac2_ndx, "LifeHistory"] = bac1_life_history.shape[0] + bac2_life_history.shape[0]
        df.at[bac2_ndx, label_col] = bac1_life_history[label_col].values.tolist()[0]
        df.at[bac2_ndx, 'transition'] = False

        if bac2_next_gen_life_history.shape[0] > 0:
            if bac2_ndx != bac2_life_history.index[-1]:

                df.at[bac2_ndx, "mother_last_to_first_bac_length_ratio"] = ''

                df.at[bac2_ndx, "next_to_first_bac_length_ratio"] = \
                    df.iloc[bac2_ndx]["AreaShape_MajorAxisLength"] / \
                    bac1_life_history["AreaShape_MajorAxisLength"].values.tolist()[0]

            else:
                df.at[bac2_ndx, "next_to_first_bac_length_ratio"] = ''

                df.at[bac2_ndx, "mother_last_to_first_bac_length_ratio"] = \
                    df.iloc[bac2_ndx]["AreaShape_MajorAxisLength"] / \
                    bac1_life_history["AreaShape_MajorAxisLength"].values.tolist()[0]
        else:
            df.at[bac2_ndx, "mother_last_to_first_bac_length_ratio"] = ''
            df.at[bac2_ndx, "next_to_first_bac_length_ratio"] = \
                df.iloc[bac2_ndx]["AreaShape_MajorAxisLength"] / \
                bac1_life_history["AreaShape_MajorAxisLength"].values.tolist()[0]

        if bac2_ndx == bac2_life_history.index[0]:
            df.at[bac2_ndx, parent_image_number_col] = bac1_life_history["ImageNumber"].values.tolist()[-1]
            df.at[bac2_ndx, parent_object_number_col] = bac1_life_history["ObjectNumber"].values.tolist()[-1]

            df.at[bac2_ndx, "difference_neighbors"] = check_num_neighbors(df, neighbor_df, bacterium1, bacterium2)

            endpoint1_1_movement = \
                np.sqrt((bac2_life_history["endppoint1_X"].values.tolist()[0] -
                         bac1_life_history["endppoint1_X"].values.tolist()[-1]) ** 2 +
                        (bac2_life_history["endppoint1_Y"].values.tolist()[0] -
                         bac1_life_history["endppoint1_Y"].values.tolist()[-1]) ** 2)

            endpoint1_endpoint2_movement = \
                np.sqrt((bac2_life_history["endppoint1_X"].values.tolist()[0] -
                         bac1_life_history["endppoint2_X"].values.tolist()[-1]) ** 2 +
                        (bac2_life_history["endppoint1_Y"].values.tolist()[0] -
                         bac1_life_history["endppoint2_Y"].values.tolist()[-1]) ** 2)

            endpoint2_2_movement = \
                np.sqrt((bac2_life_history["endppoint2_X"].values.tolist()[0] -
                         bac1_life_history["endppoint2_X"].values.tolist()[-1]) ** 2 +
                        (bac2_life_history["endppoint2_X"].values.tolist()[0] -
                         bac1_life_history["endppoint2_Y"].values.tolist()[-1]) ** 2)

            endpoint2_endpoint1_movement = \
                np.sqrt((bac2_life_history["endppoint2_X"].values.tolist()[0] -
                         bac1_life_history["endppoint1_X"].values.tolist()[-1]) ** 2 +
                        (bac2_life_history["endppoint2_X"].values.tolist()[0] -
                         bac1_life_history["endppoint1_Y"].values.tolist()[-1]) ** 2)

            center_movement = \
                np.sqrt((bac2_life_history[center_str + "Center_X"].values.tolist()[0] -
                         bac1_life_history[center_str + "Center_X"].values.tolist()[-1]) ** 2 +
                        (bac2_life_history[center_str + "Center_Y"].values.tolist()[0] -
                         bac1_life_history[center_str + "Center_Y"].values.tolist()[-1]) ** 2)

            df.at[bac2_ndx, "bacteria_movement"] = min(center_movement, endpoint1_1_movement, endpoint2_2_movement,
                                                       endpoint1_endpoint2_movement, endpoint2_endpoint1_movement)

            direction_of_motion = \
                calculate_trajectory_direction_angle(
                    np.array([bac1_life_history[center_str + "Center_X"].values.tolist()[-1],
                              bac1_life_history[center_str + "Center_Y"].values.tolist()[-1]]),
                    np.array([bac2_life_history[center_str + "Center_X"].values.tolist()[0],
                              bac2_life_history[center_str + "Center_Y"].values.tolist()[0]]))

            direction_of_motion_vector = \
                calculate_trajectory_direction(
                    np.array([bac1_life_history[center_str + "Center_X"].values.tolist()[-1],
                              bac1_life_history[center_str + "Center_Y"].values.tolist()[-1]]),
                    np.array([bac2_life_history[center_str + "Center_X"].values.tolist()[0],
                              bac2_life_history[center_str + "Center_Y"].values.tolist()[0]]))

            neighbors_dir_motion = \
                calc_neighbors_dir_motion(df, df.iloc[bac1_life_history.index.values.tolist()[-1]], neighbor_df)

            if str(neighbors_dir_motion[0]) != 'nan':
                angle_between_motion = calc_normalized_angle_between_motion(neighbors_dir_motion,
                                                                            direction_of_motion_vector)
            else:
                angle_between_motion = 0

            df.at[bac2_ndx, "direction_of_motion"] = direction_of_motion
            df.at[bac2_ndx, "angle_between_neighbor_motion_bac_motion"] = angle_between_motion

            if bac2_life_history.shape[0] > 1 or bac2_next_gen_life_history.shape[0] == 0:
                df.at[bac2_ndx, "bac_length_to_back"] = bac2_life_history["AreaShape_MajorAxisLength"].values.tolist()[
                                                            0] / \
                                                        bac1_life_history["AreaShape_MajorAxisLength"].values.tolist()[
                                                            -1]

            current_bacterium_endpoints = find_vertex(
                [bac2_life_history[center_str + "Center_X"].values.tolist()[0],
                 bac2_life_history[center_str + "Center_Y"].values.tolist()[0]],
                bac2_life_history["AreaShape_MajorAxisLength"].values.tolist()[0],
                bac2_life_history["AreaShape_Orientation"].values.tolist()[0])

            prev_bacterium_endpoints = find_vertex(
                [bac1_life_history[center_str + "Center_X"].values.tolist()[-1],
                 bac1_life_history[center_str + "Center_Y"].values.tolist()[-1]],
                bac1_life_history["AreaShape_MajorAxisLength"].values.tolist()[-1],
                bac1_life_history["AreaShape_Orientation"].values.tolist()[-1])

            # Convert points to vectors

            # Calculate slopes and intercepts
            slope1, intercept1 = calculate_slope_intercept(current_bacterium_endpoints[0],
                                                           current_bacterium_endpoints[1])
            slope2, intercept2 = calculate_slope_intercept(prev_bacterium_endpoints[0],
                                                           prev_bacterium_endpoints[1])

            orientation_angle = calculate_orientation_angle(slope1, slope2)

            df.at[bac2_ndx, "bac_length_to_back_orientation_changes"] = orientation_angle

    for bac2_next_gen_ndx, bac2_nex_gen in bac2_next_gen_life_history.iterrows():
        df.at[bac2_next_gen_ndx, 'parent_id'] = bac1_life_history['id'].values.tolist()[0]

    for bac_ndx, bac in bac2_family_from_next_gen.iterrows():
        df.at[bac_ndx, label_col] = bac1_life_history[label_col].values.tolist()[0]

    return df


def modify_fake_relation(df, bac2_fake_parent_life_history, another_daughter_life_history_from_fake_parent, neighbor_df,
                         check_daughter=False):

    if check_daughter:
        if len(another_daughter_life_history_from_fake_parent['id'].unique()) > 1:
            df = parent_modification(df, bac2_fake_parent_life_history, another_daughter_life_history_from_fake_parent,
                                     neighbor_df)

        elif len(another_daughter_life_history_from_fake_parent['id'].unique()) == 1:
            # it means that daughter & parent are same
            df = same_bacterium_modification(df, bac2_fake_parent_life_history,
                                             another_daughter_life_history_from_fake_parent, neighbor_df)

        else:
            for bac_ndx, bacterium in bac2_fake_parent_life_history.iterrows():
                df.at[bac_ndx, 'divideFlag'] = False
                df.at[bac_ndx, 'daughters_index'] = ''
                df.at[bac_ndx, 'bad_division_flag'] = False
                df.at[bac_ndx, 'division_time'] = 0
                df.at[bac_ndx, 'LifeHistory'] = bac2_fake_parent_life_history.shape[0]

                if bac_ndx == bac2_fake_parent_life_history.index[-1]:
                    if bac2_fake_parent_life_history['ImageNumber'].values.tolist()[-1] != \
                            df['ImageNumber'].values.tolist()[-1]:
                        df.at[bac_ndx, 'unexpected_end'] = True
                    df.at[bac_ndx, 'daughter_distance_to_mother'] = ''
                    df.at[bac_ndx, "max_daughter_len_to_mother"] = ''
                    df.at[bac_ndx, "daughter_orientation"] = ''
                    df.at[bac_ndx, "daughter_length_to_mother"] = ''
                    df.at[bac_ndx, 'daughters_distance'] = ''
                    df.at[bac_ndx, 'mother_last_to_first_bac_length_ratio'] = ''
                    df.at[bac_ndx, "next_to_first_bac_length_ratio"] = \
                        bac2_fake_parent_life_history.loc[bac2_fake_parent_life_history.index[-1]][
                            'AreaShape_MajorAxisLength'] / \
                        bac2_fake_parent_life_history.loc[bac2_fake_parent_life_history.index[0]][
                            'AreaShape_MajorAxisLength']

    else:
        for bac_ndx, bacterium in bac2_fake_parent_life_history.iterrows():
            df.at[bac_ndx, 'divideFlag'] = False
            df.at[bac_ndx, 'daughters_index'] = ''
            df.at[bac_ndx, 'bad_division_flag'] = False
            df.at[bac_ndx, 'division_time'] = 0
            df.at[bac_ndx, 'LifeHistory'] = bac2_fake_parent_life_history.shape[0]

            if bac_ndx == bac2_fake_parent_life_history.index[-1]:
                if bac2_fake_parent_life_history['ImageNumber'].values.tolist()[-1] != \
                        df['ImageNumber'].values.tolist()[-1]:
                    df.at[bac_ndx, 'unexpected_end'] = True
                df.at[bac_ndx, 'daughter_distance_to_mother'] = ''
                df.at[bac_ndx, "max_daughter_len_to_mother"] = ''
                df.at[bac_ndx, "daughter_orientation"] = ''
                df.at[bac_ndx, "daughter_length_to_mother"] = ''
                df.at[bac_ndx, 'daughters_distance'] = ''
                df.at[bac_ndx, 'mother_last_to_first_bac_length_ratio'] = ''
                df.at[bac_ndx, "next_to_first_bac_length_ratio"] = \
                    bac2_fake_parent_life_history.loc[bac2_fake_parent_life_history.index[-1]]['AreaShape_MajorAxisLength'] / \
                    bac2_fake_parent_life_history.loc[bac2_fake_parent_life_history.index[0]][
                        'AreaShape_MajorAxisLength']

    return df


def bacteria_modification(df, bac1, bac2_life_history, all_bac_undergo_phase_change, neighbor_df):
    # I want to assign bac2 to bac1
    # bac1 is parent

    if bac1 is not None:

        bac2_in_current_time_step = \
            bac2_life_history.loc[bac2_life_history['ImageNumber'] ==
                                  all_bac_undergo_phase_change['ImageNumber'].values.tolist()[0]]

        bac2_after_current_time_step = \
            bac2_life_history.loc[bac2_life_history['ImageNumber'] >=
                                  all_bac_undergo_phase_change['ImageNumber'].values.tolist()[0]]

        bac2_before_current_time_step = \
            bac2_life_history.loc[bac2_life_history['ImageNumber'] <
                                  all_bac_undergo_phase_change['ImageNumber'].values.tolist()[0]]

        bac1_next_gen_after_bac2_time_step = df.loc[(df['id'] == bac1['id']) &
                                                    (df['ImageNumber'] >=
                                                     bac2_in_current_time_step['ImageNumber'].values.tolist()[0])]

        # it means that the link is parent-daughter link
        same_bac1_in_bac2_time_step = df.loc[(df['id'] == bac1['id']) &
                                             (df['ImageNumber'] ==
                                              bac2_in_current_time_step['ImageNumber'].values.tolist()[0])]

        # for parent bacterium
        same_bac1_before_bac2_time_step = df.loc[(df['id'] == bac1['id']) &
                                                 (df['ImageNumber'] <
                                                  bac2_in_current_time_step['ImageNumber'].values.tolist()[0])]

        if bac1_next_gen_after_bac2_time_step.shape[0] > 0:

            df = new_parent_with_two_daughters(df, same_bac1_before_bac2_time_step, same_bac1_in_bac2_time_step,
                                               bac2_in_current_time_step)
            df = new_daughter_modification(df, bac1, bac1_next_gen_after_bac2_time_step, neighbor_df,
                                           assign_new_id=True)
            df = new_daughter_modification(df, bac1, bac2_after_current_time_step, neighbor_df, assign_new_id=False)

        else:
            df = same_bacterium_modification(df, same_bac1_before_bac2_time_step, bac2_after_current_time_step,
                                             neighbor_df)

        if bac2_before_current_time_step.shape[0] > 0:
            df = modify_fake_relation(df, bac2_before_current_time_step, None,
                                      neighbor_df=neighbor_df, check_daughter=False)

        else:
            bac2_fake_parent_life_history = df.loc[
                df['id'] == bac2_in_current_time_step['parent_id'].values.tolist()[0]]

            if bac2_fake_parent_life_history.shape[0] > 0:

                if bac2_life_history['parent_id'].values.tolist()[0] != 0:
                    if bac2_fake_parent_life_history['division_time'].values.tolist()[0] == \
                            bac2_life_history['ImageNumber'].values.tolist()[0]:
                        another_daughter_life_history_from_fake_parent = \
                            df.loc[(df['parent_id'] == bac2_life_history['parent_id'].values.tolist()[0]) &
                                   (df['id'] != bac2_life_history['id'].values.tolist()[0])]
                    else:
                        another_daughter_life_history_from_fake_parent = pd.DataFrame(columns=df.columns)
                else:
                    another_daughter_life_history_from_fake_parent = pd.DataFrame(columns=df.columns)

                df = modify_fake_relation(df, bac2_fake_parent_life_history,
                                          another_daughter_life_history_from_fake_parent, neighbor_df=neighbor_df,
                                          check_daughter=True)
    else:
        # transition
        bac2_fake_parent_life_history = df.loc[df['id'] == bac2_life_history['parent_id'].values.tolist()[0]]

        bac2_life_history_before_this_time_step = df.loc[(df['id'] == bac2_life_history['id'].values.tolist()[0]) &
                                                         (df['ImageNumber'] < bac2_life_history['ImageNumber'].values.tolist()[0])]

        if bac2_life_history_before_this_time_step.shape[0] > 0:
            bac2_fake_parent_life_history = bac2_life_history_before_this_time_step
            another_daughter_life_history_from_fake_parent = pd.DataFrame(columns=df.columns)

        elif bac2_life_history['parent_id'].values.tolist()[0] != 0:
            if bac2_fake_parent_life_history['division_time'].values.tolist()[0] == \
                    bac2_life_history['ImageNumber'].values.tolist()[0]:
                another_daughter_life_history_from_fake_parent = \
                    df.loc[(df['parent_id'] == bac2_life_history['parent_id'].values.tolist()[0]) &
                           (df['id'] != bac2_life_history['id'].values.tolist()[0])]
            else:
                another_daughter_life_history_from_fake_parent = pd.DataFrame(columns=df.columns)
        else:
            another_daughter_life_history_from_fake_parent = pd.DataFrame(columns=df.columns)

        df = new_transition_bacteria(df, bac2_life_history)

        df = modify_fake_relation(df, bac2_fake_parent_life_history,
                                  another_daughter_life_history_from_fake_parent, neighbor_df=neighbor_df,
                                  check_daughter=True)

    return df


def remove_redundant_link(dataframe, wrong_daughter_life_history, neighbor_df):
    dataframe = bacteria_modification(dataframe, None, wrong_daughter_life_history, None,
                                      neighbor_df=neighbor_df)

    return dataframe


def remove_bac(dataframe, noise_bac_ndx, noise_obj, neighbor_df):
    parent_image_number_col = [col for col in dataframe.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in dataframe.columns if 'TrackObjects_ParentObjectNumber_' in col][0]
    label_col = [col for col in dataframe.columns if 'TrackObjects_Label_' in col][0]

    life_history_before_noise = dataframe.loc[(dataframe['id'] == noise_obj['id']) &
                                              (dataframe['ImageNumber'] < noise_obj['ImageNumber'])]

    life_history_after_noise = dataframe.loc[(dataframe['id'] == noise_obj['id']) &
                                             (dataframe['ImageNumber'] > noise_obj['ImageNumber'])]

    parent_of_noise_obj = dataframe.loc[dataframe['id'] == noise_obj['parent_id']]

    daughters_of_noise_obj = dataframe.loc[dataframe['parent_id'] == noise_obj['id']]

    dataframe.at[noise_bac_ndx, 'id'] = max(dataframe['id'].values) + 1
    dataframe.at[noise_bac_ndx, 'parent_id'] = 0
    dataframe.at[noise_bac_ndx, parent_image_number_col] = 0
    dataframe.at[noise_bac_ndx, parent_object_number_col] = 0
    dataframe.at[noise_bac_ndx, label_col] = 0

    if life_history_before_noise.shape[0] > 0:
        bacterium_id = max(dataframe['id'].values) + 1

        for bac_before_ndx, bac_before in life_history_before_noise.iterrows():
            dataframe.at[bac_before_ndx, 'divideFlag'] = False
            dataframe.at[bac_before_ndx, 'daughters_index'] = ''
            dataframe.at[bac_before_ndx, 'bad_division_flag'] = False
            dataframe.at[bac_before_ndx, 'division_time'] = 0
            dataframe.at[bac_before_ndx, 'LifeHistory'] = life_history_before_noise.shape[0]
            dataframe.at[bac_before_ndx, 'id'] = bacterium_id

            if bac_before_ndx == life_history_before_noise.index[-1]:
                dataframe.at[bac_before_ndx, 'unexpected_end'] = True

    if life_history_after_noise.shape[0] > 0:
        dataframe = new_transition_bacteria(dataframe, life_history_after_noise)

    elif daughters_of_noise_obj.shape[0] > 0:
        for daughter_ndx, daughter_bac in daughters_of_noise_obj.iterrows():
            daughter_life_history = dataframe.loc[dataframe['id'] == daughter_bac['id']]
            dataframe = new_transition_bacteria(dataframe, daughter_life_history)

    if parent_of_noise_obj.shape[0] > 0:
        another_daughter_life_history_from_parent = \
            dataframe.loc[(dataframe['parent_id'] == noise_obj['parent_id']) & (dataframe['id'] != noise_obj['id'])]

        dataframe = modify_fake_relation(dataframe, parent_of_noise_obj, another_daughter_life_history_from_parent,
                                         neighbor_df=neighbor_df, check_daughter=True)

    return dataframe
