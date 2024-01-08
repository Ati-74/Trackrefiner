import numpy as np
from CellProfilerAnalysis.strain.correction.action.helperFunctions import bacteria_life_history, convert_to_um, find_vertex, \
    angle_convert_to_radian


def calculate_slope_intercept(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def calculate_orientation_angle(slope1, slope2):
    # Calculate the angle in radians between the lines
    angle_radians = np.arctan(abs((slope1 - slope2) / (1 + slope1 * slope2)))
    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def new_transition_bacteria(df, fake_daughter_life_history):

    # columns name
    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]
    label_col = [col for col in df.columns if 'TrackObjects_Label_' in col][0]

    fake_daughter_life_history_next_gens = \
        df.loc[(df[label_col] == fake_daughter_life_history[label_col].values.tolist()[0]) &
               (df['ImageNumber'] > fake_daughter_life_history['ImageNumber'].max())]

    new_label_val = max(df[label_col].values) + 1

    for bac_indx, bac in fake_daughter_life_history.iterrows():
        df.at[bac_indx, "parent_id"] = 0
        df.at[bac_indx, parent_image_number_col] = 0
        df.at[bac_indx, parent_object_number_col] = 0
        df.at[bac_indx, label_col] = new_label_val

        if bac_indx == fake_daughter_life_history.index[0]:
            df.at[bac_indx, "transition"] = True

    # Changing the label of the fake girl's family
    for bac_indx, bac in fake_daughter_life_history_next_gens.iterrows():
        df.at[bac_indx, label_col] = new_label_val

    return df


def new_parent_with_two_daughters(df, parent_life_history_before_current_time_step, old_daughter_current_time_step,
                                  new_daughter_current_time_step):

    for parent_bac_indx, parent_bac in parent_life_history_before_current_time_step.iterrows():
        df.at[parent_bac_indx, "divideFlag"] = True
        df.at[parent_bac_indx, "daughters_index"] = [old_daughter_current_time_step.index.values[0],
                                                     new_daughter_current_time_step.index.values[0]]
        df.at[parent_bac_indx, "division_time"] = new_daughter_current_time_step['ImageNumber'].values[0]
        df.at[parent_bac_indx, "bad_division_flag"] = False
        df.at[parent_bac_indx, "LifeHistory"] = parent_life_history_before_current_time_step.shape[0]

        if parent_bac_indx == parent_life_history_before_current_time_step.index[-1]:
            df.at[parent_bac_indx, "daughter_length_to_mother"] = \
                (old_daughter_current_time_step["AreaShape_MajorAxisLength"].values[0] +
                 new_daughter_current_time_step["AreaShape_MajorAxisLength"].values[0]) / \
                parent_bac["AreaShape_MajorAxisLength"]
            df.at[parent_bac_indx, "max_daughter_len_to_mother"] = \
                max(old_daughter_current_time_step["AreaShape_MajorAxisLength"].values[0],
                    new_daughter_current_time_step["AreaShape_MajorAxisLength"].values[0]) / \
                parent_bac["AreaShape_MajorAxisLength"]
            df.at[parent_bac_indx, "unexpected_end"] = False
            # df.at[parent_bac_indx, "bac_length_to_back"] = ''

    return df


def new_daughter_modification(df, parent, daughter_life_history, assign_new_id=True):

    # columns name
    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]
    label_col = [col for col in df.columns if 'TrackObjects_Label_' in col][0]

    daughter_next_genes_family = df.loc[(df[label_col] == daughter_life_history[label_col].values.tolist()[0]) &
                                       (df['ImageNumber'] > daughter_life_history['ImageNumber'].max())]

    if assign_new_id:
        bacterium_id = max(df['id'].values) + 1
    else:
        bacterium_id = daughter_life_history['id'].values[0]

    for daughter_ndx, daughter_bacterium in daughter_life_history.iterrows():

        df.at[daughter_ndx, "LifeHistory"] = daughter_life_history.shape[0]
        df.at[daughter_ndx, "parent_id"] = parent['id']
        df.at[daughter_ndx, label_col] = parent[label_col]

        if assign_new_id:
            df.at[daughter_ndx, "id"] = bacterium_id

        if daughter_ndx == daughter_life_history.index[0]:
            df.at[daughter_ndx, "bac_length_to_back"] = ''
            df.at[daughter_ndx, "bacteria_movement"] = ''
            df.at[daughter_ndx, "bac_length_to_back_orientation_changes"] = ''

        if daughter_ndx == daughter_life_history.index[-1]:
            df.at[daughter_ndx, "transition"] = False

        if daughter_ndx == daughter_life_history.index[0]:
            df.at[daughter_ndx, parent_image_number_col] = parent['ImageNumber']
            df.at[daughter_ndx, parent_object_number_col] = parent['ObjectNumber']

    # now we should change parent id information of daughters
    prev_daughters_index = daughter_life_history['daughters_index'].values[0]
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


def parent_modification(df, parent_life_history, daughters):

    division_frame_df = daughters.loc[daughters['ImageNumber'] == parent_life_history['ImageNumber'].values[-1] + 1]

    for parent_indx, parent_bac in parent_life_history.iterrows():
        df.at[parent_indx, "divideFlag"] = True
        df.at[parent_indx, "daughters_index"] = division_frame_df.index.values.tolist()
        df.at[parent_indx, "division_time"] = division_frame_df['ImageNumber'].values[0]
        df.at[parent_indx, "bad_division_flag"] = False
        df.at[parent_indx, "LifeHistory"] = parent_life_history.shape[0]

        if parent_indx == parent_life_history.index[-1]:
            df.at[parent_indx, "daughter_length_to_mother"] = \
                sum(division_frame_df["AreaShape_MajorAxisLength"].values.tolist()) / \
                parent_bac["AreaShape_MajorAxisLength"]

            df.at[parent_indx, "max_daughter_len_to_mother"] = \
                max(division_frame_df["AreaShape_MajorAxisLength"].values.tolist()) / \
                parent_bac["AreaShape_MajorAxisLength"]

            df.at[parent_indx, "unexpected_end"] = False
            # df.at[parent_indx, "bac_length_to_back"] = ''

    return df


def same_bacterium_modification(df, bac1_life_history, bac2_life_history):
    # bac2 is after bac1

    # columns name
    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]
    label_col = [col for col in df.columns if 'TrackObjects_Label_' in col][0]

    bac2_next_gen_life_gistory = df.loc[df['parent_id'] == bac2_life_history['id'].values.tolist()[0]]

    bac2_family_from_next_gen = df.loc[(df[label_col] == bac2_life_history[label_col].values.tolist()[0]) &
                                       (df['ImageNumber'] > bac2_life_history['ImageNumber'].max())]

    for bac1_indx, bacterium1 in bac1_life_history.iterrows():
        df.at[bac1_indx, "divideFlag"] = bac2_life_history["divideFlag"].values[0]
        df.at[bac1_indx, "daughters_index"] = bac2_life_history["daughters_index"].values[0]
        df.at[bac1_indx, "bad_division_flag"] = bac2_life_history["bad_division_flag"].values[0]
        df.at[bac1_indx, "division_time"] = bac2_life_history["division_time"].values[0]
        df.at[bac1_indx, "LifeHistory"] = bac1_life_history.shape[0] + bac2_life_history.shape[0]

        if bac1_indx == bac1_life_history.index[-1]:
            df.at[bac1_indx, "unexpected_end"] = False
            df.at[bac1_indx, "daughter_length_to_mother"] = ''
            df.at[bac1_indx, "max_daughter_len_to_mother"] = ''

    for bac2_ndx, bacterium2 in bac2_life_history.iterrows():
        df.at[bac2_ndx, 'id'] = bac1_life_history['id'].values[0]
        df.at[bac2_ndx, 'parent_id'] = bac1_life_history['parent_id'].values[0]
        df.at[bac2_ndx, "LifeHistory"] = bac1_life_history.shape[0] + bac2_life_history.shape[0]
        df.at[bac2_ndx, label_col] = bac1_life_history[label_col].values[0]

        if bac2_ndx == bac2_life_history.index[0]:
            df.at[bac2_ndx, parent_image_number_col] = bac1_life_history["ImageNumber"].values[-1]
            df.at[bac2_ndx, parent_object_number_col] = bac1_life_history["ObjectNumber"].values[-1]

            df.at[bac2_ndx, "bac_length_to_back"] = bac2_life_history["AreaShape_MajorAxisLength"].values[0] / \
                                                    bac1_life_history["AreaShape_MajorAxisLength"].values[-1]
            df.at[bac2_ndx, "bacteria_movement"] = \
                np.sqrt((bac2_life_history["AreaShape_Center_X"].values[0] -
                         bac1_life_history["AreaShape_Center_X"].values[-1]) ** 2 +
                        (bac2_life_history["AreaShape_Center_Y"].values[0] -
                         bac1_life_history["AreaShape_Center_Y"].values[-1]) ** 2)

            current_bacterium_endpoints = find_vertex(
                [bac2_life_history["AreaShape_Center_X"].values[0], bac2_life_history["AreaShape_Center_Y"].values[0]],
                bac2_life_history["AreaShape_MajorAxisLength"].values[0],
                bac2_life_history["AreaShape_Orientation"].values[0])

            prev_bacterium_endpoints = find_vertex(
                [bac1_life_history["AreaShape_Center_X"].values[-1], bac1_life_history["AreaShape_Center_Y"].values[-1]],
                bac1_life_history["AreaShape_MajorAxisLength"].values[-1],
                bac1_life_history["AreaShape_Orientation"].values[-1])

            # Convert points to vectors

            # Calculate slopes and intercepts
            slope1, intercept1 = calculate_slope_intercept(current_bacterium_endpoints[0],
                                                           current_bacterium_endpoints[1])
            slope2, intercept2 = calculate_slope_intercept(prev_bacterium_endpoints[0],
                                                           prev_bacterium_endpoints[1])

            orientation_angle = calculate_orientation_angle(slope1, slope2)

            df.at[bac2_ndx, "bac_length_to_back_orientation_changes"] = orientation_angle

    for bac2_next_gen_ndx, bac2_nex_gen in bac2_next_gen_life_gistory.iterrows():

        df.at[bac2_next_gen_ndx, 'parent_id'] = bac1_life_history['id'].values[0]

    for bac_ndx, bac in bac2_family_from_next_gen.iterrows():
        df.at[bac_ndx, label_col] = bac1_life_history[label_col].values[0]

    return df


def modify_fake_relation(df, bac2_fake_parent_life_history, another_daughter_life_history_from_fake_parent,
                         check_daughter=False):
    if check_daughter:
        if len(another_daughter_life_history_from_fake_parent['id'].unique()) > 1:
            df = parent_modification(df, bac2_fake_parent_life_history, another_daughter_life_history_from_fake_parent)

        elif len(another_daughter_life_history_from_fake_parent['id'].unique()) == 1:
            # it means that daughter & parent are same
            df = same_bacterium_modification(df, bac2_fake_parent_life_history,
                                             another_daughter_life_history_from_fake_parent)
    else:
        new_id = max(df['id'].values) + 1
        for bac_ndx, bacterium in bac2_fake_parent_life_history.iterrows():
            df.at[bac_ndx, 'id'] = new_id
            df.at[bac_ndx, 'divideFlag'] = False
            df.at[bac_ndx, 'daughters_index'] = ''
            df.at[bac_ndx, 'bad_division_flag'] = False
            df.at[bac_ndx, 'division_time'] = 0
            df.at[bac_ndx, 'LifeHistory'] = bac2_fake_parent_life_history.shape[0]

            if bac_ndx == bac2_fake_parent_life_history.index[-1]:
                df.at[bac_ndx, 'unexpected_end'] = True
                df.at[bac_ndx, 'daughter_distance_to_mother'] = ''
                df.at[bac_ndx, "max_daughter_len_to_mother"] = ''

    return df


def bacteria_modification(df, bac1, bac2_life_history, all_bac_undergo_phase_change):
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
                                                     bac2_in_current_time_step['ImageNumber'].values[0])]

        # it means that the link is parent-daughter link
        same_bac1_in_bac2_time_step = df.loc[(df['id'] == bac1['id']) &
                                             (df['ImageNumber'] == bac2_in_current_time_step['ImageNumber'].values[0])]

        # for parent bacterium
        same_bac1_before_bac2_time_step = df.loc[(df['id'] == bac1['id']) &
                                                 (df['ImageNumber'] < bac2_in_current_time_step['ImageNumber'].values[0])]

        if bac1_next_gen_after_bac2_time_step.shape[0] > 0:

            df = new_parent_with_two_daughters(df, same_bac1_before_bac2_time_step, same_bac1_in_bac2_time_step,
                                               bac2_in_current_time_step)
            df = new_daughter_modification(df, bac1, bac1_next_gen_after_bac2_time_step, assign_new_id=True)
            df = new_daughter_modification(df, bac1, bac2_after_current_time_step, assign_new_id=False)

        else:
            df = same_bacterium_modification(df, same_bac1_before_bac2_time_step, bac2_after_current_time_step)

        if bac2_before_current_time_step.shape[0] > 0:
            df = modify_fake_relation(df, bac2_before_current_time_step, None, check_daughter=False)

        else:
            bac2_fake_parent_life_history = df.loc[df['id'] == bac2_in_current_time_step['parent_id'].values[0]]

            if bac2_fake_parent_life_history.shape[0] > 0:

                another_daughter_life_history_from_fake_parent = \
                    df.loc[(df['parent_id'] == bac2_in_current_time_step['parent_id'].values[0]) &
                           (df['id'] != bac2_in_current_time_step['id'].values[0])]

                df = modify_fake_relation(df, bac2_fake_parent_life_history,
                                          another_daughter_life_history_from_fake_parent, check_daughter=True)
    else:
        # transition
        bac2_fake_parent_life_history = df.loc[df['id'] == bac2_life_history['parent_id'].values[0]]

        another_daughter_life_history_from_fake_parent = \
            df.loc[(df['parent_id'] == bac2_life_history['parent_id'].values[0]) &
                   (df['id'] != bac2_life_history['id'].values[0])]

        df = new_transition_bacteria(df, bac2_life_history)

        df = modify_fake_relation(df, bac2_fake_parent_life_history,
                                  another_daughter_life_history_from_fake_parent, check_daughter=True)

    return df


def remove_redundant_link(dataframe, wrong_daughter_life_history):

    dataframe = bacteria_modification(dataframe, None, wrong_daughter_life_history, None)

    return dataframe


def remove_bac(dataframe, noise_obj):

    life_history_before_noise = dataframe.loc[(dataframe['id'] == noise_obj['id']) &
                                              (dataframe['ImageNumber'] < noise_obj['ImageNumber'])]

    life_history_after_noise = dataframe.loc[(dataframe['id'] == noise_obj['id']) &
                                             (dataframe['ImageNumber'] > noise_obj['ImageNumber'])]

    parent_of_noise_obj = dataframe.loc[dataframe['id'] == noise_obj['parent_id']]

    daughters_of_noise_obj = dataframe.loc[dataframe['parent_id'] == noise_obj['id']]

    if parent_of_noise_obj.shape[0] > 0:
        another_daughter_life_history_from_parent = \
            dataframe.loc[(dataframe['parent_id'] == noise_obj['parent_id']) & (dataframe['id'] != noise_obj['id'])]

        dataframe = modify_fake_relation(dataframe, parent_of_noise_obj, another_daughter_life_history_from_parent,
                                         check_daughter=True)

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

    if daughters_of_noise_obj.shape[0] > 0:
        for daughter_ndx, daughter_bac in daughters_of_noise_obj.iterrows():
            daughter_life_history = dataframe.loc[dataframe['id'] == daughter_bac['id']]
            dataframe = new_transition_bacteria(dataframe, daughter_life_history)

    return dataframe
