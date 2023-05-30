import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix


# find vertices of an ellipse (bacteria):
# references:
# https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate
# https://math.stackexchange.com/questions/2645689/what-is-the-parametric-equation-of-a-rotated-ellipse-given-the-angle-of-rotatio
def find_vertex(center, major, angle_rotation, angle_tolerance=1e-6):

    if np.abs(angle_rotation - np.pi / 2) < angle_tolerance:  # Bacteria parallel to the vertical axis
        vertex_1_x = center[0]
        vertex_1_y = center[1] + major
        vertex_2_x = center[0]
        vertex_2_y = center[1] - major
    elif np.abs(angle_rotation) < angle_tolerance:  # Bacteria parallel to the horizontal axis
        vertex_1_x = center[0] + major
        vertex_1_y = center[1]
        vertex_2_x = center[0] - major
        vertex_2_y = center[1]
    else:
        # (x- center_x) * np.sin(angle_rotation) - (y-center_y) * np.cos(angle_rotation) = 0
        # np.power((x - center_x) * np.cos(angle_rotation) + (y - center_y) * np.sin(angle_rotation), 2) =
        # np.power(major, 2)
        semi_major = major / 2
        vertex_1_x = float(semi_major / (np.cos(angle_rotation) + np.tan(angle_rotation) * np.sin(angle_rotation)) + center[0])
        vertex_1_y = float((vertex_1_x - center[0]) * np.tan(angle_rotation) + center[1])
        vertex_2_x = float(-semi_major / (np.cos(angle_rotation) + np.tan(angle_rotation) * np.sin(angle_rotation)) + center[0])
        vertex_2_y = float((vertex_2_x - center[0]) * np.tan(angle_rotation) + center[1])

    return [[vertex_1_x, vertex_1_y], [vertex_2_x, vertex_2_y]]


def angle_convert_to_radian(df):
    # modification of bacterium orientation
    # -(angle + 90) * np.pi / 180
    df["AreaShape_Orientation"] = -(df["AreaShape_Orientation"] + 90) * np.pi / 180

    return df


def bacteria_features(df):
    """
    output: length, radius, orientation, center coordinate
    """
    major = df['AreaShape_MajorAxisLength']
    minor = df['AreaShape_MinorAxisLength']
    radius = df['AreaShape_MinorAxisLength'] / 2
    orientation = df['AreaShape_Orientation']
    try:
        center_x = df['Location_Center_X']
        center_y = df['Location_Center_Y']
    except TypeError:
        center_x = df['AreaShape_Center_X']
        center_y = df['AreaShape_Center_Y']

    features = {'major': major, 'minor': minor, 'radius': radius, 'orientation': orientation, 'center_x': center_x,
                'center_y': center_y}
    return features


def increase_rate_major_minor(bacteria_current_time_step, bacteria_next_time_step, min_increase_rate_threshold):
    """
    goal: The major and minor increase rate of bacteria from the current time step to the next time step
    @param bacteria_current_time_step dataframe bacteria in current time step
    @param bacteria_next_time_step dataframe bacteria in next time step
    @param min_increase_rate_threshold float min increase rate of major & minor length
    """

    increase_rate_major = []
    increase_rate_minor = []
    bacterium_index_current_time_step = []
    bacterium_index_next_time_step = []
    # why did I set the initial value of this variable equal to -1?
    # as we know, the index of bacteria in the data frame is equal to or greater than zero so by this initial value,
    # we can find that: does the target bacterium have an unusual neighbor?
    unusual_neighbor_index = -1
    merged_bacterium_index = -1

    for index, bacterium in bacteria_current_time_step.iterrows():

        # track bacterium it in next time step
        relative_bacteria_in_next_timestep = bacteria_next_time_step.loc[
            (bacteria_next_time_step["TrackObjects_ParentImageNumber_50"] == bacterium["ImageNumber"]) &
            (bacteria_next_time_step["TrackObjects_ParentObjectNumber_50"] == bacterium["ObjectNumber"])]

        number_of_relative_bacteria = relative_bacteria_in_next_timestep.shape[0]

        if number_of_relative_bacteria == 1:
            bacterium_index_current_time_step.append(index)
            bacterium_index_next_time_step.extend(relative_bacteria_in_next_timestep.index.values.tolist())

            increase_rate_major_len = relative_bacteria_in_next_timestep.iloc[0]['AreaShape_MajorAxisLength'] / \
                                      bacterium['AreaShape_MajorAxisLength']
            increase_rate_minor_len = relative_bacteria_in_next_timestep.iloc[0]['AreaShape_MinorAxisLength'] / \
                                      bacterium['AreaShape_MinorAxisLength']

            increase_rate_major.append(increase_rate_major_len)
            increase_rate_minor.append(increase_rate_minor_len)

    unusual_increase_rate_minor = [elem for elem in zip(bacterium_index_current_time_step,
                                                        bacterium_index_next_time_step, increase_rate_minor)
                                   if elem[2] > np.percentile(increase_rate_minor, [75]) and
                                   elem[2] >= min_increase_rate_threshold]

    unusual_increase_rate_major = [elem for elem in zip(bacterium_index_current_time_step,
                                                        bacterium_index_next_time_step, increase_rate_major)
                                   if elem[2] > np.percentile(increase_rate_major, [75]) and
                                   elem[2] >= min_increase_rate_threshold]

    if unusual_increase_rate_major and unusual_increase_rate_minor:
        max_increase_rate_major = max(unusual_increase_rate_major, key=lambda x: x[2])
        max_increase_rate_minor = max(unusual_increase_rate_minor, key=lambda x: x[2])
        if max_increase_rate_major[1] >= max_increase_rate_minor[1]:
            unusual_neighbor_index = max_increase_rate_major[0]
            merged_bacterium_index = max_increase_rate_major[1]
        else:
            unusual_neighbor_index = max_increase_rate_minor[0]
            merged_bacterium_index = max_increase_rate_minor[1]
    elif unusual_increase_rate_major:
        max_increase_rate_major = max(unusual_increase_rate_major, key=lambda x: x[2])
        unusual_neighbor_index = max_increase_rate_major[0]
        merged_bacterium_index = max_increase_rate_major[1]
    elif unusual_increase_rate_minor:
        max_increase_rate_minor = max(unusual_increase_rate_minor, key=lambda x: x[2])
        unusual_neighbor_index = max_increase_rate_minor[0]
        merged_bacterium_index = max_increase_rate_minor[1]

    return unusual_neighbor_index, merged_bacterium_index


def k_nearest_neighbors(k, target_bacterium, other_bacteria, distance_threshold=None, distance_check=True):
    """
    goal: find k nearest neighbors to target bacterium
    @param k int number of desirable nearest neighbors
    @param target_bacterium  series value of features of bacterium that we want to find its neighbors
    @param other_bacteria dataframe
    @param distance_threshold (unit: um) maximum distance threshold
    @param distance_check bool Is the distance threshold checked?
    @return nearest_neighbors list index of the nearest bacteria
    """
    # calculate distance matrix
    try:
        distance_df = adjacency_matrix(target_bacterium, other_bacteria, 'Location_Center_X', 'Location_Center_Y')
    except TypeError:
        distance_df = adjacency_matrix(target_bacterium, other_bacteria, 'AreaShape_Center_X', 'AreaShape_Center_Y')

    distance_df = distance_df.reset_index(drop=True).sort_values(by=0, ascending=True, axis=1)
    distance_val = list(zip(distance_df.columns.values, distance_df.iloc[0].values))

    if distance_check:
        nearest_neighbors_index = [elem[0] for elem in distance_val if elem[1] <= distance_threshold][
                                  :min(k, len(distance_val))]
    else:
        nearest_neighbors_index = [elem[0] for elem in distance_val][:min(k, len(distance_val))]

    return nearest_neighbors_index


def adjacency_matrix(target_bacteria, other_bacteria, col1, col2):
    """
    goal: this function is useful to create adjacency matrix (distance matrix)
    I want to find distance of one dataframe to another
    example1: distance of transition bacteria from all bacteria in previous time step
    example1: distance of unexpected-end bacterium from all bacteria in same time step

    @param target_bacteria dataframe or series The value of the features of the bacteria that we want
    to find its distance from other bacteria
    @param other_bacteria dataframe or series
    @param col1 str distance of bacteria will be calculated depending on `col1` value
    @param col2 str distance of bacteria will be calculated depending on `col2` value

    """
    # create distance matrix (rows: next time step sudden bacteria, columns: another time step bacteria)
    distance_df = pd.DataFrame(distance_matrix(target_bacteria[[col1, col2]].values,
                                               other_bacteria[[col1, col2]].values),
                               index=target_bacteria.index, columns=other_bacteria.index)

    return distance_df


def bacteria_in_specific_time_step(dataframe, t):
    """
    goal: find bacteria in specific time step with ['drop'] = False
    @param dataframe dataframe bacteria information dataframe
    @param t int timestep
    """
    correspond_bacteria = dataframe.loc[(dataframe["ImageNumber"] == t) & (dataframe["drop"] == False)]

    return correspond_bacteria


def remove_rows(df, col, true_value):
    """
        goal: remove bacteria that have no parents
        @param df    dataframe   bacteria dataframe
        @param col str column name
        @param true_value bool
        output: df   dataframe   modified dataframe

    """

    df = df.loc[df[col] == true_value].reset_index(drop=True)
    return df


def find_bad_daughters(daughters_df):
    """
    goal: find indexes of bad daughters according to daughters bacteria major axis length
    """

    sorted_daughters_df = daughters_df.sort_values(by=['AreaShape_MajorAxisLength'], ascending=False)
    sorted_daughters_index = sorted_daughters_df.index.values.tolist()
    bad_daughters_index = sorted_daughters_index[0:len(sorted_daughters_index) - 2]

    return bad_daughters_index


def bacteria_life_history(df, desired_bacterium, desired_bacterium_index, last_time_step):
    """
    goal: fine life history of bacteria
    @param df dataframe bacteria info in time steps
    @param desired_bacterium series features value of desired bacterium
    @param desired_bacterium_index int row index of desired bacterium
    @param last_time_step int
    """

    # critical bacterium info to track it in next timme step
    bacterium_time_step = desired_bacterium["ImageNumber"]
    bacterium_obj_num = desired_bacterium["ObjectNumber"]

    life_history = 1
    life_history_index = [desired_bacterium_index]

    # If division has occurred (bacterium life history has ended), then this flag will be set.
    division_occ = False
    daughters_index = []
    # means: without division
    division_time = 0
    # If division has occurred (more than two daughters), then this flag will be set.
    bad_division_occ = False
    bad_daughters_index = []
    # The flag will be true if: A bacterium's life history has ended, but its family tree has not continued
    unexpected_end = False

    while (division_occ is False) and (bad_division_occ is False) and (unexpected_end is False) and \
            (bacterium_time_step < last_time_step):
        relative_bacteria_in_next_timestep = df.loc[
            (df["TrackObjects_ParentImageNumber_50"] == bacterium_time_step) &
            (df["TrackObjects_ParentObjectNumber_50"] == bacterium_obj_num)]

        number_of_relative_bacteria = relative_bacteria_in_next_timestep.shape[0]

        if number_of_relative_bacteria == 1:
            bacterium_time_step = relative_bacteria_in_next_timestep.iloc[0]["ImageNumber"]
            bacterium_obj_num = relative_bacteria_in_next_timestep.iloc[0]["ObjectNumber"]
            bacteria_index = relative_bacteria_in_next_timestep.index.values.tolist()[0]
            life_history_index.append(bacteria_index)
            life_history += 1

        elif number_of_relative_bacteria == 2:
            # means: division
            division_occ = True
            daughters_index = relative_bacteria_in_next_timestep.index.values.tolist()
            division_time = relative_bacteria_in_next_timestep.iloc[0]["ImageNumber"]

        elif number_of_relative_bacteria > 2:
            # means: division with more than two daughters
            division_occ = True
            bad_division_occ = True
            division_time = relative_bacteria_in_next_timestep.iloc[0]["ImageNumber"]

            # find bad daughters
            bad_daughters_index = find_bad_daughters(relative_bacteria_in_next_timestep)
            daughters_index = [i for i in relative_bacteria_in_next_timestep.index.values.tolist() if i not in
                               bad_daughters_index]

        elif number_of_relative_bacteria == 0:  # interrupt
            unexpected_end = True

    bacterium_status = {"division_occ": division_occ, "daughters_index": daughters_index,
                        "bad_division_occ": bad_division_occ, "bad_daughters_index": bad_daughters_index,
                        "division_time": division_time, "unexpected_end": unexpected_end, 'life_history': life_history,
                        "lifeHistoryIndex": life_history_index,
                        'bac_in_last_time_step': [bacterium_time_step, bacterium_obj_num]}

    return bacterium_status


def find_related_bacteria(df, target_bacterium, target_bacterium_index, bacteria_index_list=None):
    """
    goal:  From the bacteria dataframe, find the row index of bacteria related (same or daughter) to
    a specific bacterium
    @param df dataframe features value of bacteria in each time step
    @param target_bacterium series target bacterium features value
    @param target_bacterium_index   int   The row index of the corresponding bacterium
    @param bacteria_index_list list row index of related bacteria
    @return  bacteria_index_list   list   row index of related bacteria
    """

    if bacteria_index_list is None:
        bacteria_index_list = [target_bacterium_index]

    # related bacteria in next time steps
    if target_bacterium['ImageNumber'] <= df.iloc[-1]['ImageNumber']:
        bac_in_next_timestep = df.loc[(df["TrackObjects_ParentImageNumber_50"] == target_bacterium['ImageNumber']) &
                                      (df["TrackObjects_ParentObjectNumber_50"] == target_bacterium['ObjectNumber'])]

        if bac_in_next_timestep.shape[0] > 0:
            bacteria_index_list.extend(bac_in_next_timestep.index.tolist())
            for bac_index, next_bacterium in bac_in_next_timestep.iterrows():
                bacteria_index_list = find_related_bacteria(df, next_bacterium, bac_index, bacteria_index_list)

    return bacteria_index_list


def convert_to_pixel(length, radius, ends, pos, um_per_pixel=0.144):

    # Convert distances to pixel (0.144 um/pixel on 63X objective)
    length = length / um_per_pixel
    radius = radius / um_per_pixel
    ends = np.array(ends) / um_per_pixel
    pos = np.array(pos) / um_per_pixel

    return length, radius, ends, pos


def convert_to_um(data_frame, um_per_pixel=0.144):

    # Convert distances to um (0.144 um/pixel on 63X objective)
    # selected columns
    cols1 = ["AreaShape_MajorAxisLength", "AreaShape_MinorAxisLength", "Location_Center_X", "Location_Center_Y"]
    cols2 = ["AreaShape_MajorAxisLength", "AreaShape_MinorAxisLength", "AreaShape_Center_X", "AreaShape_Center_Y"]

    try:
        data_frame[cols1] = data_frame[cols1] * um_per_pixel
    except TypeError:
        data_frame[cols2] = data_frame[cols2] * um_per_pixel

    return data_frame
