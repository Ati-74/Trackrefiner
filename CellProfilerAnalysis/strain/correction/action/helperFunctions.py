import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean


def calc_neighbors_dir_motion(df, source_bac, neighbor_df):

    direction_of_motion_list = {}

    neighbor_to_mother_info = find_neighbors_info(df, neighbor_df, source_bac)

    for neighbor_bac_ndx, neighbor_bac in neighbor_to_mother_info.iterrows():

        direction_of_motion_list[neighbor_bac_ndx] = []
        neighbor_bac_in_next_time_step = df.loc[(df['ImageNumber'] == neighbor_bac['ImageNumber'] + 1) &
                                                (df['id'] == neighbor_bac['id'])]

        neighbor_bac_daughters_in_next_time_step = df.loc[(df['ImageNumber'] == neighbor_bac['ImageNumber'] + 1) &
                                                          (df['parent_id'] == neighbor_bac['id'])]

        if neighbor_bac_in_next_time_step.shape[0] > 0:
            direction_of_motion = \
                calculate_trajectory_direction(
                    np.array([neighbor_bac["AreaShape_Center_X"],
                              neighbor_bac["AreaShape_Center_Y"]]),
                    np.array([neighbor_bac_in_next_time_step.iloc[0]["AreaShape_Center_X"],
                              neighbor_bac_in_next_time_step.iloc[0]["AreaShape_Center_Y"]]))

            direction_of_motion_list[neighbor_bac_ndx] = [direction_of_motion]

        elif neighbor_bac_daughters_in_next_time_step.shape[0] > 0:
            for daughter_ndx, daughter_bac in neighbor_bac_daughters_in_next_time_step.iterrows():
                direction_of_motion = \
                    calculate_trajectory_direction(
                        np.array([neighbor_bac["AreaShape_Center_X"],
                                  neighbor_bac["AreaShape_Center_Y"]]),
                        np.array([daughter_bac["AreaShape_Center_X"],
                                  daughter_bac["AreaShape_Center_Y"]]))

                direction_of_motion_list[neighbor_bac_ndx].append(direction_of_motion)

    # Extracting the first and second elements from each array
    first_elements = []
    second_elements = []

    for key, arrays in direction_of_motion_list.items():
        for arr in arrays:
            first_elements.append(arr[0])
            second_elements.append(arr[1])

    # Calculating the averages
    if len(first_elements) == 0:
        average_first = np.nan
        average_second = np.nan
    else:
        average_first = np.mean(first_elements)
        average_second = np.mean(second_elements)

    return [average_first, average_second]


def adding_features_related_to_division(dataframe, bac_ndx, bacterium_status):

    dataframe.at[bac_ndx, 'divideFlag'] = bacterium_status['division_occ']
    dataframe.at[bac_ndx, 'daughters_index'] = bacterium_status['daughters_index']
    dataframe.at[bac_ndx, 'division_time'] = bacterium_status['division_time']
    dataframe.at[bac_ndx, 'divideFlag'] = bacterium_status['division_occ']
    dataframe.at[bac_ndx, 'bad_division_flag'] = bacterium_status['bad_division_occ']
    dataframe.at[bac_ndx, 'division_time'] = bacterium_status['division_time']
    return dataframe


def adding_features_to_each_timestep_except_first(dataframe, bac_indx_in_list, bac_indx_in_df, bacterium_status,
                                                  neighbor_df):
    index_prev_stage_life = bac_indx_in_list - 1

    dataframe.at[bac_indx_in_df, "bac_length_to_back"] = \
        dataframe.iloc[bac_indx_in_df]["AreaShape_MajorAxisLength"] / \
        dataframe.iloc[bacterium_status['lifeHistoryIndex'][index_prev_stage_life]]["AreaShape_MajorAxisLength"]

    direction_of_motion = \
        calculate_trajectory_direction_angle(
            np.array([dataframe.iloc[bacterium_status['lifeHistoryIndex'][index_prev_stage_life]]["AreaShape_Center_X"],
                      dataframe.iloc[bacterium_status['lifeHistoryIndex'][index_prev_stage_life]][
                          "AreaShape_Center_Y"]]),
            np.array([dataframe.iloc[bac_indx_in_df]["AreaShape_Center_X"], dataframe.iloc[bac_indx_in_df]["AreaShape_Center_Y"]]))

    direction_of_motion_vector = \
        calculate_trajectory_direction(
            np.array([dataframe.iloc[bacterium_status['lifeHistoryIndex'][index_prev_stage_life]]["AreaShape_Center_X"],
                      dataframe.iloc[bacterium_status['lifeHistoryIndex'][index_prev_stage_life]][
                          "AreaShape_Center_Y"]]),
            np.array([dataframe.iloc[bac_indx_in_df]["AreaShape_Center_X"],
                      dataframe.iloc[bac_indx_in_df]["AreaShape_Center_Y"]]))

    neighbors_dir_motion = \
        calc_neighbors_dir_motion(dataframe,
                                  dataframe.iloc[bacterium_status['lifeHistoryIndex'][index_prev_stage_life]],
                                  neighbor_df)

    if str(neighbors_dir_motion[0]) != 'nan':
        angle_between_motion = calc_normalized_angle_between_motion(neighbors_dir_motion, direction_of_motion_vector)
    else:
        angle_between_motion = 0

    dataframe.at[bac_indx_in_df, "direction_of_motion"] = direction_of_motion
    dataframe.at[bac_indx_in_df, "angle_between_neighbor_motion_bac_motion"] = angle_between_motion

    center_movement = \
        np.sqrt((dataframe.iloc[bac_indx_in_df]["AreaShape_Center_X"] -
                 dataframe.iloc[bacterium_status['lifeHistoryIndex'][index_prev_stage_life]][
                     "AreaShape_Center_X"]) ** 2 +
                (dataframe.iloc[bac_indx_in_df]["AreaShape_Center_Y"] -
                 dataframe.iloc[bacterium_status['lifeHistoryIndex'][index_prev_stage_life]][
                     "AreaShape_Center_Y"]) ** 2)

    prev_bacterium_endpoints = find_vertex(
        [dataframe.iloc[bacterium_status['lifeHistoryIndex'][index_prev_stage_life]]["AreaShape_Center_X"],
         dataframe.iloc[bacterium_status['lifeHistoryIndex'][index_prev_stage_life]]["AreaShape_Center_Y"]],
        dataframe.iloc[bacterium_status['lifeHistoryIndex'][index_prev_stage_life]][
            "AreaShape_MajorAxisLength"],
        dataframe.iloc[bacterium_status['lifeHistoryIndex'][index_prev_stage_life]]["AreaShape_Orientation"])

    current_bacterium_endpoints = find_vertex(
        [dataframe.iloc[bac_indx_in_df]["AreaShape_Center_X"], dataframe.iloc[bac_indx_in_df]["AreaShape_Center_Y"]],
        dataframe.iloc[bac_indx_in_df]["AreaShape_MajorAxisLength"], dataframe.iloc[bac_indx_in_df]["AreaShape_Orientation"])

    endpoint1_1_movement = \
        np.sqrt((current_bacterium_endpoints[0][0] - prev_bacterium_endpoints[0][0]) ** 2 +
                (current_bacterium_endpoints[0][1] - prev_bacterium_endpoints[0][1]) ** 2)

    endpoint1_endpoint2_movement = \
        np.sqrt((current_bacterium_endpoints[0][0] - prev_bacterium_endpoints[1][0]) ** 2 +
                (current_bacterium_endpoints[0][1] - prev_bacterium_endpoints[1][1]) ** 2)

    endpoint2_2_movement = \
        np.sqrt((current_bacterium_endpoints[1][0] - prev_bacterium_endpoints[1][0]) ** 2 +
                (current_bacterium_endpoints[1][1] - prev_bacterium_endpoints[1][1]) ** 2)

    endpoint2_endpoint1_movement = \
        np.sqrt((current_bacterium_endpoints[1][0] - prev_bacterium_endpoints[0][0]) ** 2 +
                (current_bacterium_endpoints[1][1] - prev_bacterium_endpoints[0][1]) ** 2)

    dataframe.at[bac_indx_in_df, "bacteria_movement"] = min(center_movement, endpoint1_1_movement, endpoint2_2_movement,
                                                            endpoint1_endpoint2_movement, endpoint2_endpoint1_movement)

    slope2, intercept2 = calculate_slope_intercept(prev_bacterium_endpoints[0],
                                                   prev_bacterium_endpoints[1])

    # Convert points to vectors
    # Calculate slopes and intercepts
    slope1, intercept1 = calculate_slope_intercept(current_bacterium_endpoints[0],
                                                   current_bacterium_endpoints[1])

    # Calculate orientation angle
    orientation_angle = calculate_orientation_angle(slope1, slope2)

    dataframe.at[bac_indx_in_df, "bac_length_to_back_orientation_changes"] = orientation_angle

    return dataframe


def adding_features_only_for_last_time_step(dataframe, last_bacterium_in_life_history, bacterium_status):
    dataframe.at[last_bacterium_in_life_history, 'unexpected_end'] = bacterium_status['unexpected_end']

    dataframe.at[last_bacterium_in_life_history, "daughter_length_to_mother"] = \
        bacterium_status['daughter_len'] / \
        dataframe.iloc[last_bacterium_in_life_history]["AreaShape_MajorAxisLength"]

    dataframe.at[last_bacterium_in_life_history, "max_daughter_len_to_mother"] = \
        bacterium_status["max_daughter_len"] / \
        dataframe.iloc[last_bacterium_in_life_history]["AreaShape_MajorAxisLength"]

    if bacterium_status['division_occ'] and not bacterium_status['bad_division_occ']:
        daughters_df = dataframe.iloc[bacterium_status['daughters_index']]

        daughter1_endpoints = find_vertex([daughters_df["AreaShape_Center_X"].values.tolist()[0],
                                           daughters_df["AreaShape_Center_Y"].values.tolist()[0]],
                                          daughters_df["AreaShape_MajorAxisLength"].values.tolist()[0],
                                          daughters_df["AreaShape_Orientation"].values.tolist()[0])

        daughter2_endpoints = find_vertex([daughters_df["AreaShape_Center_X"].values.tolist()[1],
                                           daughters_df["AreaShape_Center_Y"].values.tolist()[1]],
                                          daughters_df["AreaShape_MajorAxisLength"].values.tolist()[1],
                                          daughters_df["AreaShape_Orientation"].values.tolist()[1])

        slope_daughter1, intercept_daughter1 = calculate_slope_intercept(daughter1_endpoints[0],
                                                                         daughter1_endpoints[1])

        slope_daughter2, intercept_daughter2 = calculate_slope_intercept(daughter2_endpoints[0],
                                                                         daughter2_endpoints[1])

        # Calculate orientation angle
        daughters_orientation_angle = calculate_orientation_angle(slope_daughter1, slope_daughter2)
        dataframe.at[last_bacterium_in_life_history, "daughter_orientation"] = daughters_orientation_angle

    return dataframe


def find_neighbors_info(df, neighbors_df, bac):
    neighbors = neighbors_df.loc[(neighbors_df['First Image Number'] == bac['ImageNumber']) &
                                 (neighbors_df['First Object Number'] == bac['ObjectNumber'])]

    if neighbors.shape[0] > 0:
        neighbors_df = df.loc[(df['ImageNumber'] == neighbors['Second Image Number'].values.tolist()[0]) &
                              (df['ObjectNumber'].isin(neighbors['Second Object Number'].values.tolist()))]
    else:
        neighbors_df = pd.DataFrame()

    return neighbors_df


def calculate_trajectory_direction(previous_position, current_position):
    # Calculate the direction vector from the previous position to the current position
    direction = current_position - previous_position

    return direction


def calculate_trajectory_direction_angle(previous_position, current_position):
    # Calculate the direction vector from the previous position to the current position
    direction = current_position - previous_position

    # Calculate the angle of the direction vector
    angle = np.arctan2(direction[1], direction[0])

    return angle


def calculate_slope_intercept(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    else:
        slope = np.nan
        intercept = np.nan
    return slope, intercept


def calculate_orientation_angle(slope1, slope2):
    # Calculate the angle in radians between the lines
    angle_radians = np.arctan(abs((slope1 - slope2) / (1 + slope1 * slope2)))
    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


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
        vertex_1_x = float(
            semi_major / (np.cos(angle_rotation) + np.tan(angle_rotation) * np.sin(angle_rotation)) + center[0])
        vertex_1_y = float((vertex_1_x - center[0]) * np.tan(angle_rotation) + center[1])
        vertex_2_x = float(
            -semi_major / (np.cos(angle_rotation) + np.tan(angle_rotation) * np.sin(angle_rotation)) + center[0])
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

    # columns name
    parent_image_number_col = [col for col in bacteria_current_time_step.columns if
                               'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in bacteria_current_time_step.columns if
                                'TrackObjects_ParentObjectNumber_' in col][0]

    for index, bacterium in bacteria_current_time_step.iterrows():

        # track bacterium it in next time step
        relative_bacteria_in_next_timestep = bacteria_next_time_step.loc[
            (bacteria_next_time_step[parent_image_number_col] == bacterium["ImageNumber"]) &
            (bacteria_next_time_step[parent_object_number_col] == bacterium["ObjectNumber"])]

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


def k_nearest_neighbors(target_bacterium, other_bacteria, distance_check=True):
    """
    goal: find k nearest neighbors to target bacterium
    @param target_bacterium  series value of features of bacterium that we want to find its neighbors
    @param other_bacteria dataframe
    @param distance_check bool Is the distance threshold checked?
    @return nearest_neighbors list index of the nearest bacteria
    """
    # calculate distance matrix
    try:
        distance_df = calc_distance_matrix(target_bacterium, other_bacteria, 'Location_Center_X', 'Location_Center_Y')
    except TypeError:
        distance_df = calc_distance_matrix(target_bacterium, other_bacteria, 'AreaShape_Center_X', 'AreaShape_Center_Y')

    distance_df = distance_df.reset_index(drop=True).sort_values(by=0, ascending=True, axis=1)
    distance_val = list(zip(distance_df.columns.values, distance_df.iloc[0].values))

    if distance_check:
        nearest_neighbors_index = [elem[0] for elem in distance_val if elem[1] <= distance_threshold][
                                  :min(k, len(distance_val))]
    else:
        nearest_neighbors_index = [elem[0] for elem in distance_val][:min(k, len(distance_val))]

    return nearest_neighbors_index


def calc_normalized_angle_between_motion(motion1, motion2):

    # Calculate the dot product
    dot_product = motion1[0] * motion2[0] + motion1[1] * motion2[1]

    # Calculate the magnitudes of the vectors
    magnitude_A = np.sqrt(motion1[0] ** 2 + motion1[1] ** 2)
    magnitude_B = np.sqrt(motion2[0] ** 2 + motion2[1] ** 2)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_A * magnitude_B)

    # Calculate the angle in radians and then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees / 180


def calculate_orientation_angle(slope1, slope2):
    # Calculate the angle in radians between the lines
    angle_radians = np.arctan(abs((slope1 - slope2) / (1 + slope1 * slope2)))
    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def distance_normalization(df, distance_df):
    # bacteria_movement = [v for v in df['bacteria_movement'].dropna().values.tolist() if v != '']

    # min_val = min(bacteria_movement)
    # max_val = max(bacteria_movement)

    # normalized_df = pd.DataFrame()
    # for column in distance_df.columns:
    #    normalized_df[column] = (distance_df[column] - min_val) / (max_val - min_val)

    # Extract 'bacteria_movement' values, drop NaNs, and filter out empty strings
    bacteria_movement = df['bacteria_movement'].dropna()
    bacteria_movement = bacteria_movement[bacteria_movement != ''].astype(float)

    # Calculate min and max
    min_val = bacteria_movement.min()
    max_val = bacteria_movement.max()

    # Normalize the entire DataFrame at once
    normalized_df = (distance_df - min_val) / (max_val - min_val)

    return normalized_df


def min_max_normalize_row(row):
    min_val = row.min()
    max_val = row.max()
    return (row - min_val) / (max_val - min_val)


def calc_distance_between_daughters(dataframe, d1_ndx, d2_ndx):

    daughter1 = dataframe.iloc[d1_ndx]
    daughter2 = dataframe.iloc[d2_ndx]

    daughter1_endpoints = find_vertex([daughter1["AreaShape_Center_X"], daughter1["AreaShape_Center_Y"]],
                                      daughter1["AreaShape_MajorAxisLength"], daughter1["AreaShape_Orientation"])

    daughter2_endpoints = find_vertex([daughter2["AreaShape_Center_X"], daughter2["AreaShape_Center_Y"]],
                                      daughter2["AreaShape_MajorAxisLength"], daughter2["AreaShape_Orientation"])

    endpoint1_2_dis = euclidean(daughter1_endpoints[0], daughter2_endpoints[1])
    endpoint2_1_dis = euclidean(daughter1_endpoints[1], daughter2_endpoints[0])

    return min(endpoint1_2_dis, endpoint2_1_dis)



def calc_distance_matrix(target_bacteria, other_bacteria, col1, col2, col3=None, col4=None, normalization=False):
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
    if col3 is None and col4 is None:
        distance_df = pd.DataFrame(distance_matrix(target_bacteria[[col1, col2]].values,
                                                   other_bacteria[[col1, col2]].values),
                                   index=target_bacteria.index, columns=other_bacteria.index)
    else:
        distance_df = pd.DataFrame(distance_matrix(target_bacteria[[col1, col2]].values,
                                                   other_bacteria[[col3, col4]].values),
                                   index=target_bacteria.index, columns=other_bacteria.index)

    if normalization:
        # Apply Min-Max normalization to each row
        distance_df = distance_df.apply(min_max_normalize_row, axis=1)

    return distance_df


def bacteria_in_specific_time_step(dataframe, t):
    """
    goal: find bacteria in specific time step with ['drop'] = False
    @param dataframe dataframe bacteria information dataframe
    @param t int timestep
    """
    # & (dataframe["drop"] == False)
    correspond_bacteria = dataframe.loc[(dataframe["ImageNumber"] == t)]

    return correspond_bacteria


# Function to modify list elements
def modify_list(lst, thresh):
    return [item - 1 if item > thresh else item for item in lst]


def remove_rows(df, col, true_value):
    """
        goal: remove bacteria that have no parents
        @param df    dataframe   bacteria dataframe
        @param col str column name
        @param true_value bool
        output: df   dataframe   modified dataframe

    """

    remove_rows_df = df.loc[df[col] != true_value]
    df = df.loc[df[col] == true_value].reset_index(drop=True)

    for remove_row_ndx, remove_row in remove_rows_df.iterrows():
        # Apply the function to each list in the DataFrame column
        threshold = remove_row_ndx
        df['daughters_index'] = df['daughters_index'].apply(modify_list, args=(threshold,))

    return df


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

    # columns name
    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    life_history = 1
    life_history_index = [desired_bacterium_index]

    # If division has occurred (bacterium life history has ended), then this flag will be set.
    division_occ = False
    daughters_index = []
    # means: without division
    division_time = 0

    daughters_len = np.nan
    max_daughter_len = np.nan

    # If division has occurred (more than two daughters), then this flag will be set.
    bad_division_occ = False
    bad_daughters_index = []
    # The flag will be true if: A bacterium's life history has ended, but its family tree has not continued
    unexpected_end = False

    while (division_occ is False) and (bad_division_occ is False) and (unexpected_end is False) and \
            (bacterium_time_step < last_time_step):
        relative_bacteria_in_next_timestep = df.loc[
            (df[parent_image_number_col] == bacterium_time_step) & (df[parent_object_number_col] == bacterium_obj_num)]

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

            # daughters length
            daughters_len = sum(relative_bacteria_in_next_timestep["AreaShape_MajorAxisLength"].values.tolist())
            max_daughter_len = max(relative_bacteria_in_next_timestep["AreaShape_MajorAxisLength"].values.tolist())

        elif number_of_relative_bacteria > 2:
            # means: division with more than two daughters
            division_occ = True
            bad_division_occ = True
            division_time = relative_bacteria_in_next_timestep.iloc[0]["ImageNumber"]

            # index of daughters
            daughters_index = relative_bacteria_in_next_timestep.index.values.tolist()

        elif number_of_relative_bacteria == 0:  # interrupt
            unexpected_end = True

    bacterium_status = {"division_occ": division_occ, "daughters_index": daughters_index,
                        "bad_division_occ": bad_division_occ,
                        "division_time": division_time, "unexpected_end": unexpected_end, 'life_history': life_history,
                        "lifeHistoryIndex": life_history_index,
                        'bac_in_last_time_step': [bacterium_time_step, bacterium_obj_num],
                        "daughter_len": daughters_len, "max_daughter_len": max_daughter_len}

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

    # columns name
    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    # related bacteria in next time steps
    if target_bacterium['ImageNumber'] <= df.iloc[-1]['ImageNumber']:
        bac_in_next_timestep = df.loc[(df[parent_image_number_col] == target_bacterium['ImageNumber']) &
                                      (df[parent_object_number_col] == target_bacterium['ObjectNumber'])]

        if bac_in_next_timestep.shape[0] > 0:
            bacteria_index_list.extend(bac_in_next_timestep.index.tolist())
            for bac_index, next_bacterium in bac_in_next_timestep.iterrows():
                bacteria_index_list = find_related_bacteria(df, next_bacterium, bac_index, bacteria_index_list)

    return bacteria_index_list


def convert_ends_to_pixel(ends, um_per_pixel=0.144):
    ends = np.array(ends) / um_per_pixel
    return ends


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
    cols3 = ["AreaShape_MajorAxisLength", "AreaShape_MinorAxisLength", "AreaShape_Center_X", "AreaShape_Center_Y",
             "Location_Center_X", "Location_Center_Y"]

    try:
        data_frame[cols3] = data_frame[cols3] * um_per_pixel
    except:
        try:
            data_frame[cols1] = data_frame[cols1] * um_per_pixel
        except KeyError:
            data_frame[cols2] = data_frame[cols2] * um_per_pixel

    return data_frame
