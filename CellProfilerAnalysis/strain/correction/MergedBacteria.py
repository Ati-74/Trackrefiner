import pandas as pd
import numpy as np
from scipy import interpolate
from CellProfilerAnalysis.strain.correction.action.processing import k_nearest_neighbors, increase_rate_major_minor, bacteria_features


def average_feature_value(bacterium_in_t, bacterium_in_t_2):

    """
    goal: using feature value of bacterium in t, and bacterium in t + 1 , I try to find feature value of bacterium
     in t +1 (x)
     @param bacterium_in_t series bacterium features value in time step `t`
     @param bacterium_in_t_2 series bacterium features value in time step `t + 1`
    """

    new_bacterium_major = np.average([bacterium_in_t['AreaShape_MajorAxisLength'],
                                     bacterium_in_t_2['AreaShape_MajorAxisLength']])
    new_bacterium_minor = np.average([bacterium_in_t['AreaShape_MinorAxisLength'],
                                     bacterium_in_t_2['AreaShape_MinorAxisLength']])
    new_bacterium_orientation = np.average([bacterium_in_t['AreaShape_Orientation'],
                                           bacterium_in_t_2['AreaShape_Orientation']])

    try:
        new_bacterium_center_x = np.average([bacterium_in_t['Location_Center_X'], bacterium_in_t_2['Location_Center_X']])
        new_bacterium_center_y = np.average([bacterium_in_t['Location_Center_Y'], bacterium_in_t_2['Location_Center_Y']])
    except TypeError:
        new_bacterium_center_x = np.average([bacterium_in_t['AreaShape_Center_X'], bacterium_in_t_2['AreaShape_Center_X']])
        new_bacterium_center_y = np.average([bacterium_in_t['AreaShape_Center_Y'], bacterium_in_t_2['AreaShape_Center_Y']])

    new_values = {'minor': new_bacterium_minor, 'major': new_bacterium_major, 'orientation': new_bacterium_orientation,
                  'center_x': new_bacterium_center_x, 'center_y': new_bacterium_center_y}

    return new_values


def predict_new_feature_value(linear_interpolations, y_bacterium):

    """
    goal: using feature value of bacterium in t, I try to predict feature value of bacterium in t +1 (x)
    """
    new_bacterium_major = linear_interpolations['major_linear_extrapolation'](y_bacterium['AreaShape_MajorAxisLength'])
    new_bacterium_minor = linear_interpolations['minor_linear_extrapolation'](y_bacterium['AreaShape_MinorAxisLength'])
    new_bacterium_orientation = linear_interpolations['orientation_linear_extrapolation'](
        y_bacterium['AreaShape_Orientation'])

    try:
        new_bacterium_center_x = linear_interpolations['center_x_linear_extrapolation'](y_bacterium['Location_Center_X'])
        new_bacterium_center_y = linear_interpolations['center_y_linear_extrapolation'](y_bacterium['Location_Center_Y'])
    except TypeError:
        new_bacterium_center_x = linear_interpolations['center_x_linear_extrapolation'](y_bacterium['AreaShape_Center_X'])
        new_bacterium_center_y = linear_interpolations['center_y_linear_extrapolation'](y_bacterium['AreaShape_Center_X'])

    new_values = {'minor': new_bacterium_minor, 'major': new_bacterium_major, 'orientation': new_bacterium_orientation,
                  'center_x': new_bacterium_center_x, 'center_y': new_bacterium_center_y}

    return new_values


def fit_linear_extrapolation(minor, major, orientation, center_x, center_y):

    # perform linear interpolation
    major_linear_extrapolation = interpolate.interp1d(major[:-1], major[1:], kind='linear', fill_value='extrapolate')
    minor_linear_extrapolation = interpolate.interp1d(minor[:-1], minor[1:], kind='linear', fill_value='extrapolate')
    orientation_linear_extrapolation = interpolate.interp1d(orientation[:-1], orientation[1:], kind='linear',
                                                            fill_value='extrapolate')
    center_x_linear_extrapolation = interpolate.interp1d(center_x[:-1], center_x[1:], kind='linear',
                                                         fill_value='extrapolate')
    center_y_linear_extrapolation = interpolate.interp1d(center_y[:-1], center_y[1:], kind='linear',
                                                         fill_value='extrapolate')

    linear_interpolations = {"major_linear_extrapolation": major_linear_extrapolation,
                             "minor_linear_extrapolation": minor_linear_extrapolation,
                             "orientation_linear_extrapolation": orientation_linear_extrapolation,
                             "center_x_linear_extrapolation": center_x_linear_extrapolation,
                             "center_y_linear_extrapolation": center_y_linear_extrapolation}

    return linear_interpolations


def assign_new_feature_value(df, new_bacterium_index, new_bacterium_values, nearest_bacterium_life_history,
                             target_bacterium_life_history, check_cell_type):

    """
    assign new values to divided bacterium and modify all other related bacteria
    """
    # columns name
    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]
    label_col = [col for col in df.columns if 'TrackObjects_Label_' in col][0]

    daughters_of_nearest_bacterium_life_history = df.loc[df['parent_id'] == nearest_bacterium_life_history.iloc[-1]['id']]

    df.at[new_bacterium_index, "AreaShape_MajorAxisLength"] = new_bacterium_values['major']
    df.at[new_bacterium_index, "AreaShape_MinorAxisLength"] = new_bacterium_values['minor']
    df.at[new_bacterium_index, "AreaShape_Orientation"] = new_bacterium_values['orientation']
    try:
        df.at[new_bacterium_index, "Location_Center_X"] = new_bacterium_values['center_x']
        df.at[new_bacterium_index, "Location_Center_Y"] = new_bacterium_values['center_y']
    except TypeError:
        df.at[new_bacterium_index, "AreaShape_Center_X"] = new_bacterium_values['center_x']
        df.at[new_bacterium_index, "AreaShape_Center_Y"] = new_bacterium_values['center_y']

    df.at[new_bacterium_index, parent_image_number_col] = \
        target_bacterium_life_history.iloc[-1]['ImageNumber']
    df.at[new_bacterium_index, parent_object_number_col] = \
        target_bacterium_life_history.iloc[-1]['ObjectNumber']
    df.at[new_bacterium_index, label_col] = \
        target_bacterium_life_history.iloc[-1][label_col]

    next_time_step = target_bacterium_life_history.iloc[-1]['ImageNumber'] + 1
    df.at[new_bacterium_index, "ImageNumber"] = next_time_step

    df.at[new_bacterium_index, 'divideFlag'] = nearest_bacterium_life_history.iloc[-1]['divideFlag']
    df.at[new_bacterium_index, 'daughters_index'] = nearest_bacterium_life_history.iloc[-1]['daughters_index']
    df.at[new_bacterium_index, 'bad_division_flag'] = nearest_bacterium_life_history.iloc[-1]['bad_division_flag']
    df.at[new_bacterium_index, 'bad_daughters_index'] = nearest_bacterium_life_history.iloc[-1]['bad_daughters_index']
    df.at[new_bacterium_index, 'division_time'] = nearest_bacterium_life_history.iloc[-1]['division_time']
    df.at[new_bacterium_index, 'unexpected_end'] = False
    df.at[new_bacterium_index, 'transition_drop'] = False
    df.at[new_bacterium_index, 'bad_daughter_drop'] = False
    if check_cell_type:
        df.at[new_bacterium_index, 'cellType'] = target_bacterium_life_history.iloc[-1]['cellType']
    df.at[new_bacterium_index, 'id'] = target_bacterium_life_history.iloc[-1]['id']
    df.at[new_bacterium_index, 'parent_id'] = target_bacterium_life_history.iloc[-1]['parent_id']
    if str(df.iloc[new_bacterium_index]['LifeHistory']) != 'nan':
        df.at[new_bacterium_index, 'LifeHistory'] = df.iloc[new_bacterium_index]['LifeHistory'] + \
                                                    nearest_bacterium_life_history.iloc[-1]['LifeHistory']
    else:
        object_num_in_next_time_step = \
            sorted(df.loc[(df["ObjectNumber"].notnull()) & (df["ImageNumber"] == next_time_step)]['ObjectNumber'])
        last_object = object_num_in_next_time_step[-1]
        df.at[new_bacterium_index, "ObjectNumber"] = last_object + 1

        df.at[new_bacterium_index, 'LifeHistory'] = target_bacterium_life_history.iloc[-1]['LifeHistory'] + \
                                                    nearest_bacterium_life_history.iloc[-1]['LifeHistory'] + 1

    # change parent image number & parent object number
    df.at[nearest_bacterium_life_history.index[0], parent_image_number_col] = \
        df.iloc[new_bacterium_index]['ImageNumber']
    df.at[nearest_bacterium_life_history.index[0], parent_object_number_col] = \
        df.iloc[new_bacterium_index]['ObjectNumber']
    for bacterium_index in nearest_bacterium_life_history.index:
        df.at[bacterium_index, 'id'] = target_bacterium_life_history.iloc[-1]['id']
        df.at[bacterium_index, 'parent_id'] = target_bacterium_life_history.iloc[-1]['parent_id']
        if str(df.iloc[new_bacterium_index]['LifeHistory']) != 'nan':
            df.at[bacterium_index, 'LifeHistory'] = df.iloc[bacterium_index]['LifeHistory'] + \
                                                    target_bacterium_life_history.iloc[-1]['LifeHistory']
        else:
            df.at[bacterium_index, 'LifeHistory'] = df.iloc[bacterium_index]['LifeHistory'] + \
                                                    target_bacterium_life_history.iloc[-1]['LifeHistory'] + 1

    if daughters_of_nearest_bacterium_life_history.shape[0] > 0:
        for daughter_index in daughters_of_nearest_bacterium_life_history.index:
            daughter_life_history = df.loc[df['id'] == df.iloc[daughter_index]['id']]
            for daughter_life_history_index in daughter_life_history.index:
                df.at[daughter_life_history_index, 'parent_id'] = target_bacterium_life_history.iloc[-1]['id']

    for bacterium_index in target_bacterium_life_history.index:
        df.at[bacterium_index, 'divideFlag'] = nearest_bacterium_life_history.iloc[-1]['divideFlag']
        df.at[bacterium_index, 'daughters_index'] = nearest_bacterium_life_history.iloc[-1]['daughters_index']
        df.at[bacterium_index, 'bad_division_flag'] = nearest_bacterium_life_history.iloc[-1]['bad_division_flag']
        df.at[bacterium_index, 'bad_daughters_index'] = nearest_bacterium_life_history.iloc[-1]['bad_daughters_index']
        df.at[bacterium_index, 'division_time'] = nearest_bacterium_life_history.iloc[-1]['division_time']
        df.at[bacterium_index, 'unexpected_end'] = False
        if str(df.iloc[new_bacterium_index]['LifeHistory']) != 'nan':
            df.at[bacterium_index, 'LifeHistory'] = df.iloc[bacterium_index]['LifeHistory'] + \
                                                    nearest_bacterium_life_history.iloc[-1]['LifeHistory']
        else:
            df.at[bacterium_index, 'LifeHistory'] = df.iloc[bacterium_index]['LifeHistory'] + \
                                                    nearest_bacterium_life_history.iloc[-1]['LifeHistory'] + 1

    return df


def correction_merged_bacteria(df, unexpected_end_bacterium_life_history,
                               unusual_neighbor_life_history_before_merged_bacterium, merged_bacterium,
                               merged_bacterium_index, check_cellType):
    # columns name
    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    unexpected_end_bac_features = bacteria_features(unexpected_end_bacterium_life_history)
    neighbor_features = bacteria_features(unusual_neighbor_life_history_before_merged_bacterium)

    # Which bacterium in the t+2 time step is related to the unexpected end bacterium?
    merged_bacterium_daughters = df.loc[(df["ImageNumber"] == (merged_bacterium["ImageNumber"] + 1)) &
                                        (df[parent_image_number_col] == merged_bacterium["ImageNumber"]) &
                                        (df[parent_object_number_col] == merged_bacterium["ObjectNumber"]) &
                                        (df["transition_drop"] == False) & (df["bad_daughter_drop"] == False)]

    # which daughter bacteria is descendant of unexpected end bacterium?
    nearest_bacterium_to_unexpected_end_bac_index = k_nearest_neighbors(1, unexpected_end_bacterium_life_history.iloc[[-1]],
                                                                        merged_bacterium_daughters, distance_check=False)[0]

    nearest_bacterium_to_unexpected_end_bac_life_history = df.loc[df['id'] ==
                                                                  df.loc[nearest_bacterium_to_unexpected_end_bac_index]['id']]

    nearest_bacterium_to_neighbor_bacterium_index = [elem for elem in merged_bacterium_daughters.index.values.tolist()
                                                     if elem != nearest_bacterium_to_unexpected_end_bac_index][0]

    nearest_bacterium_to_neighbor_bacterium_life_history = df.loc[df['id'] ==
                                                                  df.loc[nearest_bacterium_to_neighbor_bacterium_index]['id']]

    if len(unexpected_end_bac_features['minor']) > 2:
        target_bacterium_linear_extrapolations = fit_linear_extrapolation(unexpected_end_bac_features['minor'],
                                                                          unexpected_end_bac_features['major'],
                                                                          unexpected_end_bac_features['orientation'],
                                                                          unexpected_end_bac_features['center_x'],
                                                                          unexpected_end_bac_features['center_y'])
        # predict values of new bacterium
        # target bacterium
        new_bacterium_values = predict_new_feature_value(target_bacterium_linear_extrapolations,
                                                         unexpected_end_bacterium_life_history.iloc[-1])
    elif len(unexpected_end_bac_features['minor']) <= 2:
        new_bacterium_values = average_feature_value(unexpected_end_bacterium_life_history.iloc[-1],
                                                     nearest_bacterium_to_unexpected_end_bac_life_history.iloc[0])

    if new_bacterium_values['minor'] <= 0 or new_bacterium_values['major'] <= 0 or \
            new_bacterium_values['center_x'] <= 0 or new_bacterium_values['center_y'] <= 0:
        new_bacterium_values = average_feature_value(unexpected_end_bacterium_life_history.iloc[-1],
                                                     nearest_bacterium_to_unexpected_end_bac_life_history.iloc[0])

    if len(neighbor_features['minor']) > 2:
        neighbor_bacterium_linear_extrapolations = fit_linear_extrapolation(neighbor_features['minor'],
                                                                            neighbor_features['major'],
                                                                            neighbor_features['orientation'],
                                                                            neighbor_features['center_x'],
                                                                            neighbor_features['center_y'])

        new_neighbor_bacterium_values = predict_new_feature_value(neighbor_bacterium_linear_extrapolations,
                                                                  unusual_neighbor_life_history_before_merged_bacterium.iloc[-1])

    elif len(neighbor_features['minor']) <= 2:
        new_neighbor_bacterium_values = average_feature_value(unusual_neighbor_life_history_before_merged_bacterium.iloc[-1],
                                                              nearest_bacterium_to_neighbor_bacterium_life_history.iloc[0])

    if new_neighbor_bacterium_values['minor'] <= 0 or new_neighbor_bacterium_values['major'] <= 0 or \
            new_neighbor_bacterium_values['center_x'] <= 0 or new_neighbor_bacterium_values['center_y'] <= 0:
        new_neighbor_bacterium_values = average_feature_value(
            unusual_neighbor_life_history_before_merged_bacterium.iloc[-1],
            nearest_bacterium_to_neighbor_bacterium_life_history.iloc[0])

    # update features value
    # neighbor bacterium

    df = assign_new_feature_value(df, merged_bacterium_index, new_neighbor_bacterium_values,
                                  nearest_bacterium_to_neighbor_bacterium_life_history,
                                  unusual_neighbor_life_history_before_merged_bacterium, check_cellType)
    # unexpected end  bacterium
    # insert new row to dataframe
    # append empty row to dataframe
    row_index = max(df.index) + 1
    empty = pd.DataFrame(columns=df.columns, index=[row_index])
    df = pd.concat([df, empty])

    df = assign_new_feature_value(df, row_index, new_bacterium_values,
                                  nearest_bacterium_to_unexpected_end_bac_life_history,
                                  unexpected_end_bacterium_life_history, check_cellType)

    return df


def merged_bacteria(df, k=6, distance_threshold=5, min_increase_rate_threshold=1.5, check_cellType=True):
    """
    @param df dataframe bacteria features value
    @param k int maximum number of nearest neighbor
    @param distance_threshold float maximum distance of desired bacterium (unexpected end bacterium) to other bacteria
    @param min_increase_rate_threshold float min increase rate of major & minor length
    """

    # correction of segmentation error
    # goal: I want to check all bacteria and find bacteria with life history = 1 and resolve them

    # columns name
    parent_image_number_col = [col for col in df.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in df.columns if 'TrackObjects_ParentObjectNumber_' in col][0]

    # define new column
    df['last_id_before_modification'] = df['id']

    unexpected_end_bacteria = df.loc[(df['unexpected_end'] == True) & (df['transition_drop'] == False) &
                                     (df['bad_daughter_drop'] == False)]

    for unexpected_end_bacterium_index, unexpected_end_bacterium in unexpected_end_bacteria.iterrows():
        unexpected_end_bacterium_life_history = df.loc[df['id'] == unexpected_end_bacterium['id']]

        if unexpected_end_bacterium_life_history.shape[0] == 0:
            unexpected_end_bacterium_life_history = df.loc[df['last_id_before_modification'] ==
                                                           unexpected_end_bacterium['id']]

            unexpected_end_bacterium_life_history = df.loc[df['id'] == unexpected_end_bacterium_life_history.iloc[-1]['id']]

        # features value of bacteria in the last time step of unexpected end bacterium life history
        # features value of unexpected bacterium in the last time step of its life history
        unexpected_end_bac_last_time_step = unexpected_end_bacterium_life_history.iloc[[-1]]
        other_bacteria = df.loc[(df["ImageNumber"] == unexpected_end_bac_last_time_step['ImageNumber'].iloc[0]) &
                                (df["ObjectNumber"] != unexpected_end_bac_last_time_step['ObjectNumber'].iloc[0])]

        # index of k nearest neighbors bacteria
        nearest_neighbors_index = k_nearest_neighbors(k, unexpected_end_bac_last_time_step, other_bacteria,
                                                      distance_threshold, distance_check=True)
        if nearest_neighbors_index:
            # features value of k nearest neighbors bacteria
            nearest_neighbors_df = other_bacteria.loc[nearest_neighbors_index]

            next_time_step_bacteria = df.loc[df["ImageNumber"] == unexpected_end_bac_last_time_step.iloc[0]['ImageNumber'] + 1]
            # unusual neighbor
            unusual_neighbor_index, merged_bacterium_index = \
                increase_rate_major_minor(nearest_neighbors_df, next_time_step_bacteria, min_increase_rate_threshold)

            if unusual_neighbor_index != -1:
                merged_bacterium = df.iloc[merged_bacterium_index]
                relative_bacteria_to_merged_bacterium_in_next_timestep = df.loc[(df["ImageNumber"] ==
                                                                                 (merged_bacterium["ImageNumber"] + 1)) &
                                                                                (df[parent_image_number_col] ==
                                                                                 merged_bacterium["ImageNumber"]) &
                                                                                (df[parent_object_number_col] ==
                                                                                 merged_bacterium["ObjectNumber"]) &
                                                                                (df["transition_drop"] == False) &
                                                                                (df["bad_daughter_drop"] == False)]

                if relative_bacteria_to_merged_bacterium_in_next_timestep.shape[0] == 2:
                    unusual_neighbor = df.loc[unusual_neighbor_index]
                    unusual_neighbor_life_history_before_merged_bacterium = df.loc[(df['id'] == unusual_neighbor['id']) &
                                                                                   (df['ImageNumber'] <=
                                                                                    unusual_neighbor['ImageNumber'])]

                    df = correction_merged_bacteria(df, unexpected_end_bacterium_life_history,
                                                    unusual_neighbor_life_history_before_merged_bacterium, merged_bacterium,
                                                    merged_bacterium_index, check_cellType)

                    df = df.sort_values(by=['ImageNumber', 'ObjectNumber']).reset_index(drop=True)

    return df
