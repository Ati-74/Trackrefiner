import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from CellProfilerAnalysis.strain.correction.action.processing import k_nearest_neighbors, increase_rate_major_minor, bacteria_features


def predict_x_test(linear_regression, y_test):

    # y = mx + b
    coefficient = linear_regression.coef_[0][0]
    intercept = linear_regression.intercept_[0]

    # x = (y - b) / x
    if coefficient != 0:
        x_test = (y_test - intercept) / coefficient
    else:
        x_test = (y_test - intercept)

    return x_test


def predict_new_feature_value(linear_regressions, y_bacterium):

    """
    goal: using feature value of bacterium in t +2 (y), I try to predict feature value of bacterium in t +1 (x)
    """

    new_bacterium_major = predict_x_test(linear_regressions['major_linear_regressor'],
                                         y_bacterium['AreaShape_MajorAxisLength'])
    new_bacterium_minor = predict_x_test(linear_regressions['minor_linear_regressor'],
                                         y_bacterium['AreaShape_MinorAxisLength'])
    new_bacterium_orientation = predict_x_test(linear_regressions['orientation_linear_regressor'],
                                               y_bacterium['AreaShape_Orientation'])

    try:
        new_bacterium_center_x = predict_x_test(linear_regressions['center_x_linear_regressor'],
                                                y_bacterium['Location_Center_X'])
        new_bacterium_center_y = predict_x_test(linear_regressions['center_y_linear_regressor'],
                                                y_bacterium['Location_Center_Y'])
    except TypeError:
        new_bacterium_center_x = predict_x_test(linear_regressions['center_x_linear_regressor'],
                                                y_bacterium['AreaShape_Center_X'])
        new_bacterium_center_y = predict_x_test(linear_regressions['center_y_linear_regressor'],
                                                y_bacterium['AreaShape_Center_X'])

    new_values = {'minor': new_bacterium_minor, 'major': new_bacterium_major, 'orientation': new_bacterium_orientation,
                  'center_x': new_bacterium_center_x, 'center_y': new_bacterium_center_y}

    return new_values


def fit_linear_regression(minor, major, orientation, center_x, center_y):

    major_linear_regressor = LinearRegression()  # create object for the class
    minor_linear_regressor = LinearRegression()  # create object for the class
    orientation_linear_regressor = LinearRegression()  # create object for the class
    center_x_linear_regressor = LinearRegression()  # create object for the class
    center_y_linear_regressor = LinearRegression()  # create object for the class

    # perform linear regression
    major_linear_regressor.fit(np.array(major[:-1]).reshape(-1, 1), np.array(major[1:]).reshape(-1, 1))
    minor_linear_regressor.fit(np.array(minor[:-1]).reshape(-1, 1), np.array(minor[1:]).reshape(-1, 1))
    orientation_linear_regressor.fit(np.array(orientation[:-1]).reshape(-1, 1), np.array(orientation[1:]).reshape(-1, 1))
    center_x_linear_regressor.fit(np.array(center_x[:-1]).reshape(-1, 1), np.array(center_x[1:]).reshape(-1, 1))
    center_y_linear_regressor.fit(np.array(center_y[:-1]).reshape(-1, 1), np.array(center_y[1:]).reshape(-1, 1))

    linear_regressions = {"major_linear_regressor": major_linear_regressor,
                          "minor_linear_regressor": minor_linear_regressor,
                          "orientation_linear_regressor": orientation_linear_regressor,
                          "center_x_linear_regressor": center_x_linear_regressor,
                          "center_y_linear_regressor": center_y_linear_regressor}

    return linear_regressions


def assign_new_feature_value(df, new_bacterium_index, new_bacterium_values, nearest_bacterium_life_history,
                             target_bacterium_life_history):

    """
    assign new values to divided bacterium and modify all other related bacteria
    """

    df.at[new_bacterium_index, "AreaShape_MajorAxisLength"] = new_bacterium_values['major']
    df.at[new_bacterium_index, "AreaShape_MinorAxisLength"] = new_bacterium_values['minor']
    df.at[new_bacterium_index, "AreaShape_Orientation"] = new_bacterium_values['orientation']
    try:
        df.at[new_bacterium_index, "Location_Center_X"] = new_bacterium_values['center_x']
        df.at[new_bacterium_index, "Location_Center_Y"] = new_bacterium_values['center_y']
    except TypeError:
        df.at[new_bacterium_index, "AreaShape_Center_X"] = new_bacterium_values['center_x']
        df.at[new_bacterium_index, "AreaShape_Center_Y"] = new_bacterium_values['center_y']

    df.at[new_bacterium_index, "ImageNumber"] = target_bacterium_life_history.iloc[-1]['ImageNumber'] + 1
    df.at[new_bacterium_index, "ObjectNumber"] = target_bacterium_life_history.iloc[-1]['ObjectNumber'] + 1
    df.at[new_bacterium_index, "TrackObjects_ParentImageNumber_50"] = \
        target_bacterium_life_history.iloc[-1]['TrackObjects_ParentImageNumber_50']
    df.at[new_bacterium_index, "TrackObjects_ParentObjectNumber_50"] = \
        target_bacterium_life_history.iloc[-1]['TrackObjects_ParentObjectNumber_50']
    df.at[new_bacterium_index, "TrackObjects_Label_50"] = \
        target_bacterium_life_history.iloc[-1]['TrackObjects_Label_50']

    df.at[new_bacterium_index, 'divideFlag'] = nearest_bacterium_life_history.iloc[-1]['divideFlag']
    df.at[new_bacterium_index, 'daughters_index'] = nearest_bacterium_life_history.iloc[-1]['daughters_index']
    df.at[new_bacterium_index, 'bad_division_flag'] = nearest_bacterium_life_history.iloc[-1]['bad_division_flag']
    df.at[new_bacterium_index, 'bad_daughters_index'] = nearest_bacterium_life_history.iloc[-1]['bad_daughters_index']
    df.at[new_bacterium_index, 'division_time'] = nearest_bacterium_life_history.iloc[-1]['division_time']
    df.at[new_bacterium_index, 'unexpected_end'] = False
    df.at[new_bacterium_index, 'transition_drop'] = False
    df.at[new_bacterium_index, 'bad_daughter_drop'] = False
    df.at[new_bacterium_index, 'cellType'] = target_bacterium_life_history.iloc[-1]['cellType']
    df.at[new_bacterium_index, 'id'] = target_bacterium_life_history.iloc[-1]['id']
    df.at[new_bacterium_index, 'parent_id'] = target_bacterium_life_history.iloc[-1]['parent_id']
    if str(df.iloc[new_bacterium_index]['LifeHistory']) != 'nan':
        df.at[new_bacterium_index, 'LifeHistory'] = df.iloc[new_bacterium_index]['LifeHistory'] + \
                                                    nearest_bacterium_life_history.iloc[-1]['LifeHistory']
    else:
        df.at[new_bacterium_index, 'LifeHistory'] = target_bacterium_life_history.iloc[-1]['LifeHistory'] + \
                                                    nearest_bacterium_life_history.iloc[-1]['LifeHistory'] + 1

    for bacterium_index in nearest_bacterium_life_history.index:
        df.at[bacterium_index, 'id'] = target_bacterium_life_history.iloc[-1]['id']
        df.at[bacterium_index, 'parent_id'] = target_bacterium_life_history.iloc[-1]['parent_id']
        if str(df.iloc[new_bacterium_index]['LifeHistory']) != 'nan':
            df.at[bacterium_index, 'LifeHistory'] = df.iloc[bacterium_index]['LifeHistory'] + \
                                                    target_bacterium_life_history.iloc[-1]['LifeHistory']
        else:
            df.at[bacterium_index, 'LifeHistory'] = df.iloc[bacterium_index]['LifeHistory'] + \
                                                    target_bacterium_life_history.iloc[-1]['LifeHistory'] + 1

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
                               merged_bacterium_index):

    unexpected_end_bac_major, unexpected_end_bac_minor, unexpected_end_bac_orientation, unexpected_end_bac_center_x, \
        unexpected_end_bac_center_y = bacteria_features(unexpected_end_bacterium_life_history)
    neighbor_major, neighbor_minor, neighbor_orientation, neighbor_center_x, neighbor_center_y = \
        bacteria_features(unusual_neighbor_life_history_before_merged_bacterium)

    # Which bacterium in the t+2 time step is related to the unexpected end bacterium?
    merged_bacterium_daughters = df.loc[(df["ImageNumber"] == (merged_bacterium["ImageNumber"] + 1)) &
                                        (df["TrackObjects_ParentImageNumber_50"] == merged_bacterium["ImageNumber"]) &
                                        (df["TrackObjects_ParentObjectNumber_50"] == merged_bacterium["ObjectNumber"]) &
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

    if len(unexpected_end_bac_major) > 1:
        target_bacterium_linear_regressions = fit_linear_regression(unexpected_end_bac_major, unexpected_end_bac_minor,
                                                                    unexpected_end_bac_orientation,
                                                                    unexpected_end_bac_center_x,
                                                                    unexpected_end_bac_center_y)
        # predict values of new bacterium
        # target bacterium
        new_bacterium_values = predict_new_feature_value(target_bacterium_linear_regressions,
                                                         nearest_bacterium_to_unexpected_end_bac_life_history.iloc[0])

    if len(neighbor_major) > 1:
        neighbor_bacterium_linear_regressions = fit_linear_regression(neighbor_major, neighbor_minor,
                                                                      neighbor_orientation, neighbor_center_x,
                                                                      neighbor_center_y)

        new_neighbor_bacterium_values = predict_new_feature_value(neighbor_bacterium_linear_regressions,
                                                                  nearest_bacterium_to_neighbor_bacterium_life_history.iloc[0])

    if len(unexpected_end_bac_major) > 1 and len(neighbor_major) == 1:
        new_neighbor_bacterium_values = predict_new_feature_value(target_bacterium_linear_regressions,
                                                                  nearest_bacterium_to_neighbor_bacterium_life_history.iloc[0])

    elif len(neighbor_major) > 1 and len(unexpected_end_bac_major) == 1:
        new_bacterium_values = predict_new_feature_value(neighbor_bacterium_linear_regressions,
                                                         nearest_bacterium_to_unexpected_end_bac_life_history.iloc[0])

    elif len(neighbor_major) == 1 and len(unexpected_end_bac_major) == 1:
        other_neighbor_bacteria = df.loc[(df['ImageNumber'] ==
                                          unexpected_end_bacterium_life_history['ImageNumber'].iloc[0]) &
                                         (df['ObjectNumber'] !=
                                          unexpected_end_bacterium_life_history['ObjectNumber'].iloc[0]) &
                                         (df['ObjectNumber'] !=
                                          unusual_neighbor_life_history_before_merged_bacterium['ObjectNumber'].iloc[0]) &
                                         (df['LifeHistory'] > 1)]

        new_nearest_bacterium_index = k_nearest_neighbors(1, unexpected_end_bacterium_life_history.iloc[[-1]],
                                                          other_neighbor_bacteria, distance_check=False)[0]

        new_nearest_bacterium_life_history = df.loc[df['id'] == df.loc[new_nearest_bacterium_index]['id']]

        # fetch feature values
        new_nearest_bac_major, new_nearest_bac_minor, new_nearest_bac_orientation, new_nearest_bac_center_x, \
            new_nearest_bac_center_y = bacteria_features(new_nearest_bacterium_life_history)

        new_nearest_bac_linear_regressions = fit_linear_regression(new_nearest_bac_major, new_nearest_bac_minor,
                                                                   new_nearest_bac_orientation, new_nearest_bac_center_x,
                                                                   new_nearest_bac_center_y)

        # predict values
        new_bacterium_values = predict_new_feature_value(new_nearest_bac_linear_regressions,
                                                         nearest_bacterium_to_unexpected_end_bac_life_history.iloc[0])

        new_neighbor_bacterium_values = predict_new_feature_value(new_nearest_bac_linear_regressions,
                                                                  nearest_bacterium_to_neighbor_bacterium_life_history.iloc[0])

    # update features value
    # neighbor bacterium
    df = assign_new_feature_value(df, merged_bacterium_index, new_neighbor_bacterium_values,
                                      nearest_bacterium_to_neighbor_bacterium_life_history,
                                      unusual_neighbor_life_history_before_merged_bacterium)
    # unexpected end  bacterium
    # insert new row to dataframe
    # append empty row to dataframe
    row_index = max(df.index) + 1
    empty = pd.DataFrame(columns=df.columns, index=[row_index])
    df = pd.concat([df, empty])

    df = assign_new_feature_value(df, row_index, new_bacterium_values,
                                  nearest_bacterium_to_unexpected_end_bac_life_history,
                                  unexpected_end_bacterium_life_history)

    return df


def merged_bacteria(df, k=6, distance_threshold=5, min_increase_rate_threshold=1.5):
    """
    @param df dataframe bacteria features value
    @param k int maximum number of nearest neighbor
    @param distance_threshold float maximum distance of desired bacterium (unexpected end bacterium) to other bacteria
    @param min_increase_rate_threshold float min increase rate of major & minor length
    """

    # correction of segmentation error
    # goal: I want to check all bacteria and find bacteria with life history = 1 and resolve them

    # define new column
    df['las_id_before_modification'] = df['id']

    unexpected_end_bacteria = df.loc[(df['unexpected_end'] == True) & (df['transition_drop'] == False) &
                                     (df['bad_daughter_drop'] == False)]

    for unexpected_end_bacterium_index, unexpected_end_bacterium in unexpected_end_bacteria.iterrows():

        unexpected_end_bacterium_life_history = df.loc[df['id'] == unexpected_end_bacterium['id']]

        if unexpected_end_bacterium_life_history.shape[0] == 0:
            unexpected_end_bacterium_life_history = df.loc[df['las_id_before_modification'] ==
                                                           unexpected_end_bacterium['id']]

        # features value of bacteria in the last time step of unexpected end bacterium life history
        # features value of unexpected bacterium in the last time step of its life history
        unexpected_end_bac_last_life_history = unexpected_end_bacterium_life_history.iloc[[-1]]
        other_bacteria = df.loc[(df["ImageNumber"] == unexpected_end_bac_last_life_history['ImageNumber'].iloc[0]) &
                                (df["ObjectNumber"] != unexpected_end_bac_last_life_history['ObjectNumber'].iloc[0])]

        # index of k nearest neighbors bacteria
        nearest_neighbors_index = k_nearest_neighbors(k, unexpected_end_bac_last_life_history, other_bacteria,
                                                      distance_threshold, True)
        # features value of k nearest neighbors bacteria
        nearest_neighbors_df = other_bacteria.loc[nearest_neighbors_index]

        next_time_step_bacteria = df.loc[df["ImageNumber"] == unexpected_end_bac_last_life_history['ImageNumber'].iloc[0] + 1]
        # unusual neighbor
        unusual_neighbor_index, merged_bacterium_index = \
            increase_rate_major_minor(nearest_neighbors_df, next_time_step_bacteria, min_increase_rate_threshold)

        if unusual_neighbor_index >= 0:

            merged_bacterium = df.iloc[merged_bacterium_index]
            relative_bacteria_to_merged_bacterium_in_next_timestep = df.loc[(df["ImageNumber"] ==
                                                                             (merged_bacterium["ImageNumber"] + 1)) &
                                                                            (df["TrackObjects_ParentImageNumber_50"] ==
                                                                             merged_bacterium["ImageNumber"]) &
                                                                            (df["TrackObjects_ParentObjectNumber_50"] ==
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
                                                merged_bacterium_index)

    df = df.sort_values(by=['ImageNumber', 'ObjectNumber']).reset_index(drop=True)

    return df
