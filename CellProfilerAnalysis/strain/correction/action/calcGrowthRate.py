import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def calculate_average_growth_rate(division_length, birth_length, t):
    elongation_rate = round((np.log(division_length) - np.log(birth_length)) / t, 3)

    return elongation_rate


def calculate_linear_regression_growth_rate(time, length):
    linear_regressor = LinearRegression()  # create object for the class
    # perform linear regression
    linear_regressor.fit(np.array(time).reshape(-1, 1), np.log(np.array(length).reshape(-1, 1)))
    elongation_rate = round(linear_regressor.coef_[0][0], 3)

    return elongation_rate


def calculate_growth_rate(df_life_history, interval_time, growth_rate_method):

    life_history_length = df_life_history.shape[0]

    # calculation of new feature
    division_length = df_life_history.iloc[[-1]]["AreaShape_MajorAxisLength"].values[0]
    # length of bacteria when they are born
    birth_length = df_life_history.iloc[[0]]["AreaShape_MajorAxisLength"].values[0]
    # this condition checks the life history of bacteria
    # If the bacterium exists only one time step: NaN will be reported.
    if life_history_length > 1:
        if growth_rate_method == "Average":
            t = life_history_length * interval_time
            elongation_rate = calculate_average_growth_rate(division_length, birth_length, t)
        if growth_rate_method == "Linear Regression":
            # linear regression
            time = df_life_history["ImageNumber"].values * interval_time
            length = df_life_history["AreaShape_MajorAxisLength"].values
            elongation_rate = calculate_linear_regression_growth_rate(time, length)
    else:
        elongation_rate = np.nan  # shows: bacterium is present for only one timestep.

    return elongation_rate
