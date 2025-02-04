import numpy as np
from sklearn.linear_model import LinearRegression


def calculate_average_elongation_rate(division_length, birth_length, t):

    """
    Calculates the average elongation rate of a bacterium over its lifecycle based on initial and final
    logarithm lengths.

    :param float division_length:
        The length of the bacterium at the time of division (final length).
    :param float birth_length:
        The length of the bacterium at birth (initial length).
    :param float t:
        The total time duration of the lifecycle (unit: minute).

    :returns:
        float:

        The calculated average elongation rate. This represents the logarithmic elongation rate per minute.
    """

    elongation_rate = round((np.log(division_length) - np.log(birth_length)) / t, 3)

    return elongation_rate


def calculate_linear_regression_elongation_rate(time, length):
    """
    Calculates the elongation rate of a bacterium using linear regression on the logarithm of length
    as a function of time.

    :param numpy.ndarray time:
        Array of time points corresponding to the bacterium's lifecycle.
    :param numpy.ndarray length:
        Array of major length measurements for the bacterium at the corresponding time points.

    :returns:
        float:

        The calculated elongation rate (slope of the logarithm of length vs. time).

    """

    linear_regressor = LinearRegression()  # create object for the class
    # perform linear regression
    linear_regressor.fit(np.array(time).reshape(-1, 1), np.log(np.array(length).reshape(-1, 1)))
    elongation_rate = round(linear_regressor.coef_[0][0], 3)

    return elongation_rate


def calculate_elongation_rate(df_life_history, interval_time, elongation_rate_method):
    """
    Calculates the elongation rate of a bacterium based on its life history data and the specified method.

    :param pandas.DataFrame df_life_history:
        Input DataFrame containing the life history of bacteria
    :param float interval_time:
        Time interval between consecutive images (in minutes).
    :param str elongation_rate_method:
        The method used for calculating the elongation rate. Options include:
        - "Average": Calculates the average elongation rate based on initial and final lengths.
        - "Linear Regression": Fits a linear regression to the length vs. time data to estimate the elongation rate.

    :returns:
        float:

        The calculated elongation rate. If the bacterium is observed for only one time step, returns NaN.
    """

    life_history_length = df_life_history.shape[0]

    # calculation of new feature
    division_length = df_life_history["AreaShape_MajorAxisLength"].values[-1]
    # length of bacteria when they are born
    birth_length = df_life_history["AreaShape_MajorAxisLength"].values[0]
    # this condition checks the life history of bacteria
    # If the bacterium exists only one time step: NaN will be reported.
    if life_history_length > 1:
        if elongation_rate_method == "Average":
            t = life_history_length * interval_time
            elongation_rate = calculate_average_elongation_rate(division_length, birth_length, t)
        elif elongation_rate_method == "Linear Regression":
            # linear regression
            time = df_life_history["ImageNumber"].values * interval_time
            length = df_life_history["AreaShape_MajorAxisLength"].values
            elongation_rate = calculate_linear_regression_elongation_rate(time, length)
        else:

            raise ValueError(f"Invalid elongation rate method: {elongation_rate_method}. "
                             f"Valid options are 'Average' or 'Linear Regression'.")
    else:
        elongation_rate = np.nan  # shows: bacterium is present for only one timestep.

    return elongation_rate
