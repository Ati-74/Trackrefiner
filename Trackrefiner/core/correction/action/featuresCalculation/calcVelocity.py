import numpy as np


def calc_average_velocity(df_life_history, center_coord_cols, interval_time):

    """
    Calculates the velocity of a bacterium based on its life history data.

    :param pandas.DataFrame df_life_history:
        Input DataFrame containing the life history of bacteria
    :param dict center_coord_cols:
        A dictionary with keys 'x' and 'y', specifying the column names for center coordinates.
    :param float interval_time:
        Time interval between consecutive images (in minutes).

    :returns:
        float:

        The calculated velocity. If the bacterium is observed for only one time step, returns NaN.
    """

    life_history_length = df_life_history.shape[0]

    division_x_coordinate = df_life_history[center_coord_cols['x']].values[-1]
    division_y_coordinate = df_life_history[center_coord_cols['y']].values[-1]

    # coordinate of bacteria when they are born
    birth_x_coordinate = df_life_history[center_coord_cols['x']].values[0]
    birth_y_coordinate = df_life_history[center_coord_cols['y']].values[0]

    # this condition checks the life history of bacteria
    # If the bacterium exists only one time step: NaN will be reported.
    if life_history_length > 1:

        # Compute the Euclidean distance of the bacteria from the origin at initial and final positions.
        x1 = np.sqrt(birth_x_coordinate ** 2 + birth_y_coordinate ** 2)
        x2 = np.sqrt(division_x_coordinate ** 2 + division_y_coordinate ** 2)

        # Calculate the velocity over the life history length.
        velocity = round((x2 - x1) / (life_history_length * interval_time), 3)

    else:
        velocity = np.nan

    return velocity
