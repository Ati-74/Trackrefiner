from lownerJohnEllipse import welzl, plot_ellipse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg as la


def summary_statistic_plot(bac_df, ellipses, time_step, output_directory, summary_statistics_values_list,
                           summary_statistic_method):
    """
    Goal: this function plots the whole story! that means it shows bacteria with
    both end points and ellipse that are fitted to colonies' boundaries and reports
    their summary statistic value as a single number in the middle of the colony.

    @param bac_df     data frame   bacterial info like x,y center and orientation, etc.
    @param ellipses   list         ellipses' parameters that are fitted to micro-colonies
    """

    fig, ax = plt.subplots()
    ax = plt.gca()

    # bacteria information
    (bacteria_center_coord, bacteria_major, bacteria_minor, bacteria_orientation) = bac_info(bac_df)
    # number of cells
    num_cells = bac_df.shape[0]

    # draw bacteria
    for cell_index in range(num_cells):
        center = (bacteria_center_coord.iloc[cell_index]["AreaShape_Center_X"],
                  bacteria_center_coord.iloc[cell_index]["AreaShape_Center_Y"])
        minor = bacteria_minor.iloc[cell_index] / 2
        major = bacteria_major.iloc[cell_index] / 2
        # radian
        angle = -(bacteria_orientation.iloc[cell_index] + 90) * np.pi / 180
        # endpoints
        node_x1_x = center[0] + major * np.cos(angle)
        node_x1_y = center[1] + major * np.sin(angle)
        node_x2_x = center[0] - major * np.cos(angle)
        node_x2_y = center[1] - major * np.sin(angle)
        plt.plot([node_x1_x, node_x2_x], [node_x1_y, node_x2_y], lw=minor, solid_capstyle="round")

    # ellipses
    for ellipse_index, ellipse_params in enumerate(ellipses):
        plot_ellipse(ellipse_params, str="k--")
        # ellipse: a tuple (c, a, b, t), where c = (x, y) is the center, a and
        # b are the major and minor radii, and t is the rotation angle.
        center_pos, major, minor, theta = ellipse_params
        # anisotropy related to this micro colony
        anisotropy_value = summary_statistics_values_list[ellipse_index]
        # Adding text inside a rectangular box by using the keyword 'bbox'
        plt.text(center_pos[0], center_pos[1], anisotropy_value, color="red")
    plt.title(summary_statistic_method + " for each micro colonies at timestep " + str(time_step))

    # plt.show()
    fig.savefig(output_directory + "/img/" + summary_statistic_method + "/" + summary_statistic_method + "_t" +
                str(time_step) + ".png", dpi=600)
    # close fig
    fig.clf()
    plt.close()


def fit_enclosing_ellipse(points):
    # convert dataframe to numpy array
    points = points.to_numpy()
    # print(points)
    # finds the smallest ellipse covering a finite set of points
    # https://github.com/dorshaviv/lowner-john-ellipse
    enclosing_ellipse = welzl(points)
    return enclosing_ellipse


def bac_info(bac_in_micro_colony):
    """
    Goal: this function returns important features of bacteria in the micro-colony that we
    study (ex: orientation, minor/major axis length, etc.). we need this data for calculating endpoints for
    fitting ellipse and plotting bacteria.

    @param bac_in_micro_colony     data frame   information(orientation, major/minor axis length, etc.)
    about each bacterium in each micro-colony in current time-step.

    """
    # center coordinate
    bacteria_center_coord = bac_in_micro_colony[["AreaShape_Center_X", "AreaShape_Center_Y"]]
    # major axis length
    bacteria_major_axis = bac_in_micro_colony["AreaShape_MajorAxisLength"]
    # minor axis length
    bacteria_minor_axis = bac_in_micro_colony["AreaShape_MinorAxisLength"]
    # orientation
    bacteria_orientation = bac_in_micro_colony["AreaShape_Orientation"]

    return bacteria_center_coord, bacteria_major_axis, bacteria_minor_axis, bacteria_orientation


def fit_ellipse(bac_in_micro_colony):
    # bacteria info
    (bacteria_center_coord, bacteria_major, bacteria_minor, bacteria_orientation) = bac_info(bac_in_micro_colony)
    bac_angle = -(bacteria_orientation + 90) * np.pi / 180

    # endpoints
    endpoint1_x = bacteria_center_coord["AreaShape_Center_X"] + (bacteria_major / 2) * np.cos(bac_angle)
    endpoint1_y = bacteria_center_coord["AreaShape_Center_Y"] + (bacteria_major / 2) * np.sin(bac_angle)
    endpoint2_x = bacteria_center_coord["AreaShape_Center_X"] - (bacteria_major / 2) * np.cos(bac_angle)
    endpoint2_y = bacteria_center_coord["AreaShape_Center_Y"] - (bacteria_major / 2) * np.sin(bac_angle)
    endpoint1 = pd.concat([endpoint1_x, endpoint1_y], axis=1)
    endpoint2 = pd.concat([endpoint2_x, endpoint2_y], axis=1)
    endpoints = pd.concat([endpoint1, endpoint2], axis=0)
    # fit ellipse
    ellipse_params = fit_enclosing_ellipse(endpoints)

    return ellipse_params


def aspect_ratio_calc(ellipse_params):
    # calculate aspect ratio
    center_pos, major, minor, theta = ellipse_params
    aspect_ratio = round(minor / major, 3)

    return aspect_ratio


def anisotropy_calc(bac_in_micro_colony):
    # bacteria info
    (bacteria_center_coord, bacteria_major, bacteria_minor, bacteria_orientation) = bac_info(bac_in_micro_colony)

    # main idea: https://github.com/ingallslab/bsim-related/blob/main/bsim_related/data_processing/cell_data_processing.py#L184

    local_anisotropies = []

    # modification of orientation
    bac_angle = -(bacteria_orientation + 90) * np.pi / 180

    for bacterium_index in range(bac_in_micro_colony.shape[0]):
        # Projection matrix
        projection_matrix = np.zeros(shape=(2, 2))
        for neighbor_index in range(bac_in_micro_colony.shape[0]):
            if neighbor_index != bacterium_index:
                # Compute the sum of the projection matrices on the orientation vectors of the neighbouring bacteria
                # projection matrix
                """
                cos(angle)                  cos(angle)*sin(angle)
                cos(angle)*sin(angle)       sin(angle)
                """
                projection_matrix += np.matrix([[np.cos(bac_angle.iloc[neighbor_index]) ** 2,
                                                 np.cos(bac_angle.iloc[neighbor_index]) *
                                                 np.sin(bac_angle.iloc[neighbor_index])],
                                                [np.cos(bac_angle.iloc[neighbor_index]) *
                                                 np.sin(bac_angle.iloc[neighbor_index]),
                                                 np.sin(bac_angle.iloc[neighbor_index]) ** 2]])

        # Compute the mean of the projection matrices on the orientation vectors of the neighbouring bacteria
        num_neighbours = bac_in_micro_colony.shape[0] - 1
        projection_matrix = projection_matrix / num_neighbours
        # Get the max real eigenvalues of the mean projection matrix; this is the local anisotropy
        local_anisotropies.append(max(la.eigvals(projection_matrix).real))

    # calculate mean anisotropy
    mean_anisotropy = round(np.mean(local_anisotropies), 3)

    return mean_anisotropy
