import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import os


def angle_convert_to_radian(df):
    """
    Convert the orientation angles from degrees to radians and modify their values.

    @param df DataFrame containing bacteria information with 'AreaShape_Orientation' column.

    Returns:
    DataFrame: Updated DataFrame with 'AreaShape_Orientation' converted to radians.
    """
    df["AreaShape_Orientation"] = -(df["AreaShape_Orientation"] + 90) * np.pi / 180
    return df


def bac_info(bac_in_timestep):
    """
    Extract key information about bacteria from a given dataframe for a specific time step.

    @param bac_in_timestep DataFrame containing bacteria information for a particular time step.

    Returns:
    tuple: A tuple containing four elements:
           - Objects_center_x (Series): The x-coordinates of the centers of the bacteria.
           - Objects_center_y (Series): The y-coordinates of the centers of the bacteria.
           - Objects_major_axis (Series): The lengths of the major axes of the bacteria.
           - Objects_orientation (Series): The orientations of the bacteria.
    """
    # center coordinate
    try:
        Objects_center_x = bac_in_timestep["Location_Center_X"]
        Objects_center_y = bac_in_timestep["Location_Center_Y"]
    except KeyError:
        Objects_center_x = bac_in_timestep["AreaShape_Center_X"]
        Objects_center_y = bac_in_timestep["AreaShape_Center_Y"]

    # major axis length
    Objects_major_axis = bac_in_timestep["AreaShape_MajorAxisLength"]

    # orientation
    Objects_orientation = bac_in_timestep["AreaShape_Orientation"]

    return Objects_center_x, Objects_center_y, Objects_major_axis, Objects_orientation


def find_vertex(center_x, center_y, major, angle_rotation, angle_tolerance=1e-6):
    """
    Calculate the vertices of an ellipse based on its center coordinates, major axis length, and angle of rotation.

    @param center_x float The x-coordinate of the ellipse's center.
    @param center_y float The y-coordinate of the ellipse's center.
    @param major float The length of the major axis of the ellipse.
    @param angle_rotation float The angle of rotation of the ellipse from the horizontal axis in radians.
    @param angle_tolerance float(optional) A small tolerance value used to handle floating-point errors.
    Default is 1e-6.

    Returns:
    list: A list containing two lists, each representing the x and y coordinates of a vertex.
    """
    # Special case: Ellipse is nearly vertical
    if np.abs(np.abs(angle_rotation) - np.pi / 2) < angle_tolerance:  # Bacteria parallel to the vertical axis
        vertex_1_x = center_x
        vertex_1_y = center_y + major / 2
        vertex_2_x = center_x
        vertex_2_y = center_y - major / 2
    # Special case: Ellipse is nearly horizontal
    elif np.abs(angle_rotation) < angle_tolerance:  # Bacteria parallel to the horizontal axis
        vertex_1_x = center_x + major / 2
        vertex_1_y = center_y
        vertex_2_x = center_x - major / 2
        vertex_2_y = center_y
    else:
        # General case: Ellipse at an arbitrary angle
        semi_major = major
        vertex_1_x = float(semi_major / (np.cos(angle_rotation) + np.tan(angle_rotation) * np.sin(angle_rotation)) +
                           center_x)
        vertex_1_y = float((vertex_1_x - center_x) * np.tan(angle_rotation) + center_y)
        vertex_2_x = float(-semi_major / (np.cos(angle_rotation) + np.tan(angle_rotation) * np.sin(angle_rotation)) +
                           center_x)
        vertex_2_y = float((vertex_2_x - center_x) * np.tan(angle_rotation) + center_y)

    return [[vertex_1_x, vertex_1_y], [vertex_2_x, vertex_2_y]]


def lineage_life_history_plot(df_current, img_dir, TimeStep, axis, clr, prefix_raw_name, font_size, postfix_raw_name):
    """
    Plot the lineage life history of cells in a given time step on a background image.

    @param df_current DataFrame containing cell information.
    @param img_dir str Directory path where the background images are stored.
    @param TimeStep int The current time step for which the plot is generated.
    @param axis matplotlib.axes.Axes The matplotlib axis on which to plot.
    @param clr str Color used for plotting the cells.
    @param prefix_raw_name str Prefix for the raw image files.
    @param font_size int Font size for the cell ID annotations.
    @param postfix_raw_name str Postfix for the raw image files.
    """
    # draw Objects
    TimeStep = int(TimeStep)
    ax = axis

    # Adjust image file name format based on the length of TimeStep string
    if len(str(TimeStep - 1)) == 1:
        img_name = prefix_raw_name + "0" + str(TimeStep - 1) + postfix_raw_name
    elif len(str(TimeStep)) == 2:
        img_name = prefix_raw_name + "" + str(TimeStep - 1) + postfix_raw_name
    elif len(str(TimeStep)) == 3:
        img_name = prefix_raw_name + "" + str(TimeStep - 1) + postfix_raw_name

    # Read and display the background image
    img = cv2.imread(img_dir + '/' + img_name)
    plt.imshow(img)
    ax.imshow(img)

    # Extract objects' information from the current time step
    (Objects_center_coord_x, Objects_center_coord_y, Objects_major_current, Objects_orientation_current) = \
        bac_info(df_current)

    # Plot each cell
    num_cells = df_current.shape[0]
    for cell_indx in range(num_cells):
        # Get the current cell's information
        center_current = (Objects_center_coord_x[cell_indx], Objects_center_coord_y[cell_indx])
        major_current = (Objects_major_current.iloc[cell_indx]) / 2
        # radian
        angle_current = Objects_orientation_current.iloc[cell_indx]

        # Calculate the endpoints of the cell
        ends = find_vertex(center_current[0], center_current[1], major_current, angle_current)
        # endpoints
        node_x1_x_current = ends[0][0]
        node_x1_y_current = ends[0][1]
        node_x2_x_current = ends[1][0]
        node_x2_y_current = ends[1][1]

        # plot current bac
        ax.plot([node_x1_x_current, node_x2_x_current], [node_x1_y_current, node_x2_y_current], lw=1,
                solid_capstyle="round", color=clr)

        # Add cell ID and parent ID as text
        pos1x = np.abs(node_x1_x_current + center_current[0]) / 2
        pos1y = np.abs(node_x1_y_current + center_current[1]) / 2
        final_pos1x = np.abs(pos1x + center_current[0]) / 2
        final_pos1y = np.abs(pos1y + center_current[1]) / 2

        ax.text(final_pos1x, final_pos1y, int(df_current.iloc[cell_indx]["ObjectNumber"]), fontsize=font_size,
                color="#ff0000")


def tracking_bac(raw_img, csv_file_path, output_dir, color, prefix_raw_name, font_size, postfix_raw_name):
    """
    Generate tracking plots for bacteria over different time steps using data from CSV files.

    @param raw_img str Directory where the raw images are stored.
    @param csv_file_path str Path to the CSV file containing bacteria data.
    @param output_dir str Directory where the output plots should be saved.
    @param color str Color used for plotting.
    @param prefix_raw_name str Prefix for the raw image files.
    @param font_size int Font size for annotations.
    @param postfix_raw_name str Postfix for the raw image files.
    """
    # read cp output
    df = pd.read_csv(csv_file_path)
    modified_df = angle_convert_to_radian(df)

    # time steps
    t = list(set(modified_df['ImageNumber'].values))
    num_digit = len(str(t[-1]))

    os.makedirs(output_dir, exist_ok=True)

    for timestep in t:
        print('Time step ' + str(timestep))
        df_current = modified_df.loc[modified_df["ImageNumber"] == timestep]
        df_current = df_current.reset_index(drop=True)

        # plot
        fig, ax = plt.subplots()
        lineage_life_history_plot(df_current, raw_img, timestep, ax, color, prefix_raw_name, font_size,
                                  postfix_raw_name)
        plt.suptitle("Objects in time step = " + str(timestep), fontsize=14, fontweight="bold")

        obj_num_patch = mpatches.Patch(color='#ff0000', label='ObjectNumber')
        plt.legend(handles=[obj_num_patch], loc='upper right', ncol=6, bbox_to_anchor=(.75, 1.07), prop={'size': 7})

        fig.savefig(output_dir + '/' + "timeStep_t_" +
                    '0' * (num_digit - len(str(timestep))) + str(timestep) + ".png", dpi=600)
        fig.clf()
        plt.close()


if __name__ == "__main__":
    # Define the paths and parameters
    # Path to the CSV file containing bacteria data.
    cp_output_csv_file = 'CP_output/FilterObjects.csv'

    # Directory where the raw images are stored.
    raw_img_dir = 'modified_images/'

    # Directory where the output plots will be saved.
    output_dir = 'tracking_plot2/'

    # Prefix for the raw image files.
    prefix_raw_name = 'K12_Scene2_C0_T'

    # Postfix for the raw image files.
    postfix_raw_name = '.tif'

    # Color used for plotting.
    color = '#56e64e'
    
    # Font size for annotations.
    font_size = 2

    # Run the bacteria tracking plot generation
    tracking_bac(raw_img_dir, cp_output_csv_file, output_dir, color, prefix_raw_name, font_size, postfix_raw_name)
