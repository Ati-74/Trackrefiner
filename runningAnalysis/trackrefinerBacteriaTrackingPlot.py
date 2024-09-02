import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import os
import shutil
import argparse
import glob


def bac_info(bac_in_timestep, um_per_pixel):
    """
    Extract key information about bacteria from a given DataFrame for a specific time step.

    @param bac_in_timestep DataFrame DataFrame containing bacteria information for a particular time step.
    @param um_per_pixel float Conversion factor from pixels to micrometers.

    Returns:
    tuple: A tuple containing four elements:
           - Objects_center_x (Series): The x-coordinates of the centers of the bacteria.
           - Objects_center_y (Series): The y-coordinates of the centers of the bacteria.
           - Objects_major_axis (Series): The lengths of the major axes of the bacteria.
           - Objects_orientation (Series): The orientations of the bacteria.
    """

    # Calculate the center coordinates of bacteria in micrometers
    Objects_center_x = bac_in_timestep["AreaShape_Center_X"] / um_per_pixel
    Objects_center_y = bac_in_timestep["AreaShape_Center_Y"] / um_per_pixel

    # Calculate the major axis length of bacteria in micrometers
    Objects_major_axis = bac_in_timestep["length"] / um_per_pixel

    # Get the orientation of bacteria in radians
    Objects_orientation = bac_in_timestep["AreaShape_Orientation"]

    return Objects_center_x, Objects_center_y, Objects_major_axis, Objects_orientation


def find_vertex(center_x, center_y, major, angle_rotation, angle_tolerance=1e-6):
    """
    Calculate the vertices of an ellipse based on its center coordinates, major axis length, and angle of rotation.

    @param center_x float The x-coordinate of the ellipse's center.
    @param center_y float The y-coordinate of the ellipse's center.
    @param major float The length of the major axis of the ellipse.
    @param angle_rotation float The angle of rotation of the ellipse from the horizontal axis in radians.
    @param angle_tolerance float A small tolerance value used to handle floating-point errors. Default is 1e-6.

    Returns:
    list: A list containing two lists, each representing the x and y coordinates of a vertex.

    Notes:
    The function checks for two special cases:
    1. When the ellipse is nearly parallel to the vertical axis (angle_rotation ≈ π/2).
    2. When the ellipse is nearly parallel to the horizontal axis (angle_rotation ≈ 0).
    For these cases, the vertices are calculated differently to avoid numerical instability.
    """

    # Handle cases where the ellipse is nearly vertical
    if np.abs(np.abs(angle_rotation) - np.pi / 2) < angle_tolerance:
        vertex_1_x = center_x
        vertex_1_y = center_y + major / 2
        vertex_2_x = center_x
        vertex_2_y = center_y - major / 2
    # Handle cases where the ellipse is nearly horizontal
    elif np.abs(angle_rotation) < angle_tolerance:
        vertex_1_x = center_x + major / 2
        vertex_1_y = center_y
        vertex_2_x = center_x - major / 2
        vertex_2_y = center_y
    else:
        # General case for any arbitrary angle
        semi_major = major
        vertex_1_x = float(semi_major / (np.cos(angle_rotation) + np.tan(angle_rotation) * np.sin(angle_rotation)) +
                           center_x)
        vertex_1_y = float((vertex_1_x - center_x) * np.tan(angle_rotation) + center_y)
        vertex_2_x = float(-semi_major / (np.cos(angle_rotation) + np.tan(angle_rotation) * np.sin(angle_rotation)) +
                           center_x)
        vertex_2_y = float((vertex_2_x - center_x) * np.tan(angle_rotation) + center_y)

    return [[vertex_1_x, vertex_1_y], [vertex_2_x, vertex_2_y]]


def draw_tracking_plot(df_current, raw_images, timestep, axis, object_color, um_per_pixel, font_size):

    """
    Plot the lineage life history of bacteria in a given time step on a background image (raw image).

    @param df_current DataFrame DataFrame containing cell information for the current timestep.
    @param raw_images list List of paths to raw images for each timestep.
    @param timestep int The current time step for which the plot is generated.
    @param axis matplotlib.axes.Axes The matplotlib axis on which to plot.
    @param object_color str Color used for plotting the bacteria.
    @param um_per_pixel float Conversion factor from pixels to micrometers.
    @param font_size float Font size for labeling the cells.
    """

    timestep = int(timestep)
    ax = axis

    # Get the raw image for the current time step
    current_raw_img = raw_images[timestep - 1]
    print('Raw image path for this time step: ' + str(current_raw_img))

    # Read and display the background image
    img = cv2.imread(current_raw_img)
    plt.imshow(img)
    ax.imshow(img)

    # Extract objects' information from the current time step
    Objects_center_coord_x, Objects_center_coord_y, Objects_major_current, Objects_orientation_current = \
        bac_info(df_current, um_per_pixel)

    # Plot each cell
    num_cells = df_current.shape[0]
    for cell_indx in range(num_cells):
        # Get the current cell's information
        center_current = (Objects_center_coord_x[cell_indx], Objects_center_coord_y[cell_indx])
        major_current = (Objects_major_current.iloc[cell_indx]) / 2  # Half the length for plotting

        # Get the orientation in radians
        angle_current = Objects_orientation_current.iloc[cell_indx]

        # Calculate the endpoints of the cell
        ends = find_vertex(center_current[0], center_current[1], major_current, angle_current)
        # Endpoints of the major axis of the ellipse
        node_x1_x_current = ends[0][0]
        node_x1_y_current = ends[0][1]
        node_x2_x_current = ends[1][0]
        node_x2_y_current = ends[1][1]

        # Plot current bacteria
        ax.plot([node_x1_x_current, node_x2_x_current], [node_x1_y_current, node_x2_y_current], lw=1,
                solid_capstyle="round", color=object_color)

        # Add cell ID and parent ID as text
        pos1x = np.abs(node_x1_x_current + center_current[0]) / 2
        pos2x = np.abs(node_x2_x_current + center_current[0]) / 2

        pos1y = np.abs(node_x1_y_current + center_current[1]) / 2
        pos2y = np.abs(node_x2_y_current + center_current[1]) / 2

        final_pos1x = np.abs(pos1x + center_current[0]) / 2
        final_pos2x = np.abs(pos2x + center_current[0]) / 2

        final_pos1y = np.abs(pos1y + center_current[1]) / 2
        final_pos2y = np.abs(pos2y + center_current[1]) / 2

        # Plot cell ID and parent ID
        ax.text(final_pos1x, final_pos1y, int(df_current.iloc[cell_indx]["id"]), fontsize=font_size, color="#ff0000")
        ax.text(final_pos2x, final_pos2y, int(df_current.iloc[cell_indx]["parent_id"]), fontsize=font_size,
                color="#0000ff")


def tracking_bac(raw_img_dir, trackrefiner_csv_output_file, output_dir, object_color, um_per_pixel, font_size):
    """
    Generate tracking plots for bacteria over different time steps using data from CSV file.

    @param raw_img_dir str Directory path where the raw images are stored.
    @param trackrefiner_csv_output_file str Path to the CSV file output of Trackrefiner.
    @param output_dir str Directory where the output plots should be saved.
    @param object_color str Color of objects in tracking plots.
    @param um_per_pixel float Conversion factor from pixels to micrometers.
    @param font_size float Font size for labeling the cells in the plots.
    """

    # Read CSV file containing tracking data
    df = pd.read_csv(trackrefiner_csv_output_file)

    # Get list of all raw images
    raw_images = glob.glob(raw_img_dir + '/*.tif')

    # Get unique time steps
    t = np.unique(df['stepNum'].values)
    num_digit = len(str(t[-1]))  # To format filenames consistently

    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(trackrefiner_csv_output_file) + '/Trackrefiner_tracking_plot/'

    os.makedirs(output_dir, exist_ok=True)

    for timestep in t:
        print('Time Step: ' + str(timestep))

        # Get data for the current timestep
        df_current = df.loc[df["stepNum"] == timestep]
        df_current = df_current.reset_index(drop=True)

        # Plot tracking plot for the current timestep
        fig, ax = plt.subplots()
        draw_tracking_plot(df_current, raw_images, timestep, ax, object_color, um_per_pixel, font_size)

        # Add legend and title
        plt.suptitle("Trackrefiner: Tracking objects in time step = " + str(timestep), fontsize=14, fontweight="bold")
        parent_patch = mpatches.Patch(color='#0000ff', label='parent', )
        id_patch = mpatches.Patch(color='#ff0000', label='identity id', )
        plt.legend(handles=[parent_patch, id_patch], loc='upper right', ncol=6,
                   bbox_to_anchor=(.75, 1.07), prop={'size': 7})

        # Save the figure
        fig.savefig(output_dir + '/' + "Trackrefiner.timeStep t = " +
                    '0' * (num_digit - len(str(timestep))) + str(timestep) + ".png", dpi=600)
        # Close the figure
        fig.clf()
        plt.close()


if __name__ == "__main__":

    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description='Generate tracking plots for cell lineages using Trackrefiner output.')

    # Add arguments for command line interface
    parser.add_argument('-t', '--trackrefiner_csv_output_file', help='CSV file output of Trackrefiner.')

    parser.add_argument('-r', '--raw_image_dir', help='Directory containing raw images. '
                                                      'The format of raw images should be tif.')

    parser.add_argument('-o', '--output', default=None,
                        help="Directory where to save tracking plots. "
                             "Default value: Save to the Trackrefiner output file folder.")

    parser.add_argument('-u', '--umPerPixel', default=0.144,
                        help="Conversion factor from pixels to micrometers. Default value: 0.144.")

    parser.add_argument('-c', '--objectColor', default="#56e64e",
                        help="Color of objects in tracking plots. Default value: #56e64e.")

    parser.add_argument('-f', '--font_size', default="1",
                        help="Font size for labeling information on objects. Default value: 1")

    # Parse the arguments
    args = parser.parse_args()

    # Assign parsed arguments to variables
    raw_img_dir = args.raw_image_dir
    trackrefiner_csv_output_file = args.trackrefiner_csv_output_file
    output_dir = args.output
    object_color = args.objectColor
    um_per_pixel = float(args.umPerPixel)
    font_size = float(args.font_size)

    # Generate tracking plots for bacteria
    tracking_bac(raw_img_dir, trackrefiner_csv_output_file, output_dir, object_color, um_per_pixel, font_size)
