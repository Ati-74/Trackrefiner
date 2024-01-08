import pandas as pd
import numpy as np
from CellProfilerAnalysis.strain.correction.action import helperFunctions
import matplotlib.pyplot as plt
import cv2
import glob
import os
import pickle


def find_img(pickle_list, margin=10):
    """
        Function to find the minimum and maximum x and y coordinates of the bacteria in the dataset.

        @param pickle_list list The list of paths to the pickle files.
        @param margin int An additional space added to the min and max coordinates for better visualization.
        Default is 10.

        @Returns:
        - min_x, max_x, min_y, max_y (tuple): The minimum and maximum x and y coordinates of the bacteria in the images.
        """

    # Initialize lists to store the x and y coordinates of the bacteria
    bacteria_x_point = []
    bacteria_y_point = []

    for i, pickle_file in enumerate(pickle_list):

        # read current pickle file
        current_bacteria_info = pickle.load(open(pickle_file, 'rb'))
        cs = current_bacteria_info['cellStates']

        for cell_indx in cs.keys():
            [bac_endpoint1, bac_endpoint2] = processing.convert_ends_to_pixel(cs[cell_indx].ends)
            bacteria_x_point.extend([bac_endpoint1[0], bac_endpoint2[0]])
            bacteria_y_point.extend([bac_endpoint1[1], bac_endpoint2[1]])

    min_x = min(bacteria_x_point) - margin
    max_x = max(bacteria_x_point) + margin

    min_y = min(bacteria_y_point) - margin
    max_y = max(bacteria_y_point) + margin

    return min_x, max_x, min_y, max_y


def draw_bacteria(pickle_file_directory, raw_image_file_directory, output_directory=None, removing_axis=False,
                  bacteria_color='red'):

    """
    Function to draw bacteria images using data from pickle files and raw images.

    @param pickle_file_directory str The path to the directory containing the pickle files.
    @param raw_image_file_directory str The path to the directory containing the raw images.
    @param output_directory str The path to the directory where the output images will be saved.
                               If set to None, the images will be saved in the same directory as the pickle files.
    @param removing_axis bool If True, the axes are removed from the output images. Default is False.
    @param bacteria_color str The color to use for the bacteria in the output images. Default is 'red'.
    """

    # Get the list of pickle files in the directory
    pickle_list = [filename for filename in sorted(glob.glob(pickle_file_directory + '/*.pickle'))]
    # Get the list of raw image files in the directory
    raw_image_list = [filename for filename in sorted(glob.glob(raw_image_file_directory + '/*.tif'))]

    # If there are no raw images, find the image boundaries from the pickle files
    if not raw_image_list:
        min_x, max_x, min_y, max_y = find_img(pickle_list)

    # If no output directory is specified, use the directory of the pickle files
    if output_directory is None:
        output_directory = os.path.dirname(pickle_file_directory)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory + '/pickle2image')

    for i, pickle_file in enumerate(pickle_list):

        # read current pickle file
        current_bacteria_info = pickle.load(open(pickle_file, 'rb'))
        cs = current_bacteria_info['cellStates']

        # Create a plot for the current time step (pickle file)
        fig, ax = plt.subplots()

        # If there are raw images, read the current raw image and display it
        if raw_image_list:
            img = cv2.imread(raw_image_list[i])
            plt.imshow(img)
            ax.imshow(img)
        else:
            # If there are no raw images, set the plot boundaries and reverse the y-axis
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            # Reverse Y axis
            ax.invert_yaxis()

        # Plot each bacterium in the current image
        for cell_indx in cs.keys():
            [bac_endpoint1, bac_endpoint2] = processing.convert_ends_to_pixel(cs[cell_indx].ends)
            radius = cs[cell_indx].radius

            # plot current bac
            ax.plot([bac_endpoint1[0], bac_endpoint2[0]], [bac_endpoint1[1], bac_endpoint2[1]], lw=radius,
                    solid_capstyle="round", color=bacteria_color)

        if removing_axis:
            ax.axis('off')  # Turn off the axis
        # plt.show()
        # Save the plot as a PNG image in the output directory
        fig.savefig(output_directory + '/pickle2image/' + os.path.basename(pickle_file).split('.pickle')[0] + ".png",
                    dpi=1200)
        # close fig
        fig.clf()
        plt.close()


if __name__ == "__main__":
    # draw bacteria images using data from pickle files and raw images.

    # The path to the directory containing the pickle files
    pickle_file_directory = '../examples/K12/outputs'
    # The path to the directory containing the raw image
    # Optional, if you set it, the bacteria will be drawn on the raw images
    # raw images should be in tif format
    raw_image_file_directory = '../examples/K12/raw img'

    # If you set it to Non, the images will be saved in the same directory as the pickle files
    output_directory = None
    # If True, the axes are removed from the output images.
    removing_axis = True
    # The color to use for the bacteria in the output images.
    bacteria_color = 'red'

    draw_bacteria(pickle_file_directory, raw_image_file_directory, output_directory, removing_axis, bacteria_color)
