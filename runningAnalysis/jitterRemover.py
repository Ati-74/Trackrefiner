import numpy as np
import glob
from skimage import io
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import os
import argparse


def find_background_color(image):
    """
    Estimate the background color of an image.

    This function estimates the background color by averaging the colors at the borders of the image.
    It concatenates the pixel values from the top, bottom, left, and right edges and calculates their mean.

    @param image numpy array The image for which to estimate the background color.

    Returns:
        numpy array: The estimated background color.
    """

    # Extract the top edge of the image
    top_edge = image[0, :]

    # Extract the bottom edge of the image
    bottom_edge = image[-1, :]

    # Extract the left edge of the image
    left_edge = image[:, 0]

    # Extract the right edge of the image
    right_edge = image[:, -1]

    # Concatenate the edges
    edges = np.concatenate((top_edge, bottom_edge, left_edge, right_edge))

    # Calculate the mean of the edges to get the background color
    estimated_bg_color = np.mean(edges, axis=0)

    return estimated_bg_color


if __name__ == '__main__':

    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description='jitter remover')

    # Add arguments
    parser.add_argument('-i', '--input', help='This folder contains input images.')
    parser.add_argument('-o', '--output', default=None,
                        help="Where to save output images. Default value: save to the input images folder")

    # Parse the arguments
    args = parser.parse_args()

    # Define paths for input and output images
    images_path = args.input
    output_path = args.output

    if output_path is not None:
        output_path = output_path + "/"
    else:
        # Create the directory if it does not exist
        os.makedirs(images_path + '/modified_images', exist_ok=True)
        output_path = images_path + '/modified_images/'

    # Sort and store the list of image file paths
    images_list = sorted(glob.glob(images_path + '/*.tif'))

    # Iterate through images in reverse order
    for img_indx in range(len(images_list)-1, 0, -1):

        # Iterate through images in reverse order
        if img_indx == len(images_list)-1:
            reference_image = io.imread(images_list[img_indx])
            io.imsave(output_path + os.path.basename(images_list[img_indx]), reference_image)

            print("Reference Image: ")
            print(images_list[img_indx])
        else:

            print("Reference Image: ")
            print(output_path + os.path.basename(images_list[img_indx]))
            reference_image = io.imread(output_path + os.path.basename(images_list[img_indx]))

        # Read the current image to align
        moving_image = io.imread(images_list[img_indx - 1])

        print("Moving Image: ")
        print(images_list[img_indx - 1])

        # Calculate the phase cross-correlation between the reference and the moving images
        # In some scenarios, setting normalization to None (normalization=None) may yield improved results
        # default value: normalization="phase".
        shifted, error, diffphase = phase_cross_correlation(reference_image, moving_image, upsample_factor=100)

        # Estimate the background color of the moving image
        background_color = find_background_color(moving_image)

        # Apply shift to the moving image and save
        corrected_image = shift(moving_image, shift=(shifted[0], shifted[1]), mode='constant',
                                cval=background_color)

        io.imsave(output_path + os.path.basename(images_list[img_indx - 1]), corrected_image)
