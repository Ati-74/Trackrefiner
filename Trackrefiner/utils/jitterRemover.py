import numpy as np
import glob
from skimage import io
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from skimage import color
import cv2
import os
import argparse


def find_background_color(image):
    """
    Estimate the background color of an image.

    This function calculates the background color by averaging the pixel values along the borders of the image
    (top, bottom, left, and right edges).

    :param numpy.ndarray image:
        The image for which to estimate the background color. It should be a 2D or 3D array representing
        grayscale or color image data.

    :returns:
        numpy.ndarray:

        The estimated background color as a single value (for grayscale) or an array (for color images).
    """

    if image.ndim == 3:  # RGB image

        top_edge = image[0, :, :]

        bottom_edge = image[-1, :, :]

        left_edge = image[:, 0, :]

        right_edge = image[:, -1, :]

    else:  # Grayscale image
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


def main():
    """
    Main function for the jitter remover utils.

    This utils aligns images in a specified folder by removing jitter using phase cross-correlation and
    saves the corrected images to an output folder.

    :param str --input (-i):
        Path to the folder containing input images in `.tif` format.

    :param str --output (-o):
        Path to the folder where the corrected images will be saved. If not specified, a `modified_images`
        folder will be created inside the input images folder.

    :returns:
        None. Processed images are saved in the output folder.

    :rtype: None
    """

    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description='jitter remover')

    # Add arguments
    parser.add_argument('-i', '--input', required=True, help='This folder contains input images.')
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
    for img_indx in range(len(images_list) - 1, 0, -1):

        # Iterate through images in reverse order
        if img_indx == len(images_list) - 1:
            reference_image = cv2.imread(images_list[img_indx], cv2.IMREAD_UNCHANGED)
            if reference_image.ndim == 3:
                reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

            # io.imsave(output_path + os.path.basename(images_list[img_indx]), reference_image)

            if len(reference_image.shape) == 2:  # Grayscale image
                cv2.imwrite(output_path + os.path.basename(images_list[img_indx]), reference_image)
            elif reference_image.shape[2] == 3:  # RGB Image (convert to BGR)
                reference_image_bgr = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path + os.path.basename(images_list[img_indx]), reference_image_bgr)

            print("Reference Image: ")
            print(images_list[img_indx])
        else:

            print("Reference Image: ")
            print(output_path + os.path.basename(images_list[img_indx]))
            reference_image = cv2.imread(output_path + os.path.basename(images_list[img_indx]), cv2.IMREAD_UNCHANGED)

            if reference_image.ndim == 3:
                reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

        # Read the current image to align
        moving_image = cv2.imread(images_list[img_indx - 1], cv2.IMREAD_UNCHANGED)

        if moving_image.ndim == 3:
            moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB)

        print("Moving Image: ")
        print(images_list[img_indx - 1])

        # Calculate the phase cross-correlation between the reference and the moving images
        # In some scenarios, setting normalization to None (normalization=None) may yield improved results
        # default value: normalization="phase".
        shifted, error, diffphase = phase_cross_correlation(reference_image, moving_image, upsample_factor=100)

        # Estimate the background color of the moving image
        background_color = find_background_color(moving_image)

        # Apply shift to the moving image and save
        if moving_image.ndim == 3 and moving_image.shape[2] == 3:
            # Apply shift to each channel and stack them back together
            corrected_image = np.stack([shift(moving_image[..., i], shift=(shifted[0], shifted[1]),
                                              mode='constant', cval=background_color[i])
                                        for i in range(3)], axis=-1)
        else:
            corrected_image = shift(moving_image, shift=(shifted[0], shifted[1]), mode='constant',
                                    cval=background_color)

        # io.imsave(output_path + os.path.basename(images_list[img_indx - 1]), corrected_image)
        if len(corrected_image.shape) == 2:  # Grayscale image
            cv2.imwrite(output_path + os.path.basename(images_list[img_indx - 1]), corrected_image)
        elif corrected_image.shape[2] == 3:  # RGB Image (convert to BGR)
            corrected_image_bgr = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path + os.path.basename(images_list[img_indx - 1]), corrected_image_bgr)


if __name__ == "__main__":
    main()
