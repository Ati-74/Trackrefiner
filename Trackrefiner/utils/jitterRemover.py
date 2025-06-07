import numpy as np
import glob
import pandas as pd
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
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


def reverse_jitter_removing(images_list, output_path):

    shift_dict = {'ImageNumber': [], 'Shift_X': [], 'Shift_Y': []}

    # Iterate through images in reverse order
    for img_idx in range(len(images_list) - 1, 0, -1):

        # Iterate through images in reverse order
        if img_idx == len(images_list) - 1:

            reference_image = cv2.imread(images_list[img_idx], cv2.IMREAD_UNCHANGED)

            cv2.imwrite(output_path + os.path.basename(images_list[img_idx]), reference_image)

            print("Reference Image: ")
            print(images_list[img_idx])

            shift_dict['ImageNumber'].append(img_idx + 1)
            shift_dict['Shift_X'].append(0)
            shift_dict['Shift_Y'].append(0)

        else:

            print("Reference Image: ")
            print(output_path + os.path.basename(images_list[img_idx]))
            reference_image = cv2.imread(output_path + os.path.basename(images_list[img_idx]), cv2.IMREAD_UNCHANGED)

            if reference_image.ndim == 3:
                reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

        # Read the current image to align
        moving_image = cv2.imread(images_list[img_idx - 1], cv2.IMREAD_UNCHANGED)

        if moving_image.ndim == 3:
            moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB)

        print("Moving Image: ")
        print(images_list[img_idx - 1])
        print('=============================================================')

        # Calculate the phase cross-correlation between the reference and the moving images
        # In some scenarios, setting normalization to None (normalization=None) may yield improved results
        # default value: normalization="phase".
        if reference_image.ndim == 3:

            reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
            moving_image_gray = cv2.cvtColor(moving_image, cv2.COLOR_RGB2GRAY)
            shifted, error, diffphase = phase_cross_correlation(reference_image_gray, moving_image_gray,
                                                                upsample_factor=100)
        else:
            shifted, error, diffphase = phase_cross_correlation(reference_image, moving_image, upsample_factor=100)

        shift_dict['ImageNumber'].append(img_idx)
        shift_dict['Shift_X'].append(shifted[1])
        shift_dict['Shift_Y'].append(shifted[0])

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

        if len(corrected_image.shape) == 2:  # Grayscale image
            cv2.imwrite(output_path + os.path.basename(images_list[img_idx - 1]), corrected_image)

        elif corrected_image.shape[2] == 3:  # RGB Image (convert to BGR)
            corrected_image_bgr = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path + os.path.basename(images_list[img_idx - 1]), corrected_image_bgr)

    shift_df = pd.DataFrame.from_dict(shift_dict)
    # sort
    shift_df = shift_df.sort_values(by='ImageNumber')
    shift_df.to_csv(output_path + 'shift_dataframe_R.csv', index=False)


def forward_jitter_removing(images_list, output_path):

    shift_dict = {'ImageNumber': [], 'Shift_X': [], 'Shift_Y': []}

    # Iterate through images
    for img_idx in range(len(images_list) - 1):

        # Iterate through images in reverse order
        if img_idx == 0:

            reference_image = cv2.imread(images_list[img_idx], cv2.IMREAD_UNCHANGED)
            cv2.imwrite(output_path + os.path.basename(images_list[img_idx]), reference_image)

            shift_dict['ImageNumber'].append(img_idx + 1)
            shift_dict['Shift_X'].append(0)
            shift_dict['Shift_Y'].append(0)

            print("Reference Image: ")
            print(images_list[img_idx])
        else:

            print("Reference Image: ")
            print(output_path + os.path.basename(images_list[img_idx]))
            reference_image = cv2.imread(output_path + os.path.basename(images_list[img_idx]), cv2.IMREAD_UNCHANGED)

            if reference_image.ndim == 3:
                reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

        # Read the current image to align
        moving_image = cv2.imread(images_list[img_idx + 1], cv2.IMREAD_UNCHANGED)

        if moving_image.ndim == 3:
            moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB)

        print("Moving Image: ")
        print(images_list[img_idx + 1])
        print('=============================================================')

        # Calculate the phase cross-correlation between the reference and the moving images
        # In some scenarios, setting normalization to None (normalization=None) may yield improved results
        # default value: normalization="phase".
        if reference_image.ndim == 3:
            reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
            moving_image_gray = cv2.cvtColor(moving_image, cv2.COLOR_RGB2GRAY)
            shifted, error, diffphase = phase_cross_correlation(reference_image_gray, moving_image_gray,
                                                                upsample_factor=100)
        else:
            shifted, error, diffphase = phase_cross_correlation(reference_image, moving_image, upsample_factor=100)

        shift_dict['ImageNumber'].append(img_idx + 2)
        shift_dict['Shift_X'].append(shifted[1])
        shift_dict['Shift_Y'].append(shifted[0])

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

        if len(corrected_image.shape) == 2:  # Grayscale image
            cv2.imwrite(output_path + os.path.basename(images_list[img_idx + 1]), corrected_image)
        elif corrected_image.shape[2] == 3:  # RGB Image (convert to BGR)
            corrected_image_bgr = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path + os.path.basename(images_list[img_idx + 1]), corrected_image_bgr)

    shift_df = pd.DataFrame.from_dict(shift_dict)
    # sort
    shift_df = shift_df.sort_values(by='ImageNumber')
    shift_df.to_csv(output_path + 'shift_dataframe_F.csv', index=False)


def jitter_removing_predefined_shifts(images_list, output_path, shift_df):

    # Iterate through images in reverse order
    for img_idx in range(len(images_list)):

        # Read the current image to align
        moving_image = cv2.imread(images_list[img_idx], cv2.IMREAD_UNCHANGED)

        print("Moving Image: ")
        print(images_list[img_idx])
        print('=============================================================')

        shift_x, shift_y = \
            shift_df.loc[shift_df['ImageNumber'] == (img_idx + 1)][['Shift_X', 'Shift_Y']].values.tolist()[0]

        if shift_x != 0 or shift_y != 0:

            if moving_image.ndim == 3:
                moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB)

            # Estimate the background color of the moving image
            background_color = find_background_color(moving_image)

            # Apply shift to the moving image and save
            if moving_image.ndim == 3 and moving_image.shape[2] == 3:

                # Apply shift to each channel and stack them back together
                corrected_image = np.stack([shift(moving_image[..., i], shift=(shift_y, shift_x),
                                                  mode='constant', cval=background_color[i])
                                            for i in range(3)], axis=-1)
            else:
                corrected_image = shift(moving_image, shift=(shift_y, shift_x), mode='constant',
                                        cval=background_color)

            if len(corrected_image.shape) == 2:  # Grayscale image
                cv2.imwrite(output_path + os.path.basename(images_list[img_idx]), corrected_image)
            elif corrected_image.shape[2] == 3:  # RGB Image (convert to BGR)
                corrected_image_bgr = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path + os.path.basename(images_list[img_idx]), corrected_image_bgr)

        else:
            cv2.imwrite(output_path + os.path.basename(images_list[img_idx]), moving_image)


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
    parser.add_argument('-d', '--processing_order', default='R',
                        help="Specifies the processing order for jitter removal. "
                             "'R' (Reverse) processes images from last to first (high-density to low-density), "
                             "which typically yields better results. "
                             "'F' (Forward) processes from first to last (low-density to high-density). "
                             "Default: 'R'. Allowed values: 'R' (Reverse), 'F' (Forward). "
                             "This argument is used when no predefined shift values are provided via"
                             " '--shift_dataframe'.")
    parser.add_argument('-s', '--shift_dataframe', default=None,
                        help="Optional: Path to a CSV file containing predefined shift values for jitter removal. "
                             "The file must have three columns: 'imageNumber' (to match images in order), "
                             "'Shift_X' (horizontal shift), and 'Shift_Y' (vertical shift). "
                             "If provided, the processing will follow the order in the file, "
                             "starting from the first image. "
                             "In this case, specifying '--processing_order' is not necessary.")

    parser.add_argument('-o', '--output', default=None,
                        help="Where to save output images. Default value: save to the input images folder")

    # Parse the arguments
    args = parser.parse_args()

    # Define paths for input and output images
    images_path = args.input
    output_path = args.output
    processing_order = args.processing_order
    shift_dataframe = args.shift_dataframe

    if output_path is not None:
        output_path = output_path + "/"
    else:
        # Create the directory if it does not exist
        os.makedirs(images_path + '/jitter_removed_images', exist_ok=True)
        output_path = images_path + '/jitter_removed_images/'

    # Sort and store the list of image file paths
    images_list = sorted(glob.glob(images_path + '/*.tif'))

    if shift_dataframe is None:

        if processing_order == 'R':
            reverse_jitter_removing(images_list, output_path)
        else:
            forward_jitter_removing(images_list, output_path)

    else:

        shift_df = pd.read_csv(shift_dataframe)
        shift_df = shift_df.sort_values(by='ImageNumber')

        if shift_df.shape[0] == len(images_list):
            required_columns = {'ImageNumber', 'Shift_X', 'Shift_Y'}

            if required_columns.issubset(shift_df.columns):
                # Ensure all columns are numeric
                if (shift_df['ImageNumber'].apply(lambda x: isinstance(x, (int, float))).all() and
                        shift_df['Shift_X'].apply(lambda x: isinstance(x, (int, float))).all() and
                        shift_df['Shift_Y'].apply(lambda x: isinstance(x, (int, float))).all()):
                    # Ensure ImageNumber is positive
                    if (shift_df['ImageNumber'] > 0).all():
                        jitter_removing_predefined_shifts(images_list, output_path, shift_df)
                    else:
                        raise ValueError("Error: 'ImageNumber' column must contain only positive values.")
                else:
                    raise ValueError(
                        "Error: 'ImageNumber', 'Shift_X', and 'Shift_Y' columns must contain only numeric values.")
            else:
                raise ValueError("Error: The provided shift dataframe must contain the columns "
                                 "'ImageNumber', 'Shift_X', and 'Shift_Y'.")
        else:
            raise ValueError(f"Error: The number of rows in the shift dataframe ({shift_df.shape[0]}) does not "
                             f"match the number of images ({len(images_list)}).")

    print(f"Jitter removal completed. The processed images are saved in the folder: {output_path}.")


if __name__ == "__main__":
    main()
