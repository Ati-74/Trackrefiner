import numpy as np
import pandas as pd
from PIL import Image
import glob
from Scripts.densityCalculation.Script.action.multiRegionsDetection import multi_region_detection
from Scripts.densityCalculation.Script.action.CalcDensity import calculate_colony_density
import logging

# Configure logging to capture warnings
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def checking_columns(data_frame):

    """
    Identify and return key column names for bacterial tracking and positioning in the provided dataframe.

    :param pd.DataFrame data_frame:
        Dataframe containing bacterial tracking and measurement data.

    :return:
        - **center_coordinate_columns** (*dict*): Dictionary specifying x and y coordinate column names for the center.
        - **all_center_coordinate_columns** (*dict*): Dictionary containing all potential x and y center coordinates.
        - **parent_image_number_col** (*str*): Column name for the parent image number.
        - **parent_object_number_col** (*str*): Column name for the parent object number.
        - **label_col** (*str*): Column name for tracking object labels.

    """

    center_coordinate_columns = []
    all_center_coordinate_columns = []

    parent_image_number_col = [col for col in data_frame.columns if 'TrackObjects_ParentImageNumber_' in col][0]
    parent_object_number_col = [col for col in data_frame.columns if 'TrackObjects_ParentObjectNumber_' in col][0]
    # column name
    label_col = [col for col in data_frame.columns if 'TrackObjects_Label_' in col][0]

    if 'Location_Center_X' in data_frame.columns and 'Location_Center_Y' in data_frame.columns:
        center_coordinate_columns = {'x': 'Location_Center_X', 'y': 'Location_Center_Y'}
        all_center_coordinate_columns = {'x': ['Location_Center_X'], 'y': ['Location_Center_Y']}

        if 'AreaShape_Center_X' in data_frame.columns and 'AreaShape_Center_Y' in data_frame.columns:
            all_center_coordinate_columns = {'x': ['Location_Center_X', 'AreaShape_Center_X'],
                                             'y': ['Location_Center_Y', 'AreaShape_Center_Y']}

    elif 'AreaShape_Center_X' in data_frame.columns and 'AreaShape_Center_Y' in data_frame.columns:
        center_coordinate_columns = {'x': 'AreaShape_Center_X', 'y': 'AreaShape_Center_Y'}
        all_center_coordinate_columns = {'x': ['AreaShape_Center_X'], 'y': ['AreaShape_Center_Y']}

    else:
        print('There was no column corresponding to the center of bacteria.')
        breakpoint()

    return (center_coordinate_columns, all_center_coordinate_columns, parent_image_number_col, parent_object_number_col,
            label_col)


def angle_convert_to_radian(df):

    """
    Convert the bacterial orientation angle from degrees to radians.

    The function modifies the `AreaShape_Orientation` column by applying the formula:
    `-(angle + 90) * np.pi / 180`, which adjusts for the convention used in the dataset.

    :param pd.DataFrame df:
        Dataframe containing bacterial shape and orientation data.

    """

    # modification of bacterium orientation
    # -(angle + 90) * np.pi / 180
    df["AreaShape_Orientation"] = -(df["AreaShape_Orientation"] + 90) * np.pi / 180

    return df


def convert_to_um(data_frame, um_per_pixel, all_center_coordinate_columns):

    """
    Convert pixel-based measurements to micrometers.

    :param pd.DataFrame data_frame:
        Dataframe containing spatial measurements in pixels.
    :param float um_per_pixel:
        Conversion factor to transform pixel-based values into micrometers.
    :param dict all_center_coordinate_columns:
        Dictionary specifying column names for x and y coordinates.

    :return:
        - **data_frame** (*pd.DataFrame*): Updated dataframe with measurements converted to micrometers.
    """

    # Convert distances to um (0.144 um/pixel on 63X objective)

    data_frame[all_center_coordinate_columns['x']] *= um_per_pixel

    data_frame[all_center_coordinate_columns['y']] *= um_per_pixel

    data_frame['AreaShape_MajorAxisLength'] *= um_per_pixel
    data_frame['AreaShape_MinorAxisLength'] *= um_per_pixel

    return data_frame


def data_conversion(dataframe, um_per_pixel, all_center_coordinate_columns):
    """
     Apply unit conversion and orientation angle adjustment to bacterial data.

     :param pd.DataFrame dataframe:
         Dataframe containing bacterial measurement data.
     :param float um_per_pixel:
         Conversion factor from pixels to micrometers.
     :param dict all_center_coordinate_columns:
         Dictionary containing x and y coordinate column names.

     :return:
         - **dataframe** (*pd.DataFrame*): Updated dataframe with converted measurements and adjusted angles.
     """

    dataframe = convert_to_um(dataframe, um_per_pixel, all_center_coordinate_columns)
    dataframe = angle_convert_to_radian(dataframe)

    return dataframe


def process_datasets(gt_out_path, cp_out_path, npy_files_dir, um_per_pixel=0.144, output_directory=None,
                     time_step_list=None):

    """
    Process bacterial tracking and density analysis by integrating CP tracking and ground truth data.

    This function reads CP tracking and ground truth datasets, applies spatial conversions, detects multi-region
    objects, and calculates colony density.

    :param str gt_out_path:
        Path to the ground truth CSV file.
    :param str cp_out_path:
        Path to the CellProfiler output CSV file.
    :param str npy_files_dir:
        Directory containing segmentation `.npy` files.
    :param float um_per_pixel:
        Conversion factor from pixels to micrometers (default: 0.144).
    :param str output_directory:
        Directory where processed outputs and density calculations will be stored.
    :param list time_step_list:
        List of time steps to process. If 'all', all time steps from the dataset are used.

    :return:
        - **df_density** (*pd.DataFrame*): Dataframe containing computed colony densities per time step.
    """

    sorted_npy_files_list = sorted(glob.glob(npy_files_dir + '/*.npy'))

    gt_df = pd.read_csv(gt_out_path)
    cp_df = pd.read_csv(cp_out_path)

    (center_coordinate_columns, all_center_coordinate_columns, parent_image_number_col, parent_object_number_col,
     label_col) = checking_columns(cp_df)

    cp_out_df = data_conversion(cp_df, um_per_pixel, all_center_coordinate_columns)

    # useful for only final comparison between original & modified dataframe
    cp_out_df['prev_index'] = cp_out_df.index
    cp_out_df['coordinate'] = ''

    if time_step_list == 'all':
        time_step_list = cp_out_df['ImageNumber'].unique()

    # correction of multi regions
    cp_out_df = multi_region_detection(cp_out_df, sorted_npy_files_list, um_per_pixel,
                                       center_coordinate_columns,
                                       all_center_coordinate_columns,
                                       parent_image_number_col, parent_object_number_col,
                                       time_step_list)

    merged_df = gt_df.merge(cp_out_df, on=['ImageNumber', 'ObjectNumber'], how='inner', suffixes=('_gr', '_cp'))

    density_dict = {}

    if merged_df.shape[0] != gt_df.shape[0]:
        raise ValueError(f"Mismatch in the number of rows: merged_df has {merged_df.shape[0]} rows while "
                         f"gt_df has {gt_df.shape[0]} rows.")
    else:

        img_array = np.load(sorted_npy_files_list[0])
        h, w, d = img_array.shape

        for t in merged_df['ImageNumber'].unique():

            if t in time_step_list:

                current_bac = merged_df.loc[merged_df['ImageNumber'] == t]

                # Initialize a blank image (black background)
                bw_image = np.zeros((h, w), dtype=np.uint8)  # Image dimensions are (height, width)

                # Loop through each row in the DataFrame
                for _, row in current_bac.iterrows():
                    coordinates = row['coordinate']  # Extract the set of (x, y) tuples

                    for x, y in coordinates:
                        if 0 <= x < w and 0 <= y < h:  # Ensure the coordinates are within bounds
                            bw_image[int(y), int(x)] = 255  # Set the pixel to white (255)

                # Convert the NumPy array to a PIL Image
                img = Image.fromarray(bw_image, mode='L')  # 'L' mode is for grayscale images

                # Display the image
                # img.show()

                # Define export path and compute colony densityCalculation
                density = calculate_colony_density(img, fig_export_path=f'{output_directory}/density_plot/t ={t}.')
                density_dict[t] = {'ImageNumber': t, 'Density': density}

    df_density = pd.DataFrame.from_dict(density_dict, orient='index')
    df_density.to_csv(output_directory + '/density.csv', index=False)
