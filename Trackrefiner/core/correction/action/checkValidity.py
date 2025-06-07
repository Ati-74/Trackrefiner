import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QMessageBox


def show_error_message(msg):
    error_box = QMessageBox()
    error_box.setIcon(QMessageBox.Critical)
    error_box.setWindowTitle("Input Error")
    error_box.setText(msg)
    error_box.setStandardButtons(QMessageBox.Ok)
    error_box.exec_()


def validate_cp_output_csv(is_gui_mode, cp_output_csv):
    """
    Validate the CP output CSV file.
    """
    # Check if the file exists
    if not os.path.isfile(cp_output_csv):
        msg = f"The CellProfiler output CSV file does not exist: {cp_output_csv}"
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise FileNotFoundError(msg)

    # Load the file as a DataFrame
    try:
        df = pd.read_csv(cp_output_csv)
    except Exception as e:
        msg = f"Unable to read the CellProfiler output CSV file: {cp_output_csv}. Error: {e}"
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Check if the DataFrame is empty
    if df.shape[0] == 0:
        msg = f"The {cp_output_csv} is empty."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Required columns
    required_columns = [
        "ImageNumber",
        "ObjectNumber",
        "AreaShape_MajorAxisLength",
        "AreaShape_MinorAxisLength",
        "AreaShape_Orientation",
    ]
    for col in required_columns:
        if col not in df.columns:
            msg = f"Missing required column in CellProfiler output CSV: {col}"
            if is_gui_mode:
                show_error_message(msg)
                return  # stop further execution
            else:
                raise ValueError(msg)

    # Validate ImageNumber and ObjectNumber (int and > 0)
    for col in ["ImageNumber", "ObjectNumber"]:
        if not df[col].apply(lambda x: isinstance(x, (int, float)) and x > 0).all():
            msg = f"All values in column '{col}' must be positive integers in {cp_output_csv}."
            if is_gui_mode:
                show_error_message(msg)
                return  # stop further execution
            else:
                raise ValueError(msg)

    # Validate AreaShape_MajorAxisLength and AreaShape_MinorAxisLength (positive > 0)
    for col in ["AreaShape_MajorAxisLength", "AreaShape_MinorAxisLength"]:
        if not df[col].apply(lambda x: isinstance(x, (int, float)) and x > 0).all():
            msg = f"All values in column '{col}' must be positive numbers in {cp_output_csv}."
            if is_gui_mode:
                show_error_message(msg)
                return  # stop further execution
            else:
                raise ValueError(msg)

    # Validate AreaShape_Orientation (must be a number)
    if not df["AreaShape_Orientation"].apply(lambda x: isinstance(x, (int, float))).all():
        msg = f"All values in column 'AreaShape_Orientation' must be numbers in {cp_output_csv}."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Check for parent columns
    parent_image_number_col = [col for col in df.columns if "TrackObjects_ParentImageNumber_" in col]
    if not parent_image_number_col:
        msg = "Missing required 'TrackObjects_ParentImageNumber_' column in CellProfiler output CSV."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    parent_object_number_col = [col for col in df.columns if "TrackObjects_ParentObjectNumber_" in col]
    if not parent_object_number_col:
        msg = "Missing required 'TrackObjects_ParentObjectNumber_' column in CellProfiler output CSV."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Validate parent columns (int and > 0)
    for col in parent_image_number_col + parent_object_number_col:
        if not df[col].apply(lambda x: isinstance(x, (int, float)) and x >= 0).all():
            msg = f"All values in column '{col}' must be positive integers in {cp_output_csv}."
            if is_gui_mode:
                show_error_message(msg)
                return  # stop further execution
            else:
                raise ValueError(msg)

    # Check for label columns
    label_col_list = [col for col in df.columns if "TrackObjects_Label_" in col]
    if not label_col_list:
        msg = "Missing required 'TrackObjects_Label_' column(s) in CellProfiler output CSV."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Check for center columns
    center_columns_1 = {"Location_Center_X", "Location_Center_Y"}
    center_columns_2 = {"AreaShape_Center_X", "AreaShape_Center_Y"}
    if not (center_columns_1.issubset(df.columns) or center_columns_2.issubset(df.columns)):
        msg = ("Missing required center columns in CellProfiler output CSV. "
               "Either ('Location_Center_X', 'Location_Center_Y') or "
               "('AreaShape_Center_X', 'AreaShape_Center_Y') must be present.")
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Validate center columns (>= 0)
    for col_set in [center_columns_1, center_columns_2]:
        for col in col_set:
            if col in df.columns and not df[col].apply(lambda x: isinstance(x, (int, float)) and x >= 0).all():
                msg = f"All values in column '{col}' must be non-negative numbers in {cp_output_csv}."
                if is_gui_mode:
                    show_error_message(msg)
                    return  # stop further execution
                else:
                    raise ValueError(msg)


def validate_segmentation_results(is_gui_mode, segmentation_dir, cp_output_csv):
    """Validate the segmentation results directory."""
    if not segmentation_dir:
        return  # If segmentation_results is not provided, no validation needed

    # Check if the directory exists
    if not os.path.isdir(segmentation_dir):
        msg = f"The segmentation results directory does not exist: {segmentation_dir}"
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise FileNotFoundError(msg)

    # Check if the directory contains `.npy` files
    npy_files = [f for f in os.listdir(segmentation_dir) if f.endswith('.npy')]
    if not npy_files:
        msg = f"The segmentation results directory does not contain any `.npy` files: {segmentation_dir}"
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Check the number of `.npy` files matches the highest ImageNumber in the CP output CSV
    try:
        df = pd.read_csv(cp_output_csv)
        if "ImageNumber" not in df.columns:
            msg = "The CellProfiler output CSV file must include the 'ImageNumber' column."
            if is_gui_mode:
                show_error_message(msg)
                return  # stop further execution
            else:
                raise ValueError(msg)
        max_image_number = int(df["ImageNumber"].max())
    except Exception as e:
        msg = (f"Error reading or validating 'ImageNumber' in CellProfiler output CSV: "
               f"{cp_output_csv}. Error: {e}")
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    if len(npy_files) != max_image_number:
        msg = (
            f"The number of `.npy` files in {segmentation_dir}: ({len(npy_files)}) does not match "
            f"the number of time steps in {cp_output_csv}: ({max_image_number})."
        )
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    for file in npy_files:
        arr = np.load(os.path.join(segmentation_dir, file))
        if np.all(arr == 0):
            msg = f"The segmentation file `{file}` contains only zeros (empty mask)."
            if is_gui_mode:
                show_error_message(msg)
                return  # stop further execution
            else:
                raise ValueError(msg)


def validate_neighbor_csv(is_gui_mode, neighbor_csv):
    """
    Validate the neighbor CSV file.
    """
    # Check if the file exists
    if not os.path.isfile(neighbor_csv):
        msg = f"The neighbor CSV file does not exist: {neighbor_csv}"
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise FileNotFoundError(msg)

    # Load the file as a DataFrame
    try:
        df = pd.read_csv(neighbor_csv)
    except Exception as e:
        msg = f"Unable to read the neighbor CSV file: {neighbor_csv}. Error: {e}"
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Required columns
    required_columns = [
        'Relationship',
        'First Image Number',
        'First Object Number',
        'Second Image Number',
        'Second Object Number',
    ]
    for col in required_columns:
        if col not in df.columns:
            msg = f"Missing required column in neighbor CSV: {col}"
            if is_gui_mode:
                show_error_message(msg)
                return  # stop further execution
            else:
                raise ValueError(msg)

    # Filter for 'Neighbors' relationship
    neighbors_df = df.loc[df['Relationship'] == 'Neighbors']
    if neighbors_df.shape[0] == 0:
        msg = f"The filtered dataframe for 'Neighbors' relationship in {neighbor_csv} is empty."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Validate integer and positive values for required columns
    int_positive_columns = [
        'First Image Number',
        'First Object Number',
        'Second Image Number',
        'Second Object Number',
    ]
    for col in int_positive_columns:
        if not neighbors_df[col].apply(lambda x: isinstance(x, (int, float)) and x > 0).all():
            msg = f"All values in column '{col}' must be positive integers in the neighbor CSV: {neighbor_csv}."
            if is_gui_mode:
                show_error_message(msg)
                return  # stop further execution
            else:
                raise ValueError(msg)


def validate_time_arguments(is_gui_mode, interval_time, doubling_time):
    """
    Validate interval_time and doubling_time arguments.
    """
    # Ensure both times are numbers (int or float)
    if not isinstance(interval_time, (int, float)):
        msg = f"Invalid value for --interval_time: {interval_time}. Must be an integer or float."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    if not isinstance(doubling_time, (int, float)):
        msg = f"Invalid value for --doubling_time: {doubling_time}. Must be an integer or float."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Ensure both times are positive
    if interval_time <= 0:
        msg = f"Invalid value for --interval_time: {interval_time}. Must be a positive number."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    if doubling_time <= 0:
        msg = f"Invalid value for --doubling_time: {doubling_time}. Must be a positive number."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Ensure doubling_time is greater than interval_time
    if doubling_time <= interval_time:
        msg = (
            f"Invalid value for --doubling_time: {doubling_time}. "
            f"It must be greater than --interval_time: {interval_time}."
        )
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)


def validate_elongation_rate_method(is_gui_mode, elongation_rate_method):
    """
    Validate elongation_rate_method.
    """
    valid_methods = ['Average', 'Linear Regression']
    if elongation_rate_method not in valid_methods:
        msg = f"Invalid elongation_rate_method: {elongation_rate_method}. Must be one of {valid_methods}."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)


def validate_pixel_per_micron(is_gui_mode, pixel_per_micron):
    """
    Validate pixel_per_micron.
    """

    # Ensure pixel_per_micron is float or int
    if not isinstance(pixel_per_micron, (float, int)):
        msg = (f"Invalid type for pixel_per_micron: {type(pixel_per_micron).__name__}. "
               f"Must be float or int.")
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    if pixel_per_micron < 0:
        msg = f"Invalid pixel_per_micron: {pixel_per_micron}. Must be a positive number."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)


def validate_intensity_threshold(is_gui_mode, intensity_threshold):
    """
    Validate intensity_threshold.
    """

    # Ensure intensity_threshold is float or int
    if not isinstance(intensity_threshold, (float, int)):
        msg = (f"Invalid type for intensity_threshold: {type(intensity_threshold).__name__}. "
               f"Must be float or int.")
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    if intensity_threshold < 0 or intensity_threshold > 1:
        msg = f"Invalid intensity_threshold: {intensity_threshold}. It must be a number between 0 and 1 ."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)


def validate_classifier(is_gui_mode, clf):
    """
    Validate classifier.
    """
    valid_classifiers = ['LogisticRegression', 'C-Support Vector Classifier']
    if clf not in valid_classifiers:
        msg = f"Invalid classifier: {clf}. Must be one of {valid_classifiers}."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)


def validate_num_cpus(is_gui_mode, num_cpus):
    """
    Validate num_cpus.
    """

    # Ensure num_cpus is int
    if not isinstance(num_cpus, int):
        msg = f"Invalid type for num_cpus: {type(num_cpus).__name__}. Must be int."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    if num_cpus != -1 and num_cpus < 0:
        msg = f"Invalid num_cpus: {num_cpus}. Must be -1 (all CPUs) or a positive integer."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)


def validate_boundary_limits(is_gui_mode, image_boundaries):
    """
    Validate boundary_limits.
    """
    try:
        values = list(map(int, image_boundaries.split(',')))
        if len(values) != 4:
            msg = "Boundary limits must contain exactly 4 integers."
            if is_gui_mode:
                show_error_message(msg)
                return  # stop further execution
            else:
                raise ValueError(msg)
        if not all(v > 0 for v in values):
            msg = "All boundary limits must be positive integers."
            if is_gui_mode:
                show_error_message(msg)
                return  # stop further execution
            else:
                raise ValueError(msg)
        if not (values[0] < values[1] and values[2] < values[3]):
            msg = "Boundary limits must specify valid ranges: lower_x < upper_x and lower_y < upper_y."
            if is_gui_mode:
                show_error_message(msg)
                return  # stop further execution
            else:
                raise ValueError(msg)
    except ValueError:
        msg = "Boundary limits must be a comma-separated list of 4 integers."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)


def validate_dynamic_boundaries(is_gui_mode, dynamic_boundaries):
    """
    Validate dynamic_boundaries.
    """
    # Check if the file exists
    if not os.path.isfile(dynamic_boundaries):
        msg = f"The dynamic boundaries file does not exist: {dynamic_boundaries}"
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise FileNotFoundError(msg)

    # Load the file as a DataFrame
    try:
        df = pd.read_csv(dynamic_boundaries)
    except Exception as e:
        msg = f"Unable to read the dynamic boundaries CSV file: {dynamic_boundaries}. Error: {e}"
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Required columns
    required_columns = ['Time Step', 'Lower X Limit', 'Upper X Limit', 'Lower Y Limit', 'Upper Y Limit']
    for col in required_columns:
        if col not in df.columns:
            msg = f"Missing required column in dynamic boundaries CSV: {col}"
            if is_gui_mode:
                show_error_message(msg)
                return  # stop further execution
            else:
                raise ValueError(msg)

    # Validate column values
    if not all(df[col].apply(lambda x: isinstance(x, int) and x >= 0).all() for col in required_columns):
        msg = "All values in dynamic boundaries CSV must be positive integers."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)

    # Validate boundary ranges
    if not all(df['Upper X Limit'] > df['Lower X Limit']):
        msg = "All 'Upper X Limit' values must be greater than 'Lower X Limit'."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)
    if not all(df['Upper Y Limit'] > df['Lower Y Limit']):
        msg = "All 'Upper Y Limit' values must be greater than 'Lower Y Limit'."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)


def check_validity(is_gui_mode, cp_output_csv, segmentation_res_dir, neighbor_csv, interval_time, doubling_time,
                   elongation_rate_method, pixel_per_micron, assigning_cell_type, intensity_threshold,
                   disable_tracking_correction, clf, n_cpu, image_boundaries, dynamic_boundaries):

    validate_cp_output_csv(is_gui_mode, cp_output_csv)

    if not disable_tracking_correction:
        validate_segmentation_results(is_gui_mode, segmentation_res_dir, cp_output_csv)

    validate_neighbor_csv(is_gui_mode, neighbor_csv)
    validate_time_arguments(is_gui_mode, interval_time, doubling_time)
    validate_elongation_rate_method(is_gui_mode, elongation_rate_method)
    validate_pixel_per_micron(is_gui_mode, pixel_per_micron)
    if assigning_cell_type:
        validate_intensity_threshold(is_gui_mode, intensity_threshold)
    validate_classifier(is_gui_mode, clf)
    validate_num_cpus(is_gui_mode, n_cpu)

    if image_boundaries is not None:
        validate_boundary_limits(is_gui_mode, image_boundaries)

    if dynamic_boundaries is not None:
        validate_dynamic_boundaries(is_gui_mode, dynamic_boundaries)

    if image_boundaries is not None and dynamic_boundaries is not None:
        msg = "You cannot specify both --boundary_limits and --dynamic_boundaries at the same time."
        if is_gui_mode:
            show_error_message(msg)
            return  # stop further execution
        else:
            raise ValueError(msg)
