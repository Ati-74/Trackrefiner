import os
import pandas as pd


def validate_cp_output_csv(cp_output_csv):
    """
    Validate the CP output CSV file.
    """
    # Check if the file exists
    if not os.path.isfile(cp_output_csv):
        raise FileNotFoundError(f"The CP output CSV file does not exist: {cp_output_csv}")

    # Load the file as a DataFrame
    try:
        df = pd.read_csv(cp_output_csv)
    except Exception as e:
        raise ValueError(f"Unable to read the CP output CSV file: {cp_output_csv}. Error: {e}")

    # Check if the DataFrame is empty
    if df.shape[0] == 0:
        raise ValueError(f"The {cp_output_csv} is empty.")

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
            raise ValueError(f"Missing required column in CP output CSV: {col}")

    # Validate ImageNumber and ObjectNumber (int and > 0)
    for col in ["ImageNumber", "ObjectNumber"]:
        if not df[col].apply(lambda x: isinstance(x, (int, float)) and x > 0).all():
            raise ValueError(f"All values in column '{col}' must be positive integers in {cp_output_csv}.")

    # Validate AreaShape_MajorAxisLength and AreaShape_MinorAxisLength (positive > 0)
    for col in ["AreaShape_MajorAxisLength", "AreaShape_MinorAxisLength"]:
        if not df[col].apply(lambda x: isinstance(x, (int, float)) and x > 0).all():
            raise ValueError(f"All values in column '{col}' must be positive numbers in {cp_output_csv}.")

    # Validate AreaShape_Orientation (must be a number)
    if not df["AreaShape_Orientation"].apply(lambda x: isinstance(x, (int, float))).all():
        raise ValueError(f"All values in column 'AreaShape_Orientation' must be numbers in {cp_output_csv}.")

    # Check for parent columns
    parent_image_number_col = [col for col in df.columns if "TrackObjects_ParentImageNumber_" in col]
    if not parent_image_number_col:
        raise ValueError("Missing required 'TrackObjects_ParentImageNumber_' column in CP output CSV.")

    parent_object_number_col = [col for col in df.columns if "TrackObjects_ParentObjectNumber_" in col]
    if not parent_object_number_col:
        raise ValueError("Missing required 'TrackObjects_ParentObjectNumber_' column in CP output CSV.")

    # Validate parent columns (int and > 0)
    for col in parent_image_number_col + parent_object_number_col:
        if not df[col].apply(lambda x: isinstance(x, (int, float)) and x >= 0).all():
            raise ValueError(f"All values in column '{col}' must be positive integers in {cp_output_csv}.")

    # Check for label columns
    label_col_list = [col for col in df.columns if "TrackObjects_Label_" in col]
    if not label_col_list:
        raise ValueError("Missing required 'TrackObjects_Label_' column(s) in CP output CSV.")

    # Check for center columns
    center_columns_1 = {"Location_Center_X", "Location_Center_Y"}
    center_columns_2 = {"AreaShape_Center_X", "AreaShape_Center_Y"}
    if not (center_columns_1.issubset(df.columns) or center_columns_2.issubset(df.columns)):
        raise ValueError(
            "Missing required center columns in CP output CSV. "
            "Either ('Location_Center_X', 'Location_Center_Y') or "
            "('AreaShape_Center_X', 'AreaShape_Center_Y') must be present."
        )

    # Validate center columns (>= 0)
    for col_set in [center_columns_1, center_columns_2]:
        for col in col_set:
            if col in df.columns and not df[col].apply(lambda x: isinstance(x, (int, float)) and x >= 0).all():
                raise ValueError(f"All values in column '{col}' must be non-negative numbers in {cp_output_csv}.")


def validate_segmentation_results(segmentation_dir, cp_output_csv):
    """Validate the segmentation results directory."""
    if not segmentation_dir:
        return  # If segmentation_results is not provided, no validation needed

    # Check if the directory exists
    if not os.path.isdir(segmentation_dir):
        raise FileNotFoundError(f"The segmentation results directory does not exist: {segmentation_dir}")

    # Check if the directory contains `.npy` files
    npy_files = [f for f in os.listdir(segmentation_dir) if f.endswith('.npy')]
    if not npy_files:
        raise ValueError(f"The segmentation results directory does not contain any `.npy` files: {segmentation_dir}")

    # Check the number of `.npy` files matches the highest ImageNumber in the CP output CSV
    try:
        df = pd.read_csv(cp_output_csv)
        if "ImageNumber" not in df.columns:
            raise ValueError("The CP output CSV file must include the 'ImageNumber' column.")
        max_image_number = int(df["ImageNumber"].max())
    except Exception as e:
        raise ValueError(f"Error reading or validating 'ImageNumber' in CP output CSV: {cp_output_csv}. Error: {e}")

    if len(npy_files) != max_image_number:
        raise ValueError(
            f"The number of `.npy` files in {segmentation_dir}: ({len(npy_files)}) does not match "
            f"the number of time steps in {cp_output_csv}: ({max_image_number})."
        )


def validate_neighbor_csv(neighbor_csv):
    """
    Validate the neighbor CSV file.
    """
    # Check if the file exists
    if not os.path.isfile(neighbor_csv):
        raise FileNotFoundError(f"The neighbor CSV file does not exist: {neighbor_csv}")

    # Load the file as a DataFrame
    try:
        df = pd.read_csv(neighbor_csv)
    except Exception as e:
        raise ValueError(f"Unable to read the neighbor CSV file: {neighbor_csv}. Error: {e}")

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
            raise ValueError(f"Missing required column in neighbor CSV: {col}")

    # Filter for 'Neighbors' relationship
    neighbors_df = df.loc[df['Relationship'] == 'Neighbors']
    if neighbors_df.shape[0] == 0:
        raise ValueError(f"The filtered dataframe for 'Neighbors' relationship in {neighbor_csv} is empty.")

    # Validate integer and positive values for required columns
    int_positive_columns = [
        'First Image Number',
        'First Object Number',
        'Second Image Number',
        'Second Object Number',
    ]
    for col in int_positive_columns:
        if not neighbors_df[col].apply(lambda x: isinstance(x, (int, float)) and x > 0).all():
            raise ValueError(
                f"All values in column '{col}' must be positive integers in the neighbor CSV: {neighbor_csv}."
            )


def validate_time_arguments(interval_time, doubling_time):
    """
    Validate interval_time and doubling_time arguments.
    """
    # Ensure both times are numbers (int or float)
    if not isinstance(interval_time, (int, float)):
        raise ValueError(f"Invalid value for --interval_time: {interval_time}. Must be an integer or float.")

    if not isinstance(doubling_time, (int, float)):
        raise ValueError(f"Invalid value for --doubling_time: {doubling_time}. Must be an integer or float.")

    # Ensure both times are positive
    if interval_time <= 0:
        raise ValueError(f"Invalid value for --interval_time: {interval_time}. Must be a positive number.")

    if doubling_time <= 0:
        raise ValueError(f"Invalid value for --doubling_time: {doubling_time}. Must be a positive number.")

    # Ensure doubling_time is greater than interval_time
    if doubling_time <= interval_time:
        raise ValueError(
            f"Invalid value for --doubling_time: {doubling_time}. "
            f"It must be greater than --interval_time: {interval_time}."
        )


def validate_elongation_rate_method(elongation_rate_method):
    """
    Validate elongation_rate_method.
    """
    valid_methods = ['Average', 'Linear Regression']
    if elongation_rate_method not in valid_methods:
        raise ValueError(f"Invalid elongation_rate_method: {elongation_rate_method}. Must be one of {valid_methods}.")


def validate_pixel_per_micron(pixel_per_micron):
    """
    Validate pixel_per_micron.
    """

    # Ensure pixel_per_micron is float or int
    if not isinstance(pixel_per_micron, (float, int)):
        raise ValueError(f"Invalid type for pixel_per_micron: {type(pixel_per_micron).__name__}. "
                         f"Must be float or int.")

    if pixel_per_micron < 0:
        raise ValueError(f"Invalid pixel_per_micron: {pixel_per_micron}. Must be a positive number.")


def validate_intensity_threshold(intensity_threshold):
    """
    Validate intensity_threshold.
    """

    # Ensure intensity_threshold is float or int
    if not isinstance(intensity_threshold, (float, int)):
        raise ValueError(f"Invalid type for intensity_threshold: {type(intensity_threshold).__name__}. "
                         f"Must be float or int.")

    if intensity_threshold < 0:
        raise ValueError(f"Invalid intensity_threshold: {intensity_threshold}. Must be a non-negative number.")


def validate_classifier(clf):
    """
    Validate classifier.
    """
    valid_classifiers = ['LogisticRegression', 'GaussianProcessClassifier', 'C-Support Vector Classifier']
    if clf not in valid_classifiers:
        raise ValueError(f"Invalid classifier: {clf}. Must be one of {valid_classifiers}.")


def validate_num_cpus(num_cpus):
    """
    Validate num_cpus.
    """

    # Ensure num_cpus is int
    if not isinstance(num_cpus, int):
        raise ValueError(f"Invalid type for num_cpus: {type(num_cpus).__name__}. Must be int.")

    if num_cpus != -1 and num_cpus < 0:
        raise ValueError(f"Invalid num_cpus: {num_cpus}. Must be -1 (all CPUs) or a positive integer.")


def validate_boundary_limits(image_boundaries):
    """
    Validate boundary_limits.
    """
    try:
        values = list(map(int, image_boundaries.split(',')))
        if len(values) != 4:
            raise ValueError("Boundary limits must contain exactly 4 integers.")
        if not all(v > 0 for v in values):
            raise ValueError("All boundary limits must be positive integers.")
        if not (values[0] < values[1] and values[2] < values[3]):
            raise ValueError("Boundary limits must specify valid ranges: lower_x < upper_x and lower_y < upper_y.")
    except ValueError:
        raise ValueError("Boundary limits must be a comma-separated list of 4 integers.")


def validate_dynamic_boundaries(dynamic_boundaries):
    """
    Validate dynamic_boundaries.
    """
    # Check if the file exists
    if not os.path.isfile(dynamic_boundaries):
        raise FileNotFoundError(f"The dynamic boundaries file does not exist: {dynamic_boundaries}")

    # Load the file as a DataFrame
    try:
        df = pd.read_csv(dynamic_boundaries)
    except Exception as e:
        raise ValueError(f"Unable to read the dynamic boundaries CSV file: {dynamic_boundaries}. Error: {e}")

    # Required columns
    required_columns = ['Time Step', 'Lower X Limit', 'Upper X Limit', 'Lower Y Limit', 'Upper Y Limit']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dynamic boundaries CSV: {col}")

    # Validate column values
    if not all(df[col].apply(lambda x: isinstance(x, int) and x >= 0).all() for col in required_columns):
        raise ValueError("All values in dynamic boundaries CSV must be positive integers.")

    # Validate boundary ranges
    if not all(df['Upper X Limit'] > df['Lower X Limit']):
        raise ValueError("All 'Upper X Limit' values must be greater than 'Lower X Limit'.")
    if not all(df['Upper Y Limit'] > df['Lower Y Limit']):
        raise ValueError("All 'Upper Y Limit' values must be greater than 'Lower Y Limit'.")


def check_validity(cp_output_csv, segmentation_res_dir, neighbor_csv, interval_time, doubling_time,
                   elongation_rate_method, pixel_per_micron, intensity_threshold,
                   disable_tracking_correction, clf, n_cpu, image_boundaries, dynamic_boundaries):

    validate_cp_output_csv(cp_output_csv)

    if not disable_tracking_correction:
        validate_segmentation_results(segmentation_res_dir, cp_output_csv)

    validate_neighbor_csv(neighbor_csv)
    validate_time_arguments(interval_time, doubling_time)
    validate_elongation_rate_method(elongation_rate_method)
    validate_pixel_per_micron(pixel_per_micron)
    validate_intensity_threshold(intensity_threshold)
    validate_classifier(clf)
    validate_num_cpus(n_cpu)

    if image_boundaries is not None:
        validate_boundary_limits(image_boundaries)

    if dynamic_boundaries is not None:
        validate_dynamic_boundaries(dynamic_boundaries)

    if image_boundaries is not None and dynamic_boundaries is not None:
        raise ValueError("You cannot specify both --boundary_limits and --dynamic_boundaries at the same time.")
