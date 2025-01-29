import numpy as np


def mark_cell_types_by_intensity(df, cell_type_array, cols, intensity_threshold):

    """
    Marks cell types as present (1) or absent (0) based on intensity threshold values.

    :param pandas.DataFrame df:
        Input DataFrame containing intensity data for each cell type across multiple columns.
    :param numpy.ndarray cell_type_array:
        A 2D numpy array where each row represents a bacterial cell, and each column represents
        a cell type. This array will be updated with binary values (0 or 1) based on the intensity threshold.
    :param list cols:
        List of column names in `df` corresponding to cell types whose intensities need to be checked.
    :param float intensity_threshold:
        Threshold value above which the cell type is considered present (marked as 1).

    :returns:
        numpy.ndarray:

        Updated 2D numpy array with binary marks indicating cell type presence for each bacterial cell.
    """

    for col_idx, col in enumerate(cols):
        condition = df[col] > intensity_threshold
        cell_type_array[condition, col_idx] = 1

    return cell_type_array


def get_fluorescence_intensity_columns(dataframe_col):

    """
    Retrieves a sorted list of fluorescence intensity column names from the provided DataFrame column names.

    :param pandas.Series dataframe_col:
        A Pandas Series containing column names from a DataFrame.

    :returns:
        list:

        A sorted list of column names that contain the substring `Intensity_MeanIntensity_`.
    """

    # fluorescence intensity columns
    fluorescence_intensities_col = \
        sorted(dataframe_col[dataframe_col.str.contains('Intensity_MeanIntensity_')].values.tolist())

    return fluorescence_intensities_col


def assign_cell_types(dataframe, intensity_threshold):
    """
    Assigns binary cell type labels based on fluorescence intensity thresholds.

    :param pandas.DataFrame dataframe:
        Input DataFrame containing intensity data and other features.
    :param float intensity_threshold:
        Threshold value above which a cell type is marked as present (1).

    :returns:
        numpy.ndarray:

        A 2D numpy array where each row represents a cell, and each column represents a cell type, marked as 1 or 0
        based on intensity threshold.
    """

    # If the CSV file has two mean intensity columns, the cosine similarity is calculated
    intensity_col_names = get_fluorescence_intensity_columns(dataframe.columns)

    # cell type
    if len(intensity_col_names) >= 1:

        cell_type_array = np.zeros((dataframe.shape[0], len(intensity_col_names)))
        # check  fluorescence intensity
        cell_type_array = mark_cell_types_by_intensity(dataframe, cell_type_array, intensity_col_names,
                                                       intensity_threshold)
    else:
        cell_type_array = np.zeros((dataframe.shape[0], 1))

    return cell_type_array


def determine_final_cell_type(dataframe, cell_type_array):

    """
    Determines and assigns the final cell type classification for each bacterium based on fluorescence intensity data.

    :param pandas.DataFrame dataframe:
        Input DataFrame containing bacterial data and fluorescence intensity columns.
    :param numpy.ndarray cell_type_array:
        A 2D numpy array where each row represents a bacterial cell, and each column represents
        a binary classification (1 or 0) of a specific cell type based on intensity thresholds.

    :returns:
        pandas.DataFrame:
        Updated DataFrame with a new column 'cellType' representing the final cell type
        classification for each bacterium:

            - 3: Assigned if the fluorescence intensities of at least two channels exceed the defined threshold.
            - Specific values (e.g., 2, 3, etc.): Assigned if only one channel’s intensity exceeds the threshold.
            - 0: Assigned if none of the fluorescence intensities across all channels exceed the defined threshold.
            - 1: Default cell type assigned when only one intensity column exists.
    """

    num_intensity_cols = len(get_fluorescence_intensity_columns(dataframe.columns))

    if num_intensity_cols > 1:

        dataframe['cellType'] = 0

        num_1_value_per_bac = np.sum(cell_type_array, axis=1)
        num_0_value_per_bac = cell_type_array.shape[1] - num_1_value_per_bac

        cond1_more_than_2_1_value = num_1_value_per_bac >= 2
        cond2_more_than_2_0_value = num_0_value_per_bac >= 2

        cond3 = ~ cond1_more_than_2_1_value & ~ cond2_more_than_2_0_value

        indices_of_ones_in_columns = np.where(cell_type_array[cond3] == 1)[0]
        cond3_value = indices_of_ones_in_columns + 1

        dataframe.loc[cond1_more_than_2_1_value, 'cellType'] = 3
        dataframe.loc[cond3, 'cellType'] = cond3_value

    else:
        dataframe['cellType'] = 1

    return dataframe
