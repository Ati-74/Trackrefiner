import numpy as np


def find_bacterial_length_outliers(values):
    """
    Identifies bacterial length outliers

    :param numpy.ndarray values:
        Array of bacterial length values to analyze.

    :returns:
        float: The maximum outlier value below the calculated lower bound, if any exist.
        Otherwise, returns NaN.
    """

    avg_val = np.average(values)
    std_val = np.std(values)

    lower_bound_val = calculate_lower_statistical_bound({'avg': avg_val, 'std': std_val})

    outlier_list = values[(values < lower_bound_val) & (values < 1)]

    if len(outlier_list) > 0:
        max_outlier_val = max(outlier_list)
    else:
        max_outlier_val = np.nan

    return max_outlier_val


def detect_length_change_ratio_outliers(df):
    """
    Detects bacterial length change ratio outliers

    :param pandas.DataFrame df:
        DataFrame containing the column "Length_Change_Ratio" for analysis.

    :returns:
        float: The maximum outlier value for bacterial length change ratios, or NaN if none exist.
    """

    bac_change_length_ratio_list = df["Length_Change_Ratio"].dropna().values

    if len(bac_change_length_ratio_list) > 0:

        max_bac_change_length_ratio_list_outliers = \
            find_bacterial_length_outliers(bac_change_length_ratio_list)
    else:
        max_bac_change_length_ratio_list_outliers = np.nan

    return max_bac_change_length_ratio_list_outliers


def detect_daughter_to_mother_length_outliers(df):
    """
    Detects outliers in the daughter-to-mother length ratios.

    :param pandas.DataFrame df:
        DataFrame containing "Total_Daughter_Mother_Length_Ratio" and "Max_Daughter_Mother_Length_Ratio" columns.

    :returns:
        pandas.DataFrame: A subset of the input DataFrame containing rows that exceed
        the calculated upper bounds for daughter-to-mother ratios or have ratios > 1.

    """

    sum_daughter_length_ratio_list = df["Total_Daughter_Mother_Length_Ratio"].dropna().values
    max_daughter_length_ratio_list = df["Max_Daughter_Mother_Length_Ratio"].dropna().values

    avg_sum_daughter_length_ratio = np.average(sum_daughter_length_ratio_list)
    std_sum_daughter_length_ratio = np.std(sum_daughter_length_ratio_list)

    avg_max_daughter_length_ratio = np.average(max_daughter_length_ratio_list)
    std_max_daughter_length_ratio = np.std(max_daughter_length_ratio_list)

    upper_bound_sum_daughter = calculate_upper_statistical_bound({'avg': avg_sum_daughter_length_ratio,
                                                                  'std': std_sum_daughter_length_ratio})
    upper_bound_max_daughter = calculate_upper_statistical_bound({'avg': avg_max_daughter_length_ratio,
                                                                  'std': std_max_daughter_length_ratio})

    # Apply the conditions using vectorized operations
    condition = (
            (df["Total_Daughter_Mother_Length_Ratio"] > upper_bound_sum_daughter) |
            (df["Max_Daughter_Mother_Length_Ratio"] > upper_bound_max_daughter) |
            (df["Max_Daughter_Mother_Length_Ratio"] > 1)
    )

    df_outliers = df.loc[condition]

    return df_outliers


def calculate_max_daughter_to_mother_boundary(df):
    """
    Calculates the statistical boundary (average and standard deviation) for the maximum
    daughter-to-mother length ratio.

    :param pandas.DataFrame df:
        DataFrame containing the column "Max_Daughter_Mother_Length_Ratio".

    :returns:
        dict: A dictionary with 'avg' (average) and 'std' (standard deviation) for the ratio.

    """

    max_daughter_length_ratio_list = df["Max_Daughter_Mother_Length_Ratio"].dropna().values
    avg_val = np.average(max_daughter_length_ratio_list)
    std_val = np.std(max_daughter_length_ratio_list)

    return {'avg': avg_val, 'std': std_val}


def calculate_sum_daughter_to_mother_len_boundary(df):
    """
    Calculates the statistical boundary (average and standard deviation) for the
    sum_daughter-to-mother length ratio.

    :param pandas.DataFrame df:
        DataFrame containing the column "Total_Daughter_Mother_Length_Ratio".

    :returns:
        dict: A dictionary with 'avg' (average) and 'std' (standard deviation) for the ratio.
    """

    sum_daughters_length_ratio_list = df["Total_Daughter_Mother_Length_Ratio"].dropna().values
    avg_val = np.average(sum_daughters_length_ratio_list)
    std_val = np.std(sum_daughters_length_ratio_list)

    return {'avg': avg_val, 'std': std_val}


def calculate_length_change_ratio_boundary(df):
    """
    Calculates the statistical boundary for bacterial length change ratios below 1.

    :param pandas.DataFrame df:
        DataFrame containing the column "Length_Change_Ratio".

    :returns:
        dict: A dictionary with 'avg' (average) and 'std' (standard deviation) for the ratios below 1.
    """

    bac_change_length_ratio_list = \
        df["Length_Change_Ratio"].dropna().values[df["Length_Change_Ratio"].dropna().values < 1]

    if len(bac_change_length_ratio_list) > 0:
        avg_val = np.average(bac_change_length_ratio_list)
        std_val = np.std(bac_change_length_ratio_list)
    else:
        avg_val = 0
        std_val = 0

    return {'avg': avg_val, 'std': std_val}


def calculate_bacterial_length_boundary(df):
    """
    Calculates the statistical boundary (average and standard deviation) for bacterial length values.

    :param pandas.DataFrame df:
        DataFrame containing the column "AreaShape_MajorAxisLength".

    :returns:
        dict: A dictionary with 'avg' (average) and 'std' (standard deviation) for bacterial lengths.
    """

    df = df.loc[df['noise_bac'] == False]

    bacteria_length = df['AreaShape_MajorAxisLength'].dropna().values

    avg_val = np.average(bacteria_length)
    std_val = np.std(bacteria_length)

    return {'avg': avg_val, 'std': std_val}


def calculate_upper_statistical_bound(feature_boundary_dict):

    """
    Calculates the upper statistical boundary using the formula: avg + 1.96 * std.

    :param dict feature_boundary_dict:
        Dictionary containing 'avg' (average) and 'std' (standard deviation).

    :returns:
        float: The upper statistical boundary.
    """

    return feature_boundary_dict['avg'] + 1.96 * feature_boundary_dict['std']


def calculate_lower_statistical_bound(feature_boundary_dict):
    """
    Calculates the lower statistical boundary using the formula: avg - 1.96 * std.

    :param dict feature_boundary_dict:
        Dictionary containing 'avg' (average) and 'std' (standard deviation).

    :returns:
        float: The lower statistical boundary.
    """

    return feature_boundary_dict['avg'] - 1.96 * feature_boundary_dict['std']
