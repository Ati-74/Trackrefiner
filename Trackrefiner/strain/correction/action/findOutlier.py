import numpy as np


def find_outlier_traditional_bac_change_length(values):

    avg_val = np.average(values)
    std_val = np.std(values)

    lower_bound_val = find_lower_bound({'avg': avg_val, 'std': std_val})

    outlier_list = values[(values < lower_bound_val) & (values < 1)]

    if len(outlier_list) > 0:
        max_outlier_val = max(outlier_list)
    else:
        max_outlier_val = np.nan

    return max_outlier_val


def find_outlier_traditional_sum_daughter_len_to_mother_ratio(values, higher_than_one_is_outlier=False):

    avg_val = np.average(values)
    std_val = np.std(values)

    upper_bound_val = find_upper_bound({'avg': avg_val, 'std': std_val})

    if not higher_than_one_is_outlier:
        outlier_list = values[values > upper_bound_val]
    else:
        outlier_list = values[(values > upper_bound_val) | (values >= 1)]

    if len(outlier_list) == 0:
        outlier_list = [None]

    return outlier_list


def find_final_bac_change_length_ratio_outliers(df):

    bac_change_length_ratio_list = df["LengthChangeRatio"].dropna().values
    max_bac_change_length_ratio_list_outliers = \
        find_outlier_traditional_bac_change_length(bac_change_length_ratio_list)

    return max_bac_change_length_ratio_list_outliers


def find_bac_change_length_ratio_outliers(df):

    bac_change_length_ratio_list = df["LengthChangeRatio"].dropna().values

    if len(bac_change_length_ratio_list) > 0:

        max_bac_change_length_ratio_list_outliers = \
            find_outlier_traditional_bac_change_length(bac_change_length_ratio_list)
    else:
        max_bac_change_length_ratio_list_outliers = np.nan

    return max_bac_change_length_ratio_list_outliers


def find_daughter_len_to_mother_ratio_outliers(df):

    sum_daughter_length_ratio_list = df["daughter_length_to_mother"].dropna().values
    max_daughter_length_ratio_list = df["max_daughter_len_to_mother"].dropna().values

    avg_sum_daughter_length_ratio = np.average(sum_daughter_length_ratio_list)
    std_sum_daughter_length_ratio = np.std(sum_daughter_length_ratio_list)

    avg_max_daughter_length_ratio = np.average(max_daughter_length_ratio_list)
    std_max_daughter_length_ratio = np.std(max_daughter_length_ratio_list)

    upper_bound_sum_daughter = find_upper_bound({'avg': avg_sum_daughter_length_ratio,
                                                 'std': std_sum_daughter_length_ratio})
    upper_bound_max_daughter = find_upper_bound({'avg': avg_max_daughter_length_ratio,
                                                 'std': std_max_daughter_length_ratio})

    # Apply the conditions using vectorized operations
    condition = (
            (df["daughter_length_to_mother"] > upper_bound_sum_daughter) |
            (df["max_daughter_len_to_mother"] > upper_bound_max_daughter) |
            (df["max_daughter_len_to_mother"] > 1)
    )

    df_outliers = df.loc[condition]

    return df_outliers


def find_max_daughter_len_to_mother_ratio_boundary(df):

    max_daughter_length_ratio_list = df["max_daughter_len_to_mother"].dropna().values
    avg_val = np.average(max_daughter_length_ratio_list)
    std_val = np.std(max_daughter_length_ratio_list)

    return {'avg': avg_val, 'std': std_val}


def find_sum_daughter_len_to_mother_ratio_boundary(df):

    sum_daughters_length_ratio_list = df["daughter_length_to_mother"].dropna().values
    avg_val = np.average(sum_daughters_length_ratio_list)
    std_val = np.std(sum_daughters_length_ratio_list)

    return {'avg': avg_val, 'std': std_val}


def find_bac_len_to_bac_ratio_boundary(df):

    bac_change_length_ratio_list = \
        df["LengthChangeRatio"].dropna().values[df["LengthChangeRatio"].dropna().values < 1]

    if len(bac_change_length_ratio_list) > 0:
        avg_val = np.average(bac_change_length_ratio_list)
        std_val = np.std(bac_change_length_ratio_list)
    else:
        avg_val = 0
        std_val = 0

    return {'avg': avg_val, 'std': std_val}


def find_bac_len_boundary(df):

    df = df.loc[df['noise_bac'] == False]

    bacteria_length = df['AreaShape_MajorAxisLength'].dropna().values

    avg_val = np.average(bacteria_length)
    std_val = np.std(bacteria_length)

    return {'avg': avg_val, 'std': std_val}


def find_upper_bound(feature_boundary_dict):

    return feature_boundary_dict['avg'] + 1.96 * feature_boundary_dict['std']


def find_lower_bound(feature_boundary_dict):

    return feature_boundary_dict['avg'] - 1.96 * feature_boundary_dict['std']
