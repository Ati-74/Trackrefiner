import numpy as np
from sklearn.ensemble import IsolationForest


def find_outlier_unsupervised(values):

    data = np.array(values).reshape(-1, 1)

    # Initialize the Isolation Forest model
    iso_forest = IsolationForest(contamination=0.1)

    # Fit the model to the data
    iso_forest.fit(data)

    # Predicting if each data point is an outlier
    predictions = iso_forest.predict(data)

    # Extracting the outliers
    outliers = data[predictions == -1]

    # Convert the outliers to a list for easier interpretation
    outlier_list = outliers.flatten().tolist()

    return outlier_list


def find_outlier_traditional_bac_change_length(values):

    avg_val = np.average(values)
    std_val = np.std(values)

    outlier_list = [v for v in values if v < avg_val - std_val]

    return outlier_list


def find_outlier_traditional_sum_daughter_len_to_mother_ratio(values, higher_than_one_is_outlier=False):

    avg_val = np.average(values)
    std_val = np.std(values)

    if not higher_than_one_is_outlier:
        outlier_list = [v for v in values if v > avg_val + std_val]
    else:
        outlier_list = [v for v in values if v > avg_val + std_val or v >= 1]

    if outlier_list is None:
        outlier_list = [None]

    return outlier_list


def find_bac_change_length_ratio_outliers(df):

    bac_change_length_ratio_list = [v for v in df["bac_length_to_back"].dropna().values.tolist() if v != '' and v < 1]
    bac_change_length_ratio_list_outliers = find_outlier_traditional_bac_change_length(bac_change_length_ratio_list)

    return bac_change_length_ratio_list_outliers


def find_daughter_len_to_mother_ratio_outliers(df):

    sum_daughter_length_ratio_list = [v for v in df["daughter_length_to_mother"].dropna().values.tolist() if v != '']
    sum_daughter_length_ratio_list_outliers = \
        find_outlier_traditional_sum_daughter_len_to_mother_ratio(sum_daughter_length_ratio_list)

    max_daughter_length_ratio_list = [v for v in df["max_daughter_len_to_mother"].dropna().values.tolist() if v != '']
    max_daughter_length_ratio_list_outliers = \
        find_outlier_traditional_sum_daughter_len_to_mother_ratio(max_daughter_length_ratio_list,
                                                                  higher_than_one_is_outlier=True)

    daughters_outlier = {'daughter_length_to_mother': sum_daughter_length_ratio_list_outliers,
                         "max_daughter_len_to_mother": max_daughter_length_ratio_list_outliers}

    return daughters_outlier


def find_max_daughter_len_to_mother_ratio_boundary(df):

    max_daughter_length_ratio_list = [v for v in df["max_daughter_len_to_mother"].dropna().values.tolist() if v != '']
    avg_val = np.average(max_daughter_length_ratio_list)
    std_val = np.std(max_daughter_length_ratio_list)

    return {'avg': avg_val, 'std': std_val}


def find_sum_daughter_len_to_mother_ratio_boundary(df):
    sum_daughters_length_ratio_list = [v for v in df["daughter_length_to_mother"].dropna().values.tolist() if v != '']
    avg_val = np.average(sum_daughters_length_ratio_list)
    std_val = np.std(sum_daughters_length_ratio_list)

    return {'avg': avg_val, 'std': std_val}


def find_bac_len_to_bac_ratio_boundary(df):
    bac_change_length_ratio_list = [v for v in df["bac_length_to_back"].dropna().values.tolist() if v != '' and v < 1]
    avg_val = np.average(bac_change_length_ratio_list)
    std_val = np.std(bac_change_length_ratio_list)

    return {'avg': avg_val, 'std': std_val}


def find_bac_movement_boundary(df):

    bacteria_movement = [v for v in df['bacteria_movement'].dropna().values.tolist() if v != '']

    avg_val = np.average(bacteria_movement)
    std_val = np.std(bacteria_movement)
    max_val = np.max(bacteria_movement)

    return {'avg': avg_val, 'std': std_val, 'max': max_val}


def find_bac_len_boundary(df):

    bacteria_length = [v for v in df['AreaShape_MajorAxisLength'].dropna().values.tolist() if v != '']

    avg_val = np.average(bacteria_length)
    std_val = np.std(bacteria_length)

    return {'avg': avg_val, 'std': std_val}
