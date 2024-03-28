import numpy as np
from Trackrefiner.strain.correction.action.findOutlier import  find_daughter_len_to_mother_ratio_outliers


def does_it_cause_error(df, bac1, bac2):

    # I want to check if linking bac2 to bac1 is causing an error.

    bac1_life_history_after_bac2 = df.loc[(df['ImageNumber'] >= bac2["ImageNumber"]) & (df['id'] == bac1['id'])]

    if bac1_life_history_after_bac2.shape[0] > 0:
        # it means that the relation of bac1 & bac2 is mother - daughter

        daughter_length_to_mother_under_stress = [v for v in df["daughter_length_to_mother"].dropna().values.tolist()
                                                  if v != '']
        max_daughter_length_to_mother_under_stress = \
            [v for v in df["max_daughter_len_to_mother"].dropna().values.tolist() if v != '']

        avg_sum_daughters_length_to_mother_ratio = np.mean(daughter_length_to_mother_under_stress)
        std_sum_daughters_length_to_mother_ratio = np.std(daughter_length_to_mother_under_stress)

        avg_max_daughters_length_to_mother_ratio = np.mean(max_daughter_length_to_mother_under_stress)
        std_max_daughters_length_to_mother_ratio = np.std(max_daughter_length_to_mother_under_stress)

        this_bacterium_sum_daughters_length_to_mother_ratio = \
            (bac1_life_history_after_bac2.iloc[0]['AreaShape_MajorAxisLength'] + bac2['AreaShape_MajorAxisLength']) / \
            bac1['AreaShape_MajorAxisLength']

        daughter1_length_to_mother_ratio = \
            bac1_life_history_after_bac2.iloc[0]['AreaShape_MajorAxisLength'] / bac1['AreaShape_MajorAxisLength']

        daughter2_length_to_mother_ratio = bac2['AreaShape_MajorAxisLength'] / bac1['AreaShape_MajorAxisLength']

        if this_bacterium_sum_daughters_length_to_mother_ratio <= \
                avg_sum_daughters_length_to_mother_ratio + 1.96 * std_sum_daughters_length_to_mother_ratio and \
                daughter1_length_to_mother_ratio >= 1 and daughter2_length_to_mother_ratio >= 1 and \
                daughter1_length_to_mother_ratio <= \
                avg_max_daughters_length_to_mother_ratio + 1.96 * std_max_daughters_length_to_mother_ratio and \
                daughter2_length_to_mother_ratio <= \
                avg_max_daughters_length_to_mother_ratio + 1.96 * std_max_daughters_length_to_mother_ratio:
            bad_relation_flag = False
        else:
            bad_relation_flag = True

    else:
        # it means that bac2 is same as bac1
        bac_length_to_back_under_stress = [v for v in df["bac_length_to_back"].dropna().values.tolist()
                                           if v != '' and v < 1]
        avg_bac_length_to_back_ratio = np.mean(bac_length_to_back_under_stress)
        std_bac_length_to_back_ratio = np.std(bac_length_to_back_under_stress)

        this_bacterium_bac_length_to_back = bac2['AreaShape_MajorAxisLength'] / bac1['AreaShape_MajorAxisLength']

        if avg_bac_length_to_back_ratio - 1.96 * std_bac_length_to_back_ratio <= this_bacterium_bac_length_to_back:
            bad_relation_flag = False
        else:
            bad_relation_flag = True

    return bad_relation_flag


def check_errors_for_redundant_parent_link(current_df):

    selected_rows = current_df.loc[current_df['divideFlag'] == False]

    return selected_rows


def check_errors_for_missing_parent_link(df, current_df):
    daughter_len_to_mother_ratio_outliers = find_daughter_len_to_mother_ratio_outliers(df)
    bacteria_with_redundant_parent_link_error = \
        df.loc[
            (df["daughter_length_to_mother"].isin(daughter_len_to_mother_ratio_outliers['daughter_length_to_mother']))
            | (df["max_daughter_len_to_mother"].isin(daughter_len_to_mother_ratio_outliers['max_daughter_len_to_mother']))]

    bad_division_df = df.loc[df['bad_division_flag'] == True]

    daughters_with_redundant_parent = []
    bad_daughters = []

    for redundant_parent_indx, redundant_parent in bacteria_with_redundant_parent_link_error.iterrows():
        daughters_with_redundant_parent.extend(redundant_parent['daughters_index'])

    for bad_division_parent_indx, bad_division_parent in bad_division_df.iterrows():
        bad_daughters.extend(bad_division_parent['bad_daughters_index'])

    rows_indx_with_error = []
    if len(daughters_with_redundant_parent) > 0:
        rows_indx_with_error.extend(daughters_with_redundant_parent)

    if len(bad_daughters) > 0:
        rows_indx_with_error.extend(bad_daughters)

    if len(rows_indx_with_error) > 0:
        rows_indx_with_error = set(rows_indx_with_error)
        selected_rows = current_df.loc[current_df.index.isin(rows_indx_with_error) |
                                       (current_df['unexpected_end'] == True) | (current_df['transition'] == True)]
    else:
        selected_rows = current_df.loc[(current_df['unexpected_end'] == True) | (current_df['transition'] == True)]

    return selected_rows
