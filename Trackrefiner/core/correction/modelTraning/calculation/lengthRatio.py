import numpy as np


def check_len_ratio(df, selected_rows, col_target, col_source):

    """
    Calculates the length change ratio between target and source bacteria and computes the dynamic length change.

    **Details**:
        - **LengthChangeRatio**:
            - Computed as the ratio of the major axis lengths of the target and source bacteria.
        - **Dynamic Length Change**:
            - For bacteria where the target is smaller than the source (`LengthChangeRatio < 1`):
                - If the source bacterium's age is less than 2, the dynamic length is `1 - LengthChangeRatio`.
                - If the source bacterium's age is 2 or greater, the dynamic length is adjusted using
                  the average changes in the source bacterium's history.
            - For bacteria where the target is larger than or equal to the source (`LengthChangeRatio >= 1`),
              the dynamic length is set to `0`.

    :param pandas.DataFrame df:
        The DataFrame containing all bacterial data.
    :param pandas.DataFrame selected_rows:
        Subset of `df` containing the rows to process.
    :param str col_target:
        Column suffix for the target bacterium.
    :param str col_source:
        Column suffix for the source bacterium.

    **Returns**:
        pandas.DataFrame:

        The updated `selected_rows` DataFrame with a new column `length_dynamic` appended for the target bacteria.
    """

    selected_rows['Length_Change_Ratio'] = selected_rows['AreaShape_MajorAxisLength' + col_target] / selected_rows[
        'AreaShape_MajorAxisLength' + col_source]

    condition1 = selected_rows['Length_Change_Ratio'] >= 1
    condition2 = selected_rows['Length_Change_Ratio'] < 1
    condition3 = selected_rows['age' + col_source] < 2
    condition4 = selected_rows['age' + col_source] >= 2

    selected_rows.loc[condition1, 'length_dynamic' + col_target] = 0

    selected_rows.loc[condition2 & condition3, 'length_dynamic' + col_target] = \
        1 - selected_rows.loc[condition2 & condition3, 'Length_Change_Ratio']

    other_bac_should_cal_view = selected_rows.loc[condition2 & condition4]

    other_bac_should_cal = other_bac_should_cal_view.reset_index(names='real_index')[
            ['ImageNumber' + col_target, 'ObjectNumber' + col_target,
             'ImageNumber' + col_source, 'ObjectNumber' + col_source, 'id' + col_source,
             'Length_Change_Ratio', 'real_index']].merge(
            df[['ImageNumber', 'Length_Change_Ratio', 'id']],
            left_on='id' + col_source, right_on='id', how='left', suffixes=('', '_org_df'))

    condition5 = other_bac_should_cal['ImageNumber_org_df'] < other_bac_should_cal['ImageNumber' + col_target]

    other_bac_should_cal.loc[condition5, 'avg_source_bac_changes'] = \
        other_bac_should_cal.loc[condition5].groupby(
            ['ImageNumber' + col_target, 'ObjectNumber' + col_target,
             'id' + col_source])['Length_Change_Ratio_org_df'].transform('mean')

    other_bac_should_cal = \
        other_bac_should_cal.loc[condition5].groupby(['ImageNumber' + col_target, 'ObjectNumber' + col_target,
                                                      'ImageNumber' + col_source, 'ObjectNumber' + col_source
                                                      ]).head(1)[['real_index', 'avg_source_bac_changes',
                                                                  'Length_Change_Ratio']]

    other_bac_should_cal['final_changes'] = (other_bac_should_cal['avg_source_bac_changes'] -
                                             other_bac_should_cal['Length_Change_Ratio'])

    other_bac_should_cal.loc[other_bac_should_cal['avg_source_bac_changes'] < 1, 'final_changes'] = \
        1 - other_bac_should_cal['Length_Change_Ratio']

    selected_rows.loc[other_bac_should_cal['real_index'], 'length_dynamic' + col_target] = \
        other_bac_should_cal['final_changes'].values

    if np.isnan(selected_rows['length_dynamic' + col_target].values).any():
        raise ValueError("NaN values found in 'length_dynamic' calculation.")

    return selected_rows
