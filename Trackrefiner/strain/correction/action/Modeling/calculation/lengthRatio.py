import numpy as np
import pandas as pd


def check_len_ratio(df, selected_rows, col_target, col_source):

    selected_rows['LengthChangeRatio'] = \
        selected_rows['AreaShape_MajorAxisLength' + col_target] / selected_rows[
            'AreaShape_MajorAxisLength' + col_source]

    condition1 = selected_rows['LengthChangeRatio'] >= 1
    condition2 = selected_rows['LengthChangeRatio'] < 1
    condition3 = selected_rows['age' + col_source] < 2
    condition4 = selected_rows['age' + col_source] >= 2

    selected_rows.loc[condition1, 'length_dynamic' + col_target] = 0

    selected_rows.loc[condition2 & condition3, 'length_dynamic' + col_target] = \
        1 - selected_rows.loc[condition2 & condition3, 'LengthChangeRatio']

    # other_bac_should_cal = selected_rows.loc[condition2 & condition4].copy()
    other_bac_should_cal_view = selected_rows.loc[condition2 & condition4]
    # other_bac_should_cal['real_index'] = other_bac_should_cal.index.values

    other_bac_should_cal = \
        other_bac_should_cal_view.reset_index(
            names='real_index')[
            ['ImageNumber' + col_target, 'ObjectNumber' + col_target,
             'ImageNumber' + col_source, 'ObjectNumber' + col_source, 'id' + col_source,
             'LengthChangeRatio', 'real_index']].merge(
            df[['ImageNumber', 'LengthChangeRatio', 'id']],
            left_on='id' + col_source, right_on='id', how='left', suffixes=('', '_org_df'))

    # other_bac_should_cal = other_bac_should_cal.loc[other_bac_should_cal['ImageNumber_org_df'] <
    #                                                other_bac_should_cal['ImageNumber' + col_target]].copy()

    condition5 = other_bac_should_cal['ImageNumber_org_df'] < other_bac_should_cal['ImageNumber' + col_target]

    other_bac_should_cal.loc[condition5, 'avg_source_bac_changes'] = \
        other_bac_should_cal.loc[condition5].groupby(['ImageNumber' + col_target, 'ObjectNumber' + col_target,
                                                      'id' + col_source])['LengthChangeRatio_org_df'].transform('mean')

    # other_bac_should_cal = other_bac_should_cal.drop_duplicates(['ImageNumber' + col_target,
    #                                                             'ObjectNumber' + col_target,
    #                                                             'ImageNumber' + col_source,
    #                                                             'ObjectNumber' + col_source
    #                                                             ])

    other_bac_should_cal = other_bac_should_cal.loc[condition5].groupby(['ImageNumber' + col_target,
                                                                         'ObjectNumber' + col_target,
                                                                         'ImageNumber' + col_source,
                                                                         'ObjectNumber' + col_source
                                                                         ]).head(1)[['real_index',
                                                                                     'avg_source_bac_changes',
                                                                                     'LengthChangeRatio']]

    other_bac_should_cal['final_changes'] = (other_bac_should_cal['avg_source_bac_changes'] -
                                             other_bac_should_cal['LengthChangeRatio'])

    other_bac_should_cal.loc[other_bac_should_cal['avg_source_bac_changes'] < 1, 'final_changes'] = \
        1 - other_bac_should_cal['LengthChangeRatio']

    selected_rows.loc[other_bac_should_cal['real_index'], 'length_dynamic' + col_target] = \
        other_bac_should_cal['final_changes'].values

    if np.isnan(selected_rows['length_dynamic' + col_target].values).any():
        other_bac_should_cal.to_csv('other_bac_should_cal.csv')
        selected_rows.to_csv('selected_rows.csv')
        df.to_csv('df.csv')
        breakpoint()

    return selected_rows
