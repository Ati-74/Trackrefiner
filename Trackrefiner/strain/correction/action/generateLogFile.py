import pandas as pd


def find_noise_bac(raw_df, final_df, logs_list, parent_image_number_col, parent_object_number_col):
    noise_bac_df = raw_df.loc[~ raw_df['prev_index'].isin(final_df['prev_index'])].copy()
    noise_bac_df['noise_bac'] = True
    noise_bac_df['changed'] = False
    noise_bac_df['unexpected_end'] = False
    noise_bac_df['unexpected_beginning'] = False

    noise_bac_df[[parent_image_number_col, parent_object_number_col, 'id', 'parent_id']] = 0

    msg = 'Number of Noise Objects: ' + str(noise_bac_df.shape[0])
    logs_list.append(msg)

    return noise_bac_df, logs_list


def find_unexpected_beginning(final_df, logs_list):
    unexpected_beginning_bac_df = final_df.loc[final_df['unexpected_beginning']].copy()
    unexpected_beginning_bac_df['noise_bac'] = False
    unexpected_beginning_bac_df['changed'] = False
    unexpected_beginning_bac_df['unexpected_end'] = False

    msg = 'Number of Unexpected Beginning Bacteria: ' + str(unexpected_beginning_bac_df.shape[0])
    logs_list.append(msg)

    return unexpected_beginning_bac_df, logs_list


def find_unexpected_end(final_df, logs_list):
    unexpected_bac_df = final_df.loc[final_df['unexpected_end']].copy()
    unexpected_bac_df['noise_bac'] = False
    unexpected_bac_df['changed'] = False
    unexpected_bac_df['unexpected_beginning'] = False

    msg = 'Number of Unexpected end Bacteria: ' + str(unexpected_bac_df.shape[0])
    logs_list.append(msg)

    return unexpected_bac_df, logs_list


def bacteria_with_change_relation(raw_df, final_df, logs_list, parent_image_number_col, parent_object_number_col):
    merged_df = raw_df.merge(final_df, on='prev_index', how='inner', suffixes=('_raw', ''))

    identified_tracking_errors_df = \
        merged_df.loc[
            (merged_df[parent_image_number_col + '_raw'] !=
             merged_df[parent_image_number_col]) |
            (merged_df[parent_object_number_col + '_raw'] !=
             merged_df[parent_object_number_col])].copy()

    identified_tracking_errors_df['noise_bac'] = False
    identified_tracking_errors_df['changed'] = True
    identified_tracking_errors_df['unexpected_end'] = False
    identified_tracking_errors_df['unexpected_beginning'] = False

    msg = 'The number of bacteria whose links have changed: ' + str(identified_tracking_errors_df.shape[0])
    logs_list.append(msg)

    return identified_tracking_errors_df, logs_list


def generate_log_file(raw_df, final_df, logs_list, parent_image_number_col, parent_object_number_col,
                      center_coordinate_columns):
    # find noise objects
    noise_bac_df, logs_list = find_noise_bac(raw_df, final_df, logs_list, parent_image_number_col,
                                             parent_object_number_col)

    # find unexpected beginning
    unexpected_beginning_bac_df, logs_list = find_unexpected_beginning(final_df, logs_list)

    # find unexpected end bacteria
    unexpected_bac_df, logs_list = find_unexpected_end(final_df, logs_list)

    # now we should merge dataframes
    # noise_bac	unexpected_end	unexpected_beginning	changed
    merged_ub_bac_ue_bac = \
        unexpected_beginning_bac_df[
            ['ImageNumber', 'ObjectNumber', center_coordinate_columns['x'], center_coordinate_columns['y'],
             'AreaShape_MajorAxisLength', parent_image_number_col, parent_object_number_col, 'id', 'parent_id',
             'noise_bac', 'unexpected_end', 'unexpected_beginning', 'changed']
        ].merge(unexpected_bac_df[
                    ['ImageNumber', 'ObjectNumber', center_coordinate_columns['x'], center_coordinate_columns['y'],
                     'AreaShape_MajorAxisLength', parent_image_number_col, parent_object_number_col, 'id',
                     'parent_id', 'noise_bac', 'unexpected_end', 'unexpected_beginning', 'changed']
                ],
                on=['ImageNumber', 'ObjectNumber', center_coordinate_columns['x'], center_coordinate_columns['y'],
                    'AreaShape_MajorAxisLength', parent_image_number_col, parent_object_number_col, 'id',
                    'parent_id'],
                how='outer', suffixes=('_ub', '_ue'))

    merged_ub_bac_ue_bac = merged_ub_bac_ue_bac.map(lambda x: False if str(x) in ['nan', "b''"] else x)

    merged_ub_bac_ue_bac['noise_bac'] = merged_ub_bac_ue_bac['noise_bac_ub'] + merged_ub_bac_ue_bac['noise_bac_ue']

    merged_ub_bac_ue_bac['unexpected_end'] = \
        merged_ub_bac_ue_bac['unexpected_end_ub'] + merged_ub_bac_ue_bac['unexpected_end_ue']

    merged_ub_bac_ue_bac['unexpected_beginning'] = \
        merged_ub_bac_ue_bac['unexpected_beginning_ub'] + merged_ub_bac_ue_bac['unexpected_beginning_ue']

    merged_ub_bac_ue_bac['changed'] = \
        merged_ub_bac_ue_bac['changed_ub'] + merged_ub_bac_ue_bac['changed_ue']

    merged_ub_bac_ue_bac_noise = \
        pd.concat([noise_bac_df[
                       ['ImageNumber', 'ObjectNumber', center_coordinate_columns['x'], center_coordinate_columns['y'],
                        'AreaShape_MajorAxisLength', parent_image_number_col, parent_object_number_col, 'id',
                        'parent_id', 'noise_bac', 'unexpected_end', 'unexpected_beginning', 'changed']
                   ],
                   merged_ub_bac_ue_bac[
                       ['ImageNumber', 'ObjectNumber', center_coordinate_columns['x'], center_coordinate_columns['y'],
                        'AreaShape_MajorAxisLength', parent_image_number_col, parent_object_number_col, 'id',
                        'parent_id', 'noise_bac', 'unexpected_end', 'unexpected_beginning', 'changed']
                   ]
                   ])

    # Bacteria with a change of link
    bac_with_changes_df, logs_list = bacteria_with_change_relation(raw_df, final_df, logs_list,
                                                                   parent_image_number_col, parent_object_number_col)

    logs_df = merged_ub_bac_ue_bac_noise.merge(
        bac_with_changes_df[
            ['ImageNumber', 'ObjectNumber', center_coordinate_columns['x'], center_coordinate_columns['y'],
             'AreaShape_MajorAxisLength', parent_image_number_col, parent_object_number_col, 'id', 'parent_id',
             'noise_bac', 'unexpected_end', 'unexpected_beginning', 'changed']
        ], on=['ImageNumber', 'ObjectNumber', center_coordinate_columns['x'], center_coordinate_columns['y'],
               'AreaShape_MajorAxisLength', parent_image_number_col, parent_object_number_col, 'id', 'parent_id'],
        how='outer', suffixes=('_1', '_2'))

    logs_df = logs_df.map(lambda x: False if str(x) in ['nan', "b''"] else x)

    for col in ['noise_bac', 'unexpected_end', 'unexpected_beginning', 'changed']:
        logs_df[col] = logs_df[col + '_1'] + logs_df[col + '_2']

    logs_df = logs_df.sort_values(by=['ImageNumber', 'ObjectNumber'])
    logs_df.rename(columns={'ImageNumber': 'stepNum'}, inplace=True)
    logs_df = logs_df[["stepNum", "ObjectNumber", center_coordinate_columns['x'], center_coordinate_columns['y'],
                       'AreaShape_MajorAxisLength', parent_image_number_col, parent_object_number_col, 'id',
                       'parent_id', "noise_bac", "unexpected_end", "unexpected_beginning", 'changed']]

    identified_tracking_errors_df = logs_df.loc[logs_df['changed'] == True]

    msg = 'The number of identified tracking errors: ' + str(identified_tracking_errors_df.shape[0])
    logs_list.append(msg)

    fixed_errors = identified_tracking_errors_df.loc[identified_tracking_errors_df[parent_image_number_col] != 0]

    msg = 'The number of fixed errors: ' + str(fixed_errors.shape[0])
    logs_list.append(msg)

    remaining_errors_df = identified_tracking_errors_df.loc[identified_tracking_errors_df[parent_image_number_col] == 0]

    msg = 'The number of remaining errors: ' + str(remaining_errors_df.shape[0])
    logs_list.append(msg)

    return logs_df, identified_tracking_errors_df, fixed_errors, remaining_errors_df, logs_list
