import pandas as pd
from Trackrefiner.strain.correction.action.findOutlier import find_bac_len_boundary
from Trackrefiner.strain.correction.action.bacteriaModification import remove_bac


def remove_from_neighbors_df(neighbors_df, bad_obj):

    bad_obj_neighbor = neighbors_df.loc[(neighbors_df['First Image Number'] == bad_obj['ImageNumber']) &
                                        (neighbors_df['First Object Number'] == bad_obj['ObjectNumber'])
                                        ]

    bac_neighbor_bad = neighbors_df.loc[(neighbors_df['Second Image Number'] == bad_obj['ImageNumber']) &
                                        (neighbors_df['Second Object Number'] == bad_obj['ObjectNumber'])
                                        ]

    should_be_remove_index = []
    if bad_obj_neighbor.shape[0] > 0:
        should_be_remove_index.extend(bad_obj_neighbor.index.values.tolist())

    if bac_neighbor_bad.shape[0] > 0:
        should_be_remove_index.extend(bac_neighbor_bad.index.values.tolist())

    if len(should_be_remove_index) > 0:
        neighbors_df = neighbors_df.loc[~neighbors_df.index.isin(should_be_remove_index)]

    return neighbors_df


def noise_remover(df, neighbors_df, parent_image_number_col, parent_object_number_col, label_col,
                  center_coordinate_columns, logs_df):

    num_noise_obj = None

    while num_noise_obj != 0:

        bac_len_boundary = find_bac_len_boundary(df)

        noise_objects_df = df.loc[(df['AreaShape_MajorAxisLength'] < bac_len_boundary['avg'] -
                                   1.96 * bac_len_boundary['std']) & (df['noise_bac'] == False)]

        num_noise_obj = noise_objects_df.shape[0]

        if noise_objects_df.shape[0] > 0:
            noise_objects_log = ["The objects listed below are identified as noise and have been removed.: "
                                 "\n ImageNumber\tObjectNumber"]
        else:
            noise_objects_log = ['']

        for noise_bac_ndx, noise_bac in noise_objects_df.iterrows():

            df = remove_bac(df, noise_bac_ndx, noise_bac, neighbors_df, parent_image_number_col,
                            parent_object_number_col, label_col, center_coordinate_columns)

            df.at[noise_bac_ndx, 'noise_bac'] = True

            logs_df = pd.concat([logs_df, df.iloc[noise_bac_ndx].to_frame().transpose()], ignore_index=True)

            neighbors_df = remove_from_neighbors_df(neighbors_df, noise_bac)

            noise_objects_log.append(str(noise_bac['ImageNumber']) + '\t' + str(noise_bac['ObjectNumber']))

        # df = remove_rows(df, 'noise_bac', False)

    return df, neighbors_df, noise_objects_log, logs_df
