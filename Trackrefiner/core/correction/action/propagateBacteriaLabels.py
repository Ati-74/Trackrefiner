import numpy as np


def propagate_bacteria_labels(df, parent_image_number_col, parent_object_number_col, label_col):

    """
    Assign or propagate labels for bacteria based on parent-child relationships.

    This function assigns unique labels to bacteria and propagates them across frames based on
    parent-child relationships in tracking data. It ensures consistent labeling of bacteria
    across multiple time steps by leveraging parent image and object identifiers.

    :param pd.DataFrame df:
        Input dataframe containing bacterial tracking data, including parent-child relationships.
    :param str parent_image_number_col:
        Name of the column representing the parent image number.
    :param str parent_object_number_col:
        Name of the column representing the parent object number.
    :param str label_col:
        Name of the column to assign or propagate labels.

    :return:
        pd.DataFrame

        Updated dataframe with assigned or propagated labels in the specified column.
        Also adds a `checked` column to indicate whether a row has been processed.
    """

    df['checked'] = False
    df[label_col] = np.nan
    cond1 = df[parent_image_number_col] == 0

    df.loc[cond1, label_col] = df.loc[cond1, 'index'].values + 1
    df.loc[cond1, 'checked'] = True

    # other bacteria
    other_bac_df = df.loc[~ df['checked']]

    temp_df = df.copy()
    temp_df.index = (temp_df['ImageNumber'].astype(str) + '_' + temp_df['ObjectNumber'].astype(str))

    bac_index_dict = temp_df['index'].to_dict()

    label_list = []

    same_bac_dict = {}

    for row_index, row in other_bac_df.iterrows():

        image_number, object_number, parent_img_num, parent_obj_num = \
            row[['ImageNumber', 'ObjectNumber', parent_image_number_col, parent_object_number_col]]

        if f'{int(parent_img_num)}_{int(parent_obj_num)}' not in same_bac_dict.keys():
            source_link = df.iloc[bac_index_dict[f'{int(parent_img_num)}_{int(parent_obj_num)}']]

            this_bac_label = source_link[label_col]

        else:

            this_bac_label = same_bac_dict[f'{int(parent_img_num)}_{int(parent_obj_num)}']

        label_list.append(this_bac_label)

        # same bacteria
        same_bac_dict[f'{int(image_number)}_{int(object_number)}'] = this_bac_label

    df.loc[other_bac_df.index, label_col] = label_list

    return df
