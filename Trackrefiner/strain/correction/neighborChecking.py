def neighbor_checking(df, neighbor_df, parent_image_number_col, parent_object_number_col):

    for bac_id in df['id'].unique():
        bac_life_history = df.loc[df['id'] == bac_id]
        bac_life_history_index = bac_life_history.index.values.tolist()

        for i, indx in enumerate(bac_life_history_index):

            neighbor_bac1_id_list = []
            parent_bac1_id_list = []
            daughter_bac1_id_list = []

            neighbor_bac2_id_list = []

            if i != 0:
                # bac1 & bac2 are same (bac2 is after bac1)
                bac1 = df.iloc[bac_life_history_index[i - 1]]
                bac2 = df.iloc[bac_life_history_index[i]]
            else:
                # bac1 & bac2 are same (bac2 is after bac1)
                bac2 = df.iloc[bac_life_history_index[i]]
                # bac1 is parent of bac2
                bac1 = df.loc[(df['ImageNumber'] == bac2[parent_image_number_col]) &
                              (df['ObjectNumber'] == bac2[parent_object_number_col])]
                if bac1.shape[0] > 0:
                    bac1 = df.loc[bac1.index.values.tolist()[0]]
                    bac2_daughter_id = df.loc[(df['parent_id'] == bac1['id']) & (df['id'] != bac2['id'])]['id'].values.tolist()[0]
                    daughter_bac1_id_list.append(bac2_daughter_id)
                else:
                    bac1 = None

            if bac1 is not None:
                neighbor_bac1_df = neighbor_df.loc[(neighbor_df['First Image Number'] == bac1['ImageNumber']) &
                                                   (neighbor_df['First Object Number'] == bac1['ObjectNumber'])]

                if neighbor_bac1_df.shape[0] > 0:
                    neighbor_bac1_features_df = \
                        df.loc[(df['ImageNumber'] == neighbor_bac1_df['Second Image Number'].values.tolist()[0]) &
                               (df['ObjectNumber'].isin(neighbor_bac1_df['Second Object Number'].values.tolist()))]

                    for neighbor_bac1_ndx, neighbor_bac1 in neighbor_bac1_features_df.iterrows():
                        if (neighbor_bac1[parent_image_number_col] != 0 and neighbor_bac1[parent_image_number_col] != 0) or \
                                neighbor_bac1['ImageNumber'] == 1:
                            neighbor_bac1_id_list.append(neighbor_bac1['id'])
                            daughter_indx = neighbor_bac1['daughters_index']
                            if daughter_indx != '':
                                daughters = df.loc[df.index.isin(daughter_indx)]
                                daughter_bac1_id_list.extend(daughters['id'].values.tolist())
                                parent_bac1_id_list.append(neighbor_bac1['id'])

                neighbor_bac2_df = neighbor_df.loc[(neighbor_df['First Image Number'] == bac2['ImageNumber']) &
                                                   (neighbor_df['First Object Number'] == bac2['ObjectNumber'])]

                if neighbor_bac2_df.shape[0] > 0:
                    neighbor_bac2_features_df = \
                        df.loc[(df['ImageNumber'] == neighbor_bac2_df['Second Image Number'].values.tolist()[0]) &
                               (df['ObjectNumber'].isin(neighbor_bac2_df['Second Object Number'].values.tolist()))]

                    for neighbor_bac2_ndx, neighbor_bac2 in neighbor_bac2_features_df.iterrows():
                        if (neighbor_bac2[parent_image_number_col] != 0 and neighbor_bac2[parent_image_number_col] != 0) or \
                                neighbor_bac2['ImageNumber'] == 1:
                            neighbor_bac2_id_list.append(neighbor_bac2['id'])

                # Convert lists to sets
                neighbor_bac1_id_list = set(neighbor_bac1_id_list)
                neighbor_bac2_id_list = set(neighbor_bac2_id_list)

                # Find symmetric difference
                unique_elements = neighbor_bac1_id_list ^ neighbor_bac2_id_list

                # Convert the result back to a list, if needed
                unique_elements = list(unique_elements)
                unique_elements = [v for v in unique_elements if v not in daughter_bac1_id_list and v not in
                                   parent_bac1_id_list]

                df.at[indx, "difference_neighbors"] = len(unique_elements)

    return df


def check_num_neighbors(df, neighbor_df, bac1, bac2, parent_image_number_col, return_common_elements=False):

    # Note: bac2 is after bac1

    neighbor_bac1_id_list = []
    parent_bac1_id_list = []
    daughter_bac1_id_list = []

    neighbor_bac2_id_list = []

    bac1_daughter = df.loc[(df['ImageNumber'] == bac2['ImageNumber']) & (df['parent_id'] == bac1['id']) &
                           (df['id'] != bac2['id'])]

    if bac1_daughter.shape[0] > 0:
        daughter_bac1_id_list.append(bac1_daughter['id'].values.tolist()[0])

    neighbor_bac1_df = neighbor_df.loc[(neighbor_df['First Image Number'] == bac1['ImageNumber']) &
                                       (neighbor_df['First Object Number'] == bac1['ObjectNumber'])]

    if neighbor_bac1_df.shape[0] > 0:
        neighbor_bac1_features_df = \
            df.loc[(df['ImageNumber'] == neighbor_bac1_df['Second Image Number'].values.tolist()[0]) &
                   (df['ObjectNumber'].isin(neighbor_bac1_df['Second Object Number'].values.tolist()))]

        for neighbor_bac1_ndx, neighbor_bac1 in neighbor_bac1_features_df.iterrows():
            if (neighbor_bac1[parent_image_number_col] != 0 and neighbor_bac1[parent_image_number_col] != 0) or \
                    neighbor_bac1['ImageNumber'] == 1:
                neighbor_bac1_id_list.append(neighbor_bac1['id'])
                daughter_indx = neighbor_bac1['daughters_index']
                if daughter_indx != '':
                    daughters = df.loc[df.index.isin(daughter_indx)]
                    daughter_bac1_id_list.extend(daughters['id'].values.tolist())
                    parent_bac1_id_list.append(neighbor_bac1['id'])

    neighbor_bac2_df = neighbor_df.loc[(neighbor_df['First Image Number'] == bac2['ImageNumber']) &
                                       (neighbor_df['First Object Number'] == bac2['ObjectNumber'])]

    if neighbor_bac2_df.shape[0] > 0:
        neighbor_bac2_features_df = \
            df.loc[(df['ImageNumber'] == neighbor_bac2_df['Second Image Number'].values.tolist()[0]) &
                   (df['ObjectNumber'].isin(neighbor_bac2_df['Second Object Number'].values.tolist()))]

        for neighbor_bac2_ndx, neighbor_bac2 in neighbor_bac2_features_df.iterrows():
            if (neighbor_bac2[parent_image_number_col] != 0 and neighbor_bac2[parent_image_number_col] != 0) or \
                    neighbor_bac2['ImageNumber'] == 1:
                neighbor_bac2_id_list.append(neighbor_bac2['id'])

    if len(neighbor_bac1_id_list) > 0 and len(neighbor_bac2_id_list) > 0:
        # Convert lists to sets
        neighbor_bac1_id_list = set(neighbor_bac1_id_list)
        neighbor_bac2_id_list = set(neighbor_bac2_id_list)

        # Find symmetric difference
        unique_elements = neighbor_bac1_id_list ^ neighbor_bac2_id_list
        common_elements = neighbor_bac1_id_list.intersection(neighbor_bac2_id_list)

        common_elements = list(common_elements)

        # Convert the result back to a list, if needed
        unique_elements = list(unique_elements)
        unique_elements = [v for v in unique_elements if v not in daughter_bac1_id_list and v not in
                           parent_bac1_id_list]

        result_division = [v for v in unique_elements if v in daughter_bac1_id_list or v in
                                             parent_bac1_id_list]

        common_elements.extend(result_division)

    else:
        unique_elements = []
        common_elements = []

    if not return_common_elements:
        return len(unique_elements)
    else:
        return len(unique_elements), len(common_elements)
