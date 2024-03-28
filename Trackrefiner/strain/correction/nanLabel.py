from Trackrefiner.strain.correction.action.helperFunctions import find_related_bacteria

def modify_nan_labels(df, label_col):

    # Correct the labels of bacteria whose labels are nan.
    nan_label_bacteria = df.loc[df[label_col].isnull()]
    modified_bacteria_label_index = []
    if nan_label_bacteria.shape[0] > 0:
        for bac_index, bacterium in nan_label_bacteria.iterrows():
            if bac_index not in modified_bacteria_label_index:
                # assign label
                related_bacteria_index = find_related_bacteria(df, bacterium, bac_index, bacteria_index_list=None)
                related_bacteria = df.iloc[related_bacteria_index]
                unique_label = related_bacteria[label_col].unique()
                # remove nan label if exist
                unique_label = [elem for elem in unique_label if str(elem) != 'nan']
                if unique_label:
                    for index in related_bacteria_index:
                        if str(df.iloc[index][label_col]) == 'nan':
                            modified_bacteria_label_index.append(index)
                        df.at[index, label_col] = unique_label[0]
                else:
                    # assign new label
                    bacteria_labels = df[label_col].unique()
                    new_label = sorted([int(elem) for elem in bacteria_labels if str(elem) != 'nan'])[-1] + 1
                    for index in related_bacteria_index:
                        modified_bacteria_label_index.append(index)
                        df.at[index, label_col] = new_label
    return df
