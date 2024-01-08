import pandas as pd
from CellProfilerAnalysis.strain.correction.action.findOutlier import find_bac_len_boundary
from CellProfilerAnalysis.strain.correction.action.bacteriaModification import remove_bac


def noise_remover(df):
    daughters_to_mother_ratio_list_outliers = find_bac_len_boundary(df)

    noise_objects_df = df.loc[df['AreaShape_MajorAxisLength'] <
                              daughters_to_mother_ratio_list_outliers['avg'] -
                              1.96 * daughters_to_mother_ratio_list_outliers['std']]

    if noise_objects_df.shape[0] > 0:
        noise_objects_log = ["The objects listed below are identified as noise and have been removed.: " \
                             "\n ImageNumber\tObjectNumber"]
    else:
        noise_objects_df = ['']

    # print("number of noise objects: ")
    # print(noise_objects_df.shape[0])

    for noise_bac_ndx, noise_bac in noise_objects_df.iterrows():
        df = remove_bac(df, noise_bac)
        # change noise flag
        df.at[noise_bac_ndx, 'noise_bac'] = True

        noise_objects_log.append(str(noise_bac['ImageNumber']) + '\t' + str(noise_bac['ObjectNumber']))

    return df, noise_objects_log
