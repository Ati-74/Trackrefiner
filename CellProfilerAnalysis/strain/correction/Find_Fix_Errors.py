from CellProfilerAnalysis.strain.correction.action.processing import bacteria_life_history, convert_to_um, remove_rows, angle_convert_to_radian
from CellProfilerAnalysis.strain.correction.action.FluorescenceIntensity import assign_cell_type
from CellProfilerAnalysis.strain.correction.NanLabel import modify_nan_labels
from CellProfilerAnalysis.strain.correction.TransitionError import correction_transition
from CellProfilerAnalysis.strain.correction.BadDaughters import correction_bad_daughters
from CellProfilerAnalysis.strain.correction.MergedBacteria import merged_bacteria


def assign_feature_find_errors(dataframe, intensity_threshold):
    """
    goal: assign new features like: `id`, `divideFlag`, `daughters_index`, `bad_division_flag`, `bad_daughters_index`,
    `unexpected_end`, `division_time`, `transition`, `LifeHistory`, `parent_id` to bacteria and find errors

    @param dataframe dataframe bacteria features value
    @param intensity_threshold float min intensity value of channel
    """
    dataframe['checked'] = False
    dataframe["id"] = ''
    dataframe["divideFlag"] = False
    dataframe['daughters_index'] = ''
    dataframe['bad_division_flag'] = False
    dataframe['bad_daughters_index'] = ''
    dataframe['unexpected_end'] = False
    dataframe['division_time'] = 0
    dataframe['transition'] = False
    dataframe["LifeHistory"] = ''
    dataframe["parent_id"] = ''

    last_time_step = sorted(dataframe['ImageNumber'].unique())[-1]
    # bacterium id
    bacterium_id = 1
    transition = False

    for row_index, row in dataframe.iterrows():
        if not dataframe.iloc[row_index]["checked"]:
            bacterium_status = bacteria_life_history(dataframe, row, row_index, last_time_step)
            parent_img_num = row['TrackObjects_ParentImageNumber_50']
            parent_obj_num = row['TrackObjects_ParentObjectNumber_50']
            if parent_img_num != 0 and parent_obj_num != 0:
                parent = dataframe.loc[(dataframe["ImageNumber"] == parent_img_num) & (dataframe["ObjectNumber"] ==
                                                                                       parent_obj_num)]
                parent_id = parent['id'].values.tolist()[0]
                transition = False
            else:
                parent_id = 0
                if row['ImageNumber'] > 1:
                    transition = True

            for indx in bacterium_status['lifeHistoryIndex']:
                dataframe.at[indx, 'checked'] = True

                dataframe.at[indx, 'id'] = bacterium_id
                dataframe.at[indx, 'LifeHistory'] = bacterium_status['life_history']
                if bacterium_status['division_occ']:
                    dataframe.at[indx, 'divideFlag'] = bacterium_status['division_occ']
                    dataframe.at[indx, 'daughters_index'] = bacterium_status['daughters_index']
                    dataframe.at[indx, 'division_time'] = bacterium_status['division_time']
                elif bacterium_status['division_occ']:
                    dataframe.at[indx, 'divideFlag'] = bacterium_status['division_occ']
                    dataframe.at[indx, 'bad_division_flag'] = bacterium_status['bad_division_occ']
                    dataframe.at[indx, 'bad_daughters_index'] = bacterium_status['bad_daughters_index']
                    dataframe.at[indx, 'division_time'] = bacterium_status['division_time']

                dataframe.at[indx, 'parent_id'] = parent_id

            last_bacterium_in_life_history = bacterium_status['lifeHistoryIndex'][-1]
            dataframe.at[last_bacterium_in_life_history, 'unexpected_end'] = bacterium_status['unexpected_end']
            dataframe.at[row_index, 'transition'] = transition

            bacterium_id += 1

    # assign cell type
    dataframe = assign_cell_type(dataframe, intensity_threshold)
    dataframe.drop(labels='checked', axis=1, inplace=True)
    return dataframe


def data_cleaning(raw_df):
    """
    goal:   1. remove related rows to bacteria with zero MajorAxisLength
            2. Correct the labels of bacteria whose labels are nan.

    @param raw_df dataframe bacteria features value
    """

    # remove related rows to bacteria with zero MajorAxisLength
    raw_df = raw_df.loc[raw_df["AreaShape_MajorAxisLength"] != 0].reset_index(drop=True)

    modified_df = modify_nan_labels(raw_df)

    return modified_df


def data_modification(dataframe, intensity_threshold=0.1):

    # 1. remove related rows to bacteria with zero MajorAxisLength
    # 2. Correct the labels of bacteria whose labels are nan.
    dataframe = data_cleaning(dataframe)
    dataframe = assign_feature_find_errors(dataframe, intensity_threshold)

    return dataframe


def data_conversion(dataframe, um_per_pixel=0.144):

    dataframe = convert_to_um(dataframe, um_per_pixel)
    dataframe = angle_convert_to_radian(dataframe)

    return dataframe


def find_fix_errors(dataframe, number_of_gap=0, um_per_pixel=0.144, intensity_threshold=0.1):

    dataframe = data_modification(dataframe, intensity_threshold)
    dataframe = data_conversion(dataframe, um_per_pixel)

    # modification of errors:
    # 1. transition error
    # 2. more than two daughters
    # 3. merged bacteria

    df = correction_transition(dataframe, number_of_gap)
    df = correction_bad_daughters(df, number_of_gap)
    df = merged_bacteria(df)
    # remove incorrect bacteria
    df = remove_rows(df, 'transition_drop', False)
    df = remove_rows(df, 'bad_daughter_drop', False)

    return df
