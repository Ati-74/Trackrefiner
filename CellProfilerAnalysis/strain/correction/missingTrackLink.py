import numpy as np


# , sorted_npy_files_list, um_per_pixel
def missing_track_link(dataframe):

    all_bacteria_movement = [v for v in dataframe['bacteria_movement'].dropna().values.tolist() if v != '']
    avg_all_bacteria_movement = np.average(all_bacteria_movement)
    std_all_bacteria_movement = np.std(all_bacteria_movement)

    for time_step in dataframe["ImageNumber"].unique():

        if time_step > 1:
            bac_in_current_time_step = dataframe.loc[dataframe["ImageNumber"] == time_step]

            bacteria_movement = [v for v in bac_in_current_time_step['bacteria_movement'].dropna().values.tolist()
                                 if v != '']

            avg_bacteria_movement = np.average(bacteria_movement)
            std_bacteria_movement = np.std(bacteria_movement)

            outliers = [v for v in bacteria_movement if (v > avg_bacteria_movement + std_bacteria_movement) and
                        (v > avg_all_bacteria_movement + std_all_bacteria_movement)]

            if len(outliers) > 0:
                print(time_step)
                print(outliers)
                print(avg_bacteria_movement + std_bacteria_movement)
                print(avg_all_bacteria_movement + std_all_bacteria_movement)
