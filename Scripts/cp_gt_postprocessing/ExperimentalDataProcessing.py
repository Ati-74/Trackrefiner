import math
import numpy as np
import pandas as pd


def lineage_bacteria_after_this_time_step(data_frame, bacteria, index):
    """
    Retrieve the lineage of a specified bacteria based on its time step and onwards.

    @param data_frame DataFrame containing features of bacteria.
    @param bacteria pd.Series A row from the DataFrame representing the bacteria of interest.
    @param index int The index of the bacteria in the data_frame.

    Returns:
    data_frame_of_lineage DataFrame containing only the lineage (ancestors and descendants) of the specified bacteria.
    """

    # Initialize a list with the provided index to store the lineage of the bacteria
    bacteria_index = [index]

    # Store the image and object numbers of the current bacteria
    parent_img_number = [bacteria['ImageNumber']]
    parent_obj_number = [bacteria['ObjectNumber']]

    # Continue to loop until no more parent object numbers are available
    while len(parent_obj_number) > 0:
        # Identify bacteria entries in the data_frame that have the current bacteria
        # as their parent (using parent image and object numbers)
        selected_bacteria = data_frame.loc[
            (data_frame["TrackObjects_ParentImageNumber_50"] == parent_img_number[0]) &
            (data_frame["TrackObjects_ParentObjectNumber_50"] == parent_obj_number[0])
            ]

        # Remove the current image and object numbers since they've been processed
        parent_img_number.pop(0)
        parent_obj_number.pop(0)

        # For each bacteria identified, append its index to the lineage list
        # and its image and object numbers to the lists for further tracking
        for index, row in selected_bacteria.iterrows():
            bacteria_index.append(index)
            parent_img_number.append(row['ImageNumber'])
            parent_obj_number.append(row['ObjectNumber'])

    # Filter the main data_frame to get only the rows related to the lineage of the given bacteria
    data_frame_of_lineage = data_frame[data_frame.index.isin(bacteria_index)]

    return data_frame_of_lineage


def find_division_occurrence(data_frame_of_lineage, bacteria, bacteria_id):
    """
    Determine if and when a specified bacteria divides based on its lineage.

    @param data_frame_of_lineage DataFrame The DataFrame containing lineage information for the bacteria.
    @param bacteria pd.Series A row from the DataFrame representing the bacteria of interest.
    @param bacteria_id int The unique identifier of the bacteria.

    Returns:
    - division_status dict A dictionary containing:
        * division_occ bool Whether a division occurred or not.
        * last_timestep_before_division int The last timestep before division or interruption.
        * lifeHistoryIndex list[int] List of indices tracing the life history of the bacteria.
    """

    # Starting time step and object number of the bacteria
    parent_time_step_of_cell = bacteria["ImageNumber"]
    parent_obj_number = bacteria["ObjectNumber"]

    division_occ = False  # A flag to track if division occurred
    life_history_index = []  # List to store indices of the bacteria's life history

    # Start by adding the given bacteria's ID to the life history list
    bacteria_index = bacteria_id
    life_history_index.append(bacteria_index)

    # Find the last time step in the provided lineage data
    last_time_step = data_frame_of_lineage["ImageNumber"].iloc[-1]

    # Continue to loop until a division is detected or until the last time step is reached
    while (division_occ is False) and (last_time_step != parent_time_step_of_cell):
        # Find relative bacteria in the next time step that have the current bacteria as their parent
        relative_bacteria_in_next_timestep = data_frame_of_lineage.loc[
            (data_frame_of_lineage["TrackObjects_ParentImageNumber_50"] == parent_time_step_of_cell) &
            (data_frame_of_lineage["TrackObjects_ParentObjectNumber_50"] == parent_obj_number)]

        number_of_relative_bacteria = relative_bacteria_in_next_timestep.shape[0]

        # If there's only one relative bacteria, update parent object number and time step for the next iteration
        if number_of_relative_bacteria == 1:
            parent_obj_number = relative_bacteria_in_next_timestep.iloc[0]["ObjectNumber"]
            parent_time_step_of_cell = relative_bacteria_in_next_timestep.iloc[0]["ImageNumber"]
            bacteria_index = relative_bacteria_in_next_timestep.index.tolist()[0]
            life_history_index.append(bacteria_index)

        # If there are two or more relative bacteria, a division has occurred
        elif number_of_relative_bacteria >= 2:
            division_occ = True
            last_timestep_before_division = parent_time_step_of_cell

        # If no relative bacteria are found, an interruption is detected
        elif number_of_relative_bacteria == 0:
            last_timestep_before_division = parent_time_step_of_cell
            break

    # If no division is detected by the end of the lineage, set the last timestep as the final timestep
    if division_occ is False and last_time_step == parent_time_step_of_cell:
        last_timestep_before_division = data_frame_of_lineage["ImageNumber"].iloc[-1]

    # Compile the results into a dictionary
    division_status = {
        "division_occ": division_occ,
        "last_timestep_before_division": last_timestep_before_division,
        "lifeHistoryIndex": life_history_index
    }

    return division_status


def life_history(data_frame_of_lineage, life_history_index):
    """
    Extracts the life history of a bacteria based on specified indices.

    @param data_frame_of_lineage DataFrame containing lineage information for the bacteria.
    @param life_history_index list[int] List of indices representing the bacteria's life history.

    Returns:
    - df_life_history DataFrame A subset of data_frame_of_lineage containing the life history of the bacteria.
    """

    df_life_history = data_frame_of_lineage.loc[life_history_index]

    return df_life_history


def average_growth_rate(division_length, birth_length, t):
    """
    Calculates the average elongation rate of a bacteria over time.

    @param division_length float The length of the bacteria at the time of division.
    @param birth_length float The initial length of the bacteria at birth.
    @param t float Time duration between birth and division.

    Returns:
    - elongation_rate float The average elongation rate, rounded to three decimal places.
    """

    elongation_rate = round((division_length - birth_length) / t, 3)

    return elongation_rate


def calc_growth_rate(df_life_history, interval_time):
    """
    Calculates the growth rate of bacteria based on its life history and the interval time.

    @param df_life_history DataFrame containing the life history of a bacteria.
    @param interval_time float Time interval between frames

    Returns:
    - elongation_rate float The growth rate of the bacteria. If the bacteria is present for only one timestep,
        it returns 'NaN'.
    """
    life_history_length = df_life_history.shape[0]

    # Extract the major axis length of bacteria at the end of its life history (division).
    division_length = df_life_history.iloc[[-1]]["AreaShape_MajorAxisLength"].values[0]
    # Extract the major axis length of bacteria at the start of its life history (birth).
    birth_length = df_life_history.iloc[[0]]["AreaShape_MajorAxisLength"].values[0]

    # If the bacteria exists for more than one timestep, compute the elongation rate. Otherwise, report NaN.
    if life_history_length > 1:
        t = life_history_length * interval_time
        elongation_rate = average_growth_rate(division_length, birth_length, t)
    else:
        elongation_rate = "NaN"

    return elongation_rate


def calc_average_velocity(pos1, pos2, life_history_length, interval_time):
    """
    Calculates the average velocity of a bacteria based on its initial and final positions.

    @param pos1 pd.Series Initial position of the bacteria.
    @param pos2 pd.Series Final position of the bacteria.
    @param life_history_length int Number of time intervals the bacteria has been tracked.
    @param interval_time float Time interval between frames.

    Returns:
    - average_velocity float The average velocity of the bacteria, rounded to three decimal places.
    """
    # Compute the Euclidean distance of the bacteria from the origin at initial and final positions.
    x1 = math.sqrt(pos1["AreaShape_Center_X"] ** 2 + pos1["AreaShape_Center_Y"] ** 2)
    x2 = math.sqrt(pos2["AreaShape_Center_X"] ** 2 + pos2["AreaShape_Center_Y"] ** 2)

    # Calculate the average velocity over the life history length.
    average_velocity = round((x2 - x1) / (life_history_length * interval_time), 3)

    return average_velocity


def assign_parent_daughter(df):
    """
    Assigns daughter cells to each parent cell in the DataFrame.

    @param df DataFrame containing cell data with columns "id" and "parent id".

    Returns:
    - df DataFrame The modified DataFrame with the 'daughters' column added, containing a list of daughter ids
        for each parent cell.
    """
    # Initialize a new column 'checked' with False values.
    df['checked'] = False

    # Iterate over each row in the DataFrame.
    for index, row in df.iterrows():
        # Only process cells that haven't been checked yet.
        if not df.iloc[index]['checked']:
            # Find all entries (rows) for the same cell (identified by "id").
            same_bac = df.loc[df["id"] == df.iloc[index]["id"]]
            same_bac_index = same_bac.index.values.tolist()

            # Identify potential daughter cells based on the 'parent_id' column.
            daughters = df.loc[df['parent_id'] == df.iloc[index]["id"]].reset_index(drop=True)

            # If the cell has one or no daughters, set daughters to 0.
            # Otherwise, set 'daughters' to a list of unique daughter "id" values.
            if daughters.shape[0] <= 1:
                daughters = 0
            else:
                daughters = list(set(daughters["id"].values.tolist()))

            # Update the 'daughters' and 'checked' columns for the same cell in all relevant rows.
            for row_index in same_bac_index:
                df.at[row_index, 'daughters'] = daughters
                df.at[row_index, 'checked'] = True

    return df


def bacteria_analysis(data_frame, interval_time, um_per_pixel):
    """
        Conducts analysis on bacteria based on given DataFrame.

        @param data_frame DataFrame containing bacteria data (output of Cellprofiler).
        @param interval_time float Time interval between measurements.
        @param um_per_pixel float Conversion factor for pixels to microns

        Returns:
        - final_data_frame (pd.DataFrame): Modified version of the input DataFrame with calculated features.
        - results (pd.DataFrame): Results of the analysis containing summary features for each bacteria.
        """

    # Initialize the result dictionary and the columns in the input DataFrame.
    result_dict = {"CellId": [], "label": [], "birth_length": [], "AverageLength": [], "AverageVelocity": [],
                   "LifeHistory": [], "GrowthRate": [], "AverageOrientation": []}

    # Initialization of columns in the DataFrame.
    # These columns will store analysis results and intermediary calculations.
    data_frame["id"] = ""
    data_frame["divideFlag"] = False
    data_frame["growthRate"] = ""
    data_frame["LifeHistory"] = ""
    data_frame["birth_Length"] = ""
    data_frame["lineage"] = ""
    data_frame["parent_id"] = ""
    data_frame["daughters"] = ""
    data_frame["InstantaneousGrowthRate"] = ""
    data_frame["InstantaneousVelocity"] = ""
    data_frame["new_label"] = ""
    data_frame['TrackObjects_Label_50'] = np.nan

    # Convert the orientation values from degrees to radians. (we should modify orientation)
    data_frame["AreaShape_Orientation"] = -(data_frame["AreaShape_Orientation"] + 90) * np.pi / 180

    data_frame['AreaShape_Center_X'] *= um_per_pixel
    data_frame['AreaShape_Center_Y'] *= um_per_pixel
    data_frame['Location_Center_X'] *= um_per_pixel
    data_frame['Location_Center_Y'] *= um_per_pixel

    data_frame['AreaShape_MajorAxisLength'] *= um_per_pixel
    data_frame['AreaShape_MinorAxisLength'] *= um_per_pixel

    # Initialization for tracking the IDs and labels.
    id_of_bacteria = 1
    last_time_step = data_frame['ImageNumber'].unique()[-1]
    try:
        max_label = max([x for x in data_frame['TrackObjects_Label_50'].values.tolist() if str(x) != 'nan'])
    except:
        max_label = 0

    # Iterate through the DataFrame row-by-row.
    for index, row in data_frame.iterrows():

        if row['ImageNumber'] < last_time_step:
            next_time_step_bacteria = \
                data_frame.loc[(data_frame['TrackObjects_ParentImageNumber_50'] == row['ImageNumber']) &
                               (data_frame['TrackObjects_ParentObjectNumber_50'] ==
                                row['ObjectNumber'])].reset_index(drop=True)
            if next_time_step_bacteria.shape[0] == 1:
                next_time_step_bacteria = next_time_step_bacteria.iloc[0]
                instantaneous_growth_rate = average_growth_rate(next_time_step_bacteria['AreaShape_MajorAxisLength'],
                                                                row['AreaShape_MajorAxisLength'], interval_time)
                # average velocity
                pos1 = row[["AreaShape_Center_X", "AreaShape_Center_Y"]]
                pos2 = next_time_step_bacteria[["AreaShape_Center_X", "AreaShape_Center_Y"]]
                instantaneous_velocity = calc_average_velocity(pos1, pos2, 1, interval_time)
            else:
                instantaneous_growth_rate = np.nan
                instantaneous_velocity = np.nan
        else:
            instantaneous_growth_rate = np.nan
            instantaneous_velocity = np.nan

        data_frame.at[index, "InstantaneousGrowthRate"] = instantaneous_growth_rate
        data_frame.at[index, "InstantaneousVelocity"] = instantaneous_velocity

        if not data_frame.iloc[index]["growthRate"] and data_frame.iloc[index]["growthRate"] != 0:

            data_frame_of_lineage = lineage_bacteria_after_this_time_step(data_frame, row, index)

            these_bac_label = list(set(data_frame_of_lineage['TrackObjects_Label_50'].values.tolist()))
            if 'nan' in [str(x) for x in these_bac_label]:
                these_bac_label = max_label + 1
                for idx in data_frame_of_lineage.index:
                    data_frame.at[idx, "TrackObjects_Label_50"] = these_bac_label
                max_label += 1

            division_status = find_division_occurrence(data_frame_of_lineage, row, index)

            df_life_history = life_history(data_frame_of_lineage, division_status["lifeHistoryIndex"])

            elongation_rate = calc_growth_rate(df_life_history, interval_time)

            life_history_index = df_life_history.index.tolist()
            life_history_length = df_life_history.shape[0]
            division_length = df_life_history.iloc[[-1]]["AreaShape_MajorAxisLength"].values[0]
            # length of bacteria when they are born
            birth_length = df_life_history.iloc[[0]]["AreaShape_MajorAxisLength"].values[0]

            # mean Length
            mean_length = np.mean(df_life_history["AreaShape_MajorAxisLength"].values)

            # mean orientation (radian)
            life_history_bacteria_orientation = df_life_history["AreaShape_Orientation"].values.tolist()
            # modification of orientation
            life_history_bacteria_orientation = [life_history_bacteria_orientation[i] for i in
                                                 range(len(life_history_bacteria_orientation))]
            mean_orientation = np.mean(life_history_bacteria_orientation)

            # average velocity
            pos1 = df_life_history.iloc[0][["AreaShape_Center_X", "AreaShape_Center_Y"]]
            pos2 = df_life_history.iloc[-1][["AreaShape_Center_X", "AreaShape_Center_Y"]]
            average_velocity = calc_average_velocity(pos1, pos2, life_history_length, interval_time)

            result_dict["CellId"].append(id_of_bacteria)
            result_dict["label"].append(df_life_history.iloc[[0]]["TrackObjects_Label_50"].values[0])
            result_dict["birth_length"].append(birth_length)
            result_dict["LifeHistory"].append(life_history_length)
            result_dict["AverageOrientation"].append(mean_orientation)
            if life_history_length > 1:
                result_dict["AverageLength"].append(mean_length)
                result_dict["AverageVelocity"].append(average_velocity)
            else:
                result_dict["AverageLength"].append(np.nan)
                result_dict["AverageVelocity"].append(np.nan)
            result_dict["GrowthRate"].append(elongation_rate)

            # parent id
            parent_img_number = data_frame.iloc[index]['TrackObjects_ParentImageNumber_50']
            parent_obj_number = data_frame.iloc[index]['TrackObjects_ParentObjectNumber_50']
            if parent_obj_number == 0:
                parent = 0
            else:
                parent = data_frame.loc[(data_frame['ImageNumber'] == parent_img_number) &
                                        (data_frame['ObjectNumber'] == parent_obj_number)]
                parent = parent["id"].values[0]
            for idx in life_history_index:
                data_frame.at[idx, "id"] = id_of_bacteria
                data_frame.at[idx, "growthRate"] = elongation_rate
                data_frame.at[idx, "parent_id"] = parent

            if division_status["division_occ"]:
                last_time_step_of_bacteria = life_history_index[-1]
                data_frame.at[last_time_step_of_bacteria, "divideFlag"] = True

                # daughters
                division_time = division_status["last_timestep_before_division"] + 1
                data_frame_of_daughters = data_frame_of_lineage.loc[(data_frame_of_lineage["ImageNumber"] ==
                                                                     division_time)]
                daughter_index_list = data_frame_of_daughters.index.tolist()

                parent_id = id_of_bacteria

                for daughter_idx in daughter_index_list:
                    data_frame.at[daughter_idx, "lineage"] = parent_id

            # Update bacteria ID for next iteration.
            id_of_bacteria += 1

    # Assign daughter cells to each parent cell.
    data_frame = assign_parent_daughter(data_frame)
    # Rename columns for clarity and select a subset of columns for the final DataFrame.
    final_data_frame = data_frame.rename(columns={"ImageNumber": "TimeStep", "AreaShape_Orientation": "Orientation",
                                                  "TrackObjects_Label_50": "label", "AreaShape_Center_X": "Center_X",
                                                  "AreaShape_Center_Y": "Center_Y",
                                                  "AreaShape_MajorAxisLength": "Major_axis",
                                                  "AreaShape_MinorAxisLength": "Minor_axis"})

    final_data_frame = final_data_frame[["TimeStep", 'ObjectNumber', "id", "Orientation", "label", "Center_X",
                                         "Center_Y", "Major_axis", "InstantaneousGrowthRate", "InstantaneousVelocity",
                                         "divideFlag",
                                         "Major_axis", "Minor_axis", "parent_id",
                                         "TrackObjects_ParentImageNumber_50", "TrackObjects_ParentObjectNumber_50"]]
    # Convert the results dictionary to a DataFrame.
    results = pd.DataFrame.from_dict(result_dict, orient="index").transpose()

    return final_data_frame, results
