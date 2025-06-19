import tkinter as tk
from tkinter import filedialog, colorchooser
import tkinter.messagebox as messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as mpatches
import pandas as pd
import cv2
import numpy as np
import glob
from Trackrefiner import calculate_bac_endpoints, extract_bacteria_info
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


class TrackingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CP-PostProcessing Bacterial Tracking Visualization")

        self.geometry("450x350")

        self.frame = tk.Frame(self)
        self.frame.pack(expand=True)

        # CSV file selection
        self.open_file_frame = tk.Frame(self.frame)
        self.open_file_frame.pack(side='top', pady=(0, 0))

        self.btn_open_file = tk.Button(self.open_file_frame, text="Select CP-PostProcessing CSV Output File",
                                       command=self.load_csv_file)
        self.btn_open_file.pack(side='top', pady=(0, 0))

        # Raw image folder selection
        self.open_images_frame = tk.Frame(self.frame)
        self.open_images_frame.pack(side='top', pady=(10, 10))

        self.btn_open_images = tk.Button(self.open_images_frame, text="Select Raw Images Directory",
                                         command=self.load_images_dir)
        self.btn_open_images.pack(side='top', pady=(5, 0))

        # umPerPixel entry
        self.um_per_pixel_frame = tk.Frame(self.frame)
        self.um_per_pixel_frame.pack(side='top', pady=(10, 10))

        self.label_um_per_pixel = tk.Label(self.um_per_pixel_frame, text="umPerPixel:")
        self.label_um_per_pixel.pack(side='left', pady=(0, 0))
        self.entry_um_per_pixel = tk.Entry(self.um_per_pixel_frame)
        self.entry_um_per_pixel.insert(0, "0.144")  # Default value of 0.144
        self.entry_um_per_pixel.pack(side='left', pady=(0, 0))

        # Font size entry
        self.font_size_frame = tk.Frame(self.frame)
        self.font_size_frame.pack(side='top', pady=(10, 10))

        self.label_font_size = tk.Label(self.font_size_frame, text="Font Size for Labels:")
        self.label_font_size.pack(side='left', pady=(0, 0))
        self.entry_font_size = tk.Entry(self.font_size_frame)
        self.entry_font_size.insert(0, "12")  # Default value of 12
        self.entry_font_size.pack(side='left', pady=(0, 0))

        # Object color picker
        self.label_color_frame = tk.Frame(self.frame)
        self.label_color_frame.pack(side='top', pady=(10, 0))

        self.label_color = tk.Label(self.label_color_frame, text="Pick Color for Object:")
        self.label_color.pack(side='left', pady=(0, 0))
        self.color_button = tk.Button(self.label_color_frame, text="Select Color", command=self.select_color)
        self.color_button.pack(side='left', pady=(0, 0))

        # Show all time steps button
        self.all_button_frame = tk.Frame(self.frame)
        self.all_button_frame.pack(side='top', pady=(10, 0))
        self.show_all_button = tk.Button(self.all_button_frame, text="Show All Time Steps",
                                         command=self.show_all_time_steps)
        self.show_all_button.pack(side='top', pady=(10, 0))

        # Button to show two specific time steps
        self.two_steps_button_frame = tk.Frame(self.frame)
        self.two_steps_button_frame.pack(side='top', pady=(10, 0))
        self.show_two_steps_button = tk.Button(self.two_steps_button_frame, text="Show Two Specific Time Steps",
                                               command=self.show_two_steps)
        self.show_two_steps_button.pack(side='top', pady=(10, 0))

        # Variables for file paths
        self.csv_file = None
        self.csv_neighbor_file = None
        self.raw_images_dir = None

        # Conversion factors and settings
        self.um_per_pixel = 0.144
        self.object_color = "#56e64e"
        self.font_size = 12

        # List to track opened windows
        self.open_windows = []

        self.protocol("WM_DELETE_WINDOW", self.on_main_window_close)

    def load_csv_file(self):
        self.csv_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")],
                                                   title="Select CSV file output of CP-PostProcessing")
        print(f"Loaded CSV file: {self.csv_file}")

    def load_neighbor_csv_file(self, parent_window):
        self.csv_neighbor_file = filedialog.askopenfilename(parent=parent_window, filetypes=[("CSV files", "*.csv")],
                                                            title="Select CSV file containing neighboring data of"
                                                                  " bacteria")
        print(f"Loaded CSV file containing neighboring data of bacteria: {self.csv_neighbor_file}")

    def load_images_dir(self):
        self.raw_images_dir = filedialog.askdirectory(title="Select Raw Images Directory")
        print(f"Loaded Raw Images Directory: {self.raw_images_dir}")

    def select_color(self):
        color_code = colorchooser.askcolor(title="Choose Object Color")
        if color_code:
            self.object_color = color_code[1]  # Get the hexadecimal color code
            print(f"Selected color: {self.object_color}")

    def update_font_size(self):
        try:
            self.font_size = int(self.entry_font_size.get())  # Get font size from input and convert to integer
        except ValueError:
            self.font_size = 12  # Set a default value if input is invalid

    def update_um_per_pixel(self):
        try:
            self.um_per_pixel = float(self.entry_um_per_pixel.get())  # Get font size from input and convert to float
        except ValueError:
            self.um_per_pixel = 0.144  # Set a default value if input is invalid

    def correction_tracking(self):
        # Open a new window to specify time step 1, time step 2, ID, parent ID, and neighbor distance
        new_window = tk.Toplevel(self)
        new_window.title("CP-PostProcessing Tracking Correction")
        new_window.geometry("400x150")

        # Time step entry
        t_frame = tk.Frame(new_window)
        t_frame.pack(side='top', pady=(10, 0))

        label_t = tk.Label(t_frame, text="Time Step:")
        label_t.pack(side='left', pady=(0, 0))
        entry_t = tk.Entry(t_frame)
        entry_t.pack(side='left', pady=(0, 0))

        # ID entry
        id_frame = tk.Frame(new_window)
        id_frame.pack(side='top', pady=(10, 0))
        label_id = tk.Label(id_frame, text="Objects ID(s) (comma separated):")
        label_id.pack(side='left', pady=(0, 0))
        entry_id = tk.Entry(id_frame)
        entry_id.pack(side='left', pady=(0, 0))

        # Parent ID entry
        parent_id_frame = tk.Frame(new_window)
        parent_id_frame.pack(side='top', pady=(10, 0))
        label_parent_id = tk.Label(parent_id_frame, text="Parent ID(s) (comma separated):")
        label_parent_id.pack(side='left', pady=(0, 0))
        entry_parent_id = tk.Entry(parent_id_frame)
        entry_parent_id.pack(side='left', pady=(0, 0))

        # Submit button to handle input
        submit_frame = tk.Frame(new_window)
        submit_frame.pack(side='top', pady=(10, 0))
        submit_button = tk.Button(submit_frame, text="Correction",
                                  command=lambda: self.do_correction(entry_t.get(), entry_id.get(),
                                                                     entry_parent_id.get(), new_window))
        submit_button.pack()

    def show_two_steps(self):

        self.update_font_size()  # Update font size before performing actions
        self.update_um_per_pixel()

        # Open a new window to specify time step 1, time step 2, ID, parent ID, and neighbor distance
        new_window = tk.Toplevel(self)
        new_window.title("CP-PostProcessing Specify Time Steps and Settings")
        new_window.geometry("400x320")

        # CSV neighbor file selection
        open_neighbor_file_frame = tk.Frame(new_window)
        open_neighbor_file_frame.pack(side='top', pady=(10, 0))

        btn_open_neighbor_file = tk.Button(open_neighbor_file_frame,
                                           text="Select CSV file containing neighboring data of bacteria",
                                           command=lambda: self.load_neighbor_csv_file(new_window))
        btn_open_neighbor_file.pack(side='top', pady=(0, 0))

        # Time step 1 entry
        t1_frame = tk.Frame(new_window)
        t1_frame.pack(side='top', pady=(10, 0))

        label_t1 = tk.Label(t1_frame, text="Time Step 1:")
        label_t1.pack(side='left', pady=(0, 0))
        entry_t1 = tk.Entry(t1_frame)
        entry_t1.pack(side='left', pady=(0, 0))

        # Time step 2 entry
        t2_frame = tk.Frame(new_window)
        t2_frame.pack(side='top', pady=(10, 0))
        label_t2 = tk.Label(t2_frame, text="Time Step 2:")
        label_t2.pack(side='left', pady=(0, 0))
        entry_t2 = tk.Entry(t2_frame)
        entry_t2.pack(side='left', pady=(0, 0))

        # Object number entry
        obj_num_frame = tk.Frame(new_window)
        obj_num_frame.pack(side='top', pady=(10, 0))
        label_obj_num = tk.Label(obj_num_frame, text="Objects Number(s) (comma separated):")
        label_obj_num.pack(side='left', pady=(0, 0))
        entry_obj_num = tk.Entry(obj_num_frame)
        entry_obj_num.pack(side='left', pady=(0, 0))

        # ID entry
        id_frame = tk.Frame(new_window)
        id_frame.pack(side='top', pady=(10, 0))
        label_id = tk.Label(id_frame, text="Objects ID(s) (comma separated):")
        label_id.pack(side='left', pady=(0, 0))
        entry_id = tk.Entry(id_frame)
        entry_id.pack(side='left', pady=(0, 0))

        # Parent ID entry
        parent_id_frame = tk.Frame(new_window)
        parent_id_frame.pack(side='top', pady=(10, 0))
        label_parent_id = tk.Label(parent_id_frame, text="Parent ID(s) (comma separated):")
        label_parent_id.pack(side='left', pady=(0, 0))
        entry_parent_id = tk.Entry(parent_id_frame)
        entry_parent_id.pack(side='left', pady=(0, 0))

        # Parent ID entry
        id_not_to_see_frame = tk.Frame(new_window)
        id_not_to_see_frame.pack(side='top', pady=(10, 0))
        label_id_not_to_see = tk.Label(id_not_to_see_frame, text="ID(s) (comma separated) dont wat to see:")
        label_id_not_to_see.pack(side='left', pady=(0, 0))
        entry_id_not_to_see = tk.Entry(id_not_to_see_frame)
        entry_id_not_to_see.pack(side='left', pady=(0, 0))

        # Neighbor distance entry
        distance_frame = tk.Frame(new_window)
        distance_frame.pack(side='top', pady=(10, 0))
        label_distance = tk.Label(distance_frame, text="Neighbor Distance (unit: pixel):")
        label_distance.pack(side='left', pady=(0, 0))
        entry_distance = tk.Entry(distance_frame)
        entry_distance.pack(side='left', pady=(0, 0))

        # Dropdown menu for selecting the visualization option
        vis_mode_frame = tk.Frame(new_window)
        vis_mode_frame.pack(side='top', pady=(10, 0))
        label_option = tk.Label(vis_mode_frame, text="Visualization Mode:")
        label_option.pack(side='left', pady=(0, 0))

        selected_option = tk.StringVar(vis_mode_frame)
        selected_option.set("Two Different Slides")  # Default value

        option_menu = tk.OptionMenu(vis_mode_frame, selected_option, "Two Different Slides", "Slide Show")
        option_menu.pack(side='left', pady=(0, 0))

        # Submit button to handle input
        submit_frame = tk.Frame(new_window)
        submit_frame.pack(side='top', pady=(10, 0))
        submit_button = tk.Button(submit_frame, text="Show",
                                  command=lambda: self.handle_two_steps(entry_t1.get(), entry_t2.get(),
                                                                        entry_obj_num.get(), entry_id.get(),
                                                                        entry_parent_id.get(),
                                                                        entry_id_not_to_see.get(),
                                                                        entry_distance.get(), selected_option.get(),
                                                                        self.csv_neighbor_file,
                                                                        new_window))
        submit_button.pack()

    def handle_two_steps(self, t1, t2, object_numbers, ids, parent_ids, ids_not_to_see, distance, mode,
                         csv_neighbor_file, window):
        # Logic to handle the specified input and show the visualizations
        print(f"Time Step 1: {t1}, Time Step 2: {t2}, Object Numbers: {object_numbers}, IDs: {ids}, "
              f"Parent IDs: {parent_ids}, Distance: {distance}, Mode: {mode}, "
              f"\nCSV file containing neighboring data of bacteria: {csv_neighbor_file}")
        # window.destroy()
        # Use this data to display images based on user input
        self.display_images(t1, t2, object_numbers, ids, parent_ids, ids_not_to_see, distance, mode, csv_neighbor_file)

    def display_images(self, timestep1, timestep2, specified_object_numbers, specified_ids_str,
                       specified_parent_ids_str, specified_ids_not_to_see, neighbor_distance_str,
                       selected_mode, csv_neighbor_file):

        # Parse the object Numbers, specified IDs, and Parent IDs

        object_numbers = specified_object_numbers.replace(" ", "").split(',') if specified_object_numbers else []
        object_numbers = [int(x) for x in object_numbers if x.isdigit()]  # Convert to list of integers

        specified_ids = specified_ids_str.replace(" ", "").split(',') if specified_ids_str else []
        specified_ids = [int(x) for x in specified_ids if x.isdigit()]  # Convert to list of integers

        specified_parent_ids = specified_parent_ids_str.replace(" ", "").split(',') if specified_parent_ids_str else []
        specified_parent_ids = [int(x) for x in specified_parent_ids if x.isdigit()]  # Convert to list of integers

        specified_ids_not_to_see = specified_ids_not_to_see.replace(" ", "").split(',') if specified_ids_not_to_see else []
        specified_ids_not_to_see = [int(x) for x in specified_ids_not_to_see if x.isdigit()]  # Convert to list of integers

        # Parse the neighbor distance if provided
        try:
            neighbor_distance = float(neighbor_distance_str) if neighbor_distance_str else None
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for neighbor distance.")
            return

        # Load the data
        if not self.csv_file:
            messagebox.showerror("No CSV File", "Please select a CSV file.")
            return

        df = pd.read_csv(self.csv_file)
        df['index'] = df.index.values

        if self.csv_neighbor_file is not None:
            df_neighbor = pd.read_csv(self.csv_neighbor_file)
            df_neighbor = df_neighbor.loc[df_neighbor['Relationship'] == 'Neighbors']
        else:
            df_neighbor = pd.DataFrame()

        # Get the unique time steps from the dataset
        available_time_steps = df['TimeStep'].unique()

        # Check if the time steps are valid
        if int(timestep1) not in available_time_steps or int(timestep2) not in available_time_steps:
            max_time_step = available_time_steps.max()
            messagebox.showerror("Invalid Time Step", f"Specified time step is incorrect. "
                                                      f"The dataset has time steps ranging from 1 to {max_time_step}.")
            return

        if not self.raw_images_dir:
            messagebox.showerror("No Image Directory", "Please select a raw images directory.")
            return

        raw_images = sorted(glob.glob(self.raw_images_dir + '/*.tif'))

        # Filter for timestep 1 and 2
        df_t1 = df[df['TimeStep'] == int(timestep1)].reset_index(drop=True)
        df_t2 = df[df['TimeStep'] == int(timestep2)].reset_index(drop=True)

        # Close any previously opened windows
        self.close_previous_windows()

        # Check the selected mode and display images accordingly
        if selected_mode == "Two Different Slides":
            # Display each time step in a separate window
            if specified_ids:
                self.show_image(df_t1, raw_images[int(timestep1) - 1], f"Time Step {timestep1}",
                                object_numbers, specified_ids.copy(), specified_parent_ids, specified_ids_not_to_see,
                                neighbor_distance,
                                df_neighbor)
                self.show_image(df_t2, raw_images[int(timestep2) - 1], f"Time Step {timestep2}",
                                object_numbers, specified_ids.copy(), specified_parent_ids, specified_ids_not_to_see,
                                neighbor_distance,
                                df_neighbor)
            else:
                self.show_image(df_t1, raw_images[int(timestep1) - 1], f"Time Step {timestep1}",
                                object_numbers, specified_ids, specified_parent_ids, specified_ids_not_to_see,
                                neighbor_distance, df_neighbor)
                self.show_image(df_t2, raw_images[int(timestep2) - 1], f"Time Step {timestep2}",
                                object_numbers, specified_ids, specified_parent_ids, specified_ids_not_to_see,
                                neighbor_distance, df_neighbor)

        elif selected_mode == "Slide Show":
            # Display both time steps in a single window with slider control
            self.show_sliding_image(df_t1, df_t2, raw_images[int(timestep1) - 1], raw_images[int(timestep2) - 1],
                                    object_numbers, specified_ids, specified_parent_ids, specified_ids_not_to_see,
                                    neighbor_distance,
                                    timestep1, timestep2, df_neighbor)

    def show_image(self, df_current, image_path, window_title, specified_object_numbers=None, specified_ids=None,
                   specified_parent_ids=None, specified_ids_not_to_see=None, neighbor_distance=None, df_neighbor=None):

        # Create a new window for the plot
        img_window = tk.Toplevel(self)
        img_window.title(window_title)

        # Keep track of this window
        self.open_windows.append(img_window)

        # Configure the window to resize
        img_window.grid_rowconfigure(0, weight=1)
        img_window.grid_columnconfigure(0, weight=1)

        # Create a frame to hold the canvas and toolbar
        frame = tk.Frame(img_window)
        frame.grid(row=0, column=0, sticky='nsew')

        # Read the image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create a figure
        fig, ax = plt.subplots()

        # Display the raw image
        ax.imshow(img_rgb)

        # Extract objects and plot as per the logic from your existing utils
        objects_center_coord_x, objects_center_coord_y, objects_major_current, objects_orientation_current = \
            extract_bacteria_info(df_current, self.um_per_pixel, {'x': 'Center_X', 'y': 'Center_Y'},
                                  "Major_axis", "Orientation")

        if specified_object_numbers or specified_ids or specified_parent_ids or specified_ids_not_to_see:

            selected_bac_by_user = df_current.loc[(df_current['ObjectNumber'].isin(specified_object_numbers)) |
                                                  (df_current['id'].isin(specified_ids)) |
                                                  (df_current['parent_id'].isin(specified_parent_ids))]

            if df_neighbor.shape[0] > 0:
                selected_bac_with_neighbors = \
                    selected_bac_by_user.merge(df_neighbor, left_on=['TimeStep', 'ObjectNumber'],
                                               right_on=['First Image Number', 'First Object Number'], how='left')
                selected_bac_with_neighbors_with_info = \
                    selected_bac_with_neighbors.merge(df_current,
                                                      left_on=['Second Image Number', 'Second Object Number'],
                                                      right_on=['TimeStep', 'ObjectNumber'], how='left',
                                                      suffixes=('_1', '_2'))

                neighbors_bac_id = [v for v in selected_bac_with_neighbors_with_info['id_2'].values.tolist() if
                                    str(v) != 'nan']
                specified_ids.extend(neighbors_bac_id)

            # Check if the object is within the neighbor distance from the specified objects
            if neighbor_distance:

                other_bac = df_current.loc[~ df_current['id'].isin(selected_bac_by_user['id'].values)]

                merged_df = selected_bac_by_user.merge(other_bac, how='cross', suffixes=('_1', '_2'))

                dist_cond = (
                        np.sqrt(np.power((merged_df['Center_X_1'] -
                                          merged_df['Center_X_2']), 2) +
                                np.power((merged_df['Center_Y_1'] -
                                          merged_df['Center_Y_2']), 2)) <= neighbor_distance)

                neighbors_bac_id = merged_df[dist_cond]['id_2'].values.tolist()
                specified_ids.extend(neighbors_bac_id)

        # Filter objects by ID, Parent ID, or distance
        print("===================== Time Step: " + str(df_current['TimeStep'].values[0]) + '=====================')
        for cell_indx in range(df_current.shape[0]):
            cell_obj_num = df_current.iloc[cell_indx]['ObjectNumber']
            cell_id = df_current.iloc[cell_indx]['id']
            parent_id = df_current.iloc[cell_indx]['parent_id']
            center_current = (objects_center_coord_x[cell_indx], objects_center_coord_y[cell_indx])

            # Skip if the object doesn't match the specified IDs or Parent IDs
            if specified_ids and specified_parent_ids and specified_object_numbers:
                if ((cell_id not in specified_ids) and (parent_id not in specified_parent_ids) and
                        (cell_obj_num not in specified_object_numbers)):
                    continue
            elif specified_ids and specified_parent_ids:
                if (cell_id not in specified_ids) and (parent_id not in specified_parent_ids):
                    continue
            elif specified_ids and specified_object_numbers:
                if (cell_id not in specified_ids) and (cell_obj_num not in specified_object_numbers):
                    continue
            elif specified_object_numbers and specified_parent_ids:
                if (cell_obj_num not in specified_object_numbers) and (parent_id not in specified_parent_ids):
                    continue
            elif specified_ids:
                if cell_id not in specified_ids:
                    continue
            elif specified_parent_ids:
                if parent_id not in specified_parent_ids:
                    continue
            elif specified_object_numbers:
                if cell_obj_num not in specified_object_numbers:
                    continue
            if specified_ids_not_to_see:
                if cell_id in specified_ids_not_to_see:
                    continue

            print("Object Number: " + str(cell_obj_num) + " Id: " + str(cell_id) + " parent id: " + str(parent_id))

            # Plot bacteria and labels
            major_current = (objects_major_current.iloc[cell_indx]) / 2
            angle_current = objects_orientation_current.iloc[cell_indx]
            ends = calculate_bac_endpoints([center_current[0], center_current[1]], major_current, angle_current)

            # Endpoints of the major axis of the ellipse
            node_x1_x_current = ends[0][0]
            node_x1_y_current = ends[0][1]
            node_x2_x_current = ends[1][0]
            node_x2_y_current = ends[1][1]

            ax.plot([node_x1_x_current, node_x2_x_current], [node_x1_y_current, node_x2_y_current], lw=1,
                    solid_capstyle="round", color=self.object_color)

            # Add cell ID and parent ID as text
            pos1x = np.abs(node_x1_x_current + center_current[0]) / 2
            pos2x = np.abs(node_x2_x_current + center_current[0]) / 2

            pos1y = np.abs(node_x1_y_current + center_current[1]) / 2
            pos2y = np.abs(node_x2_y_current + center_current[1]) / 2

            final_pos1x = np.abs(pos1x + center_current[0]) / 2
            final_pos2x = np.abs(pos2x + center_current[0]) / 2

            final_pos1y = np.abs(pos1y + center_current[1]) / 2
            final_pos2y = np.abs(pos2y + center_current[1]) / 2

            # Add ID and Parent ID as text on the object
            ax.text(final_pos1x, final_pos1y, cell_id, fontsize=self.font_size, color="#ff0000")
            ax.text(final_pos2x, final_pos2y, parent_id, fontsize=self.font_size, color="#0000ff")

        # Add legend inside the window
        parent_patch = mpatches.Patch(color='#0000ff', label='parent')
        id_patch = mpatches.Patch(color='#ff0000', label='identity id')
        plt.legend(handles=[parent_patch, id_patch], loc='upper center', ncol=6,
                   bbox_to_anchor=(.5, 1.1), prop={'size': 7})

        # Embed the figure into the Tkinter window with zoom functionality
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar for zooming
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        toolbar.pack()

        # Resize image dynamically with the window
        fig.tight_layout()

    def show_sliding_image(self, df_t1, df_t2, image_path_t1, image_path_t2, specified_object_numbers=None,
                           specified_ids=None, specified_parent_ids=None, specified_ids_not_to_see= None,
                           neighbor_distance=None, timestep1=None,
                           timestep2=None, df_neighbor=None):
        # Create a new window for the sliding plot
        slide_window = tk.Toplevel(self)
        slide_window.title("CP-PostProcessing Slider Control")

        # Keep track of this window
        self.open_windows.append(slide_window)

        # Configure the window to resize
        slide_window.grid_rowconfigure(0, weight=1)
        slide_window.grid_columnconfigure(0, weight=1)

        # Create a frame to hold the canvas and another frame for the toolbar
        canvas_frame = tk.Frame(slide_window)
        toolbar_frame = tk.Frame(slide_window)

        canvas_frame.grid(row=0, column=0, sticky='nsew')
        toolbar_frame.grid(row=1, column=0, sticky='ew')

        # Make sure the canvas_frame and canvas expand when the window is resized
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Read the images for both timesteps
        img_t1 = cv2.imread(image_path_t1)
        img_t1_rgb = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)

        img_t2 = cv2.imread(image_path_t2)
        img_t2_rgb = cv2.cvtColor(img_t2, cv2.COLOR_BGR2RGB)

        # Create a figure and adjust it to fit the canvas
        fig, ax = plt.subplots()

        # Label for showing time step name dynamically
        time_step_label = tk.Label(slide_window, text=f"Time Step {timestep1}")
        time_step_label.grid(row=3, column=0, sticky='ew')

        # Variables to store zoom level and position
        current_xlim = None
        current_ylim = None

        # Function to update the image based on slider value
        def update_image(val):

            nonlocal current_xlim, current_ylim

            time_step = int(val)

            # Store the current zoom level if it's the first time switching
            if current_xlim is None and current_ylim is None:
                current_xlim = ax.get_xlim()
                current_ylim = ax.get_ylim()

            ax.clear()

            # Choose image and dataframe based on slider value
            if time_step == 1:
                current_img = img_t1_rgb
                current_df = df_t1
                title = "CP-PostProcessing Time Step " + str(timestep1)
                time_step_label.config(text=f"Time Step {timestep1}")  # Update label to show timestep1
            else:
                current_img = img_t2_rgb
                current_df = df_t2
                title = "CP-PostProcessing Time Step " + str(timestep2)
                time_step_label.config(text=f"Time Step {timestep2}")  # Update label to show timestep2

            ax.imshow(current_img)
            # ax.set_title(title)

            # Plot bacteria objects (if specified) for the current time step
            if specified_ids:
                self.plot_bacteria_on_image(ax, current_df, specified_object_numbers, specified_ids.copy(),
                                            specified_parent_ids, specified_ids_not_to_see, neighbor_distance, df_neighbor)
            else:
                self.plot_bacteria_on_image(ax, current_df, specified_object_numbers, specified_ids,
                                            specified_parent_ids, specified_ids_not_to_see,
                                            neighbor_distance, df_neighbor)

            # Reapply the stored zoom level
            ax.set_xlim(current_xlim)
            ax.set_ylim(current_ylim)

            canvas.draw()

        # Display the initial image (time step 1)
        ax.imshow(img_t1_rgb)
        # ax.set_title("Time Step 1")

        # Plot bacteria objects for time step 1
        if specified_ids:
            self.plot_bacteria_on_image(ax, df_t1, specified_object_numbers, specified_ids.copy(), specified_parent_ids,
                                        specified_ids_not_to_see, neighbor_distance, df_neighbor)
        else:
            self.plot_bacteria_on_image(ax, df_t1, specified_object_numbers, specified_ids, specified_parent_ids,
                                        specified_ids_not_to_see, neighbor_distance, df_neighbor)

        # Embed the figure into the Tkinter window with zoom functionality
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Configure resizing behavior for the figure
        def on_resize(event):
            # Adjust figure layout on resize
            fig.tight_layout()
            canvas.draw()

        # Bind the resize event to dynamically update the layout
        canvas.get_tk_widget().bind("<Configure>", on_resize)

        # Add toolbar for zooming, in the separate toolbar frame
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Create a slider widget to switch between the two time steps, using grid instead of pack
        slider = tk.Scale(slide_window, from_=1, to=2, orient=tk.HORIZONTAL, label="Slide", command=update_image)
        slider.grid(row=2, column=0, sticky='ew')

        # Allow the slider to resize along the X-axis
        slide_window.grid_rowconfigure(2, weight=0)
        slide_window.grid_columnconfigure(0, weight=1)

    def plot_bacteria_on_image(self, ax, df_current, specified_object_numbers, specified_ids, specified_parent_ids,
                               specified_ids_not_to_see, neighbor_distance, df_neighbor):

        # Extract information about the bacteria: center coordinates, major axis length, and orientation
        objects_center_coord_x, objects_center_coord_y, objects_major_current, objects_orientation_current = \
            extract_bacteria_info(df_current, self.um_per_pixel, {'x': 'Center_X', 'y': 'Center_Y'},
                                  "Major_axis", "Orientation")

        if specified_object_numbers or specified_ids or specified_parent_ids:

            selected_bac_by_user = df_current.loc[(df_current['ObjectNumber'].isin(specified_object_numbers)) |
                                                  (df_current['id'].isin(specified_ids)) |
                                                  (df_current['parent_id'].isin(specified_parent_ids))]

            if df_neighbor.shape[0] > 0:
                selected_bac_with_neighbors = \
                    selected_bac_by_user.merge(df_neighbor, left_on=['TimeStep', 'ObjectNumber'],
                                               right_on=['First Image Number', 'First Object Number'], how='left')
                selected_bac_with_neighbors_with_info = \
                    selected_bac_with_neighbors.merge(df_current,
                                                      left_on=['Second Image Number', 'Second Object Number'],
                                                      right_on=['TimeStep', 'ObjectNumber'], how='left',
                                                      suffixes=('_1', '_2'))

                neighbors_bac_id = [v for v in selected_bac_with_neighbors_with_info['id_2'].values.tolist() if
                                    str(v) != 'nan']
                specified_ids.extend(neighbors_bac_id)

            # Check if the object is within the neighbor distance from the specified objects
            if neighbor_distance:

                other_bac = df_current.loc[~ df_current['id'].isin(selected_bac_by_user['id'].values)]

                merged_df = selected_bac_by_user.merge(other_bac, how='cross', suffixes=('_1', '_2'))

                dist_cond = (
                        np.sqrt(np.power((merged_df['Center_X_1'] -
                                          merged_df['Center_X_2']), 2) +
                                np.power((merged_df['Center_Y_1'] -
                                          merged_df['Center_Y_2']), 2)) <= neighbor_distance)

                neighbors_bac_id = merged_df[dist_cond]['id_2'].values.tolist()
                specified_ids.extend(neighbors_bac_id)

        # Loop through each bacteria object and plot it
        print("===================== Time Step: " + str(df_current['TimeStep'].values[0]) + '=====================')
        for cell_indx in range(df_current.shape[0]):
            cell_obj_num = df_current.iloc[cell_indx]['ObjectNumber']
            cell_id = df_current.iloc[cell_indx]['id']
            parent_id = df_current.iloc[cell_indx]['parent_id']
            center_current = (objects_center_coord_x[cell_indx], objects_center_coord_y[cell_indx])

            # Skip objects that don't match the specified IDs or Parent IDs
            if specified_ids and specified_parent_ids and specified_object_numbers:
                if ((cell_id not in specified_ids) and (parent_id not in specified_parent_ids) and
                        (cell_obj_num not in specified_object_numbers)):
                    continue
            elif specified_ids and specified_parent_ids:
                if (cell_id not in specified_ids) and (parent_id not in specified_parent_ids):
                    continue
            elif specified_ids and specified_object_numbers:
                if (cell_id not in specified_ids) and (cell_obj_num not in specified_object_numbers):
                    continue
            elif specified_object_numbers and specified_parent_ids:
                if (cell_obj_num not in specified_object_numbers) and (parent_id not in specified_parent_ids):
                    continue
            elif specified_ids:
                if cell_id not in specified_ids:
                    continue
            elif specified_parent_ids:
                if parent_id not in specified_parent_ids:
                    continue
            elif specified_object_numbers:
                if cell_obj_num not in specified_object_numbers:
                    continue
            if specified_ids_not_to_see:
                if cell_id in specified_ids_not_to_see:
                    continue

            print("Object Number: " + str(cell_obj_num) + " Id: " + str(cell_id) + " parent id: " + str(parent_id))

            # Plot the bacteria's major axis as a line
            major_current = (objects_major_current.iloc[cell_indx]) / 2
            angle_current = objects_orientation_current.iloc[cell_indx]
            ends = calculate_bac_endpoints([center_current[0], center_current[1]], major_current, angle_current)

            # Endpoints of the major axis of the ellipse
            node_x1_x_current = ends[0][0]
            node_x1_y_current = ends[0][1]
            node_x2_x_current = ends[1][0]
            node_x2_y_current = ends[1][1]

            ax.plot([node_x1_x_current, node_x2_x_current], [node_x1_y_current, node_x2_y_current], lw=1,
                    solid_capstyle="round", color=self.object_color)

            # Add the bacteria's ID and parent ID as text labels
            pos1x = np.abs(node_x1_x_current + center_current[0]) / 2
            pos2x = np.abs(node_x2_x_current + center_current[0]) / 2

            pos1y = np.abs(node_x1_y_current + center_current[1]) / 2
            pos2y = np.abs(node_x2_y_current + center_current[1]) / 2

            final_pos1x = np.abs(pos1x + center_current[0]) / 2
            final_pos2x = np.abs(pos2x + center_current[0]) / 2

            final_pos1y = np.abs(pos1y + center_current[1]) / 2
            final_pos2y = np.abs(pos2y + center_current[1]) / 2

            # Add ID and Parent ID labels to the image
            ax.text(final_pos1x, final_pos1y, cell_id, fontsize=self.font_size, color="#ff0000")
            ax.text(final_pos2x, final_pos2y, parent_id, fontsize=self.font_size, color="#0000ff")

        # Add a legend for the ID and Parent ID colors
        parent_patch = mpatches.Patch(color='#0000ff', label='parent')
        id_patch = mpatches.Patch(color='#ff0000', label='identity id')
        ax.legend(handles=[parent_patch, id_patch], loc='upper center', ncol=6,
                  bbox_to_anchor=(.5, 1.1), prop={'size': 7})

    def show_all_time_steps(self):

        self.update_font_size()  # Update font size before performing actions
        self.update_um_per_pixel()

        # Close any previously opened windows
        self.close_previous_windows()

        # Load the data
        df = pd.read_csv(self.csv_file)

        # Get all available time steps from the dataset
        available_time_steps = df['TimeStep'].unique()

        raw_images = sorted(glob.glob(self.raw_images_dir + '/*.tif'))

        if not available_time_steps.size:
            messagebox.showerror("No Time Steps", "No time steps found in the dataset.")
            return

        # Create a new window for the sliding plot
        slide_window = tk.Toplevel(self)
        slide_window.title("CP-PostProcessing Show All Time Steps")

        # Keep track of this window
        self.open_windows.append(slide_window)

        # Create a frame to hold the canvas and toolbar
        canvas_frame = tk.Frame(slide_window)
        toolbar_frame = tk.Frame(slide_window)

        canvas_frame.grid(row=0, column=0, sticky='nsew')
        toolbar_frame.grid(row=1, column=0, sticky='ew')

        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Create a figure and adjust it to fit the canvas
        fig, ax = plt.subplots()

        # Label for showing time step name dynamically
        time_step_label = tk.Label(slide_window, text=f"Time Step {available_time_steps[0]}")
        time_step_label.grid(row=3, column=0, sticky='ew')

        # Variables to store zoom level and position
        current_xlim = None
        current_ylim = None

        # Function to update the image based on the slider value
        def update_image(val):

            nonlocal current_xlim, current_ylim

            time_step_index = int(val)
            timestep = available_time_steps[time_step_index]

            # Store the current zoom level if it's the first time switching
            if current_xlim is None and current_ylim is None:
                current_xlim = ax.get_xlim()
                current_ylim = ax.get_ylim()

            # Clear the axes
            ax.clear()

            # Update the time step label dynamically
            time_step_label.config(text=f"Time Step {timestep}")

            # Filter data for the current time step
            df_current = df[df['TimeStep'] == timestep].reset_index(drop=True)

            # Load the corresponding image
            img_path = raw_images[timestep - 1]
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax.imshow(img_rgb)
            self.plot_bacteria_on_image(ax, df_current, [], [], [],
                                        None, None, pd.DataFrame())

            # Reapply the stored zoom level
            ax.set_xlim(current_xlim)
            ax.set_ylim(current_ylim)

            # Draw the updated canvas
            canvas.draw()

        # Display the first image (corresponding to the first time step)
        first_time_step = available_time_steps[0]
        img = cv2.imread(raw_images[first_time_step - 1])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        # Plot bacteria objects for the first time step
        df_t1 = df[df['TimeStep'] == first_time_step].reset_index(drop=True)
        self.plot_bacteria_on_image(ax, df_t1, [], [], [],
                                    None, None, pd.DataFrame())

        # Embed the figure into the Tkinter window with zoom functionality
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Add toolbar for zooming
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        toolbar.pack()

        # Create a slider widget to switch between time steps
        slider = tk.Scale(slide_window, from_=0, to=len(available_time_steps) - 1, orient=tk.HORIZONTAL, label="Slide",
                          command=update_image)
        slider.grid(row=2, column=0, sticky='ew')

    def do_correction(self, t, ids, parent_ids, window):
        print(f"Time Step: {t}, IDs: {ids}, Parent IDs: {parent_ids}")

        # window.destroy()
        # Use this data to display images based on user input
        self.calc_tracking_features(t, ids, parent_ids)

    def calc_tracking_features(self, timestep, specified_ids_str, specified_parent_ids_str):

        # Parse the object Numbers, specified IDs, and Parent IDs
        specified_ids = specified_ids_str.replace(" ", "").split(',') if specified_ids_str else []
        specified_ids = [int(x) for x in specified_ids if x.isdigit()]  # Convert to list of integers

        specified_parent_ids = specified_parent_ids_str.replace(" ", "").split(',') if specified_parent_ids_str else []
        specified_parent_ids = [int(x) for x in specified_parent_ids if x.isdigit()]  # Convert to list of integers

        # Parse the neighbor distance if provided
        try:
            messagebox.showerror("Invalid Input", "Please enter a valid number for neighbor distance.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for neighbor distance.")
            return

    def close_previous_windows(self):
        """Close all previously opened windows."""
        for window in self.open_windows:
            if window.winfo_exists():  # Check if the window still exists before destroying it
                window.destroy()
        # Clear the list after closing all windows
        self.open_windows.clear()

    def check_if_all_closed(self):
        """Check if all windows are closed and close the main app if true."""
        open_windows_exist = any(window.winfo_exists() for window in self.open_windows)
        if not open_windows_exist:
            self.quit()

    def on_main_window_close(self):
        """Close all open windows and then close the main app."""
        # Close all the opened windows first
        self.close_previous_windows()
        # Quit the main app
        self.quit()


# Run the GUI application
if __name__ == "__main__":
    app = TrackingGUI()
    app.mainloop()
