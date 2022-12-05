from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from CellProfilerAnalysis.strain.ProcessCellProfilerData import process_data


def start_processing():
    msg = messagebox.showinfo("Processing is Started", "Processing is Started")
    print("interval time: " + str(interval_time.get()))
    print("1 pixel is equal to: " + str(um_per_pixel.get()))
    print("growth rate method" + str(growth_rate_method_value.get()))
    print("input file: " + input_file)
    if CheckVar1.get() == 1:
        output_directory_path_as_input_folder = '/'.join(input_file.split('/')[:-1])
        print("output directory: " + output_directory_path_as_input_folder)
        # start processing csv file
        process_data(input_file, output_directory_path_as_input_folder, interval_time=float(interval_time.get()),
                     growth_rate_method=growth_rate_method_value.get(), um_per_pixel=float(um_per_pixel.get()))

        msg = messagebox.showinfo('Analysis were done', "Analysis were done")
        msg = messagebox.showinfo('The calculation files were written into ' + output_directory_path_as_input_folder,
                                  "The calculation files were written into " + output_directory_path_as_input_folder)

    else:
        print("output directory: " + output_directory_path)
        # start processing csv file
        process_data(input_file, output_directory_path, interval_time=float(interval_time.get()),
                     growth_rate_method=growth_rate_method_value.get(), um_per_pixel=float(um_per_pixel.get()))

        msg = messagebox.showinfo("Analysis were done", "Analysis were done")
        msg = messagebox.showinfo("The calculation files were written into " + output_directory_path,
                                  "The calculation files were written into " + output_directory_path)


def browse_files_func():
    # define global variable
    global input_file
    input_file = filedialog.askopenfilename(initialdir="/", title="Select CellProfiler tracking file",
                                            filetypes=(("CSV files", "*.CSV"),))
    # Change label contents
    browse_value.configure(text=input_file.split("/")[-1].split("\\")[-1])
    input_file = input_file


def browse_directory_func():
    # define global variable
    global output_directory_path
    output_directory = filedialog.askdirectory()
    # Change label contents
    browse_directory_value.configure(text=output_directory)
    output_directory_path = output_directory


# Change the label text
def show():
    label.config(text=system_type_value.get())


if __name__ == "__main__":
    top = Tk()
    top.geometry("700x300")
    top.title("Image Processing")

    # interval Time
    interval_time_lable = Label(top, text="Interval Time: ")
    interval_time_lable.pack(side=LEFT)
    interval_time_lable.place(x=30, y=40)
    interval_time = Entry(top)
    interval_time.insert(1, '1')
    interval_time.pack(side=LEFT)
    interval_time.place(x=130, y=40)

    # growth rate method
    # Dropdown menu options
    options_growth_rate_method = ["Average", "Linear Regression"]
    # datatype of menu text
    growth_rate_method_value = StringVar()
    # initial menu text
    growth_rate_method_value.set("Average")
    growth_rate_method_label = Label(top, text="Growth rate method:")
    growth_rate_method_label.pack(side=LEFT)
    growth_rate_method_label.place(x=350, y=40)
    growth_rate_method = OptionMenu(top, growth_rate_method_value, *options_growth_rate_method)
    growth_rate_method.pack()
    growth_rate_method.place(x=530, y=35)

    # browse File
    browse_file_label = Label(top, text="Browse File:")
    browse_file_label.pack(side=LEFT)
    browse_file_label.place(x=30, y=100)
    button_browse_file = Button(top, text="CellProfiler tracking file", command=browse_files_func)
    button_browse_file.pack()
    button_browse_file.place(x=130, y=95)
    browse_value = Label(top, text="")
    browse_value.pack(side=LEFT)
    browse_value.place(x=130, y=135)

    # browse directory
    browse_directory_label = Label(top, text="Browse Output Directory:")
    browse_directory_label.pack(side=LEFT)
    browse_directory_label.place(x=350, y=100)

    button_browse_directory = Button(top, text="Output directory", command=browse_directory_func)
    button_browse_directory.pack()
    button_browse_directory.place(x=530, y=95)
    browse_directory_value = Label(top, text="")
    browse_directory_value.pack(side=LEFT)
    browse_directory_value.place(x=350, y=135)

    # check box
    CheckVar1 = IntVar()
    checkBox_output_directory = Checkbutton(top, text="Use input folder as output folder ", variable=CheckVar1,
                                            onvalue=1, offvalue=0)
    checkBox_output_directory.pack()
    checkBox_output_directory.place(x=115, y=160)

    # Convert distances to um
    um_per_pixel_label = Label(top, text="1 pixel is equal to: ")
    um_per_pixel_label.pack(side=LEFT)
    um_per_pixel_label.place(x=30, y=200)
    um_per_pixel = Entry(top)
    um_per_pixel.insert(1, '0.144')
    um_per_pixel.pack(side=LEFT)
    um_per_pixel.place(x=160, y=200)
    # continuation
    continuation_um_per_pixel_label = Label(top, text="um")
    continuation_um_per_pixel_label.pack(side=LEFT)
    continuation_um_per_pixel_label.place(x=330, y=200)
    continuation_um_per_pixel_label = Entry(top)

    # start process button
    start_process = Button(top, text="Start Processing", command=start_processing)
    start_process.place(x=250, y=250)

    top.mainloop()
