import sys
from PyQt5.QtCore import Qt, QUrl, pyqtSignal, QObject
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, \
    QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit
from PyQt5.QtWidgets import QMessageBox
import os
from Trackrefiner import process_objects_data


class OutputStream(QObject):
    output_written = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.terminal = sys.__stdout__  # Keep the original stdout

    def write(self, text):
        self.output_written.emit(text)
        self.terminal.write(text)  # Also write to the terminal
        self.terminal.flush()  # Ensure immediate output to the terminal

    def flush(self):
        self.terminal.flush()  # Flush the original stdout


class TrackRefinerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # Redirect stdout to the log output
        self.stdout_stream = OutputStream()
        self.stdout_stream.output_written.connect(self.append_log)
        sys.stdout = self.stdout_stream

    def init_ui(self):
        self.setWindowTitle("Trackrefiner")
        self.layout = QVBoxLayout()

        self.layout.setSpacing(5)  # Consistent spacing between rows
        self.layout.setContentsMargins(10, 10, 10, 10)  # Consistent margins

        # Input file selection for CellProfiler Output CSV
        self.cp_output_csv_file = self.create_csv_file_input()
        self.add_tooltipped_widget("CellProfiler Output CSV File:",
                                   "Path to the CP-generated CSV file containing measured bacterial "
                                   "features and tracking information.", self.cp_output_csv_file)

        # Segmentation Results Folder
        self.segmentation_results_dir = self.create_folder_input()
        self.add_tooltipped_widget("Segmentation Results Folder:",
                                   "Path to folder containing .pickle files generated from "
                                   "segmentation by CellProfiler.", self.segmentation_results_dir)

        # Neighbor CSV
        self.neighbor_csv = self.create_csv_file_input()
        self.add_tooltipped_widget("Neighbor CSV File:", "Path to CSV file containing "
                                                         "bacterial neighbor information.", self.neighbor_csv)

        # Interval time and Doubling time in one row
        self.interval_time = self.create_double_spinbox(0.0, 1440.0, None)
        self.doubling_time_of_bacteria = self.create_double_spinbox(0.0, 1440.0, None)
        self.add_two_part_row(
            "Interval Time (minutes):",
            "Time interval between frames in minutes.",
            self.interval_time,
            "Doubling Time (minutes):",
            "Minimum lifespan of bacteria in minutes.",
            self.doubling_time_of_bacteria
        )

        # Elongation rate method and Pixel per micron in one row
        self.elongation_rate_method = self.create_combobox(["Average", "Linear Regression"])
        self.pixel_per_micron = self.create_double_spinbox(0.0, 10.0, 0.144)
        self.add_two_part_row(
            "Elongation Rate Method:",
            "Method to calculate elongation rate: Average or Linear Regression.",
            self.elongation_rate_method,
            "Pixel Per Micron:",
            "Conversion factor for pixels to micrometers.",
            self.pixel_per_micron
        )

        # Assign cell type and Intensity threshold in one row
        self.assigning_cell_type = self.create_checkbox()
        self.assigning_cell_type.setChecked(False)  # Default to unchecked
        self.assigning_cell_type.stateChanged.connect(self.toggle_intensity_threshold_visibility)

        self.intensity_threshold_layout = QHBoxLayout()
        self.add_tooltipped_widget_inline(
            "Assign Cell Type:",
            "Check to assign cell type to objects.",
            self.assigning_cell_type,
            self.intensity_threshold_layout
        )

        self.intensity_threshold = self.create_double_spinbox(0.0, 1.0, 0.1)
        self.intensity_threshold_label = QLabel("Intensity Threshold:")
        self.intensity_threshold_label.setToolTip("Threshold for cell intensity used in cell type assignment.")

        self.intensity_threshold_layout.addWidget(self.intensity_threshold_label)
        self.intensity_threshold_layout.addWidget(self.intensity_threshold)
        self.intensity_threshold_label.hide()
        self.intensity_threshold.hide()
        self.layout.addLayout(self.intensity_threshold_layout)

        # Classifier and Number of CPUs in one row
        self.clf = self.create_combobox([
            "LogisticRegression", "GaussianProcessClassifier", "C-Support Vector Classifier"
        ])
        self.n_cpu = self.create_spinbox(-1, 64, -1)
        self.add_two_part_row(
            "Classifier:",
            "Classifier for track refining.",
            self.clf,
            "Number of CPUs:",
            "Number of CPUs for parallel processing. Use -1 for all available CPUs.",
            self.n_cpu
        )

        # Boundary limits
        self.boundary_limits = self.create_line_edit()
        self.boundary_limits.setPlaceholderText("e.g., 0, 112, 52, 323 means X: 0–112, Y: 52–323")
        self.add_tooltipped_widget(
            "Boundary Limits:",
            "Define boundary limits to exclude objects outside the image boundary.",
            self.boundary_limits
        )

        # Dynamic boundaries
        self.dynamic_boundaries = self.create_csv_file_input()
        self.add_tooltipped_widget(
            "Dynamic Boundaries:",
            "Define time-dependent boundary limits using a CSV file.",
            self.dynamic_boundaries
        )

        # Disable tracking correction and Verbose Output in one row
        self.disable_tracking_correction = self.create_checkbox()
        self.verbose = self.create_checkbox()
        self.save_pickle = self.create_checkbox()
        self.add_three_part_row(
            "Disable Tracking Correction",
            "Check to disable tracking correction on CellProfiler output.",
            self.disable_tracking_correction,
            "Verbose Output",
            "Check to enable detailed log messages.",
            self.verbose,
            "Save in .pickle format",
            "Check to enable saving results in .pickle format.",
            self.save_pickle
        )

        # Output folder
        self.out_dir = self.create_folder_input()
        self.add_tooltipped_widget("Output Folder:", "Folder to save the output results.",
                                   self.out_dir)

        # Submit button
        self.submit_button = QPushButton("Run")
        self.submit_button.clicked.connect(self.run_analysis)
        self.layout.addWidget(self.submit_button)

        # Help and Issue buttons
        help_issue_layout = QHBoxLayout()

        # Help button
        self.help_button = QPushButton("Help")
        self.help_button.setToolTip("View the Trackrefiner wiki")
        self.help_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl("https://github.com/ingallslab/Trackrefiner/wiki")))
        help_issue_layout.addWidget(self.help_button)

        # Issue button
        self.issue_button = QPushButton("Report Issue")
        self.issue_button.setToolTip("Report an issue with the Trackrefiner")
        self.issue_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/ingallslab/Trackrefiner/issues")))
        help_issue_layout.addWidget(self.issue_button)

        # Add Help and Issue layout
        self.layout.addLayout(help_issue_layout)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.layout.addWidget(self.log_output)

        self.setLayout(self.layout)

    def toggle_intensity_threshold_visibility(self, state):
        if state == Qt.Checked:
            self.intensity_threshold_label.show()
            self.intensity_threshold.show()
        else:
            self.intensity_threshold_label.hide()
            self.intensity_threshold.hide()

        # Force layout recalculation
        self.layout.invalidate()

    def append_log(self, text):
        self.log_output.moveCursor(self.log_output.textCursor().End)
        self.log_output.insertPlainText(text)

    def closeEvent(self, event):
        sys.stdout = sys.__stdout__  # Restore original stdout
        super().closeEvent(event)

    def add_tooltipped_widget(self, label_text, tooltip_text, widget):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setToolTip(tooltip_text)

        # Adjust layout spacing and alignment
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignLeft)

        layout.addWidget(label)
        layout.addWidget(widget)
        self.layout.addLayout(layout)

    def add_tooltipped_widget_inline(self, label_text, tooltip_text, widget, layout):
        label = QLabel(label_text)
        label.setToolTip(tooltip_text)
        layout.addWidget(label)
        layout.addWidget(widget)

    def add_two_part_row(self, part1_label, part1_tooltip, part1_widget, part2_label, part2_tooltip, part2_widget):
        row_layout = QHBoxLayout()

        # Left part
        part1_layout = QHBoxLayout()
        self.add_tooltipped_widget_inline(part1_label, part1_tooltip, part1_widget, part1_layout)

        # Right part
        part2_layout = QHBoxLayout()
        self.add_tooltipped_widget_inline(part2_label, part2_tooltip, part2_widget, part2_layout)

        # Combine both parts in the row
        row_layout.addLayout(part1_layout)
        row_layout.addLayout(part2_layout)

        # Add the row layout to the main layout
        self.layout.addLayout(row_layout)

    def add_three_part_row(self, part1_label, part1_tooltip, part1_widget, part2_label, part2_tooltip, part2_widget,
                           part3_label, part3_tooltip, part3_widget):

        row_layout = QHBoxLayout()

        # Left part
        part1_layout = QHBoxLayout()
        self.add_tooltipped_widget_inline(part1_label, part1_tooltip, part1_widget, part1_layout)

        # Right part
        part2_layout = QHBoxLayout()
        self.add_tooltipped_widget_inline(part2_label, part2_tooltip, part2_widget, part2_layout)

        part3_layout = QHBoxLayout()
        self.add_tooltipped_widget_inline(part3_label, part3_tooltip, part3_widget, part3_layout)

        # Combine both parts in the row
        row_layout.addLayout(part1_layout)
        row_layout.addLayout(part2_layout)
        row_layout.addLayout(part3_layout)

        # Add the row layout to the main layout
        self.layout.addLayout(row_layout)

    def add_single_part_row(self, label_text, tooltip_text, widget):
        row_layout = QHBoxLayout()

        # Add label and widget inline
        label = QLabel(label_text)
        label.setToolTip(tooltip_text)
        row_layout.addWidget(label)
        row_layout.addWidget(widget)

        # Align all to the left
        row_layout.setAlignment(Qt.AlignLeft)

        # Add the row layout to the main layout
        self.layout.addLayout(row_layout)

    def create_csv_file_input(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        line_edit = QLineEdit()
        button = QPushButton("Browse")
        button.clicked.connect(lambda: self.browse_csv_file(line_edit))
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return container

    def create_folder_input(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        line_edit = QLineEdit()
        button = QPushButton("Browse")
        button.clicked.connect(lambda: self.browse_folder(line_edit))
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return container

    def create_double_spinbox(self, min_val, max_val, default_val=None):
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        if default_val is not None:
            spinbox.setValue(default_val)
        return spinbox

    def create_spinbox(self, min_val, max_val, default_val=0):
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default_val)
        return spinbox

    def create_combobox(self, items):
        combobox = QComboBox()
        combobox.addItems(items)
        return combobox

    def create_checkbox(self):
        return QCheckBox()

    def create_line_edit(self):
        return QLineEdit()

    def browse_csv_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", filter="CSV Files (*.csv)")
        if file_path:
            line_edit.setText(file_path)

    def browse_folder(self, line_edit):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            line_edit.setText(folder_path)

    def build_command(self):
        """
        Constructs a command-line statement based on the current GUI input values.
        """
        command = f"python {os.path.abspath(__file__)}"

        # Helper function to get text or None if empty
        def get_text_or_none(widget):
            text = widget.findChild(QLineEdit).text()
            return text if text.strip() else None

        # Map GUI inputs to arguments
        arguments = {
            '--cp-output-csv': get_text_or_none(self.cp_output_csv_file),
            '--segmentation-results': get_text_or_none(self.segmentation_results_dir),
            '--neighbor-csv': get_text_or_none(self.neighbor_csv),
            '--interval-time': str(self.interval_time.value()) if self.interval_time.value() > 0 else None,
            '--doubling-time': str(
                self.doubling_time_of_bacteria.value()) if self.doubling_time_of_bacteria.value() > 0 else None,
            '--elongation-rate-method': self.elongation_rate_method.currentText(),
            '--pixel-per-micron': str(self.pixel_per_micron.value()),
            '--intensity-threshold': str(
                self.intensity_threshold.value()) if self.assigning_cell_type.isChecked() else None,
            '--assign-cell-type': False if not self.assigning_cell_type.isChecked() else True,
            '--disable-tracking-correction': False if not self.disable_tracking_correction.isChecked() else True,
            '--clf': self.clf.currentText(),
            '--num-cpus': str(self.n_cpu.value()),
            '--boundary-limits': self.boundary_limits.text().strip() if self.boundary_limits.text().strip() else None,
            '--dynamic-boundaries': get_text_or_none(self.dynamic_boundaries),
            '--output': get_text_or_none(self.out_dir),
            '--save_pickle': False if not self.save_pickle.isChecked() else True,
            '--verbose': False if not self.verbose.isChecked() else True,
        }

        # Build the command string
        for arg, value in arguments.items():
            if value is not None:
                if value == arg:  # For boolean flags (e.g., --verbose)
                    command += f" {value}"
                else:  # For key-value pairs
                    command += f" {arg} {value}"
        return command

    def run_analysis(self):
        # Helper function to get text or None if empty
        def get_text_or_none(widget):
            text = widget.findChild(QLineEdit).text()
            return text if text.strip() else None

        # Check if required inputs are provided
        missing_inputs = []

        # Check for essential inputs
        if not get_text_or_none(self.cp_output_csv_file):
            missing_inputs.append("CellProfiler Output CSV File")
        if not get_text_or_none(self.neighbor_csv):
            missing_inputs.append("Neighbor CSV File")
        if self.interval_time.value() <= 0:
            missing_inputs.append("Interval Time")
        if self.doubling_time_of_bacteria.value() <= 0:
            missing_inputs.append("Doubling Time")

        # Check segmentation directory if tracking correction is enabled
        if not self.disable_tracking_correction.isChecked() and not get_text_or_none(self.segmentation_results_dir):
            missing_inputs.append("Segmentation Results Folder (when tracking correction is enabled)")

        # Show warning if any required input is missing
        if missing_inputs:
            QMessageBox.warning(
                self,
                "Missing Inputs",
                f"The following required inputs are missing or invalid:\n- {', '.join(missing_inputs)}"
            )
            return

        # Construct the command statement
        command = self.build_command()

        # Run process_objects_data with validated inputs
        process_objects_data(
            cp_output_csv=get_text_or_none(self.cp_output_csv_file),
            segmentation_res_dir=get_text_or_none(self.segmentation_results_dir),
            neighbor_csv=get_text_or_none(self.neighbor_csv),
            interval_time=self.interval_time.value() if self.interval_time.value() > 0 else None,
            elongation_rate_method=self.elongation_rate_method.currentText(),
            pixel_per_micron=self.pixel_per_micron.value(),
            intensity_threshold=self.intensity_threshold.value() if self.assigning_cell_type.isChecked() else None,
            assigning_cell_type=self.assigning_cell_type.isChecked(),
            doubling_time=self.doubling_time_of_bacteria.value() if self.doubling_time_of_bacteria.value() > 0 else None,
            disable_tracking_correction=self.disable_tracking_correction.isChecked(),
            clf=self.clf.currentText(),
            n_cpu=self.n_cpu.value(),
            image_boundaries=self.boundary_limits.text() if self.boundary_limits.text().strip() else None,
            dynamic_boundaries=get_text_or_none(self.dynamic_boundaries),
            out_dir=get_text_or_none(self.out_dir),
            verbose=self.verbose.isChecked(),
            save_pickle=self.save_pickle.isChecked(),
            command=command
        )


def main():
    app = QApplication(sys.argv)
    gui = TrackRefinerGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
