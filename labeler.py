from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import os
import numpy as np
import json
from config_loader import load_config, exe_dir_path
from read_classes import NpzFileIterator, FileProcessor
from scipy.optimize import curve_fit
from state_saver import load_state, save_state

global t_peak

def combined_function(t, A, k, lambda_):
    global t_peak
    sigmoid = np.exp(-k * (t - t_peak))
    exponential = np.where(t > t_peak, np.exp(-lambda_ * (t - t_peak)), 1)
    return A * sigmoid * exponential

def total(t, A1, A2, k1, k2, l1, l2):
    return combined_function(t, A1, k1, l1) + combined_function(t, A2, k2, l2)

class InteractivePlotter(QMainWindow):
    def __init__(self, reader, selection_file_path):
        super().__init__()
        self.reader = reader
        self.current_batches = []
        self.current_batch_index = 0
        self.selected_points = []
        self.start_point = None
        self.batch_index_to_save = None
        self.fitted_line = None
        self.selection_file_path = selection_file_path

        # Store current file information
        self.current_file = None
        self.batch_size = 300
        self.overlap_size = 50

        # Dictionary to store selections organized by file
        self.file_selections = {}

        # Add zoom window attributes
        self.zoom_window = None
        self.zoom_size = 0.00000003  # Size of the zoom window in seconds

        self.initUI()
        self.load_next_file()

    def initUI(self):
        self.setWindowTitle('Interactive Plot')
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create buttons
        button_layout = QVBoxLayout()

        self.next_button = QPushButton('Next Plot')
        self.next_button.clicked.connect(self.next_plot)
        button_layout.addWidget(self.next_button)

        self.save_button = QPushButton('Save Selections')
        self.save_button.clicked.connect(self.save_selections)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self.ax = self.figure.add_subplot(111)

        # Enable key press events
        self.setFocusPolicy(Qt.StrongFocus)

    def closeEvent(self, event):
        """
        This method is automatically called when the window is closed.
        It saves the current selections and state before closing.
        """
        # Save selections
        self.save_selections()

        # Save the current state
        if self.current_file is not None:
            save_state(self.current_file)

        # Call the parent class method to continue with the normal close process
        super().closeEvent(event)

    def fit_and_plot_curve(self, start_idx):
        # Find end index where data falls below 0.02
        end_idx = start_idx + 150  # Minimum 100 points
        # TODO: determine it better

        # Prepare data for fitting
        t_data = self.current_t[start_idx:end_idx]
        y_data = self.current_i[start_idx:end_idx]

        # Initial guess for parameters
        global t_peak
        t_peak = t_data[np.argmax(y_data)]  # Time at peak
        c_peak = np.max(y_data)
        initial_guess = [
            c_peak, c_peak, 5, 5, 1, 1
        ]

        try:
            # Fit the curve
            popt, _ = curve_fit(total, t_data, y_data, p0=initial_guess, maxfev=10000)
            y_fit = total(t_data, *popt)

            # Remove previous fitted line if it exists
            if self.fitted_line is not None:
                self.fitted_line.pop(0).remove()

            # Plot the fitted curve
            self.fitted_line = self.ax.plot(t_data, y_fit, 'm--', label='Fitted Curve')
            self.canvas.draw()

        except RuntimeError as e:
            print(f"Fitting error: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.next_plot()
        super().keyPressEvent(event)

    def load_next_file(self):
        # Save selections from previous file if exists
        if self.current_file is not None:
            self.save_selections()
            save_state(self.current_file)

        try:
            v, f, data, file = next(self.reader)
            self.current_file = file

            # Initialize selections for new file
            if self.current_file not in self.file_selections:
                self.file_selections[self.current_file] = {
                    'file_name': self.current_file,
                    'batch_size': self.batch_size,
                    'overlap_size': self.overlap_size,
                    'selections': []
                }

            fp = FileProcessor(
                self.batch_size, data,
                filter_func=lambda data: max(np.abs(data)) > 0.5,
                overlap_size=self.overlap_size
            )
            self.current_batches = fp.split_into_batches()
            self.current_batch_index = 0
            self.plot_current_batch()
        except StopIteration:
            print("No more files to process")
            self.save_selections()
            self.close()

    def plot_current_batch(self):
        self.ax.clear()
        batch = self.current_batches[self.current_batch_index]
        self.current_t = batch['t']
        self.current_i = batch['i']
        self.batch_index_to_save = batch['batch_index']

        # Plot main data
        self.ax.plot(self.current_t, self.current_i)
        self.ax.set_ylim((-0.5, 2))

        # Color overlap areas
        if self.overlap_size > 0:
            # Start overlap
            if self.current_batch_index > 0:
                self.ax.axvspan(self.current_t[0],
                              self.current_t[self.overlap_size],
                              color='red', alpha=0.1)
            # End overlap
            if self.current_batch_index < len(self.current_batches) - 1:
                self.ax.axvspan(self.current_t[-self.overlap_size],
                              self.current_t[-1],
                              color='red', alpha=0.1)

        # Add title with current file and batch information
        self.ax.set_title(f"File: {self.current_file}\nBatch: {batch['batch_index']}")

        # Plot previously selected points
        for start, end in self.selected_points:
            self.ax.axvline(x=start, color='g', linestyle='--')
            self.ax.axvline(x=end, color='r', linestyle='--')

        # Remove old zoom window if it exists
        if self.zoom_window is not None:
            self.zoom_window.remove()

        # Create new zoom window
        self.zoom_window = inset_axes(self.ax,
                                    width="30%",
                                    height="30%",
                                    loc='upper right')
        self.zoom_window.set_xlabel('Time (s)')
        self.zoom_window.set_ylabel('Current')
        self.zoom_window.set_ylim((
            np.min(self.current_i),
            np.max(self.current_i)
        ))

        self.canvas.draw()

    def on_mouse_move(self, event):
        if event.inaxes != self.ax or not hasattr(self, 'current_t'):
            return

        try:
            x = event.xdata
            y = event.ydata

            if x is None or y is None:  # Cursor out of plot area
                return

            # Clear and update zoom window
            self.zoom_window.clear()

            # Calculate zoom range
            half_width = self.zoom_size / 2
            x_min = x - half_width
            x_max = x + half_width

            # Find data points within zoom range
            mask = (self.current_t >= x_min) & (self.current_t <= x_max)
            if not any(mask):  # No data points in range
                return

            zoom_t = self.current_t[mask]
            zoom_i = self.current_i[mask]

            # Plot zoomed data
            self.zoom_window.plot(zoom_t, zoom_i, 'b-')

            # If fitted curve exists, plot it in zoom window
            if self.fitted_line is not None:
                # Get the fitted line data
                fitted_line = self.fitted_line[0]
                fitted_t = fitted_line.get_xdata()
                fitted_i = fitted_line.get_ydata()

                # Find fitted points within zoom range
                fitted_mask = (fitted_t >= x_min) & (fitted_t <= x_max)
                if any(fitted_mask):
                    zoom_fitted_t = fitted_t[fitted_mask]
                    zoom_fitted_i = fitted_i[fitted_mask]
                    self.zoom_window.plot(zoom_fitted_t, zoom_fitted_i, 'c--')

            # Add vertical line at cursor position
            self.zoom_window.axvline(x=x, color='r', linestyle=':')

            # Set y limits with some padding
            y_min = min(self.current_i)
            y_max = max(self.current_i)
            padding = (y_max - y_min) * 0.1
            if padding == 0:  # Handle case where y_min == y_max
                padding = 0.1
            self.zoom_window.set_ylim(y_min - padding, y_max + padding)

            # Set x limits
            self.zoom_window.set_xlim(x_min, x_max)

            # Add grid for better visibility
            self.zoom_window.grid(True, linestyle=':')

            # Force immediate update
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()

        except Exception as e:
            print(f"Error in zoom update: {e}")

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x = event.xdata

        if self.start_point is None:
            self.start_point = x
            start_idx = np.abs(self.current_t - x).argmin()

            # Fit and plot the curve
            self.fit_and_plot_curve(start_idx)

            self.ax.axvline(x=x, color='g', linestyle='--')
            self.canvas.draw()
        else:
            end_point = x
            self.ax.axvline(x=x, color='r', linestyle='--')
            self.selected_points.append((self.start_point, end_point))

            # Find indices
            start_idx = np.abs(self.current_t - self.start_point).argmin()
            end_idx = np.abs(self.current_t - end_point).argmin()

            # Get current batch
            current_batch = self.current_batches[self.current_batch_index]
            global_start_idx = current_batch['global_index'] + start_idx
            global_end_idx = current_batch['global_index'] + end_idx

            # Add to file selections
            self.file_selections[self.current_file]['selections'].append({
                'start_index': int(global_start_idx),
                'end_index': int(global_end_idx)
            })

            self.start_point = None
            # Remove fitted line
            if self.fitted_line is not None:
                self.fitted_line.pop(0).remove()
                self.fitted_line = None
            self.canvas.draw()

    def next_plot(self):
        self.selected_points = []  # Clear selected points for new plot
        if self.fitted_line is not None:
            self.fitted_line.pop(0).remove()
            self.fitted_line = None
        self.current_batch_index += 1

        if self.current_batch_index >= len(self.current_batches):
            self.load_next_file()
        else:
            self.plot_current_batch()

    def save_selections(self):
        if not self.file_selections:
            print("No selections to save")
            return

        output_file = self.selection_file_path
        existing_selections = {}

        # Load existing selections if the file exists
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                    # Convert list to dictionary with file_name as key for easier merging
                    for item in existing_data:
                        if 'file_name' in item:
                            existing_selections[item['file_name']] = item
            except json.JSONDecodeError:
                print(f"Warning: Could not parse existing {output_file}, creating new file")
            except Exception as e:
                print(f"Error reading existing selections: {e}")

        # Merge current selections with existing ones
        for file_name, selection_data in self.file_selections.items():
            existing_selections[file_name] = selection_data

        # Convert back to list for JSON storage
        selections_list = list(existing_selections.values())

        # Save the merged selections
        with open(output_file, 'w') as f:
            json.dump(selections_list, f, indent=4)

        print(f"Saved selections for {len(selections_list)} files to {output_file}")


def main():
    app = QApplication(sys.argv)

    config = load_config()
    directory = config.get("data_folder", "data_2600+")
    if not os.path.isabs(directory):
        directory = exe_dir_path(directory)
    selection_file = config.get("selections_file", "selections.json")
    if not os.path.isabs(selection_file):
        selection_file = exe_dir_path(selection_file)

    # Load the previous state
    state = load_state()
    skip = 0
    if state and 'last_file' in state:
        last_file = state['last_file']
        # Find the index of the last file to determine how many to skip
        files = sorted([file for file in os.listdir(directory) if file.endswith('.npz')])
        last_file_name = os.path.basename(last_file)
        if last_file_name in files:
            skip = files.index(last_file_name)
            print(f"Resuming from file: {last_file_name}")

    reader = NpzFileIterator(directory, skip)

    ex = InteractivePlotter(reader, selection_file)
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
