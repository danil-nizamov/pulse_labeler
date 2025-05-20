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

class SelectionLine:
    def __init__(self, x, color, ax, label=None):
        self.x = x
        self.color = color
        self.ax = ax
        self.line = ax.axvline(x=x, color=color, linestyle='--', linewidth=2.5, picker=7, zorder=5)
        self.label = label
        if label:
            # Adjust y for label so it's above axes, uses relative axis transform.
            self.text = ax.text(
                x, 1.01, label, color=color, ha='center', va='bottom',
                transform=ax.get_xaxis_transform(), fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75)
            )
        else:
            self.text = None

    def update(self, x):
        self.x = x
        self.line.set_xdata([x, x])
        if self.text:
            self.text.set_x(x)

    def remove(self):
        self.line.remove()
        if self.text:
            self.text.remove()

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

        self.current_file = None
        self.batch_size = 300
        self.overlap_size = 50

        self.file_selections = {}

        self.zoom_window = None
        self.zoom_size = 0.00000003  # Size of the zoom window in seconds

        self.current_lines = []  # Holds SelectionLine objects
        self._move_active = None # Tuple (SelectionLine, idx of current_lines)
        self._move_line_idx = None

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

        button_layout = QVBoxLayout()

        self.next_button = QPushButton('Next Plot')
        self.next_button.clicked.connect(self.next_plot)
        button_layout.addWidget(self.next_button)

        self.save_button = QPushButton('Save Selections')
        self.save_button.clicked.connect(self.save_selections)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        # Connect mouse events
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        self.ax = self.figure.add_subplot(111)

        self.setFocusPolicy(Qt.StrongFocus)

    def closeEvent(self, event):
        self.save_selections()
        if self.current_file is not None:
            save_state(self.current_file)
        super().closeEvent(event)

    def fit_and_plot_curve(self, start_idx):
        end_idx = start_idx + 150  # Minimum 100 points

        t_data = self.current_t[start_idx:end_idx]
        y_data = self.current_i[start_idx:end_idx]

        global t_peak
        t_peak = t_data[np.argmax(y_data)]  # Time at peak
        c_peak = np.max(y_data)
        initial_guess = [
            c_peak, c_peak, 5, 5, 1, 1
        ]

        try:
            popt, _ = curve_fit(total, t_data, y_data, p0=initial_guess, maxfev=10000)
            y_fit = total(t_data, *popt)

            if self.fitted_line is not None:
                self.fitted_line.pop(0).remove()

            self.fitted_line = self.ax.plot(t_data, y_fit, 'm--', label='Fitted Curve')
            self.canvas.draw()

        except RuntimeError as e:
            print(f"Fitting error: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.next_plot()
        super().keyPressEvent(event)

    def load_next_file(self):
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
                filter_func=lambda data: max(np.abs(data)) > 0.05,
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

        self.ax.plot(self.current_t, self.current_i)
        self.ax.set_ylim((-0.5, 2))

        if self.overlap_size > 0:
            if self.current_batch_index > 0:
                self.ax.axvspan(self.current_t[0],
                              self.current_t[self.overlap_size],
                              color='red', alpha=0.1)
            if self.current_batch_index < len(self.current_batches) - 1:
                self.ax.axvspan(self.current_t[-self.overlap_size],
                              self.current_t[-1],
                              color='red', alpha=0.1)

        self.ax.set_title(f"File: {self.current_file}\nBatch: {batch['batch_index']}")

        # Remove any old selection lines
        for sl in self.current_lines:
            sl.remove()
        self.current_lines = []
        self.selected_points = []

        # Plot previously selected points for this batch
        selections = self.file_selections[self.current_file].get('selections', [])
        for selection in selections:
            start_idx = selection['start_index'] - batch['global_index']
            end_idx = selection['end_index'] - batch['global_index']
            # Ensure within batch
            if 0 <= start_idx < len(self.current_t) and 0 <= end_idx < len(self.current_t):
                start_x = self.current_t[start_idx]
                end_x = self.current_t[end_idx]

                sl_start = SelectionLine(start_x, 'lime', self.ax, label="start")
                sl_end = SelectionLine(end_x, 'red', self.ax, label="end")
                self.current_lines.extend([sl_start, sl_end])
                self.selected_points.append((start_x, end_x))

        if self.zoom_window is not None:
            self.zoom_window.remove()
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

    def find_nearest_line(self, x, tol=0.00000003):
        """
        Finds the nearest line object to x within a certain tolerance.
        Returns (line_obj, index) or (None, None)
        """
        for i, sl in enumerate(self.current_lines):
            if abs(sl.x - x) < tol:
                return sl, i
        return None, None

    def on_pick(self, event):
        modifiers = QApplication.keyboardModifiers()
        # Right click: remove the region lines & selection
        if event.mouseevent.button == 3:
            for i, sl in enumerate(self.current_lines):
                if event.artist == sl.line:
                    idx_region = i // 2
                    # Remove the pair: start/end lines and selection from selected_points
                    idx_first = 2 * idx_region
                    idx_second = idx_first + 1
                    remove_indices = sorted([idx_first, idx_second], reverse=True)
                    for idx in remove_indices:
                        self.current_lines[idx].remove()
                        del self.current_lines[idx]
                    del self.selected_points[idx_region]
                    # Update file_selections
                    batch = self.current_batches[self.current_batch_index]
                    if 'selections' in self.file_selections[self.current_file]:
                        sels = self.file_selections[self.current_file]['selections']
                        if idx_region < len(sels):
                            del sels[idx_region]
                    self.canvas.draw()
                    return
        # SHIFT+LMB: start moving the line
        elif event.mouseevent.button == 1 and (modifiers & Qt.ShiftModifier):
            for i, sl in enumerate(self.current_lines):
                if event.artist == sl.line:
                    self._move_active = (sl, i)
                    break

    def on_mouse_move(self, event):
        # --- Handle line dragging with shift ---
        if self._move_active and event.inaxes == self.ax and event.xdata is not None:
            sl, idx = self._move_active
            sl.update(event.xdata)
            region_idx = idx // 2
            pair_idx = idx % 2
            # Get both new positions
            moved_pos = event.xdata
            other_line_obj = self.current_lines[region_idx * 2 + (1 - pair_idx)]
            other_pos = other_line_obj.x
            # Update selected_points (which is always [start, end])
            if pair_idx == 0:  # moved start
                start_time, end_time = moved_pos, other_pos
            else:              # moved end
                start_time, end_time = other_pos, moved_pos
            # Ensure ordering!
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            self.selected_points[region_idx] = (start_time, end_time)

            # Now convert to indices
            batch = self.current_batches[self.current_batch_index]
            # Clamp the time to t array limits to get valid index
            t_arr = self.current_t
            start_idx = (np.abs(t_arr - start_time)).argmin()
            end_idx = (np.abs(t_arr - end_time)).argmin()
            # Also ensure ordering here after rounding
            start_idx, end_idx = sorted((start_idx, end_idx))
            global_start_idx = batch['global_index'] + start_idx
            global_end_idx = batch['global_index'] + end_idx
            selection = self.file_selections[self.current_file]['selections'][region_idx]
            selection['start_index'] = int(global_start_idx)
            selection['end_index'] = int(global_end_idx)
            self.canvas.draw_idle()

        # --- The rest is the zoom window logic (unchanged) ---
        if event.inaxes != self.ax or not hasattr(self, 'current_t'):
            return

        try:
            x = event.xdata
            y = event.ydata
            if x is None or y is None:
                return
            self.zoom_window.clear()
            half_width = self.zoom_size / 2
            x_min = x - half_width
            x_max = x + half_width
            mask = (self.current_t >= x_min) & (self.current_t <= x_max)
            if not any(mask):
                return
            zoom_t = self.current_t[mask]
            zoom_i = self.current_i[mask]
            self.zoom_window.plot(zoom_t, zoom_i, 'b-')
            if self.fitted_line is not None:
                fitted_line = self.fitted_line[0]
                fitted_t = fitted_line.get_xdata()
                fitted_i = fitted_line.get_ydata()
                fitted_mask = (fitted_t >= x_min) & (fitted_t <= x_max)
                if any(fitted_mask):
                    zoom_fitted_t = fitted_t[fitted_mask]
                    zoom_fitted_i = fitted_i[fitted_mask]
                    self.zoom_window.plot(zoom_fitted_t, zoom_fitted_i, 'c--')
            self.zoom_window.axvline(x=x, color='r', linestyle=':')
            y_min = min(self.current_i)
            y_max = max(self.current_i)
            padding = (y_max - y_min) * 0.1
            if padding == 0:
                padding = 0.1
            self.zoom_window.set_ylim(y_min - padding, y_max + padding)
            self.zoom_window.set_xlim(x_min, x_max)
            self.zoom_window.grid(True, linestyle=':')
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
        except Exception as e:
            print(f"Error in zoom update: {e}")

    def on_release(self, event):
        # Finished moving
        self._move_active = None

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        # Do not handle click (selection) if SHIFT is pressed (move only)
        modifiers = QApplication.keyboardModifiers()
        if (modifiers & Qt.ShiftModifier):
            return

        # Only LMB for adding lines
        if event.button != 1:
            return

        x = event.xdata

        if self.start_point is None:
            self.start_point = x
            start_idx = np.abs(self.current_t - x).argmin()
            self.fit_and_plot_curve(start_idx)
            sl = SelectionLine(x, 'lime', self.ax, label="start")
            self.current_lines.append(sl)
            self.canvas.draw()
        else:
            end_point = x
            sl = SelectionLine(x, 'red', self.ax, label="end")
            self.current_lines.append(sl)
            self.selected_points.append((self.start_point, end_point))

            start_idx = np.abs(self.current_t - self.start_point).argmin()
            end_idx = np.abs(self.current_t - end_point).argmin()
            current_batch = self.current_batches[self.current_batch_index]
            global_start_idx = current_batch['global_index'] + start_idx
            global_end_idx = current_batch['global_index'] + end_idx

            # Add to file selections as region, keep structure consistent.
            self.file_selections[self.current_file]['selections'].append({
                'start_index': int(global_start_idx),
                'end_index': int(global_end_idx)
            })

            self.start_point = None
            if self.fitted_line is not None:
                self.fitted_line.pop(0).remove()
                self.fitted_line = None
            self.canvas.draw()

    def next_plot(self):
        # Remove all region lines
        for sl in self.current_lines:
            sl.remove()
        self.current_lines = []
        self.selected_points = []

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

        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                    for item in existing_data:
                        if 'file_name' in item:
                            existing_selections[item['file_name']] = item
            except json.JSONDecodeError:
                print(f"Warning: Could not parse existing {output_file}, creating new file")
            except Exception as e:
                print(f"Error reading existing selections: {e}")

        for file_name, selection_data in self.file_selections.items():
            existing_selections[file_name] = selection_data

        selections_list = list(existing_selections.values())
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

    state = load_state()
    skip = 0
    if state and 'last_file' in state:
        last_file = state['last_file']
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
