import os
from typing import Callable
import zipfile
import numpy as np
import re


class NpzFileIterator:
    def __init__(self, directory, skip=0):
        """
        Initialize the iterator with an option to skip files.

        Args:
            directory (str): Path to the directory containing .npz files
            skip (int): Number of files to skip from the beginning
        """
        self.directory = directory
        # Sort the files alphanumerically
        self._files = sorted([file for file in os.listdir(directory) if file.endswith('.npz')])

        # Validate skip parameter
        if skip < 0:
            raise ValueError("Skip value cannot be negative")
        if skip > len(self._files):
            raise ValueError(f"Skip value ({skip}) exceeds the number of available files ({len(self._files)})")

        # Set the index to the skip position
        self._index = skip

        # Store the original skip value for reference
        self.skipped = skip

    def __iter__(self):
        return self

    def __next__(self):
        while self._index < len(self._files):
            file = self._files[self._index]
            self._index += 1

            # Extract the voltage and frequency from the filename
            match = re.search(r'(\d+)Ohm_(\d+)V_(\d+)kHz', file)
            if match:
                resistance = int(match.group(1))
                voltage = int(match.group(2))
                frequency = int(match.group(3))
            else:
                print(f"Warning: Skipping file '{file}' - filename does not match expected pattern.")
                continue

            # Load the data from the file
            filepath = os.path.join(self.directory, file)
            try:
                data = np.load(filepath)['data']
                data = {
                    't': data[0],
                    'v': data[1],
                    'i': data[2]
                }
                return voltage, frequency, data, filepath
            except zipfile.BadZipfile:
                print(f"Warning: Skipping corrupted file '{file}'")
                continue

        raise StopIteration

    def reset(self, skip=None):
        """
        Reset the iterator with an optional new skip value.

        Args:
            skip (int, optional): New number of files to skip. If None, uses the original skip value.
        """
        if skip is None:
            skip = self.skipped

        if skip < 0:
            raise ValueError("Skip value cannot be negative")
        if skip > len(self._files):
            raise ValueError(f"Skip value ({skip}) exceeds the number of available files ({len(self._files)})")

        self._index = skip
        self.skipped = skip


class FileProcessor:
    '''
    Для обработки одного файла
    '''
    def __init__(
        self,
        batch_size: int,
        file_data: dict,
        filter_func: Callable = None,
        overlap_size: int = 0
    ) -> None:
        self.batch_size = batch_size
        self.data = file_data
        self.overlap_size = overlap_size
        if filter_func is not None:
            self.filter = filter_func
        else:
            self.filter = lambda _: True

    def split_into_batches(self):
        result = []
        data_length = len(self.data['t'])
        batch_num = 0
        for i in range(0, data_length, self.batch_size):
            batch_num += 1
            start = max(i - self.overlap_size, 0)
            end = min(i + self.batch_size + self.overlap_size, data_length)

            batch_current_t = self.data['t'][start:end]
            batch_current_i = self.data['i'][start:end]

            if self.filter(batch_current_i):
                result.append({
                    't': batch_current_t,
                    'i': batch_current_i,
                    'batch_index': batch_num,
                    'global_index': start
                })
        return result
