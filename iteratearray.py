import numpy as np
import math


class IterateArray:
    """ Iterate the elements of an array in decreasing order from given
     starting points. """

    def __init__(self, starting_points, iterated_array, term=100):
        """

        Parameters
        ----------
        starting_points : iterable of int
            An iterable object of the starting points in iterated_array
            for the iterators.
        iterated_array : ndarray
            An array that can be sorted.
        term : int
            Maximum number of iteration for a single iterator.
        """
        # Initialize iterators:
        # A single row represents a single "iterator", which consists of a
        # starting point and a step. Starting points are saved in column 0
        # and steps in column 1.
        self.iterators = np.zeros((len(starting_points), 2))
        self.iterators[:, 0] = starting_points

        # Set maximum step:
        self.term = term

        # Get mappings from array indices to sorted indices and back:
        self.sorting = np.argsort(-iterated_array)  # Note: descending
        self.idx_in_sorted = np.argsort(self.sorting)

        self.lim = self.sorting.size

    def iterate(self, iter_idx):
        """ Return current index and update step. """

        # Get new index:
        start = self.iterators[iter_idx, 0]
        step = self.iterators[iter_idx, 1]
        idx = self.get_index_at_step(start, step)
        if step >= self.term or idx is None:
            idx = None
        else:
            # Update step:
            self.iterators[iter_idx, 1] += 1

        return idx

    def get_step(self, iter_idx):
        step = self.iterators[iter_idx, 1]
        return step

    def get_index_at_step(self, start, step):
        start_in_sorted = self.idx_in_sorted[start]
        idx_in_sorted = self.get_index_at_step_in_sorted(
            start_in_sorted, step)

        if idx_in_sorted is None:
            idx = None
        else:
            idx = self.sorting[idx_in_sorted]

        return idx

    def get_index_at_step_in_sorted(self, start, step):
        """ Get index in the sorted version of the iterated array at
        a given step from a given starting point.

        Parameters
        ----------
        start : int
            Starting point of iterations in the sorted array.
        step : int
            Number of iterations completed.

        Returns
        -------
        idx : int
            Index of the element at step in the sorted array.
        """

        # Iterate outwards from start, alternating between lower and higher
        # index:
        idx = start + int(math.copysign(
            math.floor((step + 1) / 2), (step % 2) - 0.5))

        # Check that index is not out of array bounds:
        if abs(start - idx) > start:
            idx = step
        elif abs(start - idx) > self.lim - 1 - start:
            idx = self.lim - 1 - step

        # If all values of array are consumed:
        if idx < 0 or idx >= self.lim:
            idx = None

        return idx
