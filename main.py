import copy
import numpy as np
import random

# Load sudokus
sudoku = np.load("data/hard_puzzle.npy")
print("hard_puzzle.npy has been loaded into the variable sudoku")
print(f"sudoku.shape: {sudoku.shape}, sudoku[0].shape: {sudoku[0].shape}, sudoku.dtype: {sudoku.dtype}")

# Load solutions for demonstration
solutions = np.load("data/hard_solution.npy")
print()

# Print the first 9x9 sudoku...
print("First sudoku:")
print(sudoku[5], "\n")

# ...and its solution
# TODO: undo commenting
print("Solution of first sudoku:")
print(solutions[5])

"""My Code"""


class SudokuState:

    def __init__(self, final_values, parent=None):
        self.final_values = final_values
        self.parent = parent
        self.possible_values = [[[i for i in range(1, 10)] for _ in range(1, 10)] for _ in range(1, 10)]

### New Methods

    def set_final_values(self):
        """Checks each cell to see if it has only one possible value in its domain and sets that"""
        r = 0
        c = 0
        for row in self.possible_values:
            for col in row:
                if self.final_values[r][c] == 0 and len(self.possible_values[r][c]) == 1:
                    self.final_values[r][c] = self.possible_values[r][c][0]
                    self.update_domains(r, c, self.final_values[r][c])
                c += 1
            r += 1
            c = 0

    def initialise_sudoku_board(self):
        r = 0
        c = 0
        for row in self.final_values:
            for col in row:
                if self.final_values[r][c] != 0:
                    self.possible_values[r][c] = [self.final_values[r][c]]
                    self.update_domains(r, c, self.final_values[r][c])
                c += 1
            r += 1
            c = 0

    def update_domains(self, row, col, value):
        """ Given a cell and a value, removes that value from all other relevant cells' domains """

        if value not in self.possible_values[row][col]:
            raise ValueError(f"{value} is not a valid choice for cell {row, col}")

        # update all other cells possible values
        # start with cells in the same row
        for update_col in range(0, 9):
            # ignore target cell
            if update_col == col:
                continue
            # remove value
            if value in self.possible_values[row][update_col]:
                self.possible_values[row][update_col].remove(value)

        # now update cells in same column
        for update_row in range(0, 9):
            # ignore starting cell
            if update_row == row:
                continue
            # remove value
            if value in self.possible_values[update_row][col]:
                self.possible_values[update_row][col].remove(value)

        # now update cells in the same 9 x 9 block
        starting_cell_row = (row // 3) * 3
        starting_cell_col = (col // 3) * 3
        for block_row in range(starting_cell_row, starting_cell_row + 3):
            for block_col in range(starting_cell_col, starting_cell_col + 3):
                # Ignore starting cell
                if block_row == row and block_col == col:
                    continue
                if value in self.possible_values[block_row][block_col]:
                    self.possible_values[block_row][block_col].remove(value)


### END

    def is_goal(self):
        # The board is solved when there are no empty cells (0s)
        for section in self.final_values:
            for cell in section:
                if cell == 0:
                    return False
        return True

    def is_invalid(self):
        # This state is invalid if any cell on the board has no possible values
        return any(len(values) == 0 for values in self.possible_values)

    def get_possible_values(self, column, row):
        return self.possible_values[column][row].copy()

    def get_final_state(self):
        if self.is_goal():
            return self.final_values
        else:
            return np.full((9, 9), -1)

    def get_singleton_cells(self):
        """"" Returns the cells  (row and column tuple) which have no final value but exactly 1 possible value """
        # TODO: Tidy this up!
        singleton_cells = []
        r = 0
        c = 0
        for row in self.possible_values:
            for column in row:
                if len(column) == 1 and self.final_values[r][c] == 0:
                    singleton_cells.append((r, c))
                c += 1
            c = 0
            r += 1

        return singleton_cells

    def set_value(self, row, col, value, parent):
        """Returns a new state with this cell set to this value, and the change propagated to other domains"""

        if value not in self.possible_values[row][col]:
            raise ValueError(f"{value} is not a valid choice for cell {row, col}")

        # create a deep copy: the method returns a new state, does not modify the existing one
        state = copy.deepcopy(self)
        state.parent = parent

        # update this cell
        state.possible_values[row][col] = [value]
        state.final_values[row][col] = value

        # now update all other cells possible values
        # start with cells in the same row
        for update_col in range(0, 9):
            # ignore target cell
            if update_col == col:
                continue
            # remove value
            if value in state.possible_values[row][update_col]:
                state.possible_values[row][update_col].remove(value)

        # now update cells in same column
        for update_row in range(0, 9):
            # ignore target cell
            if update_row == row:
                continue
            # remove value
            if value in state.possible_values[update_row][col]:
                state.possible_values[update_row][col].remove(value)

        # if any other cells with no final value only have 1 possible value, make them final
        # singleton_cells = state.get_singleton_cells()
        # while len(singleton_cells) > 0:
        #     cell = singleton_cells[0]
        #     row = cell[0]
        #     col = cell[1]
        #     state = state.set_value(row, col, state.possible_values[row][col][0], state)
        #     singleton_cells = state.get_singleton_cells()
        return state

    def get_frequency_of_possible_values(self):
        """
        Returns a list of the least to most common outstanding possible values
        The frequency is the first item in the returned tuple, for faster sorting.
        :return:
        """
        total_possible_values = []

        # Create a raw, unsorted list of all total occurences of possible values
        for row in self.possible_values:
            for col in row:
                # Ignore singleton values
                if len(col) == 1:
                    continue
                for value in col:
                    total_possible_values.append(value)

        # Create a dictionary which records the number of occurences of each possible value
        freq_dict = {}
        for value in total_possible_values:
            if value in freq_dict:
                freq_dict[value] += 1
            else:
                freq_dict[value] = 1

        freq_list = []
        for key, value in freq_dict.items():
            temp = (value, key)
            freq_list.append(temp)

        freq_list = sorted(freq_list)

        sorted_values = []

        for (value, key) in freq_list:
            sorted_values.append(key)

        return sorted_values

    def set_all_values(self):

        # TODO: Tidy this up!
        state = copy.deepcopy(self)
        r = 0
        c = 0
        for f_row in state.final_values:
            for f_col in f_row:
                value = state.final_values[r][c]
                if value != 0:
                    state = state.set_value(r, c, value, state)
                c += 1
            r += 1
            c = 0

        return state


def pick_next_cell(partial_state):
    """
    Chooses which cell to try, chooses the cell with the least number of possible values
    :param partial_state:
    :return:
    """
    # TODO: Tidy this up!
    # Get a list of all empty cells
    empty_cells = []
    r = 0
    c = 0
    for row in partial_state.final_values:
        for column in row:
            if partial_state.final_values[r][c] == 0:
                empty_cells.append((r, c))
            c += 1
        c = 0
        r += 1

    # Create a list of empty cells with number of possible values
    ordered_empty_cells = []
    for cell in empty_cells:
        pos_vals = partial_state.get_possible_values(cell[0], cell[1])
        if len(ordered_empty_cells) == 0:
            ordered_empty_cells.append((cell, pos_vals))
        if pos_vals <= ordered_empty_cells[0][1]:
            ordered_empty_cells.insert(0, (cell, pos_vals))
        else:
            ordered_empty_cells.append((cell, pos_vals))

    cell = ordered_empty_cells[0][0]
    return cell


def order_values(partial_state, row, col):
    """
    Get values for the cell in the order we should try them in.
    Currently random.
    :param partial_state:
    :param row:
    :param col:
    :return:
    """
    values = partial_state.get_possible_values(row, col)
    value_frequencies = partial_state.get_frequency_of_possible_values()
    final_values = value_frequencies.copy()
    print(value_frequencies)
    for value in value_frequencies:
        if value not in values:
            final_values.remove(value)
    print(values)
    print(final_values)
    return final_values


def depth_first_search(partial_state):
    cell = pick_next_cell(partial_state)
    row = cell[0]
    col = cell[1]
    values = order_values(partial_state, row, col)
    print(partial_state.final_values)
    for value in values:
        new_state = partial_state.set_value(row, col, value, partial_state)
        if new_state.is_goal():
            return new_state
        if not new_state.is_invalid():
            deep_state = depth_first_search(new_state)
            if deep_state is not None and deep_state.is_goal():
                return deep_state


def sudoku_solver(sudoku):
    """
    Solves a Sudoku puzzle and returns its unique solution.

    Input
        sudoku : 9x9 numpy array
            Empty cells are designated by 0.

    Output
        9x9 numpy array of integers
            It contains the solution, if there is one. If there is no solution, all array entries should be -1.
    """
    partial_state = SudokuState(sudoku)
    partial_state = partial_state.set_all_values()
    print(partial_state.final_values)

    if partial_state.is_goal():
        return partial_state
    solution = depth_first_search(partial_state)

    # TODO: replace with all -1s sudoku for fail state
    return solution


test_sudoku = SudokuState(sudoku[5])
test_sudoku.initialise_sudoku_board()
sud
print(test_sudoku.possible_values)
# test_sudoku.set_final_values()
# print(test_sudoku.possible_values)

print(test_sudoku.final_values)

