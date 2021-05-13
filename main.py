import copy
import numpy as np
import time

# Load sudokus
sudoku = np.load("data/hard_puzzle.npy")
print("medium_puzzle.npy has been loaded into the variable sudoku")
print(f"sudoku.shape: {sudoku.shape}, sudoku[0].shape: {sudoku[0].shape}, sudoku.dtype: {sudoku.dtype}")

# Load solutions for demonstration
solutions = np.load("data/hard_solution.npy")

"""My Code"""


class SudokuState:

    def __init__(self, final_values, parent=None):
        self.final_values = final_values
        self.parent = parent
        self.possible_values = [[[i for i in range(1, 10)] for _ in range(1, 10)] for _ in range(1, 10)]

    def set_final_values(self):
        """Checks each cell to see if it has only one possible value in its domain and sets that"""

        for row in range(0, 9):
            for col in range(0, 9):
                if self.final_values[row][col] == 0 and len(self.possible_values[row][col]) == 1:
                    value = self.possible_values[row][col][0]
                    self.final_values[row][col] = value
                    self.set_value(row, col, value)

    def initialise_sudoku_board(self):

        for row in range(0, 9):
            for col in range(0, 9):
                value = self.final_values[row][col]
                if value != 0:
                    self.set_value(row, col, value)

    def set_value(self, row, col, value):
        """ Given a cell and a value, sets that value as the cell's final value and
         removes that value from all other relevant cells' domains """

        # Throw error if trying to set a value that is not possible
        if value not in self.possible_values[row][col]:
            for x in self.possible_values:
                print(x)
            raise ValueError(f"{value} is not a valid choice for cell {row, col}")

        # Reduce the possible value to the final value for the target cell
        self.possible_values[row][col] = [value]
        # Set the final value
        self.final_values[row][col] = value

        # remove the value from all other relevant cell's possible values
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
                # Ignore the updated cell
                if block_row == row and block_col == col:
                    continue
                if value in self.possible_values[block_row][block_col]:
                    self.possible_values[block_row][block_col].remove(value)

    def is_only_possibility_in_row(self, value, row, input_col):
        """Checks if a possible value is unique among a row"""
        for col in range(0, 9):
            if col == input_col:
                continue
            if value in self.possible_values[row][col]:
                return False

        return True

    def is_only_possibility_in_col(self, value, input_row, col):
        """Checks if a possible value is unique among a column"""
        for row in range(0, 9):
            if row == input_row:
                continue
            if value in self.possible_values[row][col]:
                return False

        return True

    def is_only_possibility_in_block(self, value, row, col):
        """Checks if a possible value is unique in a 9x9 block"""
        starting_cell_row = (row // 3) * 3
        starting_cell_col = (col // 3) * 3
        for block_row in range(starting_cell_row, starting_cell_row + 3):
            for block_col in range(starting_cell_col, starting_cell_col + 3):
                # Ignore starting cell
                if block_row == row and block_col == col:
                    continue
                if value in self.possible_values[block_row][block_col]:
                    return False

        return True

    def resolve_naked_x(self, x):
        """Examines cells in rows, columns and blocks for hidden pairs. If a hidden pair is found, removes the
        values of the hidden pair from the domains of relevant cells"""
        # Rows first
        for row in range(0, 9):
            for col in range(0, 9):
                # Ignore solved cells and cells with more than 2 values
                if self.final_values[row][col] != 0 or len(self.possible_values[row][col]) != x:
                    continue
                for next_col in range(col + 1, 9):
                    # Ignore solved cells
                    if self.final_values[row][next_col] != 0:
                        continue
                    if self.possible_values[row][col] == self.possible_values[row][next_col]:
                        for change_col in range(0, 9):
                            # Ignore any cells which have the same values (covers triples, quads)
                            if self.possible_values[row][col] == self.possible_values[row][change_col]:
                                continue
                            # Remove the values from all other cells if they are there
                            for value in self.possible_values[row][col]:
                                if value in self.possible_values[row][change_col]:
                                    self.possible_values[row][change_col].remove(value)

        # Then columns
        for col in range(0, 9):
            for row in range(0, 9):
                # Ignore solved cells and cells with more than 2 values
                if self.final_values[row][col] != 0 or len(self.possible_values[row][col]) != x:
                    continue
                for next_row in range(row + 1, 9):
                    # Ignore solved cells
                    if self.final_values[next_row][col] != 0:
                        continue
                    if self.possible_values[row][col] == self.possible_values[next_row][col]:
                        for change_row in range(0, 9):
                            # Ignore any cells which have the same values (covers triples, quads)
                            if self.possible_values[row][col] == self.possible_values[change_row][col]:
                                continue
                            # Remove the values from all other cells if they are there
                            for value in self.possible_values[row][col]:
                                if value in self.possible_values[change_row][col]:
                                    self.possible_values[change_row][col].remove(value)

        # Then Blocks
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                for cell_row in range(row, row + 3):
                    for cell_col in range(col, col + 3):
                        if self.final_values[cell_row][cell_col] != 0 or len(
                                self.possible_values[cell_row][cell_col]) != x:
                            continue
                        for next_row in range(row, row + 3):
                            for next_col in range(col, col + 3):
                                # Ignore solved cells and cells that have too many possible values
                                if self.final_values[next_row][col] != 0 or len(
                                        self.possible_values[cell_row][cell_col]) != 2:
                                    continue
                                # Ignore the cell we are comparing
                                if cell_row == next_row and cell_col == next_col:
                                    continue

                                if self.possible_values[cell_row][cell_col] == self.possible_values[next_row][next_col]:
                                    for change_row in range(row, row + 3):
                                        for change_col in range(col, col + 3):
                                            # Ignore the pairs
                                            if (change_row == cell_row and change_col == cell_col) or (
                                                    change_row == next_row and change_col == next_col):
                                                continue
                                            # Otherwise, remove those possible values from cells if they are there
                                            for value in self.possible_values[cell_row][cell_col]:
                                                if value in self.possible_values[change_row][change_col]:
                                                    self.possible_values[change_row][change_col].remove(value)

    def apply_rules(self):
        for row in range(0, 9):
            for col in range(0, 9):
                # Ignore already solved cells
                if len(self.possible_values[row][col]) != 1:
                    continue
                # loop every cell and try the rules on it
                for value in self.possible_values[row][col]:
                    if self.is_only_possibility_in_row(value, row, col):
                        self.set_value(row, col, value)
                        continue
                    if self.is_only_possibility_in_col(value, row, col):
                        self.set_value(row, col, value)
                        continue
                    if self.is_only_possibility_in_block(value, row, col):
                        self.set_value(row, col, value)
                        continue
        # If the rules have updated the domains so that there is only possible value for anything, set it
        self.resolve_naked_x(2)
        self.set_final_values()

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
        return self.possible_values[column][row]

    def get_final_state(self):
        if self.is_goal():
            return self.final_values
        else:
            return np.full((9, 9), -1)

    def get_frequency_of_possible_values(self):
        """
        Returns a list of the least to most common outstanding possible values
        The frequency is the first item in the returned tuple, for faster sorting.
        :return:
        """
        total_possible_values = []

        # Create a raw, unsorted list of all total occurrences of possible values
        for row in self.possible_values:
            for col in row:
                # Ignore singleton values
                if len(col) == 1:
                    continue
                for value in col:
                    total_possible_values.append(value)

        # Create a dictionary which records the number of occurrences of each possible value
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


def pick_next_cell(partial_state):
    """
    Chooses which cell to try, chooses the cell with the least number of possible values
    :param partial_state:
    :return:
    """

    # Get a list of all empty cells
    empty_cells = []

    for row in range(0, 9):
        for col in range(0, 9):
            if partial_state.final_values[row][col] == 0:
                empty_cells.append((row, col))

    # Find the empty cell with the least number of possible values and return it
    least_options_cell = empty_cells[0]
    for cell in empty_cells:
        if len(partial_state.possible_values[cell[0]][cell[1]]) <= len(partial_state.possible_values[least_options_cell[0]][least_options_cell[1]]):
            least_options_cell = cell

    return least_options_cell


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

    for value in value_frequencies:
        if value not in values:
            final_values.remove(value)

    return final_values


def depth_first_search(partial_state):
    # First apply all rules to the given state until the same output is yielded
    start_state = copy.deepcopy(partial_state)
    next_state = copy.deepcopy(partial_state)
    next_state.apply_rules()

    while not np.array_equal(next_state.final_values, start_state.final_values):
        start_state = copy.deepcopy(next_state)
        next_state.apply_rules()

    if next_state.is_goal():
        return next_state

    if next_state.is_invalid():
        return None

    cell = pick_next_cell(next_state)
    row = cell[0]
    col = cell[1]
    values = order_values(next_state, row, col)
    # values = next_state.possible_values[row][col]

    for value in values:

        # Virtual next step
        attempt_state = copy.deepcopy(next_state)
        attempt_state.set_value(row, col, value)
        if attempt_state.is_goal():
            return attempt_state

        if not attempt_state.is_invalid():
            deep_state = depth_first_search(attempt_state)
            if deep_state is not None and deep_state.is_goal():
                return deep_state

    return None


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
    partial_state.initialise_sudoku_board()

    if partial_state.is_goal():
        return partial_state

    solution = depth_first_search(partial_state)

    # TODO: replace with all -1s sudoku for fail state

    return solution.final_values


#Test Infra
def test_sudoku(number):
    start_time = time.process_time()
    result = sudoku_solver(sudoku[number])
    end_time = time.process_time()
    if np.array_equal(result, solutions[number]):
        print("Sudoku: ", number, " took ", end_time - start_time, " seconds to solve.")
    else:
        print("Sudoku: ", number, " yielded a wrong result")

for num in range(2,12):
    test_sudoku(num)

# for num in range(0, 4):
#     test_sudoku(num)