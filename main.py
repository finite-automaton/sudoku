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


def ones_counter():
    """provides a lookup list to know how many 1s are present in a binary representation of a number up to 511"""
    """Used for naked pairs, triples etc"""
    lookup_list = []
    for number in range(0, 512):
        binary_num = bin(number)
        count = 0
        for char in range(2, len(binary_num)):
            if binary_num[char] == '1':
                count += 1
        lookup_list.append(count)
    return lookup_list


ones_lookup = ones_counter()


class SudokuState:

    def __init__(self, initial_values, not_valid):
        # Invalid sudoku setting
        self.not_valid = not_valid
        if not_valid:
            self.final_values = initial_values
        else:
            self.initial_values = initial_values
            self.final_values = np.ndarray(shape=(9, 9),
                                           dtype=np.int16)  # larger data type is needed for bitwise operations

        # self.possible_values = np.full((9, 9), 511)  # 511 is binary representation of all values
        """returns numpy array of possible 'single' values if a posisble value cell contains
        one of these values then it is the only possible value for that cell"""
        self.final_possible_values = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
        self.encoding = [511, 1, 2, 4, 8, 16, 32, 64, 128, 256]

    def encode_sudoku(self):

        for row in range(0, 9):
            for col in range(0, 9):
                self.final_values[row][col] = self.encoding[self.initial_values[row][col]]

    def decode_sudoku(self):
        if self.not_valid:
            return
        for row in range(0, 9):
            for col in range(0, 9):
                self.final_values[row][col] = self.decode(self.final_values[row][col])

        # if value == 0:
        #     return 511  # This is binary 111111111 and reflects all possible options
        # if value == 1:
        #     return 1
        # if value == 2:
        #     return 2
        # if value == 3:
        #     return 4
        # if value == 4:
        #     return 8
        # if value == 5:
        #     return 16
        # if value == 6:
        #     return 32
        # if value == 7:
        #     return 64
        # if value == 8:
        #     return 128
        # if value == 9:
        #     return 256

    def decode(self, value):

        if value == 0:
            return 0
        elif value == 1:
            return 1
        elif value == 2:
            return 2
        elif value == 4:
            return 3
        elif value == 8:
            return 4
        elif value == 16:
            return 5
        elif value == 32:
            return 6
        elif value == 64:
            return 7
        elif value == 128:
            return 8
        elif value == 256:
            return 9
        else:
            return 0

    def initialise_sudoku_board(self):

        # First encode the final values into binary representations
        self.encode_sudoku()

        # Next set final values when there is a final value already
        for row in range(0, 9):
            for col in range(0, 9):
                value = self.final_values[row][col]
                if value in self.final_possible_values:
                    self.set_value(row, col)

    def set_value(self, row, col):
        """ Given a cell, removes the value of this cell from the possible values of other cells
        in the same row, column and block. Works when the 'set_value' cell has only 1 value!!"""

        # This won't work anymore!
        # Throw error if trying to set a value that is not possible
        # if value not in self.possible_values[row][col]:
        #     for x in self.possible_values:
        #         print(x)
        #     raise ValueError(f"{value} is not a valid choice for cell {row, col}")

        # Reduce the possible value to the final value for the target cell
        # Set the final value

        # remove the value from all other relevant cell's possible values
        # start with cells in the same row

        for update_col in range(0, 9):
            # ignore target cell
            if update_col == col:
                continue
            # if the value is not a possible value of the cell, then an AND will yield 0, and we can ignore it
            if self.final_values[row][update_col] & self.final_values[row][col] == 0:
                continue
            # otherwise, the value is in it and can be removed with a bitwise XOR
            self.final_values[row][update_col] = self.final_values[row][update_col] ^ self.final_values[row][col]

        # now update cells in same column
        for update_row in range(0, 9):
            # ignore starting cell
            if update_row == row:
                continue
            # if the value is not a possible value of the cell, then an AND will yield 0, and we can ignore it
            if self.final_values[update_row][col] & self.final_values[row][col] == 0:
                continue
            # remove value with a bitwise XOR
            self.final_values[update_row][col] = self.final_values[update_row][col] ^ self.final_values[row][col]

        # now update cells in the same 9 x 9 block
        starting_cell_row = (row // 3) * 3
        starting_cell_col = (col // 3) * 3
        for block_row in range(starting_cell_row, starting_cell_row + 3):
            for block_col in range(starting_cell_col, starting_cell_col + 3):
                # Ignore the updated cell
                if block_row == row and block_col == col:
                    continue

                # if the value is not a possible value of the cell, then an AND will yield 0, and we can ignore it
                if self.final_values[block_row][block_col] & self.final_values[row][col] == 0:
                    continue
                self.final_values[block_row][block_col] = self.final_values[block_row][block_col] ^ \
                                                          self.final_values[row][col]

        return True

    def set_final_values(self):
        """ Sets any value that may be final"""
        for row in range(0, 9):
            for col in range(0, 9):
                if self.final_values[row][col] in self.final_possible_values:
                    self.set_value(row, col)

    def is_only_possibility_in_row(self, row):
        """Sets possible value if unique among a row"""
        # possible value in the row
        for col in range(0, 9):
            # If this is already a final value, skip
            if self.final_values[row][col] in self.final_possible_values:
                continue
            # TODO: Would it be faster to find the way to reduce this with an and? So its only checking actual possible values
            for value in self.final_possible_values:
                is_only_possible_value = True
                # If not a possible value for this cell, move on to next possible value
                if self.final_values[row][col] & value == 0:
                    continue

                for other_col in range(0, 9):
                    # don't check against self
                    if col == other_col:
                        continue
                    # If the value is possible for another cell, then it isn't the only possible value
                    if self.final_values[row][other_col] & value != 0:
                        is_only_possible_value = False
                        break

                if is_only_possible_value:
                    # set the value
                    self.final_values[row][col] = value
                    self.set_value(row, col)
                    # break out of loop
                    break

    def is_only_possibility_in_col(self, col):
        """Sets possible value if unique among a column"""
        """Sets possible value if unique among a row"""
        # possible value in the row
        for row in range(0, 9):
            # TODO: Would it be faster to find the way to reduce this with an and? So its only checking actual possible values
            # If this is already a final value, skip
            if self.final_values[row][col] in self.final_possible_values:
                continue
            for value in self.final_possible_values:
                is_only_possible_value = True
                # If not a possible value for this cell, move on to next possible value
                if self.final_values[row][col] & value == 0:
                    continue

                for other_row in range(0, 9):
                    # don't check against self
                    if row == other_row:
                        continue
                    # If the value is possible for another cell, then it isn't the only possible value
                    if self.final_values[other_row][col] & value != 0:
                        is_only_possible_value = False
                        break

                if is_only_possible_value:
                    # set the value
                    self.final_values[row][col] = value
                    self.set_value(row, col)
                    # break out of loop
                    break

    def is_only_possibility_in_block(self, starting_cell_row, starting_cell_col):

        # possible value in the row
        """Checks if a possible value is unique in a 9x9 block"""
        # starting_cell_row = (row // 3) * 3
        # starting_cell_col = (col // 3) * 3
        for block_row in range(starting_cell_row, starting_cell_row + 3):
            for block_col in range(starting_cell_col, starting_cell_col + 3):
                # If this is already a final value, skip
                if self.final_values[block_row][block_col] in self.final_possible_values:
                    continue
                for value in self.final_possible_values:
                    is_only_possible_value = True
                    # If not a possible value for this cell, move on to next possible value
                    if self.final_values[block_row][block_col] & value == 0:
                        continue

                    for other_row in range(starting_cell_row, starting_cell_row + 3):
                        for other_col in range(starting_cell_col, starting_cell_col + 3):
                            # don't check against self
                            if block_row == other_row and block_col == other_col:
                                continue
                            # If the value is possible for another cell, then it isn't the only possible value
                            if self.final_values[other_row][other_col] & value != 0:
                                is_only_possible_value = False
                                break
                        else:
                            continue
                        break

                    if is_only_possible_value:
                        # set the value
                        self.final_values[block_row][block_col] = value
                        self.set_value(block_row, block_col)
                        # break out of loop
                        break

    def resolve_naked_x(self, x):
        """Examines cells in rows, columns and blocks for hidden pairs. If a hidden pair is found, removes the
        values of the hidden pair from the domains of relevant cells"""
        # Rows first
        for row in range(0, 9):
            for col in range(0, 9):
                # Ignore solved cells and cells with more than 2 values
                if self.final_values[row][col] in self.final_possible_values or ones_lookup[
                    self.final_values[row][col]] != x:
                    continue
                for next_col in range(col + 1, 9):
                    # Ignore solved cells and cells with more than x values
                    if self.final_values[row][next_col] in self.final_possible_values or ones_lookup[
                        self.final_values[row][next_col]] != x:
                        continue
                    if self.final_values[row][col] == self.final_values[row][next_col]:
                        for change_col in range(0, 9):
                            # Ignore any cells which have the same values
                            if self.final_values[row][col] == self.final_values[row][change_col]:
                                continue
                            # Remove the values from all other cells if they are there
                            if self.final_values[row][col] & self.final_values[row][change_col] == self.final_values[row][col]:
                                self.final_values[row][change_col] = self.final_values[row][col] ^ self.final_values[row][change_col]

        #Then columns
        for col in range(0, 9):
            for row in range(0, 9):
                # Ignore solved cells and cells with more than 2 values
                if self.final_values[row][col] in self.final_possible_values or ones_lookup[
                    self.final_values[row][col]] != x:
                    continue
                for next_row in range(row + 1, 9):
                    # Ignore solved cells and cells with more than x values
                    if self.final_values[next_row][col] in self.final_possible_values or ones_lookup[
                        self.final_values[next_row][col]] != x:
                        continue
                    if self.final_values[row][col] == self.final_values[next_row][col]:
                        for change_row in range(0, 9):
                            # Ignore any cells which have the same values
                            if self.final_values[row][col] == self.final_values[change_row][col]:
                                continue
                            # Remove the values from all other cells if they are there
                            if self.final_values[row][col] & self.final_values[change_row][col] == self.final_values[row][col]:
                                self.final_values[change_row][col] = self.final_values[row][col] ^ self.final_values[change_row][col]

        # Then Blocks
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                for cell_row in range(row, row + 3):
                    for cell_col in range(col, col + 3):
                        if self.final_values[cell_row][cell_col] in self.final_possible_values or ones_lookup[
                            self.final_values[cell_row][cell_col]] != x:
                            continue
                        for next_row in range(row, row + 3):
                            for next_col in range(col, col + 3):
                                # Ignore solved cells and cells that have too many possible values
                                if self.final_values[next_row][col] in self.final_possible_values or ones_lookup[
                                        self.final_values[cell_row][cell_col]] != x:
                                    continue
                                # Ignore the cell we are comparing
                                if cell_row == next_row and cell_col == next_col:
                                    continue

                                if self.final_values[cell_row][cell_col] == self.final_values[next_row][next_col]:
                                    for change_row in range(row, row + 3):
                                        for change_col in range(col, col + 3):
                                            # Ignore the pairs
                                            if (change_row == cell_row and change_col == cell_col) or (
                                                    change_row == next_row and change_col == next_col):
                                                continue
                                            # Otherwise, remove those possible values from cells if they are there
                                            if self.final_values[cell_row][cell_col] & self.final_values[change_row][change_col] == \
                                                    self.final_values[cell_row][cell_col]:
                                                self.final_values[change_row][change_col] = self.final_values[cell_row][cell_col] ^ \
                                                                                     self.final_values[change_row][change_col]

    def apply_rules(self):

        for row in range(0, 9):
            self.is_only_possibility_in_row(row)
        for col in range(0, 9):
            self.is_only_possibility_in_col(col)
        for starting_row in range(0, 9, 3):
            for starting_col in range(0, 9, 3):
                self.is_only_possibility_in_block(starting_row, starting_col)

        # If the rules have updated the domains so that there is only possible value for anything, set it
        self.resolve_naked_x(2)
        #self.resolve_naked_x(3)
        self.set_final_values()

    def is_goal(self):
        # The board is solved when there are no empty cells (0s)
        for row in range(0, 9):
            for col in range(0, 9):
                if self.final_values[row][col] not in self.final_possible_values:
                    return False
        return True

    def is_invalid(self):
        # TODO: Add check for multiple value in row, col, block
        # This state is invalid if any cell on the board has no possible values
        for row in range(0, 9):
            for col in range(0, 9):
                if self.final_values[row][col] == 0:
                    return True
        return False

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

    def get_first_unsolved_cell(self):
        for row in range(0, 9):
            for col in range(0, 9):
                if self.final_values[row][col] not in self.final_possible_values:
                    return (row, col)


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
        if len(partial_state.possible_values[cell[0]][cell[1]]) <= len(
                partial_state.possible_values[least_options_cell[0]][least_options_cell[1]]):
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

    # cell = pick_next_cell(next_state)
    cell = next_state.get_first_unsolved_cell()
    row = cell[0]
    col = cell[1]
    # values = order_values(next_state, row, col)
    # values = next_state.possible_values[row][col]

    for value in next_state.final_possible_values:
        if next_state.final_values[row][col] & value == 0:
            continue
        # Virtual next step
        attempt_state = copy.deepcopy(next_state)
        attempt_state.final_values[row][col] = value
        attempt_state.set_value(row, col)
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

    partial_state = SudokuState(sudoku, False)
    partial_state.initialise_sudoku_board()

    if partial_state.is_goal():
        partial_state.decode_sudoku()
        return partial_state

    solution = depth_first_search(partial_state)
    if solution is None:
        return np.full((9, 9), -1)
    solution.decode_sudoku()

    # TODO: replace with all -1s sudoku for fail state

    return solution.final_values


# Test Infra
def test_sudoku(number):
    start_time = time.process_time()
    result = sudoku_solver(sudoku[number])
    end_time = time.process_time()
    if np.array_equal(result, solutions[number]):
        print("Sudoku: ", number, " took ", end_time - start_time, " seconds to solve.")
    else:
        print("Sudoku: ", number, " yielded a wrong result")


# test_sudoku(14)

# for num in range(0, 15):
#     test_sudoku(num)

# for num in range(0, 4):
#     test_sudoku(num)

# Testing
# test_row = np.array([
#     [1, 2, 4, 8, 16, 32, 320, 448, 320],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9]
# ])
#
# trial1 = SudokuState(test_row)
# print(trial1.final_values)
# trial1.is_only_possibility_in_row(0)
# print(trial1.final_values)

# print(sudoku[5])
# print(solutions[5])
#
# test = SudokuState(sudoku[5])
# test.apply_rules()
# test.apply_rules()
# test.apply_rules()
# test.apply_rules()
# test.apply_rules()
# test.apply_rules()
# print(test.final_values)

mr_hard = np.array([
    [0, 0, 0, 0, 0, 1, 0, 0, 2],
    [0, 0, 3, 0, 0, 0, 0, 4, 0],
    [0, 5, 0, 0, 6, 0, 7, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 7, 0],
    [0, 0, 7, 0, 0, 3, 8, 0, 0],
    [9, 0, 0, 0, 5, 0, 0, 0, 1],
    [0, 0, 6, 0, 8, 0, 2, 0, 0],
    [0, 4, 0, 6, 0, 0, 0, 0, 7],
    [2, 0, 0, 0, 0, 9, 0, 6, 0]
])

mr_hard_solution = np.array([
    [8, 7, 9, 4, 3, 1, 6, 5, 2],
    [6, 2, 3, 5, 9, 7, 1, 4, 8],
    [1, 5, 4, 2, 6, 8, 7, 9, 3],
    [4, 3, 2, 8, 1, 6, 5, 7, 9],
    [5, 1, 7, 9, 4, 3, 8, 2, 6],
    [9, 6, 8, 7, 5, 2, 4, 3, 1],
    [7, 9, 6, 3, 8, 4, 2, 1, 5],
    [3, 4, 1, 6, 2, 5, 9, 8, 7],
    [2, 8, 5, 1, 7, 9, 3, 6, 4]])


def test_mr_hard():
    start_time = time.process_time()
    result = sudoku_solver(mr_hard)
    end_time = time.process_time()
    print(result)
    if np.array_equal(result, mr_hard_solution):
        print("This really hard sudoku took ", end_time - start_time, " seconds to solve.")
    else:
        print("Sudoku yielded a wrong result")

test_mr_hard()