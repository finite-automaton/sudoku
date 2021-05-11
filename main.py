import copy
import numpy as np
import random
import time

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

    def set_value(self, row, col, value):
        """Returns a new state with this cell set to this value, and the change propagated to other domains"""
        if value not in self.possible_values[row][col]:
            raise ValueError(f"{value} is not a valid choice for cell {row, col}")

        # create a deep copy: the method returns a new state, does not modify the existing one
        state = copy.deepcopy(self)

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

        # now update cells in the same 9 x 9 block
        starting_cell_row = (row // 3) * 3
        starting_cell_col = (col // 3) * 3
        for block_row in range(starting_cell_row, starting_cell_row + 3):
            for block_col in range(starting_cell_col, starting_cell_col + 3):
                # Ignore starting cell
                if block_row == row and block_col == col:
                    continue
                if value in state.possible_values[block_row][block_col]:
                    state.possible_values[block_row][block_col].remove(value)

        # if any other cells with no final value only have 1 possible value, make them final
        singleton_cells = state.get_singleton_cells()
        while len(singleton_cells) > 0:
            cell = singleton_cells[0]
            row = cell[0]
            col = cell[1]
            state = state.set_value(row, col, state.possible_values[row][col][0])
            singleton_cells = state.get_singleton_cells()

        return state

    def set_all_values(self):
        # TODO: Tidy this up!
        state = copy.deepcopy(self)
        r = 0
        c = 0
        for f_row in state.final_values:
            for f_col in f_row:
                value = state.final_values[r][c]
                if value != 0:
                    state = state.set_value(r, c, value)
                c += 1
            r += 1
            c = 0

        return state


def pick_next_cell(partial_state):
    """
    Chooses which cell to try, currently random
    :param partial_state:
    :return:
    """
    # TODO: Tidy this up!
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

    cell = random.choice(empty_cells)
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
    random.shuffle(values)
    return values

def depth_first_search(partial_state):

    cell = pick_next_cell(partial_state)
    row = cell[0]
    col = cell[1]
    values = order_values(partial_state, row, col)
    print(partial_state.final_values)
    for value in values:
        new_state = partial_state.set_value(row, col, value)
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
    solution = depth_first_search(partial_state)

    # TODO: replace with all -1s sudoku for fail state
    return solution


first_sudoku = sudoku[10]
start_time = time.process_time()
print(sudoku_solver(first_sudoku).final_values)
end_time = time.process_time()
print("This sudoku took", end_time-start_time, "seconds to solve.\n")

# print(first_sudoku.possible_values)
# print(first_sudoku.is_goal())
# print(first_solution.is_goal())
# print(first_sudoku.get_possible_values(0, 0))
# print(first_sudoku.get_singleton_cells())
# newstate = first_sudoku.set_value(0, 1, 7)
# newstate2 = newstate.set_value(1, 1, 9)
# print(newstate2.final_values)
# set_all_state = first_sudoku.set_all_values()
# for row in set_all_state.possible_values:
#     print(row)
# print(set_all_state.final_values)
