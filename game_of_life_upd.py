import sys
import time

import numpy as np

from multiprocessing import Pool
from scipy.stats import uniform



# main parameters of running
####################
WORLD_SIDE_LEN = 100
UNIT_AREA = 5

WINDOW_HEIGHT = WORLD_SIDE_LEN * UNIT_AREA
WINDOW_WIDTH = WORLD_SIDE_LEN * UNIT_AREA

BACKGROUND = (110, 110, 110)
UNIT_COLOR = (100, 200, 50)


N_STEPS = 50
N_THREADS = 4

grid_size = (WORLD_SIDE_LEN, WORLD_SIDE_LEN)
#####################

def init_grid_structure(size):
    # construct frame for grid, init neighbors, fix boundary neighbors
    # return np.array
    # with structure  element -> neighbors_indices
    # numeration from lower left corner

    h, w = grid_size
    frame = np.zeros((h, w, 8, 2), dtype=np.int32)
    frame -= 1

    for i in range(h):
        for k in range(w):
            neighbors_ = np.array(
                [
                    (i, k + 1),
                    (i + 1, k + 1),
                    (i + 1, k),
                    (i + 1, k - 1),
                    (i, k - 1),
                    (i - 1, k - 1),
                    (i - 1, k),
                    (i - 1, k + 1),
                ],
            )

            for x_ind, (_x, _y) in enumerate(neighbors_):
                if _x < h and _y < w:
                    frame[i, k, x_ind] = (_x, _y)
    return frame


def init_grid_values(frame, threshold):
    # def init_grid_values(data: np.ndarray, threshold: float):
    # initialize random values of grid with uniform distribution

    for i in range(frame.shape[0]):
        for k in range(frame.shape[1]):
            if uniform.rvs() < threshold:
                frame[i, k] = True
    return frame


# class of "game world"
class World:
    def __init__(self, grid_size: tuple, n_threads: int, n_steps: int):
        self.n_threads = n_threads
        self.n_steps = n_steps
        self.grid_size = grid_size
        self.structure = init_grid_structure(self.grid_size)
        self.grid_values = np.zeros(self.grid_size, dtype=bool)
        self.grid_values = init_grid_values(self.grid_values, 0.5)

# Vanilla game


    def make_step_vanilla(self):
        # step for unparallel game

        n_rows, n_cols = self.grid_size
        new_grid = self.grid_values.copy()

        for i in range(n_rows):
            for k in range(n_cols):
                neighbors = self.structure[i, k]
                alive_neighbors = 0
                for _x, _y in neighbors:
                    if _x != -1 and _y != -1 and self.grid_values[_x, _y]:
                        alive_neighbors += 1

                if alive_neighbors < 2:
                    new_grid[i, k] = False
                elif self.grid_values[i, k] and (alive_neighbors == 2 or alive_neighbors == 3):
                    new_grid[i, k] = True
                elif alive_neighbors > 3:
                    new_grid[i, k] = False
                elif not self.grid_values[i, k] and alive_neighbors == 3:
                    new_grid[i, k] = True

        self.grid_values = new_grid


    def play_game_vanilla(self):
        # loop of game by steps

        for step in range(self.n_steps):
            self.make_step_vanilla()
            yield self.grid_values


# parallel game

    def make_step_row(self, i):
        row = self.grid_values[i]
        new_row = row.copy()

        for k in range(row.shape[0]):
            neighbors = self.structure[i, k]
            alive_neighbors = 0
            for n_x, n_y in neighbors:
                if n_x != -1 and n_y != -1 and self.grid_values[n_x, n_y]:
                    alive_neighbors += 1

            if alive_neighbors < 2:
                new_row[k] = False
            elif row[k] and (alive_neighbors == 2 or alive_neighbors == 3):
                new_row[k] = True
            elif alive_neighbors > 3:
                new_row[k] = False
            elif not row[k] and alive_neighbors == 3:
                new_row[k] = True
        return new_row


    def play_game_parallel(self):
        # parallel game row by row threading

        n_rows, n_cols = self.grid_size
        with Pool(self.n_threads) as pool:
            for step in range(self.n_steps):
                new_grid = pool.map(self.make_step_row, np.arange(n_cols))
                new_grid = np.array(new_grid)
                self.grid_values = new_grid
                yield new_grid


def draw_step(g):
    for x in range(0, WINDOW_WIDTH, UNIT_AREA):
        for y in range(0, WINDOW_HEIGHT, UNIT_AREA):
            rect = pygame.Rect(x, y, UNIT_AREA, UNIT_AREA)
            if g[x // UNIT_AREA, y // UNIT_AREA]:
                pygame.draw.rect(SCREEN, UNIT_COLOR, rect, 0)
            else:
                pygame.draw.rect(SCREEN, BACKGROUND, rect, 0)

if __name__ == "__main__":
    import pygame

    # steps_research = [50, 100, 150, 200, 250, 300]
    # threads_research = [2 , 3, 4]
    # results = []
    #
    # for num_steps in steps_research:
    #     game_world = World(
    #         n_threads= 2, n_steps=num_steps, grid_size=(WORLD_SIDE_LEN, WORLD_SIDE_LEN)
    #     )
    #
    #     start_time = time.time()
    #
    #     time.time()
    #     for _ in game_world.play_game_vanilla():
    #         k = 'Sucsess'
    #
    #     results.append(time.time() - start_time)
    #
    # for num_threads in threads_research:
    #     for num_steps in steps_research:
    #         game_world = World(
    #             n_threads=num_threads, n_steps=num_steps, grid_size=(WORLD_SIDE_LEN, WORLD_SIDE_LEN)
    #         )
    #
    #         start_time = time.time()
    #
    #         time.time()
    #         for _ in game_world.play_game_parallel():
    #             k = 'Sucsess'
    #
    #         results.append(time.time() - start_time)
    #
    # f = open('results.txt', 'w')
    # for res in results:
    #     f.write(str(res))
    #     f.write(" ")
    # f.close()
    #
    # print(results)



    game_world = World(
        n_threads=N_THREADS, n_steps=N_STEPS, grid_size=(WORLD_SIDE_LEN, WORLD_SIDE_LEN)
    )

    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BACKGROUND)

    start_time = time.time()

    for grid in game_world.play_game_vanilla():  ###running
        draw_step(game_world.grid_values)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    print(time.time() - start_time)
