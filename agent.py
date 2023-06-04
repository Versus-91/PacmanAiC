import numpy as np


class PacmanGameState:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.pacman_pos = (0, 0)
        self.ghost_pos = (0, 0)
        self.pellets = []
        self.score = 0

    def get_state_vector(self):
        state_vector = []

        # Add Pacman position
        state_vector.extend(self.pacman_pos)

        # Add Ghost position
        state_vector.extend(self.ghost_pos)

        # Add Pellet positions
        for pellet_pos in self.pellets:
            state_vector.extend(pellet_pos)

        # Add width and height of the game grid
        state_vector.append(self.grid_size[0])
        state_vector.append(self.grid_size[1])

        # Add score
        state_vector.append(self.score)

        return np.array(state_vector)


# Example usage
grid_size = (800, 600)  # Width and height of the game grid in pixels
game_state = PacmanGameState(grid_size)
game_state.pacman_pos = (200, 300)
game_state.ghost_pos = (600, 450)
game_state.pellets = [(100, 200), (400, 300), (700, 400)]
game_state.score = 100

state_vector = game_state.get_state_vector()
print(state_vector)
