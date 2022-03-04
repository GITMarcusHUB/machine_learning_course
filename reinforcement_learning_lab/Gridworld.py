import numpy as np
np.random.seed(1) # For reproducibility
from enum import Enum


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


SIMPLE_GRID = ['############',
               '#    S     #',
               '#     #    #',
               '#     # X  #',
               '############']


class GridEnvironment:

    def __init__(self, grid_text):
        self.walls = np.array([[c == '#' for c in l] for l in grid_text])
        self.goal = np.array([[c == 'X' for c in l] for l in grid_text])
        self.starting_point = None
        self.current_position = None
        for r, l in enumerate(grid_text):
            for c, cell in enumerate(l):
                if cell == 'S':
                    self.starting_point = (r, c)
                    return

    def reset(self):
        """
        Resets the state to the starting one and returns the observation
        (current state).
        """
        self.current_position = tuple(self.starting_point)
        return np.array(self.current_position)

    def step(self, action):
        """
        Step in the direction specified by action; returns the new observation
        (state), the intermediate reward and whether the episode is terminated.
        """
        assert self.current_position is not None, 'You should call reset() first.'
        action = Direction(action)
        if action == Direction.NORTH:
            next_state = (self.current_position[0]-1, self.current_position[1])
        elif action == Direction.SOUTH:
            next_state = (self.current_position[0]+1, self.current_position[1])
        elif action == Direction.WEST:
            next_state = (self.current_position[0], self.current_position[1]-1)
        elif action == Direction.EAST:
            next_state = (self.current_position[0], self.current_position[1]+1)
        nr, nc = next_state
        H, W = self.walls.shape
        if nr < 0 or nr >= H or nc < 0 or nc >= W or self.walls[nr, nc]:
            # Die
            self.current_position = None
            return np.array(next_state), -1, True
        if self.goal[nr, nc]:
            # Win
            self.current_position = None
            return np.array(next_state), 1, True
        self.current_position = next_state
        # The cost of the step is 0.1
        return self.current_position, -0.1, False

    def visualise(self):
        """
        Prints the grid with the agent to the console.
        """
        def vis_cell(r, c):
            if self.current_position == (r, c):
                return 'A'
            if self.walls[r, c]:
                return '#'
            if self.goal[r, c]:
                return 'X'
            return ' '
        print(
            '\n'.join(''.join(vis_cell(r, c)
                              for c in range(self.walls.shape[1]))
                      for r in range(self.walls.shape[0])))

def test(num_eps):
    """
    Demo function; how to use GridEnvironment
    """
    env = GridEnvironment(SIMPLE_GRID)
    for eps in range(num_eps):
        state = env.reset()
        cumulative_reward = 0
        done = False
        env.visualise()
        while not done:
            action = np.random.randint(len(Direction))
            state, reward, done = env.step(action)
            cumulative_reward += reward
            if not done:
                env.visualise()
        print('Episode #' + str(eps), 'cumulative undiscounted reward:',
              cumulative_reward)

if __name__ == "__main__":
    test(50)