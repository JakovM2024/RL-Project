"""
environment.py — The Grid World Environment
=============================================

This file creates the ENVIRONMENT that our RL agent will learn in.

In reinforcement learning, the environment is the "world" the agent lives in.
The agent takes ACTIONS, the environment responds with a new STATE and a REWARD.

Think of it like a board game:
- The board is the environment
- The player is the agent
- Each move is an action
- The board changes after each move (new state)
- Points scored are rewards

Our Grid World:
    S . . . .       S = Start (where the agent begins)
    . X . X .       G = Goal (where the agent wants to reach)
    . . . . .       X = Wall (blocked cells the agent can't enter)
    . X . . X       . = Empty (free to move through)
    . . . . G

The agent can move: UP, DOWN, LEFT, RIGHT
"""

import numpy as np


class GridWorld:
    """
    A simple 5x5 grid world environment.

    KEY RL CONCEPTS in this class:
    - STATE: The agent's current position (row, col) on the grid.
    - ACTION: A choice the agent makes (up/down/left/right).
    - REWARD: A number the environment gives back after each action.
              Positive rewards encourage behavior, negative rewards discourage it.
    - EPISODE: One complete attempt from start to goal (or timeout).
    """

    def __init__(self):
        """Set up the grid world."""

        # Grid dimensions
        self.rows = 5
        self.cols = 5

        # Starting position (top-left corner)
        self.start = (4, 4)

        # Goal position (bottom-right corner)
        self.goal = (0, 0)

        # Wall positions — the agent cannot move into these cells
        self.walls = [(1, 1), (1, 2), (3, 1), (1, 3), (3,3)]

        # Actions the agent can take
        # We represent them as numbers: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.num_actions = 4

        # How each action changes the agent's position (row_change, col_change)
        # UP means row decreases by 1, DOWN means row increases by 1, etc.
        self.action_effects = {
            0: (-1, 0),   # UP:    move one row up
            1: (1, 0),    # DOWN:  move one row down
            2: (0, -1),   # LEFT:  move one column left
            3: (0, 1),    # RIGHT: move one column right
        }

        # Maximum steps per episode (prevents the agent from wandering forever)
        self.max_steps = 100

        # Initialize the agent's position and step counter
        self.agent_pos = self.start
        self.steps_taken = 0

    def reset(self):
        """
        Reset the environment for a new EPISODE.

        An EPISODE is one complete attempt — the agent starts at the beginning
        and tries to reach the goal. When an episode ends (success or timeout),
        we reset and start a new one.

        Returns:
            state (numpy array): The starting state [row, col] normalized to 0-1.
        """
        self.agent_pos = self.start
        self.steps_taken = 0
        return self.get_state()

    def get_state(self):
        """
        Get the current STATE as a numpy array.

        The STATE is all the information the agent needs to make a decision.
        Here, it's just the agent's position (row, col), normalized to be
        between 0 and 1. Neural networks work better with small numbers!

        Returns:
            state (numpy array): [row/4, col/4] — position normalized to 0-1.
        """
        row, col = self.agent_pos
        # Normalize: divide by (grid_size - 1) so values are between 0 and 1
        return np.array([row / (self.rows - 1), col / (self.cols - 1)],
                        dtype=np.float32)

    def step(self, action):
        """
        Take one step in the environment.

        This is the core function of any RL environment. The agent chooses
        an ACTION, and the environment returns:
        1. The new STATE (where the agent ended up)
        2. The REWARD (how good or bad that action was)
        3. Whether the episode is DONE (reached goal or timed out)

        Args:
            action (int): The action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).

        Returns:
            next_state (numpy array): The new state after taking the action.
            reward (float): The reward for this step.
            done (bool): Whether the episode is over.
        """
        self.steps_taken += 1

        # Calculate where the agent WOULD move
        row, col = self.agent_pos
        d_row, d_col = self.action_effects[action]
        new_row = row + d_row
        new_col = col + d_col
        old_pos = (row, col)

        # Check if the new position is valid (inside the grid and not a wall)
        if self._is_valid_position(new_row, new_col):
            self.agent_pos = (new_row, new_col)

        # --- REWARD DESIGN ---
        # This is one of the most important parts of RL!
        # The rewards tell the agent what behavior we want.

        # Did the agent reach the goal?
        if self.agent_pos == self.goal:
            reward = 10.0    # Big positive reward for reaching the goal!
            done = True       # Episode is over — success!

        # Did the agent run out of time?
        elif self.steps_taken >= self.max_steps:
            reward = -1.0    # Penalty for not finding the goal in time
            done = True       # Episode is over — timeout

        elif ( np.linalg.norm(np.array(old_pos) - np.array( self.goal)) 
                - np.linalg.norm(np.array(self.agent_pos) - np.array( self.goal)) ) < 0:
            reward = -0.5
            done = False
        # Otherwise, the agent is still exploring
        else:
            reward = -0.1    # Small penalty for each step (encourages efficiency)
            done = False      # Episode continues

        return self.get_state(), reward, done

    def _is_valid_position(self, row, col):
        """Check if a position is inside the grid and not a wall."""
        # Must be within grid boundaries
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        # Must not be a wall
        if (row, col) in self.walls:
            return False
        return True

    def render(self):
        """
        Print the grid to the terminal so we can see what's happening.

        This is useful for debugging and understanding what the agent is doing.
        """
        print("\n" + "=" * 15)
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                if (row, col) == self.agent_pos:
                    line += " A"    # A = Agent's current position
                elif (row, col) == self.goal:
                    line += " G"    # G = Goal
                elif (row, col) in self.walls:
                    line += " X"    # X = Wall
                else:
                    line += " ."    # . = Empty space
            print(line)
        print("=" * 15)
        print(f"Position: {self.agent_pos} | Steps: {self.steps_taken}")


# --- QUICK TEST ---
# If you run this file directly, it will show the grid and do a few random moves
if __name__ == "__main__":
    print("Testing the Grid World Environment")
    print("=" * 40)

    env = GridWorld()
    state = env.reset()
    env.render()

    print("\nTaking some random actions...")
    for i in range(5):
        action = np.random.randint(0, 4)  # Random action
        next_state, reward, done = env.step(action)
        print(f"\nAction: {env.action_names[action]} | "
              f"Reward: {reward} | Done: {done}")
        env.render()
        if done:
            break
