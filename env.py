from enum import IntEnum
from typing import Tuple, Optional, List

import numpy as np
from gym import Env, spaces, ObservationWrapper
from gym.utils import seeding
from gym.envs.registration import register
from numpy import random


def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('FourRooms-v0')
    """
    register(id="FourRooms-v0", entry_point="env:FourRoomsEnv", max_episode_steps=500)


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


class FourRoomsEnv(Env):
    """Four Rooms gym environment."""

    def __init__(self, goal_pos=(10, 10)) -> None:
        self.rows = 11
        self.cols = 11

        # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
        self.walls = [
            (0, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 0),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 9),
            (5, 10),
            (6, 4),
            (7, 4),
            (9, 4),
            (10, 4),
        ]

        self.start_pos = (0, 0)
        self.goal_pos = goal_pos
        self.agent_pos = None

        self.space = np.zeros((self.rows, self.cols))
        for wall in self.walls:
            # print(wall)
            self.space[wall] = 1
        self.space[self.goal_pos] = 3

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )
        # self.observation_space = spaces.Discrete(self.cols * self.rows)

    def render(self, mode='rgb_array'):
        # valid space = 0
        # wall = 1
        # agent = 2
        # goal = 3

        self.space[self.agent_pos] = 2

        return self.space

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> Tuple[int, int]:
        """Reset agent to the starting position.

        Returns:
            observation (Tuple[int,int]): returns the initial observation
        """
        self.agent_pos = self.start_pos
        self.space = np.zeros((self.rows, self.cols))
        for wall in self.walls:
            self.space[wall] = 1
        self.space[self.goal_pos] = 3

        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 1.0
        else:
            done = False
            reward = 0.0

        # (there are 2 perpendicular actions for each action).
        # You can reuse your code from ex0
        n = self.noise()

        if action == Action.UP:
            if n == 1:
                action_taken = Action.LEFT
            elif n == 2:
                action_taken = Action.RIGHT
            else:
                action_taken = Action.UP

        if action == Action.LEFT:
            if n == 1:
                action_taken = Action.DOWN
            elif n == 2:
                action_taken = Action.UP
            else:
                action_taken = Action.LEFT

        if action == Action.RIGHT:
            if n == 1:
                action_taken = Action.UP
            elif n == 2:
                action_taken = Action.DOWN
            else:
                action_taken = Action.RIGHT

        if action == Action.DOWN:
            if n == 1:
                action_taken = Action.RIGHT
            elif n == 2:
                action_taken = Action.LEFT
            else:
                action_taken = Action.DOWN

        # You can reuse your code from ex0
        move = actions_to_dxdy(action_taken)
        next_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        # If the next position is a wall or out of bounds, stay at current position
        # Set self.agent_pos
        if next_pos in self.walls:  # If next state is a wall, keep current state
            pass
        elif (next_pos[0] < 0) or (next_pos[0] > 10):  # If x coordinate is out of bounds, keep current state
            pass
        elif (next_pos[1] < 0) or (next_pos[1] > 10):  # If y coordinate is out of bounds, keep current state
            pass
        else:
            self.agent_pos = next_pos

        return self.render(), reward, done, {}

    def noise(self):
        r = random.rand()
        if r <= 0.1:
            return 1  # denotes perpendicular direction 1
        elif r >= 0.9:
            return 2  # denotes perpendicular direction 2
        else:
            return 0


class FourRoomsWrapper1(ObservationWrapper):
    """Wrapper for FourRoomsEnv to aggregate rows.

    This wrapper aggregates rows, i.e. the original observation of the position (x,y) becomes just y

    To use this wrapper, create the environment using gym.make() and wrap the environment with this class.

    Ex) env = gym.make('FourRooms-v0')
        env = FourRoomsWrapper1(env)

    To use different function approximation schemes, create more wrappers, following this code.
    """

    def __init__(self, env):
        super().__init__(env)
        # Since this wrapper changes the observation space from a tuple of scalars to a single scalar,
        # need to change the observation_space
        self.observation_space = spaces.Discrete(self.cols * self.rows)

    def observation(self, observation):
        """Return the y-value of position"""
        return observation[1]