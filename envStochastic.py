from gym import Env, spaces, ActionWrapper
import env
import numpy as np


class CartPoleWrapper(ActionWrapper):
    """Wrapper for CartPoleEnv to introduce stochasticity.
    To use this wrapper, create the environment using gym.make() and wrap the environment with this class.
    Ex) env = gym.make('CartPole-v0')
        env = CartPoleWrapper(env)
    """

    def __init__(self, env):
        super().__init__(env)
        # Since this wrapper changes the observation space from a tuple of scalars to a single scalar,
        # need to change the observation_space
        # self.observation_space = spaces.Discrete(self.cols * self.rows)

    def action(self, action):
        if np.random.random() < 0.2:
            action = abs(action - 1)
        return action

    def reverse_action(self, action):
        pass


