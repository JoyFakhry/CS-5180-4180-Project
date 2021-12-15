import matplotlib.pyplot as plt
from test import DQN, optim, DQN_ReplayBuffer, train
from distributional_v2 import Dist_ReplayBuffer, Agent, rolling_average, vary_atoms
import numpy as np
import gym
from tqdm import tqdm
import random

TRIALS = 6
EPISODES = 200


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if random.random() < self.epsilon:
            # print("Random!")
            if action == 0:
                return 1
            if action == 1:
                return 0
            # return self.env.action_space.sample()
        return action

# env_id = "CartPole-v0"
# env = RandomActionWrapper(gym.make(env_id), epsilon=ep)


# Run DQN
# data = np.zeros((TRIALS, EPISODES))
# for t in range(TRIALS):
#     model = DQN(env.observation_space.shape[0], env.action_space.n)
#     optimizer = optim.Adam(model.parameters())
#     replay_buffer = DQN_ReplayBuffer(1000)
#     rewards, _, output = train(model, EPISODES, env, replay_buffer, optimizer)
#     data[t] = output
#
#
# plt.figure(figsize=(16, 8))
# avg = data.mean(axis=0)
# std = data.std(axis=0)
# length = len(avg)
# y_err = 1.96 * std * np.sqrt(1 / length)
# plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2, color='maroon')
#
# plt.plot(avg, label='DQN', color='maroon')
# plt.plot(rolling_average(avg), color='lightcoral', alpha=0.5)
# plt.xlabel("Episodes")
# plt.ylabel("Number of steps per episode")
# plt.title(f'DQN vs Dist RL on Stochastic Environment')
# plt.legend()
# plt.show()

env_id = "CartPole-v0"
eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Run Dist RL
data = np.zeros((len(eps), EPISODES))
for index, ep in enumerate(eps):
    env = RandomActionWrapper(gym.make(env_id), epsilon=ep)
    print(f'Running ep {ep}')
    agent = Agent(env, EPISODES, 'env_id', 8)
    data[index] = agent.train(EPISODES)
    np.save(f'Data/Epsilon/{ep} epsilon trial 5 {EPISODES} episodes', data)

# for index, ep in enumerate(eps):
#     avg = data[index].mean(axis=0)
#     std = data[index].std(axis=0)
#     length = len(avg)
#     y_err = 1.96 * std * np.sqrt(1 / length)
#     plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
#     plt.plot(avg, label=f'eps = {ep}')
#     plt.plot(rolling_average(avg), alpha=0.5)
# plt.legend()
# plt.savefig(f'Pics/Varying epsilon {TRIALS} trial {EPISODES} episodes.png')
# plt.show()

vary_atoms()
