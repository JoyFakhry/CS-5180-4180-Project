import numpy as np

from gym_2048.envs.game2048_env import Game2048Env
from tqdm import trange

env = Game2048Env()

num_steps = 10_000

rewards = np.ndarray(num_steps)
env.render()

for t in trange(num_steps, desc='Steps'):
    a = env.action_space.sample()
    next_state, reward, done, info = env.step(a)
    rewards[t] = reward

    if done:
        env.reset()
env.render()

print(rewards)
np.savetxt("Data/rewards.csv", rewards, delimiter="   ")
