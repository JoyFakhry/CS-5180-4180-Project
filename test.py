from gym_2048.envs.game2048_env import Game2048Env
from tqdm import trange

env = Game2048Env()

num_steps = 10_000

## Testing merge

while True:
    a = env.action_space.sample()
    next_state, reward, done, info = env.step(a)
    env.render()

    if done:
        print(info)
        break

print(env.score)

#as;dfkhas;lkdfja;sdlkfj