import numpy as np
import math
import random as rand
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import tqdm


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

# Replay Buffer
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*rand.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

# Deep Q Network
class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            # TODO change here to switch between env
            nn.Flatten(),
            nn.Linear(num_inputs, 128),
            # nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = np.random.choice(env.action_space.n)
        return action


# model = DQN(env.observation_space.shape[0], env.action_space.n)
# optimizer = optim.Adam(model.parameters())
# replay_buffer = ReplayBuffer(1000)

# TD Loss
def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    with torch.no_grad():
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


# def plot(rewards, losses):
#     clear_output(True)
#     plt.figure(figsize=(16, 8))
#     plt.subplot(121)
#     plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
#     plt.plot(rewards)
#     plt.xlabel('Episode')
#     plt.xlabel('Reward')
#     plt.subplot(122)
#     plt.title('loss')
#     plt.plot(losses)
#     plt.xlabel('Episode')
#     plt.xlabel('Loss')
#     plt.show()


def train(model):
    losses = []
    # all_rewards = []
    all_rewards = np.zeros(EPISODES)

    output = []
    step = 1
    for i in tqdm(range(EPISODES)):

        # if env_id == 'FourRooms-v0':
        #     env.reset()
        #     state = env.render()
        # else:
        state = env.reset()
        episode_reward = 0
        num_steps = 0
        while True:
            num_steps += 1
            epsilon = epsilon_by_frame(step)
            action = model.act(state, epsilon)
            # print(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                # state = env.reset()
                # all_rewards.append(episode_reward)
                all_rewards[i] = episode_reward
                output.append(num_steps)
                break

            if len(replay_buffer) > batch_size:
                loss = compute_td_loss(batch_size)
                losses.append(loss.item())
            step += 1

    return all_rewards, losses, output


"""ENV SET UP (start)"""
# Cart Pole Environment
# from env import *
# register_env()
# env_id = "FourRooms-v0"
# # env_id = "CartPole-v0"
# env = gym.make(env_id)
# env.reset()
#
# print(env.observation_space.shape)


# Epsilon greedy exploration
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 10000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * \
                                     math.exp(-1. * frame_idx / epsilon_decay)
"""ENV SET UP (end)"""

steps = 10000
batch_size = 16
gamma = 0.99

if __name__ == '__main__':
    TRIALS = 20
    EPISODES = 100

    # data = np.zeros((TRIALS, EPISODES))
    # # rewards = np.zeros((TRIALS, EPISODES))
    # step = np.zeros((TRIALS, EPISODES))
    # for t in range(TRIALS):
    #     # TODO change here to switch between env
    #     shape = env.render().flatten().shape
    #
    #     model = DQN(shape[0], env.action_space.n)
    #     # model = DQN(env.observation_space.shape[0], env.action_space.n)
    #     optimizer = optim.Adam(model.parameters())
    #     replay_buffer = ReplayBuffer(10000)
    #     rewards, _, output = train(model)
    #     data[t] = output
    # plt.figure(figsize=(16, 8))
    #
    # avg = data.mean(axis=0)
    # std = data.std(axis=0)
    # length = len(avg)
    # y_err = 1.96 * std * np.sqrt(1 / length)
    # plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
    #
    # plt.plot(avg, label='DQN')
    # plt.xlabel("Episodes")
    # plt.ylabel("Number of steps per episode")
    # plt.legend() #loc=3, fontsize='small')
    # plt.title(f'{env_id} performance over {TRIALS} runs')
    # plt.savefig(f'Pics/{env_id} performance over {TRIALS} runs.png')
    #
    # plt.show()

    from typing import TypeVar
    import random

    Action = TypeVar('Action')

    class RandomActionWrapper(gym.ActionWrapper):
        def __init__(self, env, epsilon=0.1):
            super(RandomActionWrapper, self).__init__(env)
            self.epsilon = epsilon

        def action(self, action):
            if random.random() < self.epsilon:
                # print("Random!")
                return self.env.action_space.sample()
            return action

    env_id = "CartPole-v0"
    env = RandomActionWrapper(gym.make(env_id))

    data = np.zeros((TRIALS, EPISODES))
    step = np.zeros((TRIALS, EPISODES))
    for t in range(TRIALS):
        # TODO change here to switch between env
        # shape = env.render().flatten().shape

        # model = DQN(shape[0], env.action_space.n)
        model = DQN(env.observation_space.shape[0], env.action_space.n)
        optimizer = optim.Adam(model.parameters())
        replay_buffer = ReplayBuffer(1000)
        rewards, _, output = train(model)
        data[t] = output

    plt.figure(figsize=(16, 8))
    avg = data.mean(axis=0)
    std = data.std(axis=0)
    length = len(avg)
    y_err = 1.96 * std * np.sqrt(1 / length)
    plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)

    plt.plot(avg, label='DQN')
    plt.xlabel("Episodes")
    plt.ylabel("Number of steps per episode")
    plt.legend()  # loc=3, fontsize='small')
    plt.title(f'{env_id} performance over {TRIALS} runs (Stochastic Actions)')
    plt.savefig(f'Pics/{env_id} performance over {TRIALS} runs (Stochastic Actions.png')

    plt.show()

