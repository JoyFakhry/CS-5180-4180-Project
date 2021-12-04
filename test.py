import math, random

import gym
import numpy as np

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
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


# Cart Pole Environment
env_id = "CartPole-v0"
env = gym.make(env_id)
# from env import *
# register_env()
# env = gym.make('FourRooms-v0')
# print(env.observation_space)

# Epsilon greedy exploration
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 5000


epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * \
                                     math.exp(-1. * frame_idx / epsilon_decay)

# Deep Q Network


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
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
            action = random.randrange(env.action_space.n)
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


def plot(rewards, losses):
    clear_output(True)
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.xlabel('Reward')
    plt.subplot(122)
    plt.title('loss')
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.xlabel('Loss')
    plt.show()


steps = 10000
batch_size = 32
gamma = 0.99


def train(model):
    losses = []
    # all_rewards = []
    all_rewards = np.zeros(EPISODES)
    steps = np.zeros(EPISODES)
    episode_reward = 0
    step = 1
    for i in tqdm(range(EPISODES)):

        state = env.reset()
        episode_reward = 0
        while True:
            epsilon = epsilon_by_frame(step)
            action = model.act(state, epsilon)

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                # state = env.reset()
                # all_rewards.append(episode_reward)
                all_rewards[i] = episode_reward
                steps[i] = step
                break

            if len(replay_buffer) > batch_size:
                loss = compute_td_loss(batch_size)
                losses.append(loss.item())

            step += 1

    return all_rewards, losses, steps

if __name__ == '__main__':
    TRIALS = 3
    EPISODES = 100

    data = np.zeros((TRIALS, EPISODES))
    # rewards = np.zeros((TRIALS, EPISODES))
    step = np.zeros((TRIALS, EPISODES))
    for t in range(TRIALS):
        model = DQN(env.observation_space.shape[0], env.action_space.n)
        optimizer = optim.Adam(model.parameters())
        replay_buffer = ReplayBuffer(1000)
        rewards, _, step[t,:] = train(model)
        data[t] = rewards
        # rewards = np.array(rewards)


    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    # plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    std = step.std(axis=0)
    avg = step.mean(axis=0)
    length = len(std)
    y_err = 1.96 * std * np.sqrt(1 / length)
    plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)

    plt.plot(step.mean(axis=0))
    # plt.xlabel('Episode')
    # plt.xlabel('Reward')
    plt.subplot(122)
    # plt.title('loss')
    # plt.plot(losses)
    # plt.xlabel('Episode')
    # plt.xlabel('Loss')
    # plt.show()

    avg = data.mean(axis=0)
    std = data.std(axis=0)
    length = len(avg)
    y_err = 1.96 * std * np.sqrt(1 / length)
    plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)

    plt.plot(avg, label='DQN')
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend() #loc=3, fontsize='small')
    plt.title(f'Cartpole v0 Average Rewards over {TRIALS} runs')

    plt.show()

