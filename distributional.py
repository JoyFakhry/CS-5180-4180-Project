import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from tqdm import tqdm

from test import ReplayBuffer

from IPython.display import clear_output
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

"""ENV SET UP (start)"""
# # Cart Pole Environment
# import env as envir
# envir.register_env()
# env_id = "FourRooms-v0"
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

env_id = "CartPole-v1"
env = RandomActionWrapper(gym.make(env_id))
#
# # Epsilon greedy exploration
# epsilon_start = 1.0
# epsilon_final = 0.01
# epsilon_decay = 10000
# epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * \
#                                      math.exp(-1. * frame_idx / epsilon_decay)
"""ENV SET UP (end)"""


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class CategoricalDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms, Vmin, Vmax):
        super(CategoricalDQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 128)
        self.noisy1 = NoisyLinear(128, 512)
        self.noisy2 = NoisyLinear(512, self.num_actions * self.num_atoms)
        # self.layers = nn.Sequential(
        #     # TODO change here to switch between env
        #     nn.Flatten(),
        #     nn.Linear(num_inputs, 128),
        #     # nn.Linear(env.observation_space.shape[0], 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     NoisyLinear(128, 512),
        #     nn.ReLU(),
        #     NoisyLinear(512, self.num_actions * self.num_atoms),
        #     # F.softmax()
        # )

    def forward(self, x):
        # print(x.flatten().shape)
        # x = x.flatten()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        x = F.softmax(x.view(-1, self.num_atoms), dim=-1).view(-1, self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

    def act(self, state):
        with torch.no_grad():
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(Vmin, Vmax, num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action


def projection_distribution(next_state, rewards, dones):
    batch_size = next_state.size(0)

    delta_z = float(Vmax - Vmin) / (num_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_atoms)

    next_dist = target_model(next_state).data.cpu() * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist = next_dist.gather(1, next_action).squeeze(1)

    rewards = rewards.unsqueeze(1).expand_as(next_dist)
    dones = dones.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)

    Tz = rewards + (1 - dones) * 0.99 * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b = (Tz - Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long() \
        .unsqueeze(1).expand(batch_size, num_atoms)

    proj_dist = torch.zeros(next_dist.size())
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

    return proj_dist


num_atoms = 51
Vmin = -10
Vmax = 10


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    with torch.no_grad():
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(np.float32(done))

    proj_dist = projection_distribution(next_state, reward, done)
    dist = current_model(state)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
    dist = dist.gather(1, action).squeeze(1)
    dist.data.clamp_(0.01, 0.99)
    loss = - (Variable(proj_dist) * dist.log()).sum(1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    current_model.reset_noise()
    target_model.reset_noise()

    return loss


def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(12,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


num_frames = 5000
batch_size = 2
gamma = 0.99
# EPISODES = 200


def train(current_model, target_model):
    losses = []
    all_rewards = []

    output = []
    step = 1
    for i in tqdm(range(EPISODES)):
        if env_id == 'FourRooms-v0':
            env.reset()
            state = env.render()
        else:
            state = env.reset()
        episode_reward = 0
        num_steps = 0
        while True:
            num_steps += 1
            # print(state)
            action = current_model.act(state)

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                # state = env.reset()
                all_rewards.append(episode_reward)
                output.append(num_steps)
                break

            if len(replay_buffer) > batch_size:
                loss = compute_td_loss(batch_size)
                losses.append(loss.item())

            # if frame_idx % 200 == 0:
            #     plot(frame_idx, all_rewards, losses)

            if t % 100 == 0:
                update_target(current_model, target_model)

    return all_rewards, losses, output


if __name__ == '__main__':
    TRIALS = 5
    EPISODES = 150

    data = np.zeros((TRIALS, EPISODES))
    # rewards = np.zeros((TRIALS, EPISODES))
    step = np.zeros((TRIALS, EPISODES))
    for t in range(TRIALS):
        # shape = env.render().flatten().shape
        # current_model = CategoricalDQN(shape[0], env.action_space.n, num_atoms, Vmin, Vmax)
        # target_model = CategoricalDQN(shape[0], env.action_space.n, num_atoms, Vmin, Vmax)
        current_model = CategoricalDQN(env.observation_space.shape[0], env.action_space.n, num_atoms, Vmin, Vmax)
        target_model = CategoricalDQN(env.observation_space.shape[0], env.action_space.n, num_atoms, Vmin, Vmax)

        if USE_CUDA:
            print('hi')
            current_model = current_model.cuda()
            target_model = target_model.cuda()

        optimizer = optim.Adam(current_model.parameters())

        replay_buffer = ReplayBuffer(10000)
        update_target(current_model, target_model)

        rewards, _, output = train(current_model, target_model)
        # print(len(losses))
        # if np.mean(rewards[-10:]) > 30:
        data[t] = output

    plt.figure(figsize=(16, 8))
    # plt.subplot(121)
    # # plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    # std = step.std(axis=0)
    # avg = step.mean(axis=0)
    # length = len(std)
    # y_err = 1.96 * std * np.sqrt(1 / length)
    # plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
    #
    # plt.plot(step.mean(axis=0))
    # plt.xlabel('Episode')
    # plt.xlabel('Reward')
    # plt.subplot(122)
    # plt.title('loss')
    # plt.plot(losses)
    # plt.xlabel('Episode')
    # plt.xlabel('Loss')
    # plt.show()

    # data = data[~np.all(data == 0, axis=1)]
    print(data.shape)
    avg = data.mean(axis=0)
    std = data.std(axis=0)
    length = len(avg)
    y_err = 1.96 * std * np.sqrt(1 / length)
    plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)

    plt.plot(avg, label='Distributional RL')
    plt.xlabel("Episodes")
    plt.ylabel("Number of steps per episode")
    plt.legend()  # loc=3, fontsize='small')
    plt.title(f'{env_id} performance over {TRIALS} runs, DRL')
    plt.savefig(f'Pics/{env_id} performance over {TRIALS} runs, DRL.png')

    plt.show()
