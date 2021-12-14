# import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax
from tensorflow.keras.optimizers import Adam
from env import *


import gym
import argparse
import numpy as np
from collections import deque
import random
import math
import matplotlib.pyplot as plt

# tf.keras.backend.set_floatx('float32')
# wandb.init(name='C51', project="dist-rl-tf2")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--atoms', type=int, default=32)
parser.add_argument('--v_min', type=float, default=-5.)
parser.add_argument('--v_max', type=float, default=5.)

args = parser.parse_args()


class Dist_ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(
            np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionValueModel:
    def __init__(self, state_dim, action_dim, z):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = args.atoms
        self.z = z

        self.opt = Adam(args.lr)
        self.criterion = tf.keras.losses.CategoricalCrossentropy()
        self.model = self.create_model()

    def create_model(self):
        input_state = Input((self.state_dim,))
        h1 = Dense(64, activation='relu')(input_state)
        h2 = Dense(64, activation='relu')(h1)
        outputs = []
        for _ in range(self.action_dim):
            outputs.append(Dense(self.atoms, activation='softmax')(h2))
        return tf.keras.Model(input_state, outputs)

    def train(self, x, y):
        y = tf.stop_gradient(y)
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = self.criterion(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state, ep):
        state = np.reshape(state, [1, self.state_dim])
        eps = 1. / ((ep / 10) + 1)
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_optimal_action(state)

    def get_optimal_action(self, state):
        z = self.model.predict(state)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        return np.argmax(q)


class Agent:
    def __init__(self, env, episodes, id):
        self.env = env
        self.id = id
        if self.id == 'FourRooms-v0':
            env.reset()
            shape = env.render().flatten().shape
            self.state_dim = shape[0]
        else:
            self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.buffer = Dist_ReplayBuffer()
        self.batch_size = args.batch_size
        self.v_max = args.v_max
        self.v_min = args.v_min
        self.atoms = args.atoms
        self.delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]
        self.gamma = args.gamma
        self.q = ActionValueModel(self.state_dim, self.action_dim, self.z)
        self.q_target = ActionValueModel(
            self.state_dim, self.action_dim, self.z)
        self.target_update()
        self.episodes = episodes

    def target_update(self):
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)

    def replay(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        z = self.q.predict(next_states)
        z_ = self.q_target.predict(next_states)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        q = q.reshape((self.batch_size, self.action_dim), order='F')
        next_actions = np.argmax(q, axis=1)
        m_prob = [np.zeros((self.batch_size, self.atoms))
                  for _ in range(self.action_dim)]
        for i in range(self.batch_size):
            if dones[i]:
                Tz = min(self.v_max, max(self.v_min, rewards[i]))
                bj = (Tz - self.v_min) / self.delta_z
                l, u = math.floor(bj), math.ceil(bj)
                m_prob[actions[i]][i][int(l)] += (u - bj)
                m_prob[actions[i]][i][int(u)] += (bj - l)
            else:
                for j in range(self.atoms):
                    Tz = min(self.v_max, max(
                        self.v_min, rewards[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    l, u = math.floor(bj), math.ceil(bj)
                    m_prob[actions[i]][i][int(
                        l)] += z_[next_actions[i]][i][j] * (u - bj)
                    m_prob[actions[i]][i][int(
                        u)] += z_[next_actions[i]][i][j] * (bj - l)
        self.q.train(states, m_prob)

    def train(self, max_epsiodes=500):
        output = []
        for ep in range(max_epsiodes):
            if self.id == 'FourRooms-v0':
                self.env.reset()
                state = self.env.render()
            else:
                state = self.env.reset()
            done, total_reward, steps = False, 0, 0

            num_steps = 0
            while not done:
                num_steps += 1
                action = self.q.get_action(state, ep)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, -
                                1 if done else 0, next_state, done)

                if self.buffer.size() > 1000:
                    self.replay()
                if steps % 5 == 0:
                    self.target_update()

                state = next_state
                total_reward += reward
                steps += 1
                if done:
                    output.append(num_steps)
            # wandb.log({'reward': total_reward})
            if ep % 2 == 0:
                print('EP{} reward={}'.format(ep, total_reward))
        return output


def rolling_average(data, *, window_size=10):
    """Smoothen the 1-d data array using a rollin average.

    Args:
        data: 1-d numpy.array
        window_size: size of the smoothing window

    Returns:
        smooth_data: a 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]


def main():
    TRIALS = 10
    EPISODES = 150

    env_id = "CartPole-v0"
    env = gym.make(env_id)
    print(f'Running {env_id}')
    data = np.zeros((TRIALS, EPISODES))
    for t in range(TRIALS):
        print(f'{env_id} trial {t}')
        agent = Agent(env, EPISODES, 'env_id')
        data[t] = agent.train(EPISODES)

    plt.figure(figsize=(16, 8))
    avg = data.mean(axis=0)
    std = data.std(axis=0)
    length = len(avg)
    y_err = 1.96 * std * np.sqrt(1 / length)
    plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)

    plt.plot(avg, label='Original env')
    # plt.plot(rolling_average(avg))
    plt.xlabel("Episodes")
    plt.ylabel("Number of steps per episode")
    plt.title(f'Distributional RL on Original and Stochastic Environments')

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
    print(f'Running {env_id}')
    data = np.zeros((TRIALS, EPISODES))
    for t in range(TRIALS):
        print(f'{env_id} trial {t}')
        agent = Agent(env, EPISODES, 'env_id')
        data[t] = agent.train(EPISODES)

    avg = data.mean(axis=0)
    std = data.std(axis=0)
    length = len(avg)
    y_err = 1.96 * std * np.sqrt(1 / length)
    plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)

    plt.plot(avg, label='Stochastic env')
    # plt.plot(rolling_average(avg))
    plt.legend()  # loc=3, fontsize='small')
    plt.savefig(f'Pics/Distributional RL on Original and Stochastic Environments {TRIALS} runs {EPISODES} episodes.png')

    plt.show()


if __name__ == "__main__":
    main()
