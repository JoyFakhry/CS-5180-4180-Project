# import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

plt.style.use("ggplot")
import json
import numpy as np

import gym
import argparse
import numpy as np
from collections import deque
import random
import math
from tqdm import trange

tf.keras.backend.set_floatx('float64')
# wandb.init(name='C51', project="dist-rl-tf2")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--atoms', type=int, default=8)
parser.add_argument('--v_min', type=float, default=-5.)
parser.add_argument('--v_max', type=float, default=5.)

args = parser.parse_args()


class ReplayBuffer:
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

        self.opt = tf.keras.optimizers.Adam(args.lr)
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
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.buffer = ReplayBuffer()
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

    def train(self, rewards_recorder, trial, exp, max_epsiodes=500):
        rewards_vec = []
        steps_vec = []
        for ep in range(max_epsiodes):
            done, total_reward, steps = False, 0, 0
            state = self.env.reset()
            while not done:
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
            # wandb.log({'reward': total_reward})
            steps_vec.append(steps)
            rewards_vec.append(total_reward)

            # print('EP{} reward={}'.format(ep, total_reward))
        rewards_recorder[trial,] = rewards_vec
        save_ndarray(trial, max_epsiodes, f'{exp}_rewards', rewards_recorder)
        return rewards_vec, steps_vec


def save_ndarray(trial, episode, name, ndarray):
    with open(f'savedData/t{trial}_eps{episode}_{name}.json', 'w') as f:
        json.dump(ndarray.tolist(), f)


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return action

    def reverse_action(self, action):
        pass


def main():
    env = gym.make('CartPole-v0')
    sEnv = RandomActionWrapper(gym.make('CartPole-v0'))


    n_eps = 200
    n_trials = 100
    rewards_mat = np.zeros((n_trials, n_eps))
    # steps_mat = np.zeros((n_trials, n_eps))
    s_rewards_mat = np.zeros((n_trials, n_eps))
    # s_steps_mat = np.zeros((n_trials, n_eps))

    for t in trange(n_trials):
        agent = Agent(env)
        sAgent = Agent(sEnv)
        rewards, steps = agent.train(rewards_recorder=rewards_mat, trial=t, exp="det",
                                     max_epsiodes=n_eps)
        s_rewards, s_steps = sAgent.train(rewards_recorder=s_rewards_mat, trial=t,
                                          exp="sto", max_epsiodes=n_eps)
    # plt.figure()
    # ax = plt.axes()
    # ax.plot([i for i in range(len(rewards_vec))], rewards_vec)
    # ax.set_title("rewards")
    # plt.figure()
    # ax2 = plt.axes()
    # ax2.plot([i for i in range(len(steps_vec))], steps_vec)
    # ax2.set_title("steps")
    # plt.figure()
    # ax3 = plt.axes()
    # ax3.plot([i for i in range(len(s_rewards_vec))], s_rewards_vec)
    # ax3.set_title("rewards (stochastic)")
    # plt.figure()
    # ax4 = plt.axes()
    # ax4.plot([i for i in range(len(s_steps_vec))], s_steps_vec)
    # ax4.set_title("steps (stochastic)")
    # plt.show()


if __name__ == "__main__":
    main()
