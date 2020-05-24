#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy as np
import pandas as pd
from collections import deque
import random

from keras.layers import Dense, Dropout
from keras.activations import relu, linear, softmax, sigmoid
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras import Sequential

import pickle
from matplotlib import pyplot as plt

class DQN:
    def __init__(self, env, lr, gamma, epsilon, decay):

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.counter = 0

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.r_list = []

        self.replay_memory_buffer = deque(maxlen=500000)
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.num_action_space = self.action_space.n
        self.num_observation_space = env.observation_space.shape[0]
        self.model = self.initialize_model()

    def initialize_model(self):
        model = Sequential()
        # Good Network Structure
#         model.add(Dense(512, input_dim=self.num_observation_space, activation=relu))
#         model.add(Dense(256, activation=relu))
#         model.add(Dense(self.num_action_space, activation=linear))
        
        # Exp 1 Simpler 
#         model.add(Dense(128, input_dim=self.num_observation_space, activation=relu))
#         model.add(Dense(64, activation=relu))
#         model.add(Dense(self.num_action_space, activation=linear))

        # Exp 2 Dropout
#         model.add(Dense(512, input_dim=self.num_observation_space, activation=relu))
#         model.add(Dense(256, activation=relu))
#         model.add(Dropout(0.2))
#         model.add(Dense(self.num_action_space, activation=linear))

        # Exp 3 Softmax
        model.add(Dense(512, input_dim=self.num_observation_space, activation=relu))
        model.add(Dense(256, activation=relu))
        model.add(Dense(self.num_action_space, activation=softmax))

        model.compile(loss=mean_squared_error,optimizer=Adam(lr=self.lr))
        return model

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_action_space)

        predicted_actions = self.model.predict(state)
        return np.argmax(predicted_actions[0])

    def add_to_replay(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((state, action, reward, next_state, done))
        
    def update_counter(self):
        self.counter += 1
        step_size = 5
        self.counter = self.counter % step_size

    def update_weights(self):

        if len(self.replay_memory_buffer) < self.batch_size or self.counter != 0:
            return

        # Early Stopping
        if np.mean(self.r_list[-10:]) > 200: 
            return

        random_sample = self.get_random_sample()
        states, actions, rewards, next_states, done_list = self.get_attribues(random_sample)
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target[[indexes], [actions]] = targets

        self.model.fit(states, target, epochs=1, verbose=0)

    def get_attribues(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list

    def get_random_sample(self):
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample

    def train(self, num_episodes, flag = False):
        for episode in range(num_episodes):
            state = self.env.reset()
            r_for_episode = 0
            num_steps = 1000
            s = np.reshape(state, [1, self.num_observation_space])
            for step in range(num_steps):
                self.env.render()
                a = self.get_action(s)
                n_s, r, done, info = self.env.step(a)
                n_s = np.reshape(n_s, [1, self.num_observation_space])

                self.add_to_replay(s, a, r, n_s, done)
                r_for_episode += r
                s = n_s
                self.update_counter()
                self.update_weights()

                if done:
                    break
            self.r_list.append(r_for_episode)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.decay

            r_mean = np.mean(self.r_list[-100:])
            if r_mean > 200 and flag:
                print("Training Complete...")
                break
            print(episode, " : Episode: Reward: ",r_for_episode, "  Average Reward: ",r_mean, " epsilon: ", self.epsilon )

    def save(self, name):
        self.model.save(name)

