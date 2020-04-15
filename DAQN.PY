###########################################################
#   Deep Reinforcement Learning project: Deep Auto-Encoder Q-Network learning
#   by Hend Zouaoui id: 1800809
###########################################################
from rps import RPS # Rock Paper Scissors simulator
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
filepath = r"C:\Users\Hend\Documents\DLCV\DLCV_1800809\Check23" # path to laod the auto-encoder
savepath = r"C:\Users\Hend\Documents\DLCV\DLCV_1800809\Check31" # path to save the trained DAQN
n_inputs = 120
n_outputs = n_inputs
learning_rate = 0.001
N = 500 # 1000 in the paper
batch_size = 16 # 32 in the paper
epsilon = 1.0 # parameters for Epsilon-greedy
epsilon_min = 0.1
n_epochs = 5000
C = 10000 # target update periods

class DAQNAgent:
    def __init__(self, model, target_model):
        self.model = model
        self.target_model = target_model
        self.states = np.empty([N, n_inputs, n_inputs, 3])
        self.actions = np.empty((N), dtype=np.int8)
        self.rewards = np.empty((N), dtype=np.int8)
        self.next_idx = 0
        self.size_D = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min

    def train(self):
        for e in range(1, N):
            game = RPS() # initialize a new game
            self.state = game.state
            print(game.hidden_state)
            q_values = self.model.predict(self.state[None]) # input the state image to the Q-net
            action = np.argmax(q_values, axis=-1)  # select the best action
            #print(action)
            best_action = action[0,0,0]
            print('best action', best_action)
            reward = game.play(best_action)    # perform the action in the game to get the reward
            print('reward', reward)
            self.store_transition(self.state, best_action, reward)  # store that transition into replay buffer
            self.size_D = self.size_D + 1
        print('Filled the D buffer')
        for t in range(1, n_epochs):
            #print('initializing a game')
            print(game.hidden_state)
            game = RPS()
            self.state = game.state
            q_values = self.model.predict(self.state[None]) # input the state image to the Q-net
            action = np.argmax(q_values, axis=-1)  # select the best action
            #print(action)
            best_action = action[0,0,0]
            print('best action', best_action)
            exploration_action = self.epsilon_greedy(best_action)   # Epsilon-greedy to select either the best action or a random one
            if self.epsilon > self.epsilon_min: # if epsilon didn't reach its minimum value, dicrease it linearly
                self.epsilon = self.epsilon - (1 - self.epsilon_min)/1000
            reward = game.play(exploration_action)    # perform the action in the game to get the reward
            print('selected action', exploration_action)
            print('reward', reward)
            self.store_transition(self.state, exploration_action, reward)  # store that transition into replay buffer
            self.train_step() #sample a minibatch and train on it
            if t % C == 0: # update the target network every C steps
                self.target_model.set_weights(self.model.get_weights())

    def store_transition(self, state, action, reward):
        n_idx = self.next_idx
        self.states[n_idx] = state
        self.actions[n_idx] = action
        self.rewards[n_idx] = reward
        self.next_idx = (self.next_idx + 1) % N

    def epsilon_greedy(self, best_action):
        if np.random.rand() < self.epsilon:
            return np.random.random_integers(low=0, high=2)
        return best_action

    def train_step(self):
        samples_i = self.sample(batch_size)
        samples_s = self.states[samples_i]
        samples_a = self.actions[samples_i]
        samples_r = self.rewards[samples_i]
        Qs = self.model.predict(samples_s)
        for i, val in enumerate(samples_a):
            Qs[i][val] = samples_r[i]
        losses = self.model.train_on_batch(samples_s, Qs)

    def sample(self, n):
        assert n < self.size_D
        res = []
        while len(res) < n:
            num = np.random.randint(0, self.size_D)
            if num not in res:
                res.append(num)
        return res

    def evaluation(self):
        game = RPS()
        state = game.state
        q_values = self.model.predict(state[None])
        action = np.argmax(q_values, axis=-1)
        reward = game.play(action)
        return reward

if __name__ == '__main__':
    # Restore aut-encoder, remove decoder layer, add fully connected layer and save as the Q-network
    pretrainedae1 = tf.keras.models.load_model(filepath)
    pretrainedae1.pop()
    pretrainedae1.pop()
    pretrainedae1.pop()
    pretrainedae1.pop()
    pretrainedae1.pop()
    pretrainedae1.pop()
    fully_connect1 = tf.keras.layers.Dense(units=3, activation = tf.nn.relu, kernel_initializer=tf.random_uniform_initializer)
    pretrainedae1.add(fully_connect1)
    # Restore aut-encoder, remove decoder layer, add fully connected layer and save as the target network
    pretrainedae2 = tf.keras.models.load_model(filepath)
    pretrainedae2.pop()
    pretrainedae2.pop()
    pretrainedae2.pop()
    pretrainedae2.pop()
    pretrainedae2.pop()
    pretrainedae2.pop()
    fully_connect2 = tf.keras.layers.Dense(units=3, activation = tf.nn.relu, kernel_initializer=tf.random_uniform_initializer)
    pretrainedae2.add(fully_connect2)
    # Compile both networks
    pretrainedae1.compile(optimizer='adam', loss='MSE',metrics=['accuracy'])
    pretrainedae2.compile(optimizer='adam', loss='MSE',metrics=['accuracy'])
    # TensorBoard log
    tb1 = tf.keras.callbacks.TensorBoard(log_dir='logdir0')
    tb1.set_model(pretrainedae1)
    # Initialize the DAQN agent, train it and save
    agent = DAQNAgent(pretrainedae1, pretrainedae2)
    print('DAQN agent initialized... starting the training')
    agent.train()
    target_model.save(savepath)
    # Evaluate the DAQN agent
    #reward = agent.evaluation()
    #reward = reward + agent.evaluation()
    #reward = reward + agent.evaluation()
    #print("After Training: %d out of 3" % reward)
