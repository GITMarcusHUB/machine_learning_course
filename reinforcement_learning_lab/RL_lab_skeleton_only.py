"""
ML framework for a reinforcement learning problem
"""
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import gym
import matplotlib.pyplot as plt


def create_model():
    model = Sequential([
        Dense(10, input_shape = (4,), activation = 'relu'),
        Dense(10, activation = 'relu'),
        Dense(1, activation = 'sigmoid'),
        ])
    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'sgd')
    return model


def pg_loss(args):
    taken_action, pi, cum_reward = args
    # TODO implement the Policy Gradient loss
    # the gradient of which is the gradient used in the REINFORCE algorithm
    # Note: use Keras backend stuff instead of Numpy ones
    # Remember that Keras minimizes, we want to maximize
    pg_loss = K.mean(-K.log(K.sum(taken_action * pi)) * cum_reward)
    return pg_loss


def create_trainable_model(action_model):
    ins = [action_model.input]
    taken_action = Input(name = 'taken_action', shape = (1,))
    cum_reward = Input(name = 'cumulative_reward', shape = (1,))
    loss = Lambda(pg_loss, output_shape = (1,))(
        [taken_action, action_model.output, cum_reward])

    trainable_model = Model(inputs = ins + [taken_action, cum_reward],
                            outputs = [loss])
    # Loss is already calculated at the end of the trainable model
    loss = lambda _, y_pred: y_pred
    trainable_model.compile(loss = loss, optimizer = Adam(lr = 1e-5))
    return trainable_model


def train_on_episode(trainable_model, states, actions, rewards, discount_factor=0.99):
    # TODO calculate cumulative (discounted) rewards
    discounted_rewards = K.zeros_like(rewards)
    current = 0
    for t in reversed(range(len(rewards))):
        current = current * discount_factor + rewards[t]
        discounted_rewards[t] = current
    # normalize the cumlative reward
    discounted_rewards -= discounted_rewards.mean() / discounted_rewards.std()

    trainable_model.train_on_batch([states, actions, discounted_rewards], dummy_variable)


def train(num_eps, trainable_model):
    # TODO: do interaction with cartpole, collect state transitions and
    # rewards, train on the episode
    return None


def eval(num_eps, model):
    env = gym.make('CartPole-v0')
    total_history = []
    ep = 0
    model = create_trainable_model(model)
    for _ in range(num_eps):
        state = env.reset()
        done = False
        episodic_reward = 0
        episode_history = []
        while not done:
            # env.render()
            action = 1 if model.predict(state[0].reshape(1, -1), state[1].reshape(1, -1), state[2].reshape(1, -1))>0.5 else 0
            state_, reward, done, _ = env.step(action)
            episode_history.append([state, action, reward, state_])
            state = state_
            episodic_reward += reward
        ep += 1
        total_history.append(episode_history)
        train_on_episode(model, episode_history[:, 0], episode_history[:, 1], episode_history[:, 2])
    env.close()


def main():
    num_eps = 40
    eval_num = 2
    agent = create_model()
    print(f'agent model summary: ')
    agent.summary()
    # train(num_eps, agent)
    eval(eval_num, agent)


if __name__ == "__main__":
    main()