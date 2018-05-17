# coding: utf-8

import numpy as np
# import cPickle as pickle
import pickle
import gym
import tensorflow as tf
import time
import os

gamma = 0.99

width = 80
height = 40
pixels_num = width * height

batch_size = 60
step_size = 40

log_dir = "./log_gumball_2"
lr = 1e-4

def prepro(I):
    I = I[35:195]
    # I = I[::4,::4,0] # downsample by factor of 4
    I = I[::4, ::2, 0]  # downsample by factor of 4
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.3)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.02, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1xn(x, n):
    return tf.nn.max_pool(x, ksize=[1, 1, n, 1], strides=[1, 1, n, 1], padding='VALID')


def max_pool_nxn(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')


def make_network():
    with tf.variable_scope('policy'):
        pixels = tf.placeholder(dtype=tf.float32, shape=[None, pixels_num * 2])
        x_image = tf.reshape(pixels, [-1, height, width, 2])

        actions = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        rewards = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        W_conv1 = weight_variable([5, 2, 2, 4])
        b_conv1 = bias_variable([4])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_nxn(h_conv1, 2)

        W_conv2 = weight_variable([3, 1, 4, 8])
        b_conv2 = bias_variable([8])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_nxn(h_conv2, 2)

        W_fc1 = weight_variable([10 * 20 * 8, 100])
        b_fc1 = bias_variable([100])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 20 * 8])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        W_fc2 = weight_variable([100, 1])
        b_fc2 = bias_variable([1])
        out = tf.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2, name="sigmoid")

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=actions, logits=out, name="cross_entropy")
        loss = tf.reduce_sum(tf.multiply(rewards, cross_entropy, name="rewards"))

    opt = tf.train.AdamOptimizer(lr).minimize(loss)

    tf.summary.histogram("hidden_out", W_conv2)
    tf.summary.histogram("prob_out", out)
    tf.summary.histogram("loss", loss)
    merged = tf.summary.merge_all()

    return pixels, actions, rewards, out, opt, merged


tf.reset_default_graph()
pix_ph, action_ph, reward_ph, out_sym, opt_sym, merged_sym = make_network()

resume = True
render = True

sess = tf.Session()
saver = tf.train.Saver()

if resume:
    saver.restore(sess, tf.train.latest_checkpoint(log_dir + '/checkpoints'))
else:
    sess.run(tf.global_variables_initializer())

#env = gym.make("Pong-v0")
env = gym.make("PongDeterministic-v4")
observation = env.reset()
# prev_x = np.zeros((pixels_num, ))
prev_x = None
xs = []
ys = []
ws = []
ep_ws = []
batch_ws = []
batch_xs = []
batch_ys = []
step = pickle.load(open(log_dir + '/step.p', 'rb')) if resume and os.path.exists(log_dir + '/step.p') else 0
episode_number = step
reward_mean = -21.0
win = 0
lose = 0

while True:
    if render: env.render()
    cur_x = prepro(observation)
    x = prev_x if prev_x is not None else np.zeros((pixels_num,))
    # x = np.r_[prev_x, cur_x]
    x = np.r_[x, cur_x]
    x = x.reshape((-1, pixels_num * 2))

    prev_x = cur_x

    tf_probs = sess.run(out_sym, feed_dict={pix_ph: x})
    y = 1 if np.random.uniform() < tf_probs[0, 0] else 0
    # y = 1 if 0.5 < tf_probs[0, 0] else 0
    action = 2 + y
    observation, reward, done, info = env.step(action)
    if done:
        break

env.close()
