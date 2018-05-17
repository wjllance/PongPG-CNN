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
train_dir = "/train_522_312_b6s4_fc100_auto_r_std_0.3_1e-4"


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True      #程序按需申请内存

# sess = tf.Session(config = config)
reward_mean = -21

def prepro(I):
    I = I[35:195]
    # I = I[::4,::4,0] # downsample by factor of 4
    I = I[::4, ::2, 0]  # downsample by factor of 4
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


def discount_rewards(pre_r):
    r = np.zeros_like(pre_r)
    sum_r = 0
    for i in reversed(range(0, len(r))):
        if pre_r[i] != 0:
            sum_r = pre_r[i]
            if pre_r[i] > 0 and reward_mean < -1:
                sum_r = 20
            if pre_r[i] < 0 and reward_mean > 1:
                sum_r = -20
        else:
            sum_r = sum_r * gamma + pre_r[i]
        r[i] = sum_r
    return r


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

    # decay_rate=0.99
    # opt = tf.train.AdamOptimizer(lr, decay=decay_rate).minimize(loss)
    opt = tf.train.AdamOptimizer(lr).minimize(loss)

    # tf.summary.image("h_pool1", h_pool1)
    # tf.summary.image("h_pool3", h_pool3)

    tf.summary.histogram("hidden_out", W_conv2)
    # tf.summary.histogram("logits_out", logits)
    tf.summary.histogram("prob_out", out)
    tf.summary.histogram("loss", loss)
    merged = tf.summary.merge_all()

    # grads = tf.gradients(loss, [hidden_w, logit_w])
    # return pixels, actions, rewards, out, opt, merged, grads
    return pixels, actions, rewards, out, opt, merged


tf.reset_default_graph()
pix_ph, action_ph, reward_ph, out_sym, opt_sym, merged_sym = make_network()

#resume = False
resume = True
render = False

# sess = tf.Session(config = config)
sess = tf.Session()
saver = tf.train.Saver()
writer = tf.summary.FileWriter(log_dir + train_dir, sess.graph)

if resume:
    saver.restore(sess, tf.train.latest_checkpoint(log_dir + '/checkpoints'))
else:
    sess.run(tf.global_variables_initializer())

env = gym.make("PongDeterministic-v4")
# env = gym.make("Pong-v0")
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
#reward_mean = -21.0
win = 0
lose = 0

while True:
    # print("begin training...")
    if render: env.render()
    cur_x = prepro(observation)
    x = prev_x if prev_x is not None else np.zeros((pixels_num,))
    # x = np.r_[prev_x, cur_x]
    x = np.r_[x, cur_x]
    x = x.reshape((-1, pixels_num * 2))

    prev_x = cur_x

    tf_probs = sess.run(out_sym, feed_dict={pix_ph: x})
    y = 1 if np.random.uniform() < tf_probs[0, 0] else 0
    action = 2 + y
    observation, reward, done, info = env.step(action)

    xs.append(x)
    ys.append(y)
    ep_ws.append(reward)
    if len(xs) > batch_size:
        xs.pop(0)
        ys.pop(0)
        ep_ws.pop(0)
    if reward > 0:
        win = win + 1
    elif reward < 0:
        lose = lose + 1

    if reward != 0:
        step += 1
        discounted_epr = discount_rewards(ep_ws)
        # discounted_epr -= np.mean(discounted_epr)
        # discounted_epr /= np.std(discounted_epr)
        # print(type(discounted_epr), discounted_epr.shape)
        batch_ws += discounted_epr.tolist()
        batch_xs += xs
        batch_ys += ys
        xs = []
        ys = []
        ep_ws = []
        if step % step_size == 0:
            # print(batch_ys)
            # print(batch_ws)
            batch_ws -= np.mean(batch_ws)
            batch_ws /= np.std(batch_ws)
            # print(batch_ws)
            exs = np.vstack(batch_xs)
            eys = np.vstack(batch_ys)
            ews = np.vstack(batch_ws)
            # print(exs.shape)
            # print(eys.shape)
            # print(ews.shape)
            frame_size = len(batch_xs)
            batch_xs = []
            batch_ys = []
            batch_ws = []
            tf_opt, tf_summary = sess.run([opt_sym, merged_sym],
                                          feed_dict={pix_ph: exs, action_ph: eys, reward_ph: ews})
            saver.save(sess, log_dir + "/checkpoints/pg_{}.ckpt".format(step))
            writer.add_summary(tf_summary, step)
            fp = open(log_dir + '/step.p', 'wb')
            pickle.dump(step, fp)
            fp.close()
            print("datetime: {}, update step: {}, frame size: {}". \
                  format(time.strftime('%X %x %Z'), step, frame_size))

        if done:
            episode_number += 1
            reward_mean = 0.99 * reward_mean + (1 - 0.99) * (win - lose)
            rs_sum = tf.Summary(value=[tf.Summary.Value(tag="running_reward", simple_value=reward_mean)])
            writer.add_summary(rs_sum, global_step=episode_number)
            print("datetime: {}, episode: {}, reward: {}, win:lose= {}:{}". \
                  format(time.strftime('%X %x %Z'), episode_number, reward_mean, win, lose))

            observation = env.reset()
            prev_x = None
            win = 0
            lose = 0
            if render: env.render()

env.close()



git