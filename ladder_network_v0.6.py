#!usr/bin/python
from __future__ import print_function
import tensorflow as tf
import numpy as np

def generate_simulation_dataset():
    seqs = np.random.rand(50, 1000)
    lbls = np.random.randint(2, size=(50, 10))
    np.save("sequences", seqs)
    np.save("labels", lbls)


def gaussian_noise_layer(x, stddev):
    x_noise = tf.add(x, tf.random_normal(shape=tf.shape(x), stddev=stddev))
    return x_noise


def batch_norm(x):
    return tf.contrib.layers.batch_norm(x)


def activation(x, activation_type='relu'):

    if activation_type == 'relu':
        activation_out = tf.nn.relu(x)
    elif activation_type == 'sigmoid':
        activation_out = tf.nn.sigmoid(x)
    return activation_out


def encoder(x, W, L, Gammas, Betas, noise=0.0):
    h_tilde = {}
    z_tilde = {}
    batch_mean = {}
    batch_std = {}
    z_tilde_pre = {}

    h_tilde[0] = z_tilde[0] = gaussian_noise_layer(x, stddev=noise)
    # for layer in 1 ,2, ... L
    for l in xrange(1, L+1):
        z_tilde_pre[l] = tf.matmul(h_tilde[l-1], W[l])
        mean, varience = tf.nn.moments(z_tilde_pre[l], axes=[0])
        std = tf.sqrt(varience)
        batch_mean[l] = mean
        batch_std[l] = std
        z_tilde_bn = (z_tilde_pre[l] - mean)/std
        z_tilde[l]  = gaussian_noise_layer(z_tilde_bn, stddev=noise)
        if l < L:
            h_tilde[l] = activation(tf.add(z_tilde[l], Betas[l]), activation_type='relu')
        else:
            h_tilde[l] = activation(tf.multiply(Gammas[l], tf.add(z_tilde[l], Betas[l])), activation_type='sigmoid')

    return z_tilde, h_tilde[L]


def decoder(h_tilde_L, L, V, z_tildes, A, B):
    u = {}
    z_hat = {}
    z_hat_batch_norm = {}

    for l in xrange(L, -1, -1):
        if l == L:
            u[l] = batch_norm(h_tilde_L)
            mu = A[l][0]*tf.nn.sigmoid(A[l][1]*u[l] + A[l][2]) + A[l][3]*u[l] + A[l][4]
            v = A[l][5]*tf.nn.sigmoid(A[l][6]*u[l] + A[l][7]) + A[l][8]*u[l] + A[l][9]
            z_hat[l] = (z_tildes[l] - mu) * v + mu
            z_hat_batch_norm[l] = batch_norm(z_hat[l])

        else:
            u[l] = batch_norm(tf.matmul(z_hat[l+1], V[l+1]))
            mu = A[l][0]*tf.nn.sigmoid(A[l][1]*u[l] + A[l][2]) + A[l][3]*u[l] + A[l][4]
            v = A[l][5]*tf.nn.sigmoid(A[l][6]*u[l] + A[l][7]) + A[l][8]*u[l] + A[l][9]
            z_hat[l] = (z_tildes[l] - mu) * v + mu
            z_hat_batch_norm[l] = batch_norm(z_hat[l])

    return z_hat_batch_norm


def main():
    generate_simulation_dataset()
    sequences = np.load("sequences.npy")
    labels = np.load("labels.npy")
    # tf Graph input
    X = tf.placeholder(tf.float32, shape=(None, 1000))
    y_ = tf.placeholder(tf.float32, shape=(None, 10))
    # define weight for corrupted encoder,  clean encoder
    layer_units = [1000, 500, 200, 200, 10]
    lambda_weight= [1000, 10, 0.1, 0.1, 0.1]

    # define W, V, scaling Gammas, Bias Betas
    W = {}
    V = {}
    Gammas = {}
    Betas = {}

    A = {}
    # max layer number index, total layer number is len(layer_unit), first is layer 0
    L = len(layer_units) - 1
    # for l 1,2,3,...L
    for l in xrange(1, L + 1):
        W[l] = tf.Variable(tf.random_normal([layer_units[l-1], layer_units[l]]), dtype=tf.float32)
        V[l] = tf.Variable(tf.random_normal([layer_units[l], layer_units[l-1]]), dtype=tf.float32)
        Gammas[l] = tf.Variable(tf.ones([layer_units[l]]), dtype=tf.float32)
        Betas[l] = tf.Variable(tf.zeros([layer_units[l]]), dtype=tf.float32)

    for l in xrange(L+1):
        A[l] = []
        for i in xrange(10):
            A[l].append(tf.Variable(tf.ones([layer_units[l]]), dtype=tf.float32))

    B = {}

    z_tilde, h_tilde_L = encoder(X, W, L, Gammas, Betas, noise=0.5)
    z, _ = encoder(X, W, L, Gammas, Betas, noise=0.0)
    z_hat_batch_norm =  decoder(h_tilde_L, L, V, z_tilde, A, B)

    # define cost function
    Cost_c = tf.losses.sigmoid_cross_entropy(logits=h_tilde_L, multi_class_labels=y_)
    Cost_d = 0
    for l in xrange(L+1):
        Cost_d += lambda_weight[l] * tf.reduce_mean(tf.square(z[l] - z_hat_batch_norm[l]))

    Cost = Cost_c + Cost_d

    optimazer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(Cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in xrange(10000):
            _, Cost_o =  sess.run([optimazer, Cost], feed_dict={X: sequences, y_:labels})
            print(Cost_o)

if __name__=="__main__":
    main()
