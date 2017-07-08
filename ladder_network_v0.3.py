#!usr/bin/python
from __future__ import print_function
import tensorflow as tf
import numpy as np

def generate_simulation_dataset():
    seqs = np.random.rand(50, 100)
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

    z_tilde_pre[1] = tf.matmul(h_tilde[0], W[1])
    mean, varience = tf.nn.moments(z_tilde_pre[1], axes=[0])
    std = tf.sqrt(varience)
    batch_mean[1] = mean
    batch_std[1] = std

    z_tilde_bn = (z_tilde_pre[1] - mean)/std
    z_tilde[1]  = gaussian_noise_layer(z_tilde_bn, stddev=noise)
    h_tilde[1] = activation(tf.add(z_tilde[1], Betas[1]), activation_type='relu')

    z_tilde_pre[2] = tf.matmul(h_tilde[1], W[2])
    mean, varience = tf.nn.moments(z_tilde_pre[2], axes=[0])
    std = tf.sqrt(varience)
    batch_mean[2] = mean
    batch_std[2] = std

    z_tilde_bn = (z_tilde_pre[2] - mean)/std
    z_tilde[2]  = gaussian_noise_layer(z_tilde_bn, stddev=noise)
    h_tilde[2] = activation(tf.add(z_tilde[2], Betas[2]), activation_type='relu')


    z_tilde_pre[3] = tf.matmul(h_tilde[2], W[3])
    mean, varience = tf.nn.moments(z_tilde_pre[3], axes=[0])
    std = tf.sqrt(varience)
    batch_mean[3] = mean
    batch_std[3] = std

    z_tilde_bn = (z_tilde_pre[3] - mean)/std
    z_tilde[3]  = gaussian_noise_layer(z_tilde_bn, stddev=noise)
    h_tilde[3] = activation(tf.multiply(Gammas[3], tf.add(z_tilde[3], Betas[3])), activation_type='sigmoid')

    return z_tilde, h_tilde[3], batch_mean, batch_std

def decoder(h_tilde_L, V, z_tildes, A, B):
    u = {}
    z_hat = {}
    z_hat_batch_norm = {}
    u[3] = batch_norm(h_tilde_L)
    mu = A[3][0]*tf.nn.sigmoid(A[3][1]*u[3] + A[3][2]) + A[3][3]*u[3] + A[3][4]
    v = A[3][5]*tf.nn.sigmoid(A[3][6]*u[3] + A[3][7]) + A[3][8]*u[3] + A[3][9]
    z_hat[3] = (z_tildes[3] - mu) * v + mu
    z_hat_batch_norm[3] = batch_norm(z_hat[3])

    u[2] = batch_norm(tf.matmul(z_hat[3], V[3]))
    mu = A[2][0]*tf.nn.sigmoid(A[2][1]*u[2] + A[2][2]) + A[2][3]*u[2] + A[2][4]
    v = A[2][5]*tf.nn.sigmoid(A[2][6]*u[2] + A[2][7]) + A[2][8]*u[2] + A[2][9]
    z_hat[2] = (z_tildes[2] - mu) * v + mu
    z_hat_batch_norm[2] = batch_norm(z_hat[2])


    u[1] = batch_norm(tf.matmul(z_hat[2], V[2]))
    mu = A[1][0]*tf.nn.sigmoid(A[1][1]*u[1] + A[1][2]) + A[1][3]*u[1] + A[1][4]
    v = A[1][5]*tf.nn.sigmoid(A[1][6]*u[1] + A[1][7]) + A[1][8]*u[1] + A[1][9]
    z_hat[1] = (z_tildes[1] - mu) * v + mu
    z_hat_batch_norm[1] = batch_norm(z_hat[1])

    u[0] = batch_norm(tf.matmul(z_hat[1], V[1]))
    mu = A[0][0]*tf.nn.sigmoid(A[0][1]*u[0] + A[0][2]) + A[0][3]*u[0] + A[0][4]
    v = A[0][5]*tf.nn.sigmoid(A[0][6]*u[0] + A[0][7]) + A[0][8]*u[0] + A[0][9]
    z_hat[0] = (z_tildes[0] - mu) * v + mu
    z_hat_batch_norm[0] = batch_norm(z_hat[0])

    return z_hat_batch_norm

def main():
    #generate_simulation_dataset()
    sequences = np.load("sequences.npy")
    labels = np.load("labels.npy")
    # tf Graph input
    X = tf.placeholder(tf.float32, shape=(None, 100))
    Y = tf.placeholder(tf.float32, shape=(None, 10))
    # define weight for corrupted encoder,  clean encoder
    layer_units = [100, 78, 64, 10]
    lambda_weight= [1000, 10, 0,1, 0.1]

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
            A[l].append(tf.Variable(tf.random_normal([layer_units[l]]), dtype=tf.float32))

    B = {}

    z_tilde, h_tilde_L, batch_mean, batch_std = encoder(X, W, L, Gammas, Betas, noise=0.5)
    z_hat_batch_norm =  decoder(h_tilde_L, V, z_tilde, A, B)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

if __name__=="__main__":
    main()
