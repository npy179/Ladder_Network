#!usr/bin/python
from __future__ import print_function
import tensorflow as tf
import numpy as np
from dataset import Label_Dataset, SemiDataset

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

def encoder(x, W, L, Gammas, Betas, training=True, noise=0.0):
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

    labeled_dataset = np.load("label_dataset_sample.npy")
    ulabeled_dataset = np.load("unlabel_dataset_sample.npy")

    batch_size = 5
    n_epochs = 1000

    num_label_example, _ = labeled_dataset.shape
    num_iter = (num_label_example * n_epochs) / batch_size

    ld = SemiDataset(labeled_dataset, ulabeled_dataset, batch_size)

    test_data = Label_Dataset(labeled_dataset, batch_size)
    # tf Graph input
    X = tf.placeholder(tf.float32, shape=(None, 2195))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    # define weight for corrupted encoder,  clean encoder
    layer_units = [2195, 1000, 1000, 500, 200, 1]
    lambda_weight= [1000, 10, 0.1, 0.1, 0.1, 0.1]

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

    z, predict_y = encoder(X, W, L, Gammas, Betas, noise=0.0)
    z_hat_batch_norm =  decoder(h_tilde_L, L, V, z_tilde, A, B)

    h_tilde_label_L = tf.slice(h_tilde_L, [0, 0], [batch_size, -1])
    y_label_predict = tf.slice(predict_y, [0, 0], [batch_size, -1])

    # define cost function
    # NaN problem here 0.0
    Cost_c = tf.losses.sigmoid_cross_entropy(logits=h_tilde_label_L, multi_class_labels=y_)

    correct_prediction = tf.equal(tf.cast(tf.greater(y_label_predict, 0.5), tf.float32), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # NaN problem maybe coused by batch size
    Cost_d = 0
    for l in xrange(L+1):
        Cost_d += lambda_weight[l] * tf.reduce_mean(tf.square(z[l] - z_hat_batch_norm[l]))

    loss = Cost_c + Cost_d

    optimazer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)


    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in xrange(num_iter):
            sequences_batch, labels_batch = ld.next_batch()
            _, loss0, train_accuracy =  sess.run([optimazer, loss, accuracy], feed_dict={X: sequences_batch, y_:labels_batch})
            print(loss0, " ", train_accuracy)


        for j in xrange(num_iter):
            test_sequences_batch, test_labels_batch = test_data.next_batch()
            test_accuracy =  sess.run([accuracy], feed_dict={X:test_sequences_batch, y_:test_labels_batch})
            print(test_accuracy)

if __name__=="__main__":
    main()
