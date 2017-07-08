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

def activation(x, ftype='relu'):

    if ftype == 'relu':
        activation_out = tf.nn.relu(x)
    elif ftype == 'softmax':
        activation_out = tf.nn.softmax(x)
    return activation_out


def encoder(x, weights, Gammas, Betas, noise=0.0):

    encoder_latent_layers_z_tilde = {}
    # first layer
    h_tilde = z_tilde = gaussian_noise_layer(x, stddev=noise)
    encoder_latent_layers_z_tilde['z_tilde_0'] = z_tilde

    for i in xrange(1, len(weights.keys())):
        weight_key = 'encoder_h'+str(i)
        beta_key = 'beta_' + str(i)
        gamma_key = 'gamma_' + str(i)

        z_tilde_pre = tf.matmul(h_tilde, weights[weight_key])
        x_batch_norm = batch_norm(z_tilde_pre)
        z_tilde = gaussian_noise_layer(x_batch_norm, stddev=noise)

        if i < len(weights.keys()):
            h_tilde = activation(tf.add(z_tilde, Betas[beta_key]), ftype='relu')
        else:
            h_tilde = activation(tf.multiply(Gammas[gamma_key], tf.add(z_tilde, Betas[beta_key])), ftype='softmax')

        z_tilde_key = 'z_tilde_'+str(i)
        encoder_latent_layers_z_tilde[z_tilde_key] = z_tilde

    return h_tilde, encoder_latent_layers_z_tilde


def decoder(x, weights, z_tildes, A, B):

    decoder_latent_layers_z_hat_bn = {}
    for layer in xrange(len(weights.keys())-1, -1, -1):

        if layer == len(weights.keys()) - 1:
            u = batch_norm(x)
            z_tilde_key = 'z_tilde_'+str(layer)
            weight_key = 'decoder_h'+str(layer)
            z_tilde = z_tildes[z_tilde_key]


            mu = A[layer][0] * tf.nn.sigmoid(A[layer][1] * u +
                                             A[layer][2]) + A[layer][3] * u + A[layer][4]

            v = A[layer][5] * tf.nn.sigmoid(A[layer][6] * u +
                                            A[layer][7]) + A[layer][8] * u + A[layer][9]

            z_hat = (z_tilde - mu) * v + mu
            z_hat_bn = batch_norm(z_hat)
            u_next = batch_norm(tf.matmul(z_hat , weights[weight_key]))
        """
        else:
            z_tilde_key = 'z_tilde_'+str(layer)
            weight_key = 'decoder_h'+str(layer)
            decoder_latent_layers_z_hat_bn_key = 'z_hat_'+str(layer)+'_bn'
            z_tilde = z_tildes[z_tilde_key]
            mu = A[layer][0] * tf.nn.sigmoid(A[layer][1] * u +
                                             A[layer][2]) + A[layer][3] * u + A[layer][4]

            v = A[layer][5] * tf.nn.sigmoid(A[layer][6] * u +
                                            A[layer][7]) + A[layer][8] * u + A[layer][9]
            z_hat = (z_tilde - mu) * v + mu
            z_hat_bn = batch_norm(z_hat)
            u = batch_norm(tf.matmul(z_hat , weights[weight_key]))

            decoder_latent_layers_z_hat_bn[decoder_latent_layers_z_hat_bn_key] = z_hat_bn
        """
    return u, z_tilde, u_next #decoder_latent_layers_z_hat_bn


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

    for i in xrange(1, len(layer_units)):
        w_key = "encoder_h"+str(i)
        w_value = tf.Variable(tf.random_normal([layer_units[i-1], layer_units[i]]), dtype=tf.float32)
        W[w_key] = w_value

        v_key = "decoder_h"+str(i)
        v_value = tf.Variable(tf.random_normal([layer_units[i], layer_units[i-1]]), dtype=tf.float32)
        V[v_key] = v_value

        gamma_key = "gamma_"+str(i)
        gamma_value = tf.Variable(tf.ones([layer_units[i]]), dtype=tf.float32)
        Gammas[gamma_key] = gamma_value

        beta_key = "beta_"+str(i)
        beta_value = tf.Variable(tf.zeros([layer_units[i]]), dtype=tf.float32)
        Betas[beta_key] = beta_value

    # define
    A = {}
    for layer in xrange(4):
        A[layer] = []
        for a_index in xrange(11):
            unit = layer_units[layer]
            if a_index in [0, 1, 3, 5, 6, 8]:
                init_variable = tf.ones
            else:
                init_variable = tf.zeros

            value = tf.Variable(init_variable([unit]), dtype=tf.float32)
            A[layer].append(value)

    B = {

    }

    h_tilde_3, encoder_z_tildes = encoder(X, W, Gammas, Betas, noise=0.1)
    h_3, encoder_z = encoder(X, W, Gammas, Betas, noise=0.0)
    u, z_tilde, u_next = decoder(h_tilde_3, V, encoder_z_tildes, A, B)


    # define cost function
    """
    y_ = tf.placeholder(shape=(None, 10), dtype=tf.int32)

    cost_c = tf.losses.sigmoid_cross_entropy(logits=h_tilde_3, multi_class_labels=y_)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost_c)

    cost_d = 0
    for layer in xrange(len(encoder_z.keys())):
        z_key = 'z_tilde_'+str(layer)
        z_hat_bn_key = 'z_hat_'+str(layer)+'_bn'
        cost_d += lambda_weight[layer] * tf.reduce_mean(tf.square(encoder_z[z_key] - decoder_latent_layers_z_hat_bn[z_hat_bn_key]))

    cost = cost_c + cost_d

    """

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        """
        for i in xrange(100):
            _, cost_output = sess.run([optimizer, cost], feed_dict={X: sequences, y_:labels})
            print(cost_output)
        """
        u_o, z_tilde_o, u_next_o = sess.run([u, z_tilde, u_next], feed_dict={X: sequences})
        print(u_o.shape)
        print(z_tilde_o.shape)
        print(u_next_o.shape)


if __name__=="__main__":
    main()
