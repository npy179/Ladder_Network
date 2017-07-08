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

    # first layer
    x_tilde = gaussian_noise_layer(x, stddev=noise)
    z_tilde_1_pre = tf.matmul(x_tilde, weights['encoder_h1'])
    x_batch_norm_1 = batch_norm(z_tilde_1_pre)
    z_tilde_1 = gaussian_noise_layer(x_batch_norm_1, stddev=noise)
    h_tilde_1 = activation(tf.add(z_tilde_1, Betas['beta_1']), ftype='relu')

    # second layer
    x_batch_norm_2 = batch_norm(tf.matmul(h_tilde_1 , weights['encoder_h2']))
    z_tilde_2 = gaussian_noise_layer(x_batch_norm_2, stddev=noise)
    h_tilde_2 = activation(tf.add(z_tilde_2, Betas['beta_2']), ftype='relu')

    # third layer
    x_batch_norm_3 = batch_norm(tf.matmul(h_tilde_2 , weights['encoder_h3']))
    z_tilde_3 = gaussian_noise_layer(x_batch_norm_3, stddev=noise)
    h_tilde_3 = activation(tf.multiply(Gammas['gamma_3'], tf.add(z_tilde_3, Betas['beta_3'])), ftype='softmax')

    encoder_latent_layers_z_tilde = {
        'z_tilde_0': x_tilde,
        'z_tilde_1': z_tilde_1,
        'z_tilde_2': z_tilde_2,
        'z_tilde_3': z_tilde_3
    }
    return h_tilde_3, encoder_latent_layers_z_tilde


def decoder(x, weights, z_tildes, A, B):
    u_3 = batch_norm(x)
    # last layer decoder
    z_tilde_3 = z_tildes['z_tilde_3']
    mu_3 = A['a3_1'] * tf.nn.sigmoid(A['a3_2'] * u_3 +
                                      A['a3_3']) + A['a3_4'] * u_3 + A['a3_5']
    v_3 = A['a3_6'] * tf.nn.sigmoid(A['a3_7'] * u_3 +
                                      A['a3_8']) + A['a3_9'] * u_3 + A['a3_10']
    z_hat_3 = (z_tilde_3 - mu_3) * v_3 + mu_3
    z_hat_3_bn = batch_norm(z_hat_3)
    # second layer decoder
    u_2 = batch_norm(tf.matmul(z_hat_3 , weights['decoder_h3']))
    z_tilde_2 = z_tildes['z_tilde_2']
    mu_2 = A['a2_1'] * tf.nn.sigmoid(A['a2_2'] * u_2 +
                                      A['a2_3']) + A['a2_4'] * u_2 + A['a2_5']
    v_2 = A['a2_6'] * tf.nn.sigmoid(A['a2_7'] * u_2 +
                                      A['a2_8']) + A['a2_9'] * u_2 + A['a2_10']
    z_hat_2 = (z_tilde_2 - mu_2) * v_2 + mu_2
    z_hat_2_bn = batch_norm(z_hat_2)

    # first layer decoder
    u_1 = batch_norm(tf.matmul(z_hat_2 , weights['decoder_h2']))
    z_tilde_1 = z_tildes['z_tilde_1']
    mu_1 = A['a1_1'] * tf.nn.sigmoid(A['a1_2'] * u_1 +
                                      A['a1_3']) + A['a1_4'] * u_1 + A['a1_5']
    v_1 = A['a1_6'] * tf.nn.sigmoid(A['a1_7'] * u_1 +
                                      A['a1_8']) + A['a1_9'] * u_1 + A['a1_10']
    z_hat_1 = (z_tilde_1 - mu_1) * v_1 + mu_1
    z_hat_1_bn = batch_norm(z_hat_1)


    u_0 = batch_norm(tf.matmul(z_hat_1 , weights['decoder_h1']))
    z_tilde_0 = z_tildes['z_tilde_0']


    mu_0 = A['a0_1'] * tf.nn.sigmoid(A['a0_2'] * u_0 +
                                      A['a0_3']) + A['a0_4'] * u_0 + A['a0_5']
    v_0 = A['a0_6'] * tf.nn.sigmoid(A['a0_7'] * u_0 +
                                      A['a0_8']) + A['a0_9'] * u_0 + A['a0_10']
    z_hat_0 = (z_tilde_0 - mu_0) * v_0 + mu_0
    z_hat_0_bn = batch_norm(z_hat_0)

    decoder_latent_layers_z_hat_bn = {
        'z_hat_0_bn': z_hat_0_bn,
        'z_hat_1_bn': z_hat_1_bn,
        'z_hat_2_bn': z_hat_2_bn,
        'z_hat_3_bn': z_hat_3_bn
    }
    return decoder_latent_layers_z_hat_bn


def main():
    #generate_simulation_dataset()
    sequences = np.load("sequences.npy")
    labels = np.load("labels.npy")
    # tf Graph input
    X = tf.placeholder(tf.float32, shape=(None, 100))
    Y = tf.placeholder(tf.float32, shape=(None, 10))
    # define weight for corrupted encoder,  clean encoder
    layer_units = [100, 78, 64, 10]


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
        for a_index in xrange(1, 11):
            key = "a"+str(layer)+"_"+str(a_index)
            unit = layer_units[layer]
            if a_index in [1,2,4,6,7,9]:
                init_variable = tf.ones
            else:
                init_variable = tf.zeros

            value = tf.Variable(init_variable([unit]), dtype=tf.float32)
            A[key] = value

    B = {

    }

    h_tilde_3, encoder_z_tildes = encoder(X, W, Gammas, Betas, noise=0.1)
    h_3, encoder_z = encoder(X, W, Gammas, Betas, noise=0.0)
    decoder_latent_layers_z_hat_bn = decoder(h_tilde_3, V, encoder_z_tildes, A, B)


    # define cost function
    y_ = tf.placeholder(shape=(None, 10), dtype=tf.int32)

    cost_c = tf.losses.sigmoid_cross_entropy(logits=h_tilde_3, multi_class_labels=y_)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost_c)

    z_hat_0_bn = decoder_latent_layers_z_hat_bn['z_hat_0_bn']
    z_hat_1_bn = decoder_latent_layers_z_hat_bn['z_hat_1_bn']
    z_hat_2_bn = decoder_latent_layers_z_hat_bn['z_hat_2_bn']
    z_hat_3_bn = decoder_latent_layers_z_hat_bn['z_hat_3_bn']

    z_0 = encoder_z['z_tilde_0']
    z_1 = encoder_z['z_tilde_1']
    z_2 = encoder_z['z_tilde_2']
    z_3 = encoder_z['z_tilde_3']


    cost_d = 1000 * tf.reduce_mean(tf.square(z_0 - z_hat_0_bn))
    + 10 * tf.reduce_mean(tf.square(z_1 - z_hat_1_bn))
    + 0.1 * tf.reduce_mean(tf.square(z_2 - z_hat_2_bn))
    + 0.1 * tf.reduce_mean(tf.square(z_3 - z_hat_3_bn))

    cost = cost_c + cost_d

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in xrange(100):
            _, cost_output = sess.run([optimizer, cost], feed_dict={X: sequences, y_:labels})
            print(cost_output)



if __name__=="__main__":
    main()
