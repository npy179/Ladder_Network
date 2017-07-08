#!/usr/bin/python
from __future__ import print_function
import tensorflow as tf
import numpy as np


def main():

    layer_units = [100, 78, 64, 10]
    A = {}
    for layer in xrange(3, -1, -1):
        A[layer] = []
        for a_index in xrange(1, 11):
            unit = layer_units[layer]
            if a_index in [0, 1, 3, 5, 6, 8]:
                init_variable = tf.ones
            else:
                init_variable = tf.zeros

            value = tf.Variable(init_variable([unit]), dtype=tf.float32)
            A[layer].append(value)

    print(A.keys())
if __name__=="__main__":
    main()
