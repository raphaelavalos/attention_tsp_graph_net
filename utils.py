import sonnet as snt
import tensorflow as tf
import numpy as np


def load_linear(linear, dic, sess, bias=False):
    linear.w.load(dic['w'], sess)
    if bias:
        linear.b.load(dic['b'], sess)


def load_batchnorm(batchnorm, dic, sess):
    with sess.as_default():
        #batchnorm.beta = tf.convert_to_tensor(dic['beta'])
        #batchnorm.gamma = tf.convert_to_tensor(dic['gamma'])
        #batchnorm.moving_mean = tf.convert_to_tensor(dic['moving_mean'])
        #batchnorm.moving_variance = tf.convert_to_tensor(dic['moving_variance'])
        batchnorm.get_variables()[0].load(dic['beta0'])
        batchnorm.get_variables()[1].load(dic['gamma0'])


def save_linear(linear, sess, bias=False):
    dic = {'w': linear.w.eval(sess)}
    if bias:
        dic['b'] = linear.b.eval(sess)
    return dic


def save_batchnorm(batchnorm, sess):
    with sess.as_default():
        dic = {'beta': batchnorm.beta.eval(),
               'gamma': batchnorm.gamma.eval(),
               'moving_mean': batchnorm.moving_mean.eval(),
               'moving_variance': batchnorm.moving_variance.eval(),
               'beta0': batchnorm.get_variables()[0].eval(),
               'gamma0': batchnorm.get_variables()[0].eval()}
    return dic


def initializer(dim):
    m = 1 / np.sqrt(dim)
    return tf.initializers.random_uniform(-m, m)
