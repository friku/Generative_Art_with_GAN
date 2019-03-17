from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial

conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
ln = slim.layer_norm

def ChAug(y):
    y = tf.pad(y, [[0,0],[int(y.get_shape()[1])-1,int(y.get_shape()[1])-1],[int(y.get_shape()[2])-1,int(y.get_shape()[2])-1],[0,0]], "CONSTANT")
    w,h = 2,3
    for i in range():
        y[i] = tf.roll(y[i],shift=[w,h],axis=[1,2])
#    y = tf.pad(y, [[0,0],[y.shape[1]-1,y.shape[1]-1],[y.shape[2]-1,y.shape[2]-1],[0,0]], "CONSTANT")
    return y

def generator(z, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope('generator', reuse=reuse):
        y = fc_bn_relu(z, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 4, 4, dim * 8])
        y = dconv_bn_relu(y, dim * 4, 5, 2)
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        y = dconv_bn_relu(y, dim * 1, 5, 2)
        img = tf.tanh(dconv(y, 3, 5, 2))
        return img
    

def generator_ch(z,ch_input,ch_mask, dim=64, reuse=True, training=True):
    with tf.variable_scope('generator', reuse=reuse):
        bn = partial(batch_norm, is_training=training)
        dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
        fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
        y = fc_bn_relu(z, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 4, 4, dim * 8])
        
        y = tf.multiply(y,ch_mask)
        y = tf.add(y,ch_input)
        
        
        y = dconv_bn_relu(y, dim * 4, 5, 2)
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        y = dconv_bn_relu(y, dim * 1, 5, 2)
        img = tf.tanh(dconv(y, 3, 5, 2))
        return img


def discriminator(img, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('discriminator', reuse=reuse):
        y = lrelu(conv(img, dim, 5, 2))
        y = conv_bn_lrelu(y, dim * 2, 5, 2)
        y = conv_bn_lrelu(y, dim * 4, 5, 2)
        y = conv_bn_lrelu(y, dim * 8, 5, 2)
        logit = fc(y, 1)
        return logit


def discriminator_wgan_gp(img, dim=64, reuse=True, training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)
        y = lrelu(conv(img, dim, 5, 2))
        y = conv_ln_lrelu(y, dim * 2, 5, 2)
        y = conv_ln_lrelu(y, dim * 4, 5, 2)
        y = conv_ln_lrelu(y, dim * 8, 5, 2)
        logit = fc(y, 1)
        return logit


def discriminator_wgan_gp_add(img, dim=64, reuse=True, training=True):
    conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('discriminator', reuse=reuse):
        y = lrelu(conv(img, dim, 5, 2))
        y = conv_ln_lrelu(y, dim * 2, 5, 2)
        y = conv_ln_lrelu(y, dim * 4, 5, 2)
        y = conv_ln_lrelu(y, dim * 8, 5, 2)
        y = fc(y, 16)
        feature = y
        logit = fc(y, 1)
        return logit,feature