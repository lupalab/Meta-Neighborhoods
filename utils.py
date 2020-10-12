""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf
import pdb
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

def conv_bn_relu(inp, cweight, stride, reuse, scope, activation=tf.nn.relu):
    strides = [1, stride, stride, 1]
    conv_output = tf.nn.conv2d(inp, cweight, strides, 'SAME')
    normed = normalize(conv_output, activation, reuse, scope)
    return normed

def bn_relu_conv(inp, cweight, stride, reuse, scope):
    activation = tf.nn.relu
    strides = [1, stride, stride, 1]
    normed = normalize(inp, activation, reuse, scope)
    conv_output = tf.nn.conv2d(normed, cweight, strides, 'SAME')
    return conv_output
    
## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope='extractor'+scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope='extractor'+scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label) 

def inv_euclidian_distance(query, key):
    inv_dist = 1.0 / tf.reduce_sum(tf.multiply((query-key),(query-key)), axis=1)
    return inv_dist
def cos_similarity(query,key, alpha):
    # query: 1* num_features
    # key: num_slots * num_features 
    query_norm = tf.nn.l2_normalize(query, axis=1)
    key_norm = tf.nn.l2_normalize(key, axis=1)
    cos_sim = tf.reduce_sum(alpha*tf.multiply(query_norm,key_norm), axis=1)
    cos_sim = tf.nn.softmax(cos_sim)
    return cos_sim

def dot_similarity(query,key, alpha):
    # query: 1* num_features
    # key: num_slots * num_features 
    query_norm = query
    key_norm = key
    cos_sim = tf.reduce_sum(alpha*tf.multiply(query_norm,key_norm), axis=1)
    cos_sim = tf.nn.softmax(cos_sim)
    return cos_sim

def euclidian_similarity(query, key, alpha):
    euclidian_dist = alpha*tf.reduce_sum(tf.multiply((query-key),(query-key)), axis=1)
    return tf.nn.softmax(-euclidian_dist)

def weighted_xent(pred, label, weights):
    # pred: num_slots * num_classes
    # label: num_slots * num_classes
    # weights: num_slots
    y_true = label
    y_hat = pred
    y_hat_softmax = tf.nn.softmax(y_hat)
    y_cross = y_true * tf.log(y_hat_softmax + 1e-8)
    result = weights * (-tf.reduce_sum(y_cross, 1))
    return result

def weighted_xent_simplified(pred, label, weights):
    # pred: num_slots * num_classes
    # label: num_slots * num_classes
    # weights: num_slots
    y_true = label
    y_hat = pred
    cross_entropy = - tf.reduce_sum(y_true * y_hat, axis=1) + tf.math.reduce_logsumexp(y_hat, axis=1)
    return weights * cross_entropy

def orthogonality(weight, orthogonality_penalty_weight=1e-3):
    """Calculates the layer-wise penalty encouraging orthogonality."""
    w2 = tf.matmul(weight, weight, transpose_b=True)
    wn = tf.norm(weight, ord=2, axis=1, keepdims=True) + 1e-32
    correlation_matrix = w2 / tf.matmul(wn, wn, transpose_b=True)
    matrix_size = correlation_matrix.get_shape().as_list()[0]
    base_dtype = weight.dtype.base_dtype
    identity = tf.eye(matrix_size, dtype=base_dtype)
    weight_corr = tf.reduce_mean(
        tf.squared_difference(correlation_matrix, identity))
    return tf.multiply(
        tf.cast(orthogonality_penalty_weight, base_dtype),
        weight_corr)