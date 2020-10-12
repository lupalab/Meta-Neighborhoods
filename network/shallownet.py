from tensorflow.python.platform import flags
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm
FLAGS = flags.FLAGS
import tensorflow as tf
import pdb

def conv_layer(input, conv_weights, filter, kernel, stride=1, layer_name="conv"):
    strides = [1,stride,stride,1]
    regularizer = tf.contrib.layers.l2_regularizer(scale=2e-4)
    if layer_name not in conv_weights.keys():
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
        in_channel = input.get_shape().as_list()[-1]
        conv_weights[layer_name] = tf.get_variable(layer_name, [kernel[0], kernel[1], in_channel, filter], initializer=conv_initializer, dtype=tf.float32, regularizer=regularizer)
    return tf.nn.conv2d(input, conv_weights[layer_name], strides, 'SAME')

def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return batch_norm(inputs=x, is_training=training, reuse=None) if training else batch_norm(inputs=x, is_training=training, reuse=True if FLAGS.train else False)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

class ShallowNet():
    def __init__(self, conv_weights, is_training):
        self.conv_weights = conv_weights
        self.is_training = is_training

    def modulate_with_film_dict(self, inp, scope, dict_size):
        # inp is feature_map
        with tf.variable_scope(scope):
            x = tf.reduce_mean(inp, [1, 2])
            dims = x.get_shape().as_list()[-1]
            gamma = tf.get_variable(scope+'gamma', initializer=tf.constant(0.0,shape=[dict_size,dims]), trainable=True)
            beta = tf.get_variable(scope+'beta', initializer=tf.constant(0.0,shape=[dict_size, dims]), trainable=True)
            k = tf.get_variable('k', initializer=tf.random.normal(shape=[dims, dict_size], mean=0, stddev=0.05), dtype=tf.float32)
            temp = tf.get_variable('temp', initializer=tf.constant(5.0), dtype=tf.float32, trainable=True)
            k_norm =  tf.nn.l2_normalize(k, axis=0)
            x_norm = tf.nn.l2_normalize(x, axis=1)
            cos_sim = tf.matmul(x_norm, k_norm) # BatchSize * DictSize
            cos_sim = tf.nn.softmax(temp * cos_sim, axis=1) # BatchSize * DictSize
            gamma = tf.matmul(cos_sim, gamma)
            beta = tf.matmul(cos_sim, beta)

            gamma = tf.reshape(gamma, [-1,1,1,dims])
            beta = tf.reshape(beta, [-1,1,1,dims])
            tf.summary.histogram(name=scope+'gamma', values=gamma)
            tf.summary.histogram(name=scope+'beta', values=beta)
            tf.summary.histogram(name=scope+'key', values=k)
            tf.summary.histogram(name=scope+'temp', values=temp)
            out = (gamma + 1.0)*inp + beta
            return out

    def forward(self, input_x):
        with tf.variable_scope('extractor'):
            x = conv_layer(input_x, self.conv_weights, filter = 24, kernel=[3,3], layer_name='conv0')
            x = Batch_Normalization(x, training=self.is_training, scope='conv0_batch0')
            if FLAGS.modulate == 'all':
                x = self.modulate_with_film_dict(x, scope='film0', dict_size=FLAGS.film_dict_size)

            x = Relu(x)
            x = Max_Pooling(x)

            x = conv_layer(x, self.conv_weights, filter = 96, kernel=[3,3], layer_name='conv1')
            x = Batch_Normalization(x, training=self.is_training, scope='conv0_batch1')
            if FLAGS.modulate == 'all':
                x = self.modulate_with_film_dict(x, scope='film1', dict_size=FLAGS.film_dict_size)
            x = Relu(x)
            x = Max_Pooling(x)

            x = conv_layer(x, self.conv_weights, filter = 192, kernel=[3,3], layer_name='conv2')
            x = Batch_Normalization(x, training=self.is_training, scope='conv0_batch2')
            if FLAGS.modulate == 'all':
                x = self.modulate_with_film_dict(x, scope='film2', dict_size=FLAGS.film_dict_size)
            x = Relu(x)
            
            x = conv_layer(x, self.conv_weights, filter = 192, kernel=[3,3], layer_name='conv3')
            x = Batch_Normalization(x, training=self.is_training, scope='conv0_batch3')
            if FLAGS.modulate == 'all':
                x = self.modulate_with_film_dict(x, scope='film3', dict_size=FLAGS.film_dict_size)
            x = tf.reduce_mean(x, [1, 2])
            return x


   