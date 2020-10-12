from tensorflow.python.platform import flags
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm
FLAGS = flags.FLAGS
import tensorflow as tf
import pdb
dtype = tf.float32
fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
class Linear():
    def __init__(self, feature_size, conv_weights, is_training):
        self.conv_weights = conv_weights
        self.is_training = is_training
        self.conv_weights['W'] = tf.get_variable('W', [32*32*3, feature_size], initializer=fc_initializer)
        self.conv_weights['b'] = tf.Variable(tf.zeros([feature_size]), name='b')
    def forward(self, x):
        x = tf.reshape(x, [-1, 32*32*3])
        x = tf.matmul(x, self.conv_weights['W']) + self.conv_weights['b']
        return x