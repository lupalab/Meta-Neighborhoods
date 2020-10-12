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

class DenseNet():
    def __init__(self, nb_blocks, filters, conv_weights, is_training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.conv_weights = conv_weights
        self.is_training = is_training

    def modulate_with_film_dict(self, inp, scope, dict_size):
        # inp is feature_map
        with tf.variable_scope(scope):
            x = tf.reduce_mean(inp, [1, 2])
            dims = x.get_shape().as_list()[-1]
            gamma = tf.get_variable('gamma', initializer=tf.constant(0.0,shape=[dict_size,dims]), trainable=True)
            beta = tf.get_variable('beta', initializer=tf.constant(0.0,shape=[dict_size, dims]), trainable=True)
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

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.is_training, scope=scope+'_batch1')
            if FLAGS.modulate == 'all':
                x = self.modulate_with_film_dict(x, scope=scope+'film1', dict_size=FLAGS.film_dict_size)
            x = Relu(x)
            x = conv_layer(x, self.conv_weights, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=0.2, training=self.is_training)

            x = Batch_Normalization(x, training=self.is_training, scope=scope+'_batch2')
            if FLAGS.modulate == 'all':
                x = self.modulate_with_film_dict(x, scope=scope+'film2', dict_size=FLAGS.film_dict_size)
            x = Relu(x)
            x = conv_layer(x, self.conv_weights, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=0.2, training=self.is_training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.is_training, scope=scope+'_batch1')
            if FLAGS.modulate == 'last' or FLAGS.modulate == 'all':
                self.modulate_with_film_dict(x, scope=scope+'film', dict_size=FLAGS.film_dict_size)
            x = Relu(x)
            # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            
            # https://github.com/taki0112/Densenet-Tensorflow/issues/10
            
            in_channel = x.get_shape().as_list()[-1]
            x = conv_layer(x, self.conv_weights, filter=in_channel*0.5, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=0.2, training=self.is_training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def forward(self, input_x):
        with tf.variable_scope('extractor'):
                x = conv_layer(input_x, self.conv_weights, filter=2 * self.filters, kernel=[3,3], stride=1, layer_name='conv0')
                # x = Max_Pooling(x, pool_size=[3,3], stride=2)


                """
                for i in range(self.nb_blocks) :
                # 6 -> 12 -> 48
                x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
                x = self.transition_layer(x, scope='trans_'+str(i))
                """




                x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_1')
                x = self.transition_layer(x, scope='trans_1')

                x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
                x = self.transition_layer(x, scope='trans_2')

                x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_3')

                x = Batch_Normalization(x, training=self.is_training, scope='linear_batch')
                if FLAGS.dot:
                    x = Relu(x)
                # 100 Layer
                global_pool = tf.reduce_mean(x, [1, 2])

                # x = tf.reshape(x, [-1, 10])
                return global_pool