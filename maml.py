""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import os
import pdb
import re
import time
from PIL import Image
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, normalize, weighted_xent, cos_similarity, conv_bn_relu, bn_relu_conv, weighted_xent_simplified, euclidian_similarity
from network.densenet import DenseNet
from network.resnet import ResNet_V2
from network.shallownet import ShallowNet
from network.linear import Linear
from network.supershallownet import SuperShallowNet
FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = tf.placeholder(dtype=tf.float32, shape=[])
        self.test_num_updates = test_num_updates
        if FLAGS.backbone in ['resnet29', 'resnet56']:
            self.feature_size = 256
        elif FLAGS.backbone in ['densenet40']:
            self.feature_size = 258
        elif FLAGS.backbone in ['shallownet']:
            self.feature_size = 192
        elif FLAGS.backbone in ['supershallownet']:
            self.feature_size = 64
        elif FLAGS.backbone in ['linear']:
            self.feature_size = 128
        self.loss_func_weighted = weighted_xent_simplified
        self.loss_func = xent
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
        self.wd_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

        self.channels = 3
        self.img_size = int(np.sqrt(self.dim_input/self.channels))

    def construct_vanilla_model(self, input_tensors=None, prefix='metatrain_'):
        self.input = input_tensors['input']
        self.label = input_tensors['label']
        conv_weights = dict()
        is_training = True if 'train' in prefix else False
        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
                conv_weights = self.conv_weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()
            if FLAGS.backbone == 'densenet40':
                forward_net = DenseNet(nb_blocks=2, filters=12, conv_weights=conv_weights, is_training=is_training)
            elif FLAGS.backbone == 'resnet29':
                forward_net = ResNet_V2(filters=16, depths=29, conv_weights=conv_weights, is_training=is_training)
            elif FLAGS.backbone == 'resnet56':
                forward_net = ResNet_V2(filters=16, depths=56, conv_weights=conv_weights, is_training=is_training)
            elif FLAGS.backbone == 'shallownet':
                forward_net = ShallowNet(conv_weights=conv_weights, is_training=is_training)
            elif FLAGS.backbone in ['supershallownet']:
                forward_net = SuperShallowNet(conv_weights=conv_weights, is_training=is_training)
            elif FLAGS.backbone == 'linear':
                forward_net = Linear(feature_size=128, conv_weights=conv_weights, is_training=is_training)
            else:
                raise NotImplementedError(FLAGS.backbone +' is not implemented.')
            batch_features = forward_net.forward(self.input)
            self.conv_weights = conv_weights
            output = self.fc_forward(batch_features, weights)
            vanilla_loss = tf.reduce_sum(self.loss_func(output, self.label)) / FLAGS.meta_batch_size
            vanilla_accuracy = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(output), 1), tf.argmax(self.label, 1))

        if 'train' in prefix:
            self.vanilla_loss = vanilla_loss
            self.vanilla_accuracy = vanilla_accuracy
            regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            if FLAGS.metatrain_iterations > 0:
                if FLAGS.optimizer == 'sgd':
                    optimizer = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9, use_nesterov=True)
                    self.gvs = gvs = optimizer.compute_gradients(tf.add_n([self.vanilla_loss] + regu_losses))
                    self.gvs = gvs = [(grad, var) for grad, var in gvs if grad is not None]
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                    self.metatrain_op = optimizer.apply_gradients(gvs)
                else:
                    optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=self.wd_placeholder, learning_rate=self.lr_placeholder)
                    all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                    decayed_var = [var for var in all_var if 'memory' not in var.name]
                    self.gvs = gvs = optimizer.compute_gradients(tf.add_n([self.vanilla_loss]))
                    self.gvs = gvs = [(grad, var) for grad, var in gvs if grad is not None]
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                    self.metatrain_op = optimizer.apply_gradients(gvs, decay_var_list=decayed_var)
        else:
            self.vanilla_val_loss = vanilla_loss
            self.vanilla_val_accuracy = vanilla_accuracy

        ## Summaries
        tf.summary.scalar(prefix+'Post-update loss, step 1', vanilla_loss)
        tf.summary.scalar(prefix+'Post-update accuracy, step 1', vanilla_accuracy)



    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        
        self.input = input_tensors['input']
        self.label = input_tensors['label']
        conv_weights = dict()
        is_training = True if 'train' in prefix else False
        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
                memo_weights = self.memo_weights
                conv_weights = self.conv_weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()
                self.memo_weights = memo_weights = self.construct_memory()
        
            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                img, label, features = inp
                task_outputbs, task_lossesb = [], []

                task_accuraciesb = []
                if FLAGS.dropout_ratio > 0:
                    dropout_indices = tf.random.shuffle(tf.range(FLAGS.dict_size))[:int((1-FLAGS.dropout_ratio)*FLAGS.dict_size)]
                else:
                    dropout_indices = tf.range(FLAGS.dict_size)
                k_dropped = tf.gather(memo_weights['k'], indices=dropout_indices, axis=0) if 'train' in prefix else memo_weights['k']
                v_dropped = tf.gather(memo_weights['v'], indices=dropout_indices, axis=0) if 'train' in prefix else memo_weights['v']

                sim = cos_similarity(features, k_dropped, memo_weights['alpha'])
                task_outputa = self.fc_forward(k_dropped, weights)
                task_lossa = self.loss_func_weighted(task_outputa, v_dropped, sim)
                grads = tf.gradients(task_lossa, list(weights.values()))
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict()
                for key in weights.keys():
                    if key in ['w1','b1']:
                        if FLAGS.scalar_lr:
                            fast_weights[key] =  weights[key] - memo_weights['adaptive_lr']*gradients[key]
                        else:
                            fast_weights[key] =  weights[key] - tf.matmul(memo_weights['adaptive_lr_diag'],gradients[key])
                    else:
                        fast_weights[key] = weights[key]
                
                output = self.fc_forward(features, fast_weights)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, label))

                for j in range(num_updates - 1):
                    loss = self.loss_func_weighted(self.fc_forward(k_dropped, fast_weights), v_dropped, sim)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    gradients = dict(zip(fast_weights.keys(), grads))
                    
                    for key in fast_weights.keys():
                        if key in ['w1','b1']:
                            if FLAGS.scalar_lr:
                                fast_weights[key] =  fast_weights[key] - memo_weights['adaptive_lr']*gradients[key]
                            else:
                                fast_weights[key] =  fast_weights[key] - tf.matmul(memo_weights['adaptive_lr_diag'],gradients[key])
                        else:
                            fast_weights[key] = fast_weights[key]
                    output = self.fc_forward(features, fast_weights)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, label))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(v_dropped, 1))
                for j in range(num_updates):
                    task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(label, 1)))
                task_output.extend([task_accuracya, task_accuraciesb])
                if FLAGS.visualize:
                    task_output.extend([memo_weights['v'], memo_weights['k'], sim, features])
                return task_output
            if FLAGS.backbone == 'densenet40':
                forward_net = DenseNet(nb_blocks=2, filters=12, conv_weights=conv_weights, is_training=is_training)
            elif FLAGS.backbone == 'resnet29':
                forward_net = ResNet_V2(filters=16, depths=29, conv_weights=conv_weights, is_training=is_training)
            elif FLAGS.backbone == 'resnet56':
                forward_net = ResNet_V2(filters=16, depths=56, conv_weights=conv_weights, is_training=is_training)
            elif FLAGS.backbone == 'shallownet':
                forward_net = ShallowNet(conv_weights=conv_weights, is_training=is_training)
            elif FLAGS.backbone in ['supershallownet']:
                forward_net = SuperShallowNet(conv_weights=conv_weights, is_training=is_training)
            elif FLAGS.backbone in ['linear']:
                forward_net = Linear(feature_size=128, conv_weights=conv_weights, is_training=is_training)
            else:
                raise NotImplementedError(FLAGS.backbone +' is not implemented.')
            batch_features = forward_net.forward(self.input)
            self.conv_weights = conv_weights

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            if FLAGS.visualize:
                out_dtype.extend([tf.float32, tf.float32, tf.float32, tf.float32])
            self.input = tf.expand_dims(self.input, axis=1) # self.input now has shape: batch_size * 1 * w * h * c
            self.label = tf.expand_dims(self.label, axis=1) # self.label now has shape: batch_size * 1 * num_classes


            result = tf.map_fn(task_metalearn, elems=(self.input, self.label, tf.expand_dims(batch_features, axis=1)), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            
            if FLAGS.visualize:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb, labela, key, sim, features = result
            else:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
            if FLAGS.visualize:
                tf.summary.histogram(name='activation', values=features)
                tf.summary.image(name='sim', tensor=tf.expand_dims(tf.expand_dims(sim,-1),0))
                tf.summary.histogram(name='sim', values=sim)
                tf.summary.histogram(name='key', values=key)
                tf.summary.histogram(name='labela', values=labela)
                tf.summary.image(name='labela', tensor=tf.expand_dims(tf.expand_dims(labela[0,...],-1),0))
                

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=self.wd_placeholder, learning_rate=self.lr_placeholder)
                all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                decayed_var = [var for var in all_var if ('memory' not in var.name)]
                self.gvs = gvs = optimizer.compute_gradients(tf.add_n([self.total_losses2[FLAGS.num_updates-1]]))
                gvs_clipped = []
                for grad,var in gvs:
                    print(var.name)
                    if grad is not None:
                        gvs_clipped.append((tf.clip_by_value(grad, -10, 10), var))
                for i, (grad, var) in enumerate(gvs_clipped):
                    if len(var.shape.as_list())==0:
                        tf.summary.histogram(name=var.name, values=var)
                    else:
                        tf.summary.histogram(name=var.name, values=var)
                        tf.summary.histogram(name=var.name + '_grad', values=grad)
                self.metatrain_op = optimizer.apply_gradients(gvs_clipped, decay_var_list=decayed_var)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
        
            self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)
        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    ### Network construction functions (fc networks and conv networks)
    def construct_memory(self):
        memo_weights = {}
        dtype = tf.float32
        normal_init = tf.initializers.truncated_normal(dtype = dtype, stddev=0.01)
        num_slots = FLAGS.dict_size
        with tf.variable_scope('memory', reuse=None):
            if not FLAGS.fix_v:
                gamma = tf.get_variable('gamma', initializer=tf.constant(10.0), trainable=False)
                memo_weights['v'] = tf.nn.softmax(gamma*tf.get_variable('v', shape=[num_slots, self.dim_output], initializer=normal_init, dtype=dtype), axis=1)
            else:
                indices = [int(i / (num_slots / self.dim_output)) for i in range(num_slots)]
                memo_weights['v'] = tf.one_hot(indices, depth = self.dim_output)
   
            memo_weights['k'] = tf.get_variable('k', initializer=tf.random.normal(shape=[num_slots, self.feature_size], mean=0, stddev=0.05), dtype=dtype)
            memo_weights['alpha'] = tf.math.abs(tf.get_variable('alpha', initializer=tf.constant(FLAGS.alpha), trainable=False))
            memo_weights['adaptive_lr'] = tf.math.abs(tf.get_variable('adaptive_lr', initializer=tf.constant(0.05), trainable=True))
            memo_weights['adaptive_lr_diag'] = tf.diag(tf.math.abs(tf.get_variable('adaptive_lr_diag', initializer=tf.constant(0.05,shape=[self.feature_size]), trainable=True)))
        return memo_weights

    def construct_weights(self):
            weights = {}
            dtype = tf.float32
            fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
            with tf.variable_scope('memory', reuse=None):
                weights['w1'] = tf.get_variable('w1', [self.feature_size, self.dim_output], initializer=fc_initializer)
                weights['tau'] = tf.get_variable('tau', initializer=tf.constant(10.0), trainable=True)
                if FLAGS.dot:
                    weights['b1'] = tf.Variable(tf.zeros([self.dim_output]), name='b1')
            return weights

    def fc_forward(self, inp, weights):
        # reuse is for the normalization parameters.
        inp = tf.nn.l2_normalize(inp, axis=1) if not FLAGS.dot else inp
        if not FLAGS.dot:
            weight = tf.nn.l2_normalize(weights['w1'], axis=0)
        else:
            weight = weights['w1']
        if not FLAGS.dot:
            hidden1 = weights['tau'] * tf.matmul(inp, weight)
        else:
            hidden1 = tf.matmul(inp,weight) + weights['b1']
        return hidden1

