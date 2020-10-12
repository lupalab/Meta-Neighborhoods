# vanilla cifar100 79.1%
# mn cifar100 80.28%
# mn+ifilm cifar100 80.96%
import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import pdb
import os
from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
FLAGS = flags.FLAGS

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 600000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 128, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('dict_size', 5, 'size of the dictionary.')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_float('weight_decay', 7.5e-5, 'weight decay rate.')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('visualize', False, 'visualize key, values, attention in tensorboard.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')

## more
flags.DEFINE_string('dataset', 'cifar10', 'which dataset to run.')
flags.DEFINE_string('data_dir', None, 'where the dataset is stored.')
flags.DEFINE_bool('vanilla', False, 'if true, run vanilla training.')
flags.DEFINE_bool('fix_v', False, 'if true, v is no longer trainable.')
flags.DEFINE_float('alpha', 1.0, 'control the peakiness of the softmax similarity.')
flags.DEFINE_float('dropout_ratio', 0.5, 'randomly dropout slots of dictionary during training.')
flags.DEFINE_string('optimizer', 'adamw', 'ossptimizer used to train the vanilla model. Choose from "sgd" and "adamw".')
flags.DEFINE_bool('dot',False,'if true, use dot product output layer rather than cos dist output layer.')

flags.DEFINE_string('backbone', 'resnet29', 'backbone choice, must be one of [resnet29, resnet56, densenet40, shallownet, supershallownet]')
flags.DEFINE_bool('scalar_lr', True, 'if true use scalar inner loop lr, else diagonal inner loop lr')
flags.DEFINE_string('modulate', 'None', 'modulate backbone with film, all, last or None.')
flags.DEFINE_integer('film_dict_size', 50, 'film dict size.')
def train(model, saver, sess, exp_string, data_generator, prev_best_accu, resume_itr=0):

    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 100
    
    PRINT_INTERVAL = 100
    if FLAGS.meta_lr < 1e-3:
        TEST_PRINT_INTERVAL = PRINT_INTERVAL
    else:
        TEST_PRINT_INTERVAL =  50 * PRINT_INTERVAL

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []
    best_accu = prev_best_accu
    if 'cifar' in FLAGS.dataset:
        NUM_TEST_POINTS = int(5000 / FLAGS.meta_batch_size) 
        decay_steps = [120000, 160000]
    elif FLAGS.dataset == 'cinic10':
        NUM_TEST_POINTS = int(90000 / FLAGS.meta_batch_size)
        decay_steps = [160000, 260000] 
    elif FLAGS.dataset == 'tiny':
        NUM_TEST_POINTS = int(200*50 / FLAGS.meta_batch_size)
        decay_steps = [180000, 260000] 
    elif FLAGS.dataset == 'mnist+svhn':
        NUM_TEST_POINTS = int(36000 / FLAGS.meta_batch_size)
        decay_steps = [40000, 60000]
    elif FLAGS.dataset == 'mnistm':
        NUM_TEST_POINTS = int(5000 / FLAGS.meta_batch_size)
        decay_steps = [40000, 60000]
    elif FLAGS.dataset == 'pacs':
        NUM_TEST_POINTS = int(1014 / FLAGS.meta_batch_size)
        decay_steps = [40000, 60000]
    elif FLAGS.dataset == 'mnist+svhn+usps+mnistm':
        NUM_TEST_POINTS = int(48039 / FLAGS.meta_batch_size)
        decay_steps = [50000, 70000]

    for itr in tqdm(range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations)):
        
        if itr < decay_steps[0]:
            cur_lr = FLAGS.meta_lr
            cur_wd = FLAGS.weight_decay
        elif itr < decay_steps[1]:
            cur_lr = 0.1 * FLAGS.meta_lr
            cur_wd = 0.1 * FLAGS.weight_decay
            TEST_PRINT_INTERVAL = PRINT_INTERVAL if 'cinic10' not in FLAGS.dataset else 10*PRINT_INTERVAL
        else:
            break

        cur_update_lr = FLAGS.update_lr
        feed_dict = {model.wd_placeholder: cur_wd, model.lr_placeholder: cur_lr, model.update_lr:cur_update_lr}

        input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            feed_dict = {model.lr_placeholder:0, model.update_lr:cur_update_lr}
            input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1]]
            
            accu1 = 0
            accu2 = 0
            count = 0
            for i in tqdm(range(NUM_TEST_POINTS)):
                result = sess.run(input_tensors, feed_dict)
                accu1 += result[0]
                accu2 += result[1]
                count += 1 
            print('Validation results: ' + str(accu1/count) + ', ' + str(accu2/count))
            if (accu2/count) > best_accu:
                best_accu = (accu2/count)
                saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

def train_vanilla(model, saver, sess, exp_string, data_generator, prev_best_accu, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 100
    
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = 50 * PRINT_INTERVAL

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    accuracies = []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []
    best_accu = prev_best_accu
    if FLAGS.optimizer == 'sgd':
        if 'cifar' in FLAGS.dataset:
            NUM_TEST_POINTS = int(5000 / FLAGS.meta_batch_size) 
            decay_steps = [60000, 80000, 100000, 120000]
        elif FLAGS.dataset == 'cinic10':
            NUM_TEST_POINTS = int(90000 / FLAGS.meta_batch_size)
            decay_steps = [120000, 160000, 200000, 240000] 
        elif FLAGS.dataset == 'tiny':
            NUM_TEST_POINTS = int(200*50 / FLAGS.meta_batch_size)
            decay_steps = [140000, 180000, 220000, 260000] 
        elif FLAGS.dataset == 'mnist+svhn':
            NUM_TEST_POINTS = int(36000 / FLAGS.meta_batch_size)
            decay_steps = [40000, 50000, 60000]
        elif FLAGS.dataset == 'mnistm':
            NUM_TEST_POINTS = int(5000 / FLAGS.meta_batch_size)
            decay_steps = [40000, 50000, 60000]
        elif FLAGS.dataset == 'pacs':
            NUM_TEST_POINTS = int(1014 / FLAGS.meta_batch_size)
            decay_steps = [40000, 50000, 60000]
        elif FLAGS.dataset == 'mnist+svhn+usps+mnistm':
            NUM_TEST_POINTS = int(48039 / FLAGS.meta_batch_size)
            decay_steps = [50000, 60000, 70000]
    else:
        if 'cifar' in FLAGS.dataset:
            NUM_TEST_POINTS = int(5000 / FLAGS.meta_batch_size) 
            decay_steps = [120000, 160000]
        elif FLAGS.dataset == 'cinic10':
            NUM_TEST_POINTS = int(90000 / FLAGS.meta_batch_size)
            decay_steps = [160000, 260000] 
        elif FLAGS.dataset == 'tiny':
            NUM_TEST_POINTS = int(200*50 / FLAGS.meta_batch_size)
            decay_steps = [180000, 260000] 
        elif FLAGS.dataset == 'mnist+svhn':
            NUM_TEST_POINTS = int(36000 / FLAGS.meta_batch_size)
            decay_steps = [40000, 60000]
        elif FLAGS.dataset == 'mnistm':
            NUM_TEST_POINTS = int(5000 / FLAGS.meta_batch_size)
            decay_steps = [40000, 60000]
        elif FLAGS.dataset == 'pacs':
            NUM_TEST_POINTS = int(1014 / FLAGS.meta_batch_size)
            decay_steps = [40000, 60000]
        elif FLAGS.dataset == 'mnist+svhn+usps+mnistm':
            NUM_TEST_POINTS = int(48039 / FLAGS.meta_batch_size)
            decay_steps = [50000, 70000]
    
    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        if FLAGS.optimizer == 'sgd':
            if itr < decay_steps[0]:
                cur_lr = FLAGS.meta_lr
                cur_wd = FLAGS.weight_decay
            elif itr < decay_steps[1]:
                cur_lr = 0.1 * FLAGS.meta_lr
                cur_wd = 0.1 * FLAGS.weight_decay
                TEST_PRINT_INTERVAL = PRINT_INTERVAL if 'cinic10' not in FLAGS.dataset else 10*PRINT_INTERVAL
            elif itr < decay_steps[2]:
                cur_lr = 0.01 * FLAGS.meta_lr
                cur_wd = 0.01 * FLAGS.weight_decay
                TEST_PRINT_INTERVAL = PRINT_INTERVAL if 'cinic10' not in FLAGS.dataset else 10*PRINT_INTERVAL
            elif itr < decay_steps[3]:
                cur_lr = 0.001 * FLAGS.meta_lr
                cur_wd = 0.001 * FLAGS.weight_decay
                TEST_PRINT_INTERVAL = PRINT_INTERVAL if 'cinic10' not in FLAGS.dataset else 10*PRINT_INTERVAL
            else:
                break
        else:
            if itr < decay_steps[0]:
                cur_lr = FLAGS.meta_lr
                cur_wd = FLAGS.weight_decay
            elif itr < decay_steps[1]:
                cur_lr = 0.1 * FLAGS.meta_lr
                cur_wd = 0.1 * FLAGS.weight_decay
                TEST_PRINT_INTERVAL = PRINT_INTERVAL if 'cinic10' not in FLAGS.dataset else 10*PRINT_INTERVAL
            else:
                break
        feed_dict = {model.lr_placeholder:cur_lr, model.wd_placeholder:cur_wd}
        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.vanilla_loss])
            input_tensors.extend([model.vanilla_accuracy])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            accuracies.append(result[-1])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(accuracies)) 
            print(print_str)
            accuracies = []



        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            feed_dict = {model.lr_placeholder:0, model.update_lr:0}
            input_tensors = [model.vanilla_val_accuracy]
            accu = 0
            count = 0
            for i in range(NUM_TEST_POINTS):
                result = sess.run(input_tensors, feed_dict)
                accu += result[0]
                count += 1
            print('Validation results: ' + str(accu/count))
            if (accu/count) > best_accu:
                best_accu = (accu/count)
                saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))


def test_vanilla(model, saver, sess, exp_string, data_generator):

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    if 'cifar' in FLAGS.dataset:
        NUM_TEST_POINTS = int(5000 / 100) 
    elif FLAGS.dataset == 'cinic10':
        NUM_TEST_POINTS = int(90000 / 100)
    elif FLAGS.dataset == 'tiny':
        NUM_TEST_POINTS = int(200*50 / 100)
    elif FLAGS.dataset == 'pacs':
        NUM_TEST_POINTS = int(9991 / 100)
    elif FLAGS.dataset == 'mnist+svhn':
        NUM_TEST_POINTS = int(36032 / 100)
    elif FLAGS.dataset == 'mnistm':
        NUM_TEST_POINTS = int(10000 / FLAGS.meta_batch_size)
    elif FLAGS.dataset == 'mnist+svhn+usps+mnistm':
        NUM_TEST_POINTS = int(48039 / FLAGS.meta_batch_size)
      


    for _ in range(NUM_TEST_POINTS):
        feed_dict = {model.lr_placeholder:0, model.update_lr:0}
        result = sess.run(model.vanilla_val_accuracy, feed_dict)
        metaval_accuracies.append(result)
        print(result)
    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)
    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))
    return means





def test(model, saver, sess, exp_string, data_generator):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []
    if 'cifar' in FLAGS.dataset:
        NUM_TEST_POINTS = int(5000 / 100) 
    elif FLAGS.dataset == 'cinic10':
        NUM_TEST_POINTS = int(90000 / 100)
    elif FLAGS.dataset == 'tiny':
        NUM_TEST_POINTS = int(200*50 / 100)
    elif FLAGS.dataset == 'pacs':
        NUM_TEST_POINTS = int(9991 / 100)
    elif FLAGS.dataset == 'mnist+svhn':
        NUM_TEST_POINTS = int(36032 / 100)
    elif FLAGS.dataset == 'mnistm':
        NUM_TEST_POINTS = int(10000 / FLAGS.meta_batch_size)
    elif FLAGS.dataset == 'mnist+svhn+usps+mnistm':
        NUM_TEST_POINTS = int(48039 / FLAGS.meta_batch_size)
    for _ in range(NUM_TEST_POINTS):
        feed_dict = {model.lr_placeholder:0, model.update_lr:FLAGS.update_lr}
        result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        metaval_accuracies.append(result)
        print(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))
    return means[1]

def main():    
    test_num_updates = 1
    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 100 when testing.
        FLAGS.meta_batch_size = 100
    data_generator = DataGenerator(batch_size=FLAGS.meta_batch_size)

    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input
    num_classes = data_generator.num_classes

    if FLAGS.train: # only construct training model if needed
        random.seed(5)
        image_tensor, label_tensor = data_generator.make_data_tensor()
        input_tensors = {'input': image_tensor, 'label': label_tensor}

    random.seed(6)
    image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
    metaval_input_tensors = {'input': image_tensor, 'label': label_tensor}

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.vanilla:
        if FLAGS.train:
            model.construct_vanilla_model(input_tensors=input_tensors, prefix='metatrain_')
        model.construct_vanilla_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    else:
        if FLAGS.train:
            model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()
    saver = loader = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    sess = tf.InteractiveSession()
    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    exp_string = FLAGS.dataset \
                + '_backbone_' + FLAGS.backbone \
                + '_scalar_lr_' + str(FLAGS.scalar_lr) \
                + '_mbs_'+str(FLAGS.meta_batch_size) \
                + '.dict_' + str(FLAGS.dict_size) + '.numstep' \
                + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.update_lr) \
                + '.vanilla_' + str(FLAGS.vanilla) \
                + '.fix_v_' + str(FLAGS.fix_v) \
                + '.alpha_' + str(FLAGS.alpha) \


    if FLAGS.dropout_ratio != 0.5:
        exp_string += '_dropout_' + str(FLAGS.dropout_ratio)
    if FLAGS.vanilla and FLAGS.optimizer != 'sgd':
        exp_string += FLAGS.optimizer
    exp_string += '_weight_decay_' + str(FLAGS.weight_decay)
    if FLAGS.dot:
        exp_string += '_dot'
    if FLAGS.modulate in ['all', 'last', 'before_fc']:
        exp_string += '_modulate_' + FLAGS.modulate + '_size_' + str(FLAGS.film_dict_size)
    print(exp_string)
    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    prev_best_accu = 0
    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            loader.restore(sess, model_file)
            orig_train = FLAGS.train
            FLAGS.train = False
            if FLAGS.vanilla:
                prev_best_accu = test_vanilla(model, saver, sess, exp_string, data_generator)
            else:
                prev_best_accu = test(model, saver, sess, exp_string, data_generator)
            FLAGS.train = orig_train

    if FLAGS.vanilla:
        if FLAGS.train:
            train_vanilla(model, saver, sess, exp_string, data_generator, prev_best_accu, resume_itr)
        else:
            test_vanilla(model, saver, sess, exp_string, data_generator)
    else:
        if FLAGS.train:
            train(model, saver, sess, exp_string, data_generator, prev_best_accu, resume_itr)
        else:
            test(model, saver, sess, exp_string, data_generator)


if __name__ == "__main__":
    main()
