""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf
import pdb
from tensorflow.python.platform import flags
from utils import get_images
from autoaugment import CIFAR10Policy
FLAGS = flags.FLAGS

class DataGenerator(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.img_size_cropped = 32
        self.img_size = (self.img_size_cropped,self.img_size_cropped)
        self.dim_input = np.prod(self.img_size)*3
        if FLAGS.dataset == 'cifar10':
            self.num_classes = 10
        elif FLAGS.dataset == 'cinic10':
            self.num_classes = 10
        elif FLAGS.dataset == 'cifar100':
            self.num_classes = 100
        elif FLAGS.dataset == 'tiny':
            self.num_classes = 200
        elif FLAGS.dataset == 'mnist+svhn' or FLAGS.dataset == 'mnistm' or FLAGS.dataset == 'mnist+svhn+usps+mnistm':
            self.num_classes = 10
        elif FLAGS.dataset == 'pacs':
            self.num_classes = 7
        self.dim_output = self.num_classes
        
        if FLAGS.dataset == 'cifar10':
            self.metatrain_folder = os.path.join(FLAGS.data_dir, 'cifar', 'train_images')
            if FLAGS.test_set:
                self.metaval_folder = os.path.join(FLAGS.data_dir, 'cifar', 'test_images')
            else:
                self.metaval_folder = os.path.join(FLAGS.data_dir, 'cifar', 'val_images')
        elif FLAGS.dataset == 'cifar100':
            self.metatrain_folder = os.path.join(FLAGS.data_dir, 'cifar100', 'train_images')
            if FLAGS.test_set:
                self.metaval_folder = os.path.join(FLAGS.data_dir, 'cifar100', 'test_images')
            else:
                self.metaval_folder = os.path.join(FLAGS.data_dir, 'cifar100', 'val_images')
        elif FLAGS.dataset == 'cinic10':
            self.metatrain_folder = os.path.join(FLAGS.data_dir, 'cinic10', 'train_images')
            if FLAGS.test_set:
                self.metaval_folder = os.path.join(FLAGS.data_dir, 'cinic10', 'test_images')
            else:
                self.metaval_folder = os.path.join(FLAGS.data_dir,'cinic10', 'val_images')
        elif FLAGS.dataset == 'tiny':
            self.metatrain_folder = os.path.join(FLAGS.data_dir, 'tiny-imagenet-200', 'train_images')
            self.metaval_folder = os.path.join(FLAGS.data_dir, 'tiny-imagenet-200', 'val_images')
        elif FLAGS.dataset == 'pacs':
            self.metatrain_folder = os.path.join(FLAGS.data_dir, 'PACS', 'train')
            if FLAGS.test_set:
                self.metaval_folder = os.path.join(FLAGS.data_dir, 'PACS', 'test')
            else:
                self.metaval_folder = os.path.join(FLAGS.data_dir, 'PACS', 'val')
        elif FLAGS.dataset == 'mnistm':
            self.metatrain_folder = os.path.join(FLAGS.data_dir, 'mnistm', 'train')
            if FLAGS.test_set:
                self.metaval_folder = os.path.join(FLAGS.data_dir, 'mnistm', 'test')
            else:
                self.metaval_folder =  os.path.join(FLAGS.data_dir, 'mnistm', 'val')
    
    def make_data_tensor(self, train=True):

        def read_images_from_disk(input_queue):
            """Consumes a single filename and label as a ' '-delimited string.
            Args:
            filename_and_label_tensor: A scalar string tensor.
            Returns:
            Two tensors: the decoded image, and the string label.
            """
            label = input_queue[1]
            file_contents = tf.read_file(input_queue[0])
            example = tf.image.decode_png(file_contents, channels=3)
            #example = tf.cast(example, tf.float32) / 255.0
            return example, label
        
        def auto_aug(tensor):
            policy = CIFAR10Policy()
            return tf.py_func(policy, [tensor], tf.uint8)
        def cut_out(array):
            cut_size = 16
            x = np.random.randint(self.img_size_cropped)
            y = np.random.randint(self.img_size_cropped)
            x1 = np.clip(x - cut_size // 2, 0, self.img_size_cropped)
            x2 = np.clip(x + cut_size // 2, 0, self.img_size_cropped)
            y1 = np.clip(y - cut_size // 2, 0, self.img_size_cropped)
            y2 = np.clip(y + cut_size // 2, 0, self.img_size_cropped)
            array[x1:x2,y1:y2, :] = 0
            return array

        def preprocess_image(image, train):
            if train:
                if FLAGS.dataset == 'tiny' or FLAGS.dataset == 'pacs' or FLAGS.dataset == 'mnistm' or FLAGS.dataset == 'mnist+svhn+usps+mnistm':
                    image = tf.image.resize_images(image, [self.img_size_cropped, self.img_size_cropped], method=0)
                image = tf.image.resize_image_with_crop_or_pad(image, target_height=self.img_size_cropped+8, target_width=self.img_size_cropped+8)
                image = tf.random_crop(image, size=[self.img_size_cropped, self.img_size_cropped, 3])
                image = tf.image.random_flip_left_right(image)
                image = auto_aug(image)
                image.set_shape([self.img_size_cropped, self.img_size_cropped, 3])
                image = tf.py_func(cut_out, [image], tf.uint8)
                image.set_shape([self.img_size_cropped, self.img_size_cropped, 3])
                image = tf.cast(image, tf.float32) / 255.0

            else:
                if FLAGS.dataset == 'tiny' or FLAGS.dataset == 'pacs' or FLAGS.dataset == 'mnistm' or FLAGS.dataset == 'mnist+svhn+usps+mnistm':
                    image = tf.image.resize_images(image, [self.img_size_cropped, self.img_size_cropped], method=0)
                image = tf.image.resize_image_with_crop_or_pad(image,
                                                target_height=self.img_size_cropped,
                                                target_width=self.img_size_cropped)
                image = tf.cast(image, tf.float32) / 255.0
            return image

        folders = self.metatrain_folder if train else self.metaval_folder

        print("Generating filenames")
        filenames_labels = [[os.path.join(folders, each),int(each.split('_')[0])] for each in os.listdir(folders)]
        image_list = [l[0] for l in filenames_labels]
        label_list = [l[1] for l in filenames_labels]
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        # Makes an input queue
        input_queue = tf.train.slice_input_producer([images, labels],
                                            shuffle=True)
        image, label = read_images_from_disk(input_queue)
        image = preprocess_image(image, train=train)
        label = tf.one_hot(label, self.num_classes)
        min_queue_examples = 256
        parallel_threads = 1 if FLAGS.test_set else 16
        image_batch, label_batch = tf.train.batch([image,label], batch_size=self.batch_size, num_threads=parallel_threads, capacity=min_queue_examples+3*self.batch_size)

        return image_batch, label_batch
