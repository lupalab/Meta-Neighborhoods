import tensorflow as tf
from tensorflow.keras import datasets
import argparse
from PIL import Image
import os
import pdb
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', required=True, type=str)
args = parser.parse_args()
save_dir = str(args.save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(save_dir, 'cifar100'):
    os.mkdir(save_dir)
if not os.path.exists(os.path.join(save_dir, 'cifar100', 'train_images')):
    os.mkdir(os.path.join(save_dir, 'cifar100', 'train_images'))
if not os.path.exists(os.path.join(save_dir, 'cifar100', 'test_images')):
    os.mkdir(os.path.join(save_dir, 'cifar100', 'test_images'))
if not os.path.exists(os.path.join(save_dir, 'cifar100', 'val_images')):
    os.mkdir(os.path.join(save_dir, 'cifar100', 'val_images'))
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

X_train = train_images[:45000]
Y_train = train_labels[:45000]
X_val = train_images[45000:]
Y_val = train_labels[45000:]
X_test = test_images
Y_test = test_labels

count = 0
for i in tqdm(range(X_train.shape[0])):
    im = Image.fromarray(X_train[i])
    im.save(os.path.join(save_dir, 'cifar100', 'train_images', "%d_%d.png" % (Y_train[i,0], count)))
    count += 1

count = 0
for i in tqdm(range(X_val.shape[0])):
    im = Image.fromarray(X_val[i])
    im.save(os.path.join(save_dir, 'cifar100', 'val_images', "%d_%d.png" % (Y_val[i,0], count)))
    count += 1

count = 0
for i in tqdm(range(X_test.shape[0])):
    im = Image.fromarray(X_test[i])
    im.save(os.path.join(save_dir, 'cifar100', 'test_images', "%d_%d.png" % (Y_test[i,0], count)))
    count += 1


