#!/usr/bin/env python

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to download all datasets and create .tfrecord files.
"""

import collections
import gzip
import os
import tarfile
import tempfile
from urllib import request

import numpy as np
import scipy.io
import tensorflow.compat.v1 as tf
from absl import app
from tqdm import trange

from libml import data as libml_data
from libml.utils import EasyDict

import pickle
from json_creator import AnnotationJson as annotation_handler
import imageio
import cv2

URLS = {
    'svhn': 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat',
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz',
    'cifar100': 'https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz',
    'stl10': 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz',
}

use_soft_labels = True

def _encode_png(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)
        for x in trange(images.shape[0], desc='PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw


def _load_svhn():
    splits = collections.OrderedDict()
    for split in ['train', 'test', 'extra']:
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['svhn'].format(split), f.name)
            data_dict = scipy.io.loadmat(f.name)
        dataset = {}
        dataset['images'] = np.transpose(data_dict['X'], [3, 0, 1, 2])
        dataset['images'] = _encode_png(dataset['images'])
        dataset['labels'] = data_dict['y'].reshape((-1))
        # SVHN raw data uses labels from 1 to 10; use 0 to 9 instead.
        dataset['labels'] -= 1
        splits[split] = dataset
    return splits


def _load_stl10():
    def unflatten(images):
        return np.transpose(images.reshape((-1, 3, 96, 96)),
                            [0, 3, 2, 1])

    with tempfile.NamedTemporaryFile() as f:
        if tf.gfile.Exists('stl10/stl10_binary.tar.gz'):
            f = tf.gfile.Open('stl10/stl10_binary.tar.gz', 'rb')
        else:
            request.urlretrieve(URLS['stl10'], f.name)
        tar = tarfile.open(fileobj=f)
        train_X = tar.extractfile('stl10_binary/train_X.bin')
        train_y = tar.extractfile('stl10_binary/train_y.bin')

        test_X = tar.extractfile('stl10_binary/test_X.bin')
        test_y = tar.extractfile('stl10_binary/test_y.bin')

        unlabeled_X = tar.extractfile('stl10_binary/unlabeled_X.bin')

        train_set = {'images': np.frombuffer(train_X.read(), dtype=np.uint8),
                     'labels': np.frombuffer(train_y.read(), dtype=np.uint8) - 1}

        test_set = {'images': np.frombuffer(test_X.read(), dtype=np.uint8),
                    'labels': np.frombuffer(test_y.read(), dtype=np.uint8) - 1}

        _imgs = np.frombuffer(unlabeled_X.read(), dtype=np.uint8)
        unlabeled_set = {'images': _imgs,
                         'labels': np.zeros(100000, dtype=np.uint8)}

        fold_indices = tar.extractfile('stl10_binary/fold_indices.txt').read()

    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    unlabeled_set['images'] = _encode_png(unflatten(unlabeled_set['images']))
    return dict(train=train_set, test=test_set, unlabeled=unlabeled_set,
                files=[EasyDict(filename="stl10_fold_indices.txt", data=fold_indices)])


def _load_cifar10():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile(delete=False) as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        train_data_batches, train_data_labels = [], []
        for batch in range(1, 6):
            data_dict = scipy.io.loadmat(tar.extractfile(
                'cifar-10-batches-mat/data_batch_{}.mat'.format(batch)))
            train_data_batches.append(data_dict['data'])
            train_data_labels.append(data_dict['labels'].flatten())
        train_set = {'images': np.concatenate(train_data_batches, axis=0),
                     'labels': np.concatenate(train_data_labels, axis=0)}
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['labels'].flatten()}
        f.close()
        os.unlink(f.name)
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)

def _load_cifar10h():
    '''
    Loads CIFAR-10 and uses only the test set with the same split as with the fuzzy labels.
    '''
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile(delete=False) as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))
        split = np.split(data_dict['data'], [8000])
        labels = np.split(data_dict['labels'], [8000])
        train_set = {'images': split[0],
                     'labels': labels[0].flatten()}
        test_set = {'images': split[1],
                    'labels': labels[1].flatten()}
        f.close()
        os.unlink(f.name)
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)

def _load_cifar10hf():
    '''
    Loads CIFAR-10 and replaces the test set labels with the soft labels.
    Uses the last 2000 samples as test set.
    '''
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile(delete=False) as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))
        split = np.split(data_dict['data'], [8000])
        labels = np.load("../tables/cifar10h-probs.npy")
        labels = np.split(labels, [8000])
        train_set = {'images': split[0],
                     'labels': labels[0]}
        test_set = {'images': split[1],
                    'labels': labels[1]}
        f.close()
        os.unlink(f.name)
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)
    

def _load_cifar10hfsn():
    '''
    Loads CIFAR-10 and replaces the test set labels with the soft labels.
    Then adds normally distributed label noise according to the percentage in noise_level.
    Uses the last 2000 samples as test set.
    '''
    noise_level = 0.1
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile(delete=False) as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))

        split = np.split(data_dict['data'], [8000])
        labels = np.load("../tables/cifar10h-probs.npy")
        labels = np.split(labels, [8000])

        for i in range(len(labels[0])):
            noise_class = np.random.randint(0, 10)
            labels[0][i] = labels[0][i] * (1 - noise_level)
            labels[0][i][noise_class] += noise_level

        train_set = {'images': split[0],
                     'labels': labels[0]}
        test_set = {'images': split[1],
                    'labels': labels[1]}
        f.close()
        os.unlink(f.name)
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)


def _load_cifar10hfn():
    '''
    Loads CIFAR-10 and replaces the test set labels with the soft labels.
    Then adds single class noise according to the percentage in noise_level.
    Uses the last 2000 samples as test set.
    '''
    noise_level = 0.1
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile(delete=False) as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))

        split = np.split(data_dict['data'], [8000])
        labels = np.load("../tables/cifar10h-probs.npy")
        labels = np.split(labels, [8000])

        for i in range(len(labels[0])):
            noise = np.random.normal(0, 1, (10))
            # shift noise to remain only positive
            min = np.amin(noise)
            if min < 0:
                noise -= min
            labels[0][i] = labels[0][i] * (1 - noise_level) + (noise  / np.sum(noise)) * noise_level
        train_set = {'images': split[0],
                     'labels': labels[0]}
        test_set = {'images': split[1],
                    'labels': labels[1]}
        f.close()
        os.unlink(f.name)
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)

def _load_cifar10votes():
    '''
    Loads CIFAR-10 and replaces the test set labels with the soft labels.
    Then uses the soft labels as probabilities to randomly draw votes from.
    Replaces the soft labels with distribution from artifical votes.
    Uses the last 2000 samples as test set.
    '''
    votes = 3
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile(delete=False) as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))
        split = np.split(data_dict['data'], [8000])
        labels = np.load("../tables/cifar10h-probs.npy")
        train_labels = np.zeros((8000, 10), dtype=float)
        for i in range(8000):
            choices = np.random.choice(10, votes, p=labels[i])
            choices = np.bincount(choices)
            for j in range(len(choices)):
                train_labels[i][j] = choices[j] / votes
        train_labels = train_labels / train_labels.sum(axis=1, keepdims=True)
        labels = np.split(labels, [8000])
        train_set = {'images': split[0],
                     'labels': train_labels}
        test_set = {'images': split[1],
                    'labels': labels[1]}
        f.close()
        os.unlink(f.name)
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)

    
def _load_plankton_votes():
    '''
    Loads PlanktonID from numpy arrays with filtered and shuffled samples.
    Then uses the soft labels as probabilities to randomly draw votes from.
    Replaces the soft labels with distribution from artifical votes.
    Uses the first 100 samples per class as test set.
    '''
    votes = 3
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 1, 64, 64)),
                            [0, 2, 3, 1])
    
    x = np.load('../tables/plankton_x_fix_shuffle.npy')
    y = np.load('../tables/plankton_y_fix_shuffle.npy')


    images = []
    labels = []
    test_index = []
    class_count = np.zeros(10, dtype=np.int32)
    per_class = 200
    for i in range(len(x)):
        c = np.argmax(y[i])
        if class_count[c] < per_class:
            images.append(x[i])
            labels.append(y[i])
            class_count[c] += 1
            test_index.append(i)
    test_x = np.take(x, test_index, axis=0)
    test_y = np.take(y, test_index, axis=0)
    train_x = np.delete(x, test_index, axis=0)
    train_y = np.delete(y, test_index, axis=0)

    train_labels = np.zeros((len(train_y), 10), dtype=float)
    for i in range(len(train_y)):
        choices = np.random.choice(10, votes, p=train_y[i])
        choices = np.bincount(choices)
        for j in range(len(choices)):
            train_labels[i][j] = choices[j] / votes
    train_labels = train_labels / train_labels.sum(axis=1, keepdims=True)

    train_set = {'images': train_x,
                 'labels': train_labels}
    test_set = {'images': test_x,
                'labels': test_y}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)



def _load_plankton():
    '''
    Loads PlanktonID from numpy arrays with filtered and shuffled samples.
    Uses the first 100 samples per class as test set.
    '''
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 1, 64, 64)),
                            [0, 2, 3, 1])
    '''
    json = annotation_handler.from_file("E:/Uni/Masterarbeit/Plankton/tables/plankton_no_fit-s01@default/annotations.json")
    imgs, classes, data = annotation_handler.get_probability_data(json)
    images = list()
    for path in imgs:
        cur_img = cv2.imread('E:/Uni/Masterarbeit/Plankton/' + path, flags=cv2.IMREAD_GRAYSCALE)
        resize_img = cv2.resize(cur_img, dsize=(64, 64))
        #print(resize_img.shape)
        #print(cur_img.dtype)
        images.append(resize_img)
    '''

    x = np.load('../tables/plankton_x_fix_shuffle.npy')
    y = np.load('../tables/plankton_y_fix_shuffle.npy')

    images = []
    labels = []
    test_index = []
    class_count = np.zeros(10, dtype=np.int32)
    per_class = 200
    for i in range(len(x)):
        c = np.argmax(y[i])
        if class_count[c] < per_class:
            images.append(x[i])
            labels.append(y[i])
            class_count[c] += 1
            test_index.append(i)
    test_x = np.take(x, test_index, axis=0)
    test_y = np.take(y, test_index, axis=0)
    train_x = np.delete(x, test_index, axis=0)
    train_y = np.delete(y, test_index, axis=0)

    train_set = {'images': train_x,
                 'labels': train_y}
    test_set = {'images': test_x,
                'labels': test_y}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)

def _load_plankton_hard():
    '''
    Loads PlanktonID from numpy arrays with filtered and shuffled samples.
    Generates hard labels from soft lables via highest probability.
    Uses the first 100 samples per class as test set.
    '''
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 1, 64, 64)),
                            [0, 2, 3, 1])

    x = np.load('../tables/plankton_x_fix_shuffle.npy')
    y = np.load('../tables/plankton_y_fix_shuffle.npy')

    y = np.argmax(y, axis=1)

    images = []
    labels = []
    test_index = []
    class_count = np.zeros(10, dtype=np.int32)
    per_class = 200
    for i in range(len(x)):
        c = y[i]
        if class_count[c] < per_class:
            images.append(x[i])
            labels.append(y[i])
            class_count[c] += 1
            test_index.append(i)
    test_x = np.take(x, test_index, axis=0)
    test_y = np.take(y, test_index, axis=0)
    train_x = np.delete(x, test_index, axis=0)
    train_y = np.delete(y, test_index, axis=0)

    train_set = {'images': train_x,
                 'labels': train_y}
    test_set = {'images': test_x,
                'labels': test_y}

    '''
    json = annotation_handler.from_file("E:/Uni/Masterarbeit/Plankton/tables/plankton_no_fit-s01@default/annotations.json")
    imgs, classes, data = annotation_handler.get_probability_data(json)
    images = list()
    for path in imgs:
        #print(path)
        cur_img = cv2.imread('E:/Uni/Masterarbeit/Plankton/' + path, flags=cv2.IMREAD_GRAYSCALE)
        resize_img = cv2.resize(cur_img, dsize=(64, 64))
        #print(resize_img.shape)
        #print(cur_img.dtype)
        images.append(resize_img)
    
    split = np.split(images, [10000])
    labels = np.split(np.argmax(data, axis=1), [10000])
    train_set = {'images': split[0],
                 'labels': labels[0]}
    test_set = {'images': split[1],
                'labels': labels[1]}
    '''
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)


def _load_cifar100():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar100'], f.name)
        tar = tarfile.open(fileobj=f)
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-100-matlab/train.mat'))
        train_set = {'images': data_dict['data'],
                     'labels': data_dict['fine_labels'].flatten()}
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-100-matlab/test.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['fine_labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)


def _load_fashionmnist():
    def _read32(data):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(data.read(4), dtype=dt)[0]

    image_filename = '{}-images-idx3-ubyte'
    label_filename = '{}-labels-idx1-ubyte'
    split_files = [('train', 'train'), ('test', 't10k')]
    splits = {}
    for split, split_file in split_files:
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['fashion_mnist'].format(image_filename.format(split_file)), f.name)
            with gzip.GzipFile(fileobj=f, mode='r') as data:
                assert _read32(data) == 2051
                n_images = _read32(data)
                row = _read32(data)
                col = _read32(data)
                images = np.frombuffer(data.read(n_images * row * col), dtype=np.uint8)
                images = images.reshape((n_images, row, col, 1))
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['fashion_mnist'].format(label_filename.format(split_file)), f.name)
            with gzip.GzipFile(fileobj=f, mode='r') as data:
                assert _read32(data) == 2049
                n_labels = _read32(data)
                labels = np.frombuffer(data.read(n_labels), dtype=np.uint8)
        splits[split] = {'images': _encode_png(images), 'labels': labels}
    return splits


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _save_as_tfrecord(data, filename):
    assert len(data['images']) == len(data['labels'])
    filename = os.path.join(libml_data.DATA_DIR, filename + '.tfrecord')
    print('Saving dataset:', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in trange(len(data['images']), desc='Building records'):
            # Use serialization to store soft labels.
            if use_soft_labels:
                feat = dict(image=_bytes_feature(data['images'][x]),
                            label=_bytes_feature(pickle.dumps(data['labels'][x], protocol=0)))
            else:    
                feat = dict(image=_bytes_feature(data['images'][x]),
                            label=_int64_feature(data['labels'][x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    print('Saved:', filename)


def _is_installed(name, checksums):
    for subset, checksum in checksums.items():
        filename = os.path.join(libml_data.DATA_DIR, '%s-%s.tfrecord' % (name, subset))
        if not tf.gfile.Exists(filename):
            return False
    return True


def _save_files(files, *args, **kwargs):
    del args, kwargs
    for folder in frozenset(os.path.dirname(x) for x in files):
        tf.gfile.MakeDirs(os.path.join(libml_data.DATA_DIR, folder))
    for filename, contents in files.items():
        with tf.gfile.Open(os.path.join(libml_data.DATA_DIR, filename), 'w') as f:
            f.write(contents)


def _is_installed_folder(name, folder):
    return tf.gfile.Exists(os.path.join(libml_data.DATA_DIR, name, folder))


CONFIGS = dict(
    cifar10h=dict(loader=_load_cifar10h, checksums=dict(train=None, test=None), use_soft_labels=False),
    plankton_votes1=dict(loader=_load_plankton_votes, checksums=dict(train=None, test=None), use_soft_labels=True),
    cifar10votes1=dict(loader=_load_cifar10votes, checksums=dict(train=None, test=None), use_soft_labels=True),
    plankton=dict(loader=_load_plankton, checksums=dict(train=None, test=None), use_soft_labels=True),
    cifar10hf=dict(loader=_load_cifar10hf, checksums=dict(train=None, test=None), use_soft_labels=True),
    cifar10hfsn1=dict(loader=_load_cifar10hfsn, checksums=dict(train=None, test=None), use_soft_labels=True),
    cifar10hfn1=dict(loader=_load_cifar10hfn, checksums=dict(train=None, test=None), use_soft_labels=True),
    cifar10=dict(loader=_load_cifar10, checksums=dict(train=None, test=None), use_soft_labels=False),
    cifar100=dict(loader=_load_cifar100, checksums=dict(train=None, test=None), use_soft_labels=False),
    svhn=dict(loader=_load_svhn, checksums=dict(train=None, test=None, extra=None), use_soft_labels=False),
    stl10=dict(loader=_load_stl10, checksums=dict(train=None, test=None), use_soft_labels=False),
    plankton_hard=dict(loader=_load_plankton_hard, checksums=dict(train=None, test=None), use_soft_labels=False),
)




def main(argv):
    if len(argv[1:]):
        subset = set(argv[1:])
    else:
        subset = set(CONFIGS.keys())
    tf.gfile.MakeDirs(libml_data.DATA_DIR)
    for name, config in CONFIGS.items():
        if name not in subset:
            continue
        if 'is_installed' in config:
            if config['is_installed']():
                print('Skipping already installed:', name)
                continue
        elif _is_installed(name, config['checksums']):
            print('Skipping already installed:', name)
            continue
        print('Preparing', name)
        global use_soft_labels 
        use_soft_labels = config['use_soft_labels']
        datas = config['loader']()
        saver = config.get('saver', _save_as_tfrecord)
        for sub_name, data in datas.items():
            if sub_name == 'readme':
                filename = os.path.join(libml_data.DATA_DIR, '%s-%s.txt' % (name, sub_name))
                with tf.gfile.Open(filename, 'w') as f:
                    f.write(data)
            elif sub_name == 'files':
                for file_and_data in data:
                    path = os.path.join(libml_data.DATA_DIR, file_and_data.filename)
                    with tf.gfile.Open(path, "wb") as f:
                        f.write(file_and_data.data)
            else:
                saver(data, '%s-%s' % (name, sub_name))


if __name__ == '__main__':
    app.run(main)
