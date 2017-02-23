""" messy code for loading datasets (could be cleaned up) """

import numpy as np 
from sklearn.datasets import fetch_mldata
from numpy.linalg import norm
from math import sqrt
from platform import system
import os
from sklearn import preprocessing

def get_dataset_names():
    return ['mnist', 'mnist_maxscaled_second', 'mnist_big_maxscaled', 'mnist_centered', 'mnist_big', 'mnist_short', 'mnist_short_test', 'mnist_maxscaled_short_test', 'mnist_big_short', 'mnist_big_short_test', 'mnist_maxscaled', 'mnist_maxscaled_short', 'mnist_short_test', 'mnist_short_test_train', 'mnist_short_test_test', 'cifar10_gray', 'cifar10_gray_patches', 'cifar10', 'cifar10_patches', 'cifar10_color', 'cifar10_red', 'cifar10_red_maxscaled', 'cifar10_green', 'cifar10_blue', 'cifar10_red_patches', 'digits_train', 'digits_test', 'lfw_max', 'lfw_unit']

def get_color_datasets():
    return ['cifar10_color']

def load_dataset(dataset):
    if dataset == "mnist":
        ind_train = np.load('train.npy')
        ind_test = np.load('test.npy')

        X = get_dataset('mnist', half='full')
        X_train = X[ind_train, :]
        X_test = X[ind_test, :]
        return X_train, X_test
    elif dataset == "cifar10_red":
        X = get_dataset('cifar10_red', half='full')
        X_train = X[0:50000, :]
        X_test = X[50000:60000, :]
        return X_train, X_test

def get_dataset(dataset, half='full', **kwargs):
    if dataset == 'mnist':
        data = get_mnist(centered=False, **kwargs)
    elif dataset =='mnist_centered':
        data = get_mnist(centered=True, **kwargs)
    #   data = get_cifar10_color(**kwargs)
    elif dataset == 'cifar10_red':
        data = get_cifar10_onecolor(0, **kwargs)
    elif dataset == 'cifar10_red_maxscaled':
        data = get_cifar10_onecolor_maxscaled(0, **kwargs)
    elif dataset == 'cifar10_green':
        data = get_cifar10_onecolor(1, **kwargs)
    elif dataset == 'cifar10_blue':
        data = get_cifar10_onecolor(2, **kwargs)
    elif dataset == 'cifar10_red_patches':
        data = get_cifar10_onecolor_patches(0, **kwargs)
    # elif dataset == 'digits_train':
    #   data = get_digits_train(**kwargs)
    # elif dataset == 'digits_test':
    #   data = get_digits_test(**kwargs)
    else:
        data = globals()["get_%s" % dataset](**kwargs)

    if 'return_labels' in kwargs and kwargs['return_labels']:
        data, labels = data
        ndatapoints = data.shape[0]
        if half == 'full':
            return data, labels
        else:
            idx = np.load('/fastscratch/jmcohen/%s_%s.npy' % (dataset, half))
            return data[idx,:], labels[idx]
    else:
        ndatapoints = data.shape[0]
        if half == 'full':
            return data 
        else:
            idx = np.load('/fastscratch/jmcohen/%s_%s.npy' % (dataset, half))
            return data[idx,:]

def get_lfw_max(**kwargs):
    return np.load('/jukebox/norman/jmcohen/lfw_max.npy')

def get_lfw_unit(**kwargs):
    return np.load('/jukebox/norman/jmcohen/lfw_unit.npy')

def get_mnist_short(**kwargs):
    NCLASSES = 10
    NEXAMPLES_PER_CLASS = 300
    DIM = 196
    digits = np.zeros((NCLASSES * NEXAMPLES_PER_CLASS, DIM))
    X = get_mnist(return_labels=False)
    for i in range(NCLASSES):
        digits[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS, :] = X[i*7000 : i*7000 + NEXAMPLES_PER_CLASS, :]
    return digits

def get_mnist_big_short(**kwargs):
    NCLASSES = 10
    NEXAMPLES_PER_CLASS = 300
    DIM = 28 ** 2
    digits = np.zeros((NCLASSES * NEXAMPLES_PER_CLASS, DIM))
    X = get_mnist_big(return_labels=False)
    for i in range(NCLASSES):
        digits[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS, :] = X[i*7000 : i*7000 + NEXAMPLES_PER_CLASS, :]
    return digits

def get_mnist_big_short_test(return_labels=False, **kwargs):
    NCLASSES = 10
    NEXAMPLES_PER_CLASS = 6700
    DIM = 28 ** 2
    digits = np.zeros((NCLASSES * NEXAMPLES_PER_CLASS, DIM))
    digits_labels = np.zeros(NCLASSES * NEXAMPLES_PER_CLASS)
    X, labels = get_mnist_big(return_labels=True)
    for i in range(NCLASSES):
        digits[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS, :] = X[i*7000 + 300: (i+1)*7000, :]
        digits_labels[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS] = labels[i*7000 + 300: (i+1)*7000]
    if return_labels == True:
        return digits, digits_labels
    else:
        return digits

def get_mnist_short_test(return_labels=False, **kwargs):
    NCLASSES = 10
    NEXAMPLES_PER_CLASS = 6700
    DIM = 196
    digits = np.zeros((NCLASSES * NEXAMPLES_PER_CLASS, DIM))
    digits_labels = np.zeros(NCLASSES * NEXAMPLES_PER_CLASS)
    X, labels = get_mnist(return_labels=True)
    for i in range(NCLASSES):
        digits[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS, :] = X[i*7000 + 300 : (i+1)*7000, :]
        digits_labels[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS] = labels[i*7000 + 300 : (i+1)*7000]
    if return_labels == True:
        return digits, digits_labels
    else:
        return digits

def get_mnist_maxscaled_short_test(return_labels=False, **kwargs):
    NCLASSES = 10
    NEXAMPLES_PER_CLASS = 6700
    DIM = 196
    digits = np.zeros((NCLASSES * NEXAMPLES_PER_CLASS, DIM))
    digits_labels = np.zeros(NCLASSES * NEXAMPLES_PER_CLASS)
    X, labels = get_mnist_maxscaled(return_labels=True)
    for i in range(NCLASSES):
        digits[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS, :] = X[i*7000 + 300 : (i+1)*7000, :]
        digits_labels[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS] = labels[i*7000 + 300 : (i+1)*7000]
    if return_labels == True:
        return digits, digits_labels
    else:
        return digits

def get_mnist_short_test_train(**kwargs):
    NCLASSES = 10
    NEXAMPLES_PER_CLASS = 700
    DIM = 196
    digits = np.zeros((NCLASSES * NEXAMPLES_PER_CLASS, DIM))
    digits_labels = np.zeros(NCLASSES * NEXAMPLES_PER_CLASS)
    X, labels = get_mnist(return_labels=True)
    for i in range(NCLASSES):
        digits[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS, :] = X[i*7000 + 300 : i*7000 + 1000, :]
        digits_labels[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS] = labels[i*7000 + 300 : i*7000 + 1000]
    return digits, digits_labels

def get_mnist_short_test_test(**kwargs):
    NCLASSES = 10
    NEXAMPLES_PER_CLASS = 6000
    DIM = 196
    digits = np.zeros((NCLASSES * NEXAMPLES_PER_CLASS, DIM))
    digits_labels = np.zeros(NCLASSES * NEXAMPLES_PER_CLASS)
    X, labels = get_mnist(return_labels=True)
    for i in range(NCLASSES):
        digits[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS, :] = X[i*7000 + 1000 : (i+1)*7000, :]
        digits_labels[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS] = labels[i*7000 + 1000 : (i+1)*7000]
    return digits, digits_labels

def get_mnist_maxscaled_short(**kwargs):
    NCLASSES = 10
    NEXAMPLES_PER_CLASS = 300
    DIM = 196
    digits = np.zeros((NCLASSES * NEXAMPLES_PER_CLASS, DIM))
    X = get_mnist_maxscaled(return_labels=False)
    for i in range(NCLASSES):
        digits[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS, :] = X[i*7000 : i*7000 + NEXAMPLES_PER_CLASS, :]
    return digits

def get_digits_train(**kwargs):
    NCLASSES = 10
    NEXAMPLES_PER_CLASS = 30
    DIM = 196
    digits = np.zeros((NCLASSES * NEXAMPLES_PER_CLASS, DIM))
    X = get_mnist(return_labels=False)
    for i in range(NCLASSES):
        digits[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS, :] = X[i*7000 : i*7000 + NEXAMPLES_PER_CLASS, :]
    return digits

def get_digits_test(**kwargs):
    NCLASSES = 10
    NEXAMPLES_PER_CLASS = 5
    DIM = 196
    digits = np.zeros((NCLASSES * NEXAMPLES_PER_CLASS, DIM))
    X = get_mnist(return_labels=False)
    for i in range(NCLASSES):
        digits[i*NEXAMPLES_PER_CLASS : (i+1)*NEXAMPLES_PER_CLASS, :] = X[i*7000 + 30 : i*7000 + 35, :]
    return digits

def get_mnist(return_labels=False, centered=False, **kwargs):
    if system() == 'Darwin':
        mnist = fetch_mldata('MNIST original')
    else:
        mnist = fetch_mldata('MNIST original', data_home='~/skdata')
    X = downsample_images(mnist['data'], 2)
    if centered:
        # X = preprocessing.scale(X, with_std=False)
        X = X - X.mean(0)
    for i in range(X.shape[0]):
        X[i,:] = X[i,:] / norm(X[i,:])
    if return_labels:
        return (X, mnist['target'])
    else:
        return X

def get_mnist_big(return_labels=False, centered=False, **kwargs):
    if system() == 'Darwin':
        mnist = fetch_mldata('MNIST original')
    else:
        mnist = fetch_mldata('MNIST original', data_home='~/skdata')
    X = mnist['data'].astype(np.float)
    for i in range(X.shape[0]):
        X[i,:] = X[i,:] / norm(X[i,:])
    if return_labels:
        return (X, mnist['target'])
    else:
        return X

def get_mnist_big_maxscaled(return_labels=False, **kwargs):
    if system() == 'Darwin':
        mnist = fetch_mldata('MNIST original')
    else:
        mnist = fetch_mldata('MNIST original', data_home='/fastscratch/jmcohen/skdata')
    X = mnist['data'].astype(np.float)
    maxnorm = max([norm(x) for x in X])
    for i in range(X.shape[0]):
        X[i,:] = X[i,:] / maxnorm
    if return_labels:
        return (X, mnist['target'])
    else:
        return X

def get_mnist_maxscaled(return_labels=False, **kwargs):
    if system() == 'Darwin':
        mnist = fetch_mldata('MNIST original')
    else:
        mnist = fetch_mldata('MNIST original', data_home='/fastscratch/jmcohen/skdata')
    X = downsample_images(mnist['data'], 2)
    maxnorm = max([norm(x) for x in X])
    for i in range(X.shape[0]):
        X[i,:] = X[i,:] / maxnorm
    if return_labels:
        return (X, mnist['target'])
    else:
        return X

def get_cifar10_color(return_labels=False, downsample=True, **kwargs):
    if system() == 'Darwin':
        directory = '/Users/jeremy/Documents/hazan/cifar-10-batches-py'
    else:
        directory = '/jukebox/norman/jmcohen/cifar-10-batches-py'

    p = 16 ** 2 if downsample else 32 ** 2
    X = np.zeros((60000, p * 3))
    labels = np.zeros(60000)
    i = 0
    for file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
        data = unpickle(os.path.join(directory, file))
        images = data['data']
        labels[i:i+10000] = data['labels']

        for j in range(images.shape[0]):
            channels = separate_channels(images[j,:])
            X[i,:] = np.hstack([downsample_image(channels[k,:], 2) for k in range(3)])
            i += 1

    if return_labels:
        return X, labels
    else:
        return X

def get_cifar10_onecolor_patches(color_index, factor=4, **kwargs):
    if system() == 'Darwin':
        directory = '/Users/jeremy/Documents/hazan/cifar-10-batches-py'
    else:
        directory = '/jukebox/norman/jmcohen/cifar-10-batches-py'

    n = 60000
    side_length = 32
    new_side_length = side_length / factor
    X = np.zeros((n * (factor ** 2), new_side_length ** 2))

    i = 0
    for file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
        data = unpickle(os.path.join(directory, file))
        images = data['data']

        for j in range(images.shape[0]):
            channels = separate_channels(images[j,:])
            channel = channels[color_index,:]
            X[i: i+(factor**2), :] = make_patches(channel, factor)

            for l in range(factor**2):
                if norm(X[i + l, :]) > 0:
                    X[i+l,:] /= norm(X[i+l,:])

            i += (factor**2)

    return X


def get_cifar10_onecolor(color_index, return_labels=False, downsample=False, **kwargs):
    if system() == 'Darwin':
        directory = '/Users/jcohen/Downloads/cifar-10-batches-py'
    else:
        directory = '/home/jcohen/cifar-10-batches-py'

    p = 16 ** 2 if downsample else 32 ** 2
    X = np.zeros((60000, p))
    i = 0
    labels = np.zeros(60000)
    for file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
        data = unpickle(os.path.join(directory, file))
        images = data['data']
        labels[i:i+10000] = data['labels']

        for j in range(images.shape[0]):
            channels = separate_channels(images[j,:])
            channel = channels[color_index,:]
            if downsample:
                X[i,:] = downsample_image(channel, 2)
            else:
                X[i,:] = channel

            X[i,:] /= norm(X[i,:])
            i += 1

    if return_labels:
        return (X, labels)
    else:
        return X

def get_cifar10_onecolor_maxscaled(color_index, return_labels=False, downsample=True, **kwargs):
    if system() == 'Darwin':
        directory = '/Users/jeremy/Documents/hazan/cifar-10-batches-py'
    else:
        directory = '/jukebox/norman/jmcohen/cifar-10-batches-py'

    p = 16 ** 2 if downsample else 32 ** 2
    X = np.zeros((60000, p))
    i = 0
    labels = np.zeros(60000)
    for file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
        data = unpickle(os.path.join(directory, file))
        images = data['data']
        labels[i:i+10000] = data['labels']

        for j in range(images.shape[0]):
            channels = separate_channels(images[j,:])
            channel = channels[color_index,:]
            if downsample:
                X[i,:] = downsample_image(channel, 2)
            else:
                X[i,:] = channel
            i += 1

    maxnorm = max([norm(x) for x in X])
    for i in xrange(X.shape[0]):
        X[i,:] = X[i,:] / maxnorm

    if return_labels:
        return (X, labels)
    else:
        return X


def get_cifar10(return_labels=False, downsample=True, **kwargs):
    if system() == 'Darwin':
        directory = '/Users/jeremy/Documents/hazan/cifar-10-batches-py'
    else:
        directory = '/jukebox/norman/jmcohen/cifar-10-batches-py'

    p = 16 ** 2 if downsample else 32 ** 2
    X = np.zeros((60000 * 3, p))
    i = 0
    for file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
        data = unpickle(os.path.join(directory, file))
        images = data['data']

        for j in range(images.shape[0]):
            channels = separate_channels(images[j,:])
            if downsample:
                X[i:i+3,:] = downsample_images(channels, 2)
            else:
                X[i:i+3,:] = channels

            for k in range(3):
                X[i+k,:] /= norm(X[i+k,:])

            i += 3

    return X

def get_cifar10_patches(factor=4, **kwargs):
    if system() == 'Darwin':
        directory = '/Users/jeremy/Documents/hazan/cifar-10-batches-py'
    else:
        directory = '/jukebox/norman/jmcohen/cifar-10-batches-py'

    n = 60000
    side_length = 32
    new_side_length = side_length / factor
    X = np.zeros((n * (factor ** 2) * 3, new_side_length ** 2))

    i = 0
    for file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
        data = unpickle(os.path.join(directory, file))
        images = data['data']

        for j in range(images.shape[0]):
            channels = separate_channels(images[j,:])
            for k in range(3):
                X[i+k*(factor**2) : i+(k+1)*(factor**2), :] = make_patches(channels[k,:], factor)

            for l in range(3*factor**2):
                if norm(X[i + l, :]) > 0:
                    X[i + l,:] /= norm(X[i+l,:])
                
            i += 3*(factor**2)

    return X


def get_cifar10_gray(return_labels=False, downsample=True, **kwargs):
    if system() == 'Darwin':
        directory = '/Users/jeremy/Documents/hazan/cifar-10-batches-py'
    else:
        directory = '/jukebox/norman/jmcohen/cifar-10-batches-py'

    p = 16 ** 2 if downsample else 32 ** 2
    X = np.zeros((60000, p))
    labels = np.zeros(60000)
    i = 0
    for file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
        data = unpickle(os.path.join(directory, file))
        images = data['data']
        labels[i:i+10000] = data['labels']

        for j in range(images.shape[0]):
            gray = grayscale(images[j,:])

            X[i,:] = downsample_image(gray, 2) if downsample else gray
            X[i,:] /= norm(X[i,:])

            i += 1

    if return_labels:
        return X, labels
    else:
        return X

def get_cifar10_gray_patches(factor=4, **kwargs):
    if system() == 'Darwin':
        directory = '/Users/jeremy/Documents/hazan/cifar-10-batches-py'
    else:
        directory = '/jukebox/norman/jmcohen/cifar-10-batches-py'
    n = 60000
    side_length = 32

    new_side_length = side_length / factor
    X = np.zeros((n * (factor ** 2), new_side_length ** 2))

    i = 0
    for file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
        data = unpickle(os.path.join(directory, file))
        images = data['data']

        for j in range(images.shape[0]):
            gray = grayscale(images[j,:])
            patches = make_patches(gray, factor)
            for k in range(patches.shape[0]):
                if norm(patches[k,:]) > 0:
                    patches[k,:] /= norm(patches[k,:])
            X[i:i+factor**2,:] = patches
            i += factor ** 2

    return X


def grayscale(RGB):
    R = RGB[:1024]
    B = RGB[1024:2048]
    G = RGB[2048:]
    gray = (R + B + G) / 3.0
    return gray

def separate_channels(image_flat, downsample=True):
    """ separate an image into three color channels """
    p = image_flat.size

    image = np.zeros((3, p/3))
    image[0,:] = image_flat[0:p/3]
    image[1,:] = image_flat[p/3 : 2*p/3]
    image[2,:] = image_flat[2*p/3 : ]
    return image

def make_patches(x, factor):
    """ Turn an image (as vector) into a set of factor^2 image patches."""
    p = x.size
    side_length = int(sqrt(p))
    new_side_length = side_length / factor
    patches = np.zeros((factor ** 2, new_side_length ** 2))
    index = 0
    for i in range(factor):
        for j in range(factor):
            image = x.reshape((side_length, side_length))
            patches[j + factor * i, :] = image[i*new_side_length:(i+1)*new_side_length, j*new_side_length:(j+1)*new_side_length].reshape((new_side_length**2,))
    return patches

def downsample_image(image, factor):
    p = len(image)
    side_length = int(sqrt(p))
    image = image.reshape((side_length, side_length))
    new_side_length = side_length / factor
    down_image = image[::factor, ::factor].reshape((int(new_side_length ** 2),))
    return down_image

def downsample_images(images, factor):
    n, p = images.shape
    side_length = int(sqrt(p))
    new_side_length = side_length / factor
    down_images = np.zeros((n, int(new_side_length ** 2)))
    for i in range(n):
        down_images[i,:] = downsample_image(images[i,:], factor)
    return down_images

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict
