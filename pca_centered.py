#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

""" Principal Component Analysis (centered) """

import numpy as np 
from math import sqrt
from numpy.linalg import norm
import argparse
import os
from datasets import get_dataset, get_dataset_names

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PCA')
	parser.add_argument('dataset', type=str, choices=get_dataset_names())
	parser.add_argument('directory', type=str, help="directory in which to save results")
	parser.add_argument('k', type=int, help="number of components to learn")
	parser.add_argument('--half', choices=['full', 'first', 'second'], default='full', help="which half of the dataset to use")
	args = parser.parse_args()

	k = args.k

	# save results in exp_dir
	exp_dir = '%s/pca_k_%d_%s' % (args.directory, args.k, args.half)
	if args.directory and not os.path.exists(exp_dir):
		os.mkdir(exp_dir)

	ind_train = np.load('train.npy')
	ind_test = np.load('test.npy')

	X = get_dataset(args.dataset, half='full')
	X_train = X[ind_train, :]
	X_test = X[ind_test, :]

	# compute the mean
	train_center = X_train.mean(0)
	test_center = X_test.mean(0)

	# remove the mean
	Xtrain_centered = X_train - train_center
	Xtest_centered = X_test - test_center

	# compute the singular value decomposition of the data matrix
	U, S, Vt = np.linalg.svd(Xtrain_centered, full_matrices=False)

	# A = the top k right singular vectors
	A = Vt[0:k, :]

	# compute the reconstruction of X 
	recon = Xtest_centered.dot(A.T).dot(A) + test_center

	# compute reconstruction error
	recon_err = np.linalg.norm(X_test - recon, 'fro') ** 2 / X_test.shape[0]

	# save A and the reconstruction error
	print(recon_err)
