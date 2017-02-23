#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

""" Principal Component Analysis (not centered) """

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
	args = parser.parse_args()

	k = args.k

	# save results in exp_dir
	exp_dir = '%s/pca_k_%d_%s' % (args.directory, args.k)
	if args.directory and not os.path.exists(exp_dir):
		os.mkdir(exp_dir)

	ind_train = np.load('train.npy')
	ind_test = np.load('test.npy')

	X = get_dataset(args.dataset, half='full')
	X_train = X[ind_train, :]
	X_test = X[ind_test, :]

	# compute the singular value decomposition of the data matrix
	U, S, Vt = np.linalg.svd(X_train, full_matrices=False)

	# A = the top k right singular vectors
	A = Vt[0:k, :]

	# compute the reconstruction of X 
	recon = X_test.dot(A.T).dot(A)

	# compute reconstruction error
	recon_err = np.linalg.norm(X_test - recon, 'fro') ** 2 / X_test.shape[0]

	print(recon_err)

	# save A and the reconstruction error
	np.save('%s/iter0_A.npy' % exp_dir, A)
	np.savetxt('%s/iter0_recon.txt' % exp_dir, np.array([recon_err]))
