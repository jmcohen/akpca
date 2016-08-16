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

	X = get_dataset(args.dataset, half=args.half)

	# compute the mean
	center = X.mean(0)

	# remove the mean
	X_centered = X - center

	# compute the singular value decomposition of the data matrix
	U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

	# A = the top k right singular vectors
	A = Vt[0:k, :]

	# compute the reconstruction of X 
	recon = X_centered.dot(A.T).dot(A) + center

	# compute reconstruction error
	recon_err = np.linalg.norm(X - recon, 'fro') ** 2

	# save A and the reconstruction error
	np.save('%s/iter0_A.npy' % exp_dir, A)
	np.savetxt('%s/iter0_recon.txt' % exp_dir, np.array([recon_err]))

	print recon_err / X.shape[0]


