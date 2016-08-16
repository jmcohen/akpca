#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

""" Kernel PCA """

import numpy as np 
from math import sqrt
from numpy.linalg import norm
import argparse
import os
from datasets import get_dataset, get_dataset_names
from scipy.sparse.linalg import eigs
from common import unsigned_distance, poly2_preimage, poly2_kernel, poly_kernel
from models import reconstruct_kpca_direct
from scipy.spatial.distance import cdist

def center_kernel_matrix(K):
	""" Given the kernel matrix, compute the kernel matrix of the data points *centered* in feature space

	Parameters
	----------
	K : ndarray, shape (n, n)

	Returns
	-------
	ndarray, shape (n, n)

	"""
	n = K.shape[0]
	ones = np.ones((n, n)) * (1.0 / n)
	K_centered = K - ones.dot(K) - K.dot(ones) + ones.dot(K).dot(ones)
	return K_centered

def gaussian_kernel(X, Y, sigma2):
	""" Computes the Gaussian kernel function k(x, y) = exp( -||x - y||^2 / (2 sigma^2) ) 
	between each (x, y) pair

	Parameters
	----------
	X : ndarray, shape (n1, d)
	Y : ndarray, shape (n2, d)

	Returns
	-------
	K : ndarray, shape (n1, n2)

	"""
	dist = cdist(X, Y, 'sqeuclidean')
	return np.exp(-dist / (2*sigma2))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Kernel PCA')
	parser.add_argument('dataset', type=str, choices=get_dataset_names())
	parser.add_argument('directory', type=str, help="directory in which to save results")
	parser.add_argument('k', type=int, help="number of components to learn")
	parser.add_argument('--half', choices=['full', 'first', 'second'], default='full', help="which half of the dataset to use")
	parser.add_argument('--centered', action='store_true')
	parser.add_argument('--kernel', type=str, choices=['poly', 'gaussian'], help="which kernel to use")
	parser.add_argument('--degree', type=int, help='if using polynomial kernel, the degree')
	parser.add_argument('--sigma2', type=float, default=1.0, help="if using gaussian kernel, the scale parameter")
	args = parser.parse_args()

	k = args.k

	# save results in exp_dir
	exp_dir = '%s/mykpca_k_%d_%s' % (args.directory, args.k, args.half)
	if args.directory and not os.path.exists(exp_dir):
		os.mkdir(exp_dir)

	X = get_dataset(args.dataset, half=args.half)
	n, d = X.shape

	# compute the kernel
	if args.kernel == 'poly':
		K = poly_kernel(X, X, args.degree)
	elif args.kernel == 'gaussian':
		K = gaussian_kernel(X, X, args.sigma2)

	# center the kernel matrix if appropriate
	if args.centered:
		K = center_kernel_matrix(K)

	# use Lanczos to compute the leading k eigenvectors of the kernel matrix
	w, V = eigs(K, k=k)

	# each row of A are the coefficients for a PC in feature space expressed as a linear combination of the data points in feature space
	A = V[:, 0:k].T

	# divide each eigenvector by the square root of its corresponding eigenvalue.
	# this ensures that the principal components in feature space are unit norm.
	for j in range(k):
		A[j,:] = A[j,:] / sqrt(w[j])

	# save A
	np.save('%s/iter0_A.npy' % exp_dir, A)

	# if poly2 and non-centered, compute and save reconstruction error
	if not args.centered and args.kernel == 'poly' and args.degree == 2:
		recon = reconstruct_kpca_direct(X, X, A, A.T)
		recon_err = norm(recon - X, 'fro') ** 2	
		np.savetxt('%s/iter0_recon.txt' % exp_dir, np.array([recon_err]))

