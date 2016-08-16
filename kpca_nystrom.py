#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

""" Kernel PCA with the Nystrom approximation """

import numpy as np 
from math import sqrt
from numpy.linalg import norm
import argparse
import os
from datasets import get_dataset, get_dataset_names
from scipy.sparse.linalg import eigs
from common import unsigned_distance, poly2_preimage, poly2_kernel, poly_kernel, gaussian_kernel
from models import reconstruct_kpca_direct

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Kernel PCA')
	parser.add_argument('dataset', type=str, choices=get_dataset_names())
	parser.add_argument('directory', type=str, help="directory in which to save results")
	parser.add_argument('k', type=int, help="number of components to learn")
	parser.add_argument('m', type=int, help='number of data poins to use for nystrom approximation')
	parser.add_argument('--half', choices=['full', 'first', 'second'], default='full', help="which half of the dataset to use")
	parser.add_argument('--kernel', type=str, choices=['poly', 'gaussian'], help="which kernel to use")
	parser.add_argument('--degree', type=int, default=2, help='if using polynomial kernel, the degree')
	parser.add_argument('--sigma2', type=float, default=1.0, help="if using gaussian kernel, the scale parameter")
	args = parser.parse_args()

	k = args.k

	# save results in exp_dir
	exp_dir = '%s/nystrom_k_%d_%d' % (args.directory, args.k, args.m)
	if args.kernel == 'poly':
		exp_dir += '_degree_%d' % args.degree
	elif args.kernel == 'gaussian':
		exp_dir += '_gaussian_%s' % args.sigma2
	exp_dir += '_%s' % args.half
	if args.directory and not os.path.exists(exp_dir):
		os.mkdir(exp_dir)

	X = get_dataset(args.dataset, half=args.half)
	n, d = X.shape

	# sample some of the data points for the nystrom approximation
	sample = np.random.choice(n, size=(args.m), replace=False)
	X_sample = X[sample,:]

	# compute the kernel
	if args.kernel == 'poly':
		K = poly_kernel(X, X_sample, args.degree)
	elif args.kernel == 'gaussian':
		K = gaussian_kernel(X, X_sample, args.sigma2)

	# use Lanczos to compute the leading k eigenvectors of the reduced kernel matrix
	reduced_lam, reduced_V = eigs(K[sample,:], k=k)

	# approximate the eigenvalues of the full kernel matrix
	lam = (float(n) / args.m) * reduced_lam

	# each row of A are the coefficients for a PC in feature space expressed as a linear combination of the data points in feature space
	A = np.zeros((k, n))

	for j in range(k):
		print j
		# appproximate the j-th eigenvector of the full kernel matrix from the j-th
		# eigenvector of the reduced kernel matrix
		v = (sqrt(float(args.m) / n) / reduced_lam[j]) * K.dot(reduced_V[:, j])

		# divide the eigenvector by the square root of its correponding eigenvalue
		# this ensure that the PCs in feature space are unit norm
		A[j,:] = v / sqrt(lam[j])

	# save A
	np.save('%s/iter0_A.npy' % exp_dir, A)

	# # if poly2 and non-centered, compute and save reconstruction error
	if args.kernel == 'poly' and args.degree == 2:
		recon = reconstruct_kpca_direct(X, X, A, A.T)
		recon_err = norm(recon - X, 'fro') ** 2	
		np.savetxt('%s/iter0_recon.txt' % exp_dir, np.array([recon_err]))

