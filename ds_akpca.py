#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

""" Data-span AKPCA """

import numpy as np 
from math import sqrt
from numpy.linalg import norm
from datetime import datetime
import argparse
import os
from datasets import get_dataset, get_dataset_names
from scipy.sparse.linalg import svds, eigs, LinearOperator
from pesd_common import expand_poly, compute_eigenvector, poly2_preimage_kernel, poly2_kernel
from common import get_max_iter

def compute_singular_vectors(x, X, z):
	""" computes the leading left and right singular vectors of 

	xx' - \sum_i z[i] X[i,:] X[i,:]'

	Parameters
	----------
	x : ndarray, shape (d, 1)
	X : ndarray, shape (n, d)
	z : ndarray, shape (n, 1)

	Returns
	-------
	s : float
		leading singular value
	u : ndarray, shape (d, 1)
		leading left singular vector
	v : ndarray, shape (d, 1)
		leading right singular vector

	"""
	# product of M and a vector v
	def matvec(v):
		v = v.squeeze()
		return x * x.dot(v) - X.T.dot(z * X.dot(v))


	# product of M' and a vector v
	def rmatvec(v):
		return matvec(v) # the operator is symmetric

	n, d = X.shape

	M = LinearOperator((d, d), matvec=matvec, rmatvec=rmatvec)
	U, S, Vt = svds(M, k=1, tol=1e-3, maxiter=10000)

	return S[0], U[:,0], Vt[0,:]

def compute_gradient_term(i, X, Xbase, A, B, K=None, compute_recon_err=True):
	""" computes the gradient wrt A and B of 

	|| X[i] X[i]' - \sum_j X[j,:] X[j,:]' (B A X2 x2)[i]  ||

	Parameters
	---------
	x : ndarray, shape (d, 1)
	X : ndarray, shape (n, d)
	A : ndarray, shape (k, n)
	B : ndarray, shape (n, k)
	K : ndarray, shape (n, n)
		the kernel

	"""
	# the jth element of X2x2 is the dot product between X[j,:] and X[i,:] in feature space

	if K:
		X2x2 = K[i,:]
	else:
		X2x2 = poly2_kernel(Xbase, X)

	z = B.dot(A.dot(X2x2))

	loss, u, v = compute_singular_vectors(X[i,:], Xbase, z)

	dloss_dz = -Xbase.dot(u) * Xbase.dot(v)

	dloss_dA = np.outer(B.T.dot(dloss_dz), X2x2)
	dloss_dB = np.outer(dloss_dz, (A.dot(X2x2)))

	if compute_recon_err:
		recon_err = poly2_preimage_kernel(Xbase, z)
	else:
		recon_err = 0

	return (loss, recon_err, dloss_dA, dloss_dB)


def compute_gradient(sample, X, Xbase, A, B, K=None, compute_recon_err=True):
	k, n = A.shape

	total_loss = 0
	total_recon_err = 0
	total_gradA = np.zeros((k, n))
	total_gradB = np.zeros((n, k))

	for i in sample:
		(loss, recon_err, gradA, gradB) = compute_gradient_term(i, X, Xbase, A, B, K=K, compute_recon_err=compute_recon_err)
		total_loss += loss
		total_recon_err += recon_err
		total_gradA += gradA
		total_gradB += gradB

	return total_loss, total_recon_err, gradA, gradB


# well, compute the gradient on 1/4 of the training set 
def compute_full_gradient(X, Xbase, A, B, K=None, cheat_factor=4):
	n, p = X.shape
	sample = np.random.choice(range(n), size=(n/cheat_factor))
	loss, recon_err, gradA, gradB, = compute_gradient(sample, X, Xbase, A, B, K=K)
	return (loss * cheat_factor, recon_err * cheat_factor, gradA * cheat_factor, gradB * cheat_factor)

def initialize(k, n):
	A = np.random.rand(k, n) - 0.5
	A /= norm(A, 'fro')
	return A

def sgd(X, k, save_dir, base_size, initial_step_size, minibatch_size, epoch_size, nepochs, start_after, cheat_factor, half):
	""" Stochastic gradient descent for PESD.

	Use a step size schedule that decreases with the square root of the epoch number.

	Saves the current model (and the current loss, and the current reconstruction error) after every epoch.


	Parameters
	----------
	X : ndarray, shape (n, d)
		the data matrix, with each row as a data point
	k : int
		the number of components to learn
	save_dir : string 
		the directory in which to save the model
	base_size : int
		the number of data points to take as the "base" for A and B
	initial_step_size : float
	minibatch_size : int
	epoch_size : float
		number of iterations in each epoch, as a percentage of the data set
	nepochs : int
		number of epochs to run SGD for
	start_after : int
		if -1, initialize A and B randomly and start from the beginning
		if -2, initialize starting from the iteration with minimal reconstruction error
		otherwise, load the A and B from this and start from there
	cheat_factor : int
		if cheat_factor > 1, when computing the loss and reconstruction error after every epoch, 
		subsample the dataset by this factor 
	half : string
		which half of the datset 
	
	"""
	exp_dir = '%s/sgd_k_%s_step_%s_epoch_%s_%s' % (save_dir, k, initial_step_size, epoch_size, half)
	if save_dir and not os.path.exists(exp_dir):
		os.mkdir(exp_dir)

	n, d = X.shape

	if n < 10000:
		K = poly2_kernel(X, X)
	else:
		K = None

	losses = np.zeros(nepochs)
	recon_errs = np.zeros(nepochs)

	Xbase = X[:base_size, :]

	if start_after == -1:
		A = initialize(k, r)
		B = A.T

	else:
		if start_after == -2:
			start_after = get_max_iter(exp_dir)

		A = np.load('%s/iter%d_A.npy' % (exp_dir, start_after))
		B = np.load('%s/iter%d_B.npy' % (exp_dir, start_after))
		losses[:start_after] = np.loadtxt('%s/iter%s_loss.txt' % (exp_dir, start_after))
		recon_errs[:start_after] = np.loadtxt('%s/iter%s_recon.txt' % (exp_dir, start_after))

	print exp_dir

	num_iters_per_epoch = int(epoch_size * m)

	examples = range(m)
	for epoch in range(start_after + 1, nepochs):
		print datetime.now()

		step_size = initial_step_size / (sqrt(epoch + 1))

		if save_dir:
			np.save('%s/iter%d_A.npy' % (exp_dir, epoch), A)
			np.save('%s/iter%d_B.npy' % (exp_dir, epoch), B)
			np.savetxt('%s/iter%s_loss.txt' % (exp_dir, epoch), losses[:epoch])
			np.savetxt('%s/iter%s_recon.txt' % (exp_dir, epoch), recon_errs[:epoch])

		total_loss = 0
		total_recon_err = 0

		for j in range(num_iters_per_epoch):
			sample = np.random.choice(examples, size=(minibatch_size))
			t1 = datetime.now()
			(loss, _, gradA, gradB) = compute_gradient(sample, X, Xbase, A, B, K=K, compute_recon_err=False)
			t2 = datetime.now()

			total_loss += loss
			if j > 0 and j % 100 == 0:	
				print (epoch * num_iters_per_epoch + j), total_loss / (100 * minibatch_size), t2 - t1
				total_loss = 0 

			A = A - (step_size / minibatch_size) * gradA 
			B = B - (step_size / minibatch_size) * gradB 

		loss, recon_err, _, _  = compute_full_gradient(X, Xbase, A, B, K=K, cheat_factor=cheat_factor)
		print "\t", loss / n, recon_err / n

		losses[epoch] = loss
		recon_errs[epoch] = recon_err

	return A, B


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='SVRG for kernel PSED')
	parser.add_argument('dataset', type=str, choices=get_dataset_names())
	parser.add_argument('directory', type=str)
	parser.add_argument('k', type=int)
	parser.add_argument('step_size', type=float)
	parser.add_argument('--epoch_size', type=float, default=1.0)
	parser.add_argument('--nepochs', type=int, default=10)
	parser.add_argument('--start_after', type=int, default=-1)
	parser.add_argument('--patch_scale_factor', type=int, default=4)
	parser.add_argument('--abridged', action='store_true')
	parser.add_argument('--half', choices=['full', 'first', 'second'], default='full')
	parser.add_argument('--cheat_factor', type=int, default=1)
	parser.add_argument('--minibatch_size', type=int, default=10)
	parser.add_argument('--base_size', type=int, default=1000)
	args = parser.parse_args()

	X = get_dataset(args.dataset, half=args.half, factor=args.patch_scale_factor)

	if args.abridged:
		X = X[0:5000,:]

	sgd(X, args.k, args.directory, args.step_size, args.epoch_size, args.nepochs, args.start_after, args.cheat_factor, args.half, args.minibatch_size, args.base_size)