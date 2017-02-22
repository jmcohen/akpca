#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

import numpy as np 
from math import sqrt
from numpy.linalg import norm
from datetime import datetime
import argparse
import os
from datasets import get_dataset, get_dataset_names
from common import tensor_product, spectral_norm, poly2_preimage, unsigned_distance, to_matrix, pad
from scipy.sparse.linalg import svds, eigs

REPORT_PROGRESS_EVERY = 1000

def initialize(k, d):
	""" Return a (k x d2) matrix filled with random positive and negative values """
	A = np.random.rand(k, d) - 0.5
	A /= norm(A, 'fro')
	return A

def pesd_loss_approx(X, As, Bs, cheat_factor=1.0):
	""" Approximates the objective function and reconstruction error by computing it over a minibatch.

	Parameters
	----------
	X : ndarray, shape (n, d)
	A : ndarray, shape (k, d^2)
		the current encoding matrix
	B : ndarray, shape (d^2, k)
		the current decoding matrix

	Returns
	-------
	loss : float
		the value of the objective function
	recon_err : float
		the reconstruction error

	"""
	n, d = X.shape
	sample = np.random.choice(range(n), size=int(n / cheat_factor))
	return pesd_loss_exact(X[sample, :], As, Bs)

def pesd_loss_exact(X, As, Bs):
	return sum([pesd_loss_term(x, As, Bs) for x in X]) / X.shape[0]

def loss_helper(Z, x):
    w, V = eigs(0.5 * (Z + Z.T), k=1, tol=1e-4, maxiter=10000)
    eigenvalue = w[0]
    eigenvector = V[:,0]
    return [
    	unsigned_distance(x, eigenvector),
		unsigned_distance(x, eigenvector * sqrt(eigenvalue)),
		unsigned_distance(x, eigenvector * eigenvalue)
	]

def pesd_loss_term(x, As, Bs):
	""" Computes the value of PESD objective function and the reconstruction error over one data point:

	f(x) = ||B A x2 - x2||_2

	Also computes the reconstruction error over this data point.

	Parameters
	----------
	x : ndarray, shape (d,)
	A : ndarray, shape (k, d^2)
		the current encoding matrix
	B : ndarray, shape (d^2, k)
		the current decoding matrix

	Returns
	-------
	loss : float
		the value of the objective function
	recon_err : float
		the reconstruction error

	"""
	Z = sum([(norm(A.dot(x)) ** 2) * B.dot(B.T) for (A, B) in zip(As, Bs)])
	xxt = np.outer(x, x)
	loss = spectral_norm(Z - xxt)
	return np.array([loss] + loss_helper(Z, x))

def pesd_gradient_term_full(x, As, Bs):
	""" Computes the value and gradient of the PESD objective function wrt A and B over one data point:

	f(x) = ||B A x2 - x2||_2

	Also computes the reconstruction error over this data point.

	Parameters
	----------
	x : ndarray, shape (d,)
	A : ndarray, shape (k, d^2)
		the current encoding matrix
	B : ndarray, shape (d^2, k)
		the current decoding matrix
	compute_recon_err : bool
		whether to compute reconstruction error too

	Returns
	-------
	grad_A : ndarray, shape (k, d^2)
		the derivative of the objective function wrt A
	grad_B : ndarray, shape (d^2, k)
		the derivative of the objective function wrt B
	loss : float
		the value of the objective function
	recon_err : float
		the reconstruction error

	"""
	k = len(As)

	Z = sum([(norm(As[i].dot(x)) ** 2) * Bs[i].dot(Bs[i].T) for i in range(k)])
	xxt = np.outer(x, x)

	# compute the leading singular vectors
	U, S, Vt = svds(Z - xxt, k=1, tol=1e-4, maxiter=10000)
	u = U[:,0]
	v = Vt[0,:]

	uvt = np.outer(u, v)

	# the loss 
	loss = S[0]

	gradAs = []
	for (A, B) in zip(As, Bs):
		gradAs.append( u.dot(B.dot(B.T)).dot(v) * A.dot(xxt) )

	gradBs = []
	for (A, B) in zip(As, Bs):
		gradBs.append( (norm(A.dot(x)) ** 2) * (uvt + uvt.T).dot(B) )

	# compute the reconstruction of x
	# recon = poly2_preimage(Z)

	# compute the reconstruction error
	# recon_err = unsigned_distance(x, recon)

	metrics = np.array([loss] + loss_helper(Z, x))

	return gradAs, gradBs, metrics

def pesd_minibatch_gradient(minibatch, X, As, Bs):
	""" Computes the gradient of the PESD objective function wrt A and B over a minibatch of the dataset.

	Parameters
	----------
	minibatch : ndarray, shape (minibatch_size, 1) 
		the indices of the points in the dataset that belong to this minibatch
	X : ndarray, shape (n, d)
		the dataset
	A : ndarray, shape (k, d^2)
		the current encoding matrix
	B : ndarray, shape (d^2, k)
		the current decoding matrix
	compute_recon_err : bool
		whether to compute reconstruction error

	Returns
	-------
	grad_A : ndarray, shape (k, d^2)
		the derivative of the objective function wrt A
	grad_B : ndarray, shape (d^2, k)
		the derivative of the objective function wrt B
	loss : float
		the value of the objective function
	recon_err : float
		the reconstruction error

	"""
	n, d = X.shape
	gradAs = [np.zeros(A.shape) for A in As]
	gradBs = [np.zeros(B.shape) for B in Bs]
	k = len(As)

	loss = 0 
	recon_err = 0

	metrics = np.zeros(4)

	for index in minibatch:
		(gradA_terms, gradB_terms, metrics_term) = pesd_gradient_term_full(X[index,:], As, Bs)

		for i in range(k):
			gradAs[i] += gradA_terms[i]
			gradBs[i] += gradB_terms[i]

		metrics += metrics_term

	return (gradAs, gradBs, metrics)

def sgd(X_train, X_test, k, r, save_dir, initial_step_size, minibatch_size, epoch_size, nepochs, start_after, cheat_factor):
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
	initial_step_size : float
	minibatch_size : int
	epoch_size : float
		number of iterations in each epoch, as a percentage of the data set
	nepochs : int
		number of epochs to run SGD for
	start_after : int
		if -1, initialize A and B randomly and start from the beginning
		otherwise, load the A and B from this and start from there
	cheat_factor : int
		if cheat_factor > 1, when computing the loss and reconstruction error after every epoch, 
		subsample the dataset by this factor 
	half : string
		which half of the datset 
	
	"""

	# save results in exp_dir
	exp_dir = '%s/sgd_k_%s_r_%s_step_%s_epoch_%s' % (save_dir, k, r, initial_step_size, epoch_size)
	if save_dir and not os.path.exists(exp_dir):
		os.mkdir(exp_dir)

	NUM_METRICS = 8

	n, d = X_train.shape
	all_metrics = np.zeros((nepochs, NUM_METRICS))

	if start_after == -1:
		As = [initialize(r, d) for i in range(k)]
		Bs = [initialize(d, r) for i in range(k)]

	print(exp_dir)
	print("epoch\tloss\trecon err")

	num_steps_per_epoch = int(epoch_size * n)
	start_time = datetime.now()

	examples = range(n)

	for epoch in range(nepochs):

		step_size = initial_step_size / (sqrt(epoch + 1))

		total_metrics = np.zeros(4)

		for j in range(num_steps_per_epoch):
			minibatch = np.random.choice(examples, size=(minibatch_size))

			t1 = datetime.now()
			(gradAs, gradBs, metrics) = pesd_minibatch_gradient(minibatch, X_train, As, Bs)
			t2 = datetime.now()

			total_metrics += metrics

			for i in range(k):
				As[i] = As[i] - (step_size / minibatch_size) * gradAs[i]
				Bs[i] = Bs[i] - (step_size / minibatch_size) * gradBs[i]

			if j % REPORT_PROGRESS_EVERY == 0 and j > 0:
				average_metrics = metrics / REPORT_PROGRESS_EVERY

				time_elapsed = datetime.now() - start_time

				print ("\t{}\t{}\t{}".format(epoch * num_steps_per_epoch + j, time_elapsed, "\t".join(map(str, list(average_metrics)))))

		metrics_train = pesd_loss_approx(X_train, As, Bs, cheat_factor=cheat_factor)
		metrics_test = pesd_loss_exact(X_test, As, Bs)
		metrics = np.hstack((metrics_train, metrics_test))
		all_metrics[i, :] = metrics
		time_elapsed = datetime.now() - start_time
		print ("\t{}\t{}\t{}".format(epoch, time_elapsed, "\t".join(map(str, list(metrics)))))

		if save_dir:
			for i in range(k):
				np.save('%s/iter%d_A_%d.npy' % (exp_dir, epoch, i), As[i])
				np.save('%s/iter%d_B_%d.npy' % (exp_dir, epoch, i), Bs[i])
				np.savetxt('%s/iter%s_metrics.txt' % (exp_dir, epoch), all_metrics[:epoch, :])


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='SGD for PSED')
	parser.add_argument('dataset', type=str)
	parser.add_argument('directory', type=str, help="directory in which to save results")
	parser.add_argument('k', type=int, help="number of components to learn")
	parser.add_argument('rank', type=int, help="rank of each encoding / decoding matrix")
	parser.add_argument('step_size', type=float, help="initial step size")
	parser.add_argument('--epoch_size', type=float, default=1.0, help="epoch size, as a percentage of the dataset")
	parser.add_argument('--nepochs', type=int, default=10, help="run for this many epochs")
	parser.add_argument('--start_after', type=int, default=-1, help="continue a previous run, starting at this iteration")
	parser.add_argument('--cheat_factor', type=int, default=1, help="when computing the loss and reconstruction error, subsample by this factor")
	parser.add_argument('--minibatch_size', type=int, default=1, help="the size of each minibatch.  only used in SGD.")
	args = parser.parse_args()

	ind_train = np.load('train.npy')
	ind_test = np.load('test.npy')

	X = get_dataset(args.dataset, half='full')
	X_train = X[ind_train, :]
	X_test = X[ind_test, :]

	sgd(X_train, X_test, args.k, args.rank, args.directory, args.step_size, args.minibatch_size, args.epoch_size, args.nepochs, args.start_after, args.cheat_factor)


