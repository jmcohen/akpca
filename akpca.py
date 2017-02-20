#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

import numpy as np 
from math import sqrt
from numpy.linalg import norm
from datetime import datetime
import argparse
import os
from datasets import get_dataset, get_dataset_names
from common import tensor_product, spectral_norm, poly2_preimage, unsigned_distance, to_matrix, pad
from scipy.sparse.linalg import svds

REPORT_PROGRESS_EVERY = 1000

def initialize(k, d2):
	""" Return a (k x d2) matrix filled with random positive and negative values """
	A = np.random.rand(k, d2) - 0.5
	A /= norm(A, 'fro')
	return A

def pesd_loss_approx(X, A, B, cheat_factor=1.0):
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
	sample = np.random.choice(range(n), size=(n / cheat_factor))
	loss, recon_err = pesd_loss_minibatch(sample, X, A, B)
	return loss * cheat_factor, recon_err * cheat_factor

def pesd_loss_minibatch(minibatch, X, A, B):
	""" Computes the objective function and reconstruction error over a minibatch.

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
	loss = 0
	recon_err = 0
	for i, x in enumerate(sample):
		l, r = pesd_loss_term(x, A, B)
		loss += l
		recon_err += r
	return loss, recon_err

def pesd_loss_term(x, A, B):
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

	# form x^(x2)
	x2 = tensor_product(x)

	z = B.dot(A.dot(x2))

	# the loss 
	loss = spectral_norm(to_matrix(z - x2))

	# compute the reconstruction of x
	recon = poly2_preimage(to_matrix(z))

	# compute the reconstruction error
	recon_err = unsigned_distance(x, recon)

	return loss, recon_err

def compute_gradient_approx(X, A, B, cheat_factor=4):
	""" Approximates the gradient of the PESD objective function wrt A and B by computing it over a minibatch of the dataset.

	Parameters
	----------
	X : ndarray, shape (n, d)
		the dataset
	A : ndarray, shape (k, d^2)
		the current encoding matrix
	B : ndarray, shape (d^2, k)
		the current decoding matrix

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
	sample = np.random.choice(range(n), size=(n / cheat_factor))
	grad_A, grad_B, loss, recon_err = pesd_minibatch_gradient(sample, X, A, B)
	return grad_A * cheat_factor, grad_B * cheat_factor, loss * cheat_factor, recon_err * cheat_factor

def pesd_gradient_term_full(x, A, B, compute_recon_err=True):
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

	# form x^(x2)
	x2 = tensor_product(x)

	z = B.dot(A.dot(x2))

	# compute the leading singular vectors
	U, S, Vt = svds(to_matrix(z - x2), k=1, tol=1e-4, maxiter=10000)

	# the loss 
	loss = S[0]

	# uv' reshaped into a (d^2 x 1) vector
	uv = pad(np.outer(U[:,0], Vt[0,:]))

	# (k x n^2) x (n^2 x 1) x (1 x n^2)
	grad_A = np.outer(B.T.dot(uv), x2)

	# (n^2 x 1) x (1 x n^2) x (n^2 x k)
	grad_B = np.outer(uv, A.dot(x2))

	if compute_recon_err:
		# compute the reconstruction of x
		recon = poly2_preimage(to_matrix(z))

		# compute the reconstruction error
		recon_err = unsigned_distance(x, recon)
	else:
		recon_err = 0

	return grad_A, grad_B, loss, recon_err

def pesd_minibatch_gradient(minibatch, X, A, B, compute_recon_err=True):
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
	gradA = np.zeros(A.shape)
	gradB = np.zeros(B.shape)
	loss = 0 
	recon_err = 0

	for i in minibatch:
		(gradA_term, gradB_term, loss_term, recon_err_term) = pesd_gradient_term_full(X[i,:], A, B, compute_recon_err=compute_recon_err)
		gradA += gradA_term
		gradB += gradB_term
		loss += loss_term
		recon_err += recon_err_term

	return (gradA, gradB, loss, recon_err)

def sgd(X, k, save_dir, initial_step_size, minibatch_size, epoch_size, nepochs, start_after, cheat_factor, half):
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
	exp_dir = '%s/sgd_k_%s_step_%s_epoch_%s_%s' % (save_dir, k, initial_step_size, epoch_size, half)
	if save_dir and not os.path.exists(exp_dir):
		os.mkdir(exp_dir)

	n, d = X.shape
	d2 = int(d**2)

	losses = np.zeros(nepochs)
	recon_errs = np.zeros(nepochs)

	if start_after == -1:
		A = initialize(k, d2)
		B = initialize(d2, k)
	else:
		A = np.load('%s/iter%d_A.npy' % (exp_dir, start_after))
		B = np.load('%s/iter%d_B.npy' % (exp_dir, start_after))
		losses[:start_after] = np.loadtxt('%s/iter%s_loss.txt' % (exp_dir, start_after))
		recon_errs[:start_after] = np.loadtxt('%s/iter%s_recon.txt' % (exp_dir, start_after))

	print(exp_dir)
	print("epoch\tloss\trecon err")

	num_steps_per_epoch = int(epoch_size * n)

	examples = range(n)

	for epoch in range(nepochs):

		# step size = initial_step_size / sqrt(t)
		step_size = initial_step_size / (sqrt(epoch + 1))

		# NOTE TO SELF: this should really be moved to the end of the iteration for it to make sense
		# save the model, loss, and reconstruction error
		if save_dir:
			np.save('%s/iter%d_A.npy' % (exp_dir, epoch), A)
			np.save('%s/iter%d_B.npy' % (exp_dir, epoch), B)
			np.savetxt('%s/iter%s_loss.txt' % (exp_dir, epoch), losses[:epoch])
			np.savetxt('%s/iter%s_recon.txt' % (exp_dir, epoch), recon_errs[:epoch])

		total_loss = 0
		for j in range(num_steps_per_epoch):
			minibatch = np.random.choice(examples, size=(minibatch_size))

			t1 = datetime.now()
			(gradA, gradB, loss, recon_err) = pesd_minibatch_gradient(minibatch, X, A, B, compute_recon_err=False)
			t2 = datetime.now()
			total_loss += loss

			A = A - (step_size / minibatch_size) * gradA
			B = B - (step_size / minibatch_size) * gradB

			if j % REPORT_PROGRESS_EVERY == 0:
				average_loss = total_loss / REPORT_PROGRESS_EVERY
				total_loss = 0
				print ("\t{}\t{}".format(epoch * num_steps_per_epoch + j, average_loss))

		loss, recon_err = pesd_loss_approx(X, A, B, cheat_factor=cheat_factor)
		print("%f\t%f\t%f" % (epoch, loss / n, recon_err / n))

		losses[epoch] = loss
		recon_errs[epoch] = recon_err

	return A, B

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='SGD / SVRG for PSED')
	parser.add_argument('dataset', type=str)
	parser.add_argument('directory', type=str, help="directory in which to save results")
	parser.add_argument('k', type=int, help="number of components to learn")
	parser.add_argument('step_size', type=float, help="initial step size")
	parser.add_argument('--half', choices=['full', 'first', 'second'], default='full', help="which half of the dataset to use")
	parser.add_argument('--epoch_size', type=float, default=1.0, help="epoch size, as a percentage of the dataset")
	parser.add_argument('--nepochs', type=int, default=10, help="run for this many epochs")
	parser.add_argument('--start_after', type=int, default=-1, help="continue a previous run, starting at this iteration")
	parser.add_argument('--abridged', action='store_true', help="'abridge' the dataset by taking only the first 1000 data points -- useful for debugging on a machine with limited RAM")
	parser.add_argument('--cheat_factor', type=int, default=1, help="when computing the loss and reconstruction error, subsample by this factor")
	parser.add_argument('--minibatch_size', type=int, default=1, help="the size of each minibatch.  only used in SGD.")
	args = parser.parse_args()

	if args.dataset in get_dataset_names():
		X = get_dataset(args.dataset, half=args.half)
	else:
		X = np.load(args.dataset)

	if args.abridged:
		X = X[0:1000]

	sgd(X, args.k, args.directory, args.step_size, args.minibatch_size, args.epoch_size, args.nepochs, args.start_after, args.cheat_factor, args.half)


