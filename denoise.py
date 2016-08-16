#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

""" Denoise data using PCA / KPCA / AKPCA / DS-AKPCA

	We add two types of noise: Gaussian and "speckle"

	Gaussian noise:  x <- x + eps, eps ~ N(0, sigma^2)

	Speckle noise:   x <- x with prob 1 - p
	                      0 with prob p

	The noisy data is created by running `corrupt.py` and saved in /fastscratch/jmcohen/corrupt/[DATASET_NAME]_full_[NOISE_SD]_[PROB_ZERO].npy

	where NOISE_SD is the parameter of the Gaussian noise and PROB_ZERO is the parameter of the speckle noise.

"""

import numpy as np
from datasets import get_dataset, get_dataset_names
import argparse
from numpy.linalg import norm
from common import get_max_iter, get_best_iter
from models import reconstruct

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Denoising experiment')
	parser.add_argument('directory', type=str, help='directory where model is saved')
	parser.add_argument('iter', type=int, help='iteration of optimization to load model from.  if iter=-1, take the iteration with the lowest reconstruction error.')
	parser.add_argument('dataset', choices=get_dataset_names())
	parser.add_argument('algorithm', choices=['pca', 'pca_centered', 'kpca', 'pesd', 'ds_pesd'])
	parser.add_argument('k', type=int, help='number of components in model')
	parser.add_argument('--prob_zero', type=float, default=0.0, help='probability of setting value to zero')
	parser.add_argument('--noise_sd', type=float, default=0.0, help='standard deviation of gaussian noise')
	parser.add_argument('--train_dataset', choices=get_dataset_names(), help='dataset used in training the model')
	parser.add_argument('--train_dataset_half', choices=['full', 'first', 'second'], default='full', help="which half of the training dataset was used")
	parser.add_argument('--output_file', type=str, help='file to output results to')
	parser.add_argument('--thin_factor', type=int, default=1, help='only use one out of every thin_factor points in the datase')
	parser.add_argument('--half', choices=['full', 'first', 'second'], default='full', help="which half of the dataset to use")
	args = parser.parse_args()

	X = get_dataset(args.dataset, half=args.half)
	X_noise = np.load('/fastscratch/jmcohen/corrupt/%s_%s_%s_%s.npy' % (args.dataset, args.half, args.noise_sd, args.prob_zero))

	# if thin_factor > 1, thin the dataset by taking one out of every thin_factor points
	if args.thin_factor != 1:
		X = X[np.array(range(X.shape[0])) % args.thin_factor == 0, :]
		X_noise = X_noise[np.array(range(X_noise.shape[0])) % args.thin_factor == 0, :]

	n, d = X.shape

	# if iter == -1, get the iteration with the lowest reconstruction error	
	iter = args.iter if args.iter != -1 else get_best_iter(args.directory)

	Xtrain = get_dataset(args.train_dataset, half=args.train_dataset_half) if args.train_dataset else None

	reconstructions = reconstruct(X_noise, args.directory, iter, args.algorithm, Xtrain=Xtrain)
	avg_recon_err = norm(X - reconstructions, 'fro') ** 2 / float(n)

	if args.output_file:
		f = open(args.output_file, 'a')
		f.write('%s\t%d\t%s\t%s\t%d\t%f\n' % (args.algorithm, args.k, args.noise_sd, args.prob_zero, iter, avg_recon_err))
		f.close()
	else:
		print avg_recon_err




