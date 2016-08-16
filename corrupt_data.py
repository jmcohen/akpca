#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

""" Create a "corrupted" version of a dataset by adding noise to the data """

import numpy as np
from datasets import get_dataset, get_dataset_names
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Corrupt data')
	parser.add_argument('dataset', choices=get_dataset_names())
	parser.add_argument('half', choices=['full', 'first', 'second'])
	args = parser.parse_args()

	X = get_dataset(args.dataset, half=args.half)

	n, d = X.shape

	gaussian_noise = np.random.randn(n, d)
	speckle_noise = np.random.rand(n, d)

	for noise_sd in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
		X_noise = X + gaussian_noise * noise_sd
		np.save('/fastscratch/jmcohen/corrupt/%s_%s_%s_%s.npy' % (args.dataset, args.half, noise_sd, 0.0), X_noise)

	for prob_zero in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
		X_noise = np.multiply(X, speckle_noise > prob_zero)
		np.save('/fastscratch/jmcohen/corrupt/%s_%s_%s_%s.npy' % (args.dataset, args.half, 0.0, prob_zero), X_noise)


