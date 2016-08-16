""" Visualize the denoising of a noisy MNIST digit across a range of model orders. """

import numpy as np 
import argparse
import matplotlib.pyplot as plt
from numpy.linalg import norm
from datasets import get_dataset
import os
from math import sqrt
from models import reconstruct_pca_centered, reconstruct_kpca, reconstruct_pesd, reconstruct_pesd_ds
from common import get_best_iter


def draw(image):
	side_length = int(sqrt(image.size))
	image_shape = (side_length, side_length)
	plt.imshow(image.reshape((image_shape)), cmap=plt.cm.gray, interpolation='nearest')
	plt.xticks(())
	plt.yticks(())


if __name__ == '__main__':
	ks = [5, 10, 15, 20, 25]
	# PCA = '/jukebox/norman/jmcohen/pca/mnist2/pca_k_%d_first'
	# KPCA = '/jukebox/norman/jmcohen/kpca/mnist2/kpca_k_%d_first'
	# PESD = '/fastscratch/jmcohen/pesd_svrg/mnist2/svrg_k_%d_step_0.02_epoch_1.0_False'

	PCA = '/jukebox/norman/jmcohen/pca_centered/mnist_maxscaled_short/pca_k_%d_full'
	KPCA = '/jukebox/norman/jmcohen/kpca/mnist_maxscaled_short/mykpca_k_%d_full'
	PESD = '/fastscratch/jmcohen/pesd_svrg/mnist_maxscaled_short/svrg_k_%d_step_0.1_epoch_1.0_full'
	DS_PESD = '/fastscratch/jmcohen/kernel_pesd/mnist_maxscaled_short/sgd_k_%d_step_0.06_epoch_1.0_full'

	train_dataset = 'mnist_maxscaled_short'
	train_half = 'full'

	test_dataset = 'mnist_maxscaled_short_test'
	test_half = 'full'

	parser = argparse.ArgumentParser(description='Reconstruct denoised digits')
	parser.add_argument('outdir', type=str)
	parser.add_argument('--prob_zero', type=float, default=0.0)
	parser.add_argument('--noise_sd', type=float, default=0.0)
	parser.add_argument('--nimages', type=int, default=50)
	args = parser.parse_args()

	X = get_dataset(test_dataset, half=test_half)
	X_noise = np.load('/fastscratch/jmcohen/corrupt/%s_%s_%s_%s.npy' % (test_dataset, test_half, args.noise_sd, args.prob_zero))

	n, d = X.shape

	Xtrain = get_dataset(train_dataset, half=train_half)

	if not os.path.exists(args.outdir):
		os.mkdir(args.outdir)

	for i in range(args.nimages):
		ind = (i / 10) + 6700 * (i % 10)

		print "MNIST image %d" % ind

		x = X[ind,:]
		x_noise = X_noise[ind,:].reshape((d, 1)).T

		ncol = 4
		nrow = 7

		side_length = int(sqrt(x.size))
		plt.figure(figsize=(2. * ncol, 2.26 * nrow))
		plt.suptitle('Image %d' % ind, fontsize=25)

		for i in range(4):
			plt.subplot(nrow, ncol, i+1)
			draw(x)
			plt.subplot(nrow, ncol, i+5)
			draw(x_noise)

		for ik, k in enumerate(ks):
			pca_centered_recon = reconstruct_pca_centered(Xtrain, x_noise, PCA % k, 0)
			kpca_recon = reconstruct_kpca(Xtrain, x_noise, KPCA % k, 0)
			pesd_recon = reconstruct_pesd(x_noise, PESD % k, get_best_iter(PESD % k))
			ds_pesd_recon = reconstruct_pesd_ds(Xtrain, x_noise, DS_PESD % k, get_best_iter(DS_PESD % k))

			plt.subplot(nrow, ncol, 9 + ik*4)
			draw(pca_centered_recon)

			plt.subplot(nrow, ncol, 10 + ik*4)
			draw(kpca_recon)

			plt.subplot(nrow, ncol, 11 + ik*4)
			draw(pesd_recon)

			plt.subplot(nrow, ncol, 12 + ik*4)
			draw(ds_pesd_recon)

		plt.savefig('%s/%s_%s_%d.png' % (args.outdir, args.noise_sd, args.prob_zero, ind))


