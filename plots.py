import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
import re
import os
from math import sqrt
from numpy.linalg import norm
from scipy.sparse.linalg import eigs
from common import compute_eigenvector, get_best_iter, get_max_iter
from datasets import get_dataset, get_dataset_names

def plot_trend(x, title, xlabel, ylabel, filename):
	plt.figure()
	plt.plot(x)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig(filename)

def save_gallery(title, images, n_col, n_row, filename):
	side_length = int(sqrt(images.shape[1]))
	image_shape = (side_length, side_length)
	plt.figure(figsize=(2. * n_col, 2.26 * n_row))
	plt.suptitle(title, size=16)
	for i, comp in enumerate(images):
		plt.subplot(n_row, n_col, i + 1)
		vmax = max(comp.max(), -comp.min())
		plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
				   interpolation='nearest',
				   vmin=-vmax, vmax=vmax)
		plt.xticks(())
		plt.yticks(())
	plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
	plt.savefig(filename)


def get_filters(directory, iter, algorithm, Xtrain=None):
	if algorithm == 'pesd' or algorithm == 'rankone':
		A = np.load('%s/iter%d_A.npy' % (directory, iter))
		k, n2 = A.shape # n is the number of pixels (so side_length ^ 2)
		n = int(sqrt(n2))
		filters = np.zeros((k, n))
		for i in range(k):
			Q = np.reshape(A[i,:], (n, n))
			Q = 0.5 * (Q + Q.T)
			filters[i,:] = np.real(eigs(Q, 1)[1][:,0])
	elif algorithm == 'kpca' or algorithm == 'ds_pesd':
		A = np.load('%s/iter%d_A.npy' % (directory, iter))
		k, n = A.shape
		d = Xtrain.shape[1]
		if algorithm == 'ds_pesd':
			Xtrain = Xtrain[:1000,:]
		filters = np.zeros((k, d))
		for i in range(k):
			filters[i,:] = np.real(compute_eigenvector(Xtrain, A[i,:])[0])
	elif algorithm == 'autoencoder' or algorithm == 'pca':
		A = np.load('%s/iter%d_A.npy' % (directory, iter))
		print A.shape
		k, n = A.shape
		filters = np.zeros((k, n))
		for i in range(k):
			filters[i,:] = A[i,:] / norm(A[i,:])

	return filters


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Plot loss and accuracy')
	parser.add_argument('directory', type=str)
	parser.add_argument('algorithm', choices=['autoencoder', 'pca', 'kpca', 'pesd', 'fw', 'rankone', 'ds_pesd'])
	parser.add_argument('--train_dataset', choices=get_dataset_names())
	args = parser.parse_args()

	for subdir in os.listdir(args.directory):
		try:
			print subdir
			directory = os.path.join(args.directory, subdir)

			title = subdir

			if args.train_dataset:
				Xtrain = get_dataset(args.train_dataset)
			else:
				Xtrain = None


			if args.algorithm in ['pca', 'kpca']:
				max_iter = 0
				best_iter = 0
			else:
				max_iter = get_max_iter(directory)
				best_iter = get_best_iter(directory)

			loss_file = '%s/iter%d_loss.txt' % (directory, max_iter)
			recon_file = '%s/iter%d_recon.txt' % (directory, max_iter)

			if os.path.exists(loss_file):
				loss = np.loadtxt(loss_file)
				plot_trend(loss, title, 'epoch', 'loss', '%s/loss.png' % directory)

			if os.path.exists(recon_file):
				recon = np.loadtxt(recon_file)
				plot_trend(recon, title, 'epoch', 'recon', '%s/recon.png' % directory)

			filters = get_filters(directory, best_iter, args.algorithm, Xtrain=Xtrain)
			save_gallery(title, filters, 5, filters.shape[0] / 5, '%s/filters.png' % directory)
			print '%s/filters.png' % directory
		except Exception:
			pass


