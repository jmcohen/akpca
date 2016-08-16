""" Draw a dataset, optionally with (Gaussian or speckle) noise. """

import numpy as np 
import argparse
import matplotlib.pyplot as plt
from numpy.linalg import norm
from datasets import get_dataset, get_dataset_names
import os
from math import sqrt
from models import reconstruct_pca_centered, reconstruct_kpca, reconstruct_pesd, reconstruct_pesd_ds
from common import get_best_iter

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Reconstruct denoised digits')
	parser.add_argument('dataset', choices=get_dataset_names())
	parser.add_argument('half', choices=['full', 'first', 'second'])
	parser.add_argument('outdir', type=str)
	parser.add_argument('--nimages', type=int, default=50)
	parser.add_argument('--noise_sd', type=float, default=0.0)
	parser.add_argument('--prob_zero', type=float, default=0.0)
	args = parser.parse_args()

	X_noise = np.load('/fastscratch/jmcohen/corrupt/%s_%s_%s_%s.npy' % (args.dataset, args.half, args.noise_sd, args.prob_zero))

	n, d = X_noise.shape

	if not os.path.exists(args.outdir):
		os.mkdir(args.outdir)

	for i in range(args.nimages):
		ind = (i / 10) + 6700 * (i % 10)

		print "MNIST image %d" % ind

		side_length = int(sqrt(d))
		plt.figure()
		image = X_noise[ind,:].reshape((side_length, side_length))
		plt.imshow(image, cmap=plt.cm.gray)
		plt.xticks(())
		plt.yticks(())
		# plt.savefig('%s/%d_%s_%s.png' % (args.outdir, ind, args.noise_sd, args.prob_zero), bbox_inches='tight')
		plt.savefig('%s/digit_%d_prob_zero_%d.png' % (args.outdir, ind, args.prob_zero * 10), bbox_inches='tight')
		plt.clf()

