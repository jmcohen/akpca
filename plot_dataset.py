import numpy as np 
import matplotlib.pyplot as plt
from datasets import get_dataset, get_dataset_names
import argparse
import os
from math import sqrt
import random

def corrupt(x, prob_zero, noise_sd):
	""" corrupt a vector in two possible ways -- randomly setting an entry to zero or adding gaussian noise """
	n = x.size
	if noise_sd != 0:
		x = x + np.random.randn(n) * noise_sd
	if prob_zero != 0:
		x = np.multiply(x, np.random.rand(n) > prob_zero)

	# if np.linalg.norm(x) > 0:
	# 	x /= np.linalg.norm(x)

	return x

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Plot dataset')
	parser.add_argument('dataset', type=str, choices=get_dataset_names())
	parser.add_argument('directory', type=str)
	parser.add_argument('--nexamples', type=int, default=300)
	parser.add_argument('--rgb', action='store_true')
	parser.add_argument('--prob_zero', type=float, default=0)
	parser.add_argument('--noise_sd', type=float, default=0)
	args = parser.parse_args()

	if not os.path.exists(args.directory):
		os.mkdir(args.directory)

	X = get_dataset(args.dataset, downsample=False)

	m, n = X.shape

	# corrupt
	for i in range(m):
		X[i,:] = corrupt(X[i,:], args.prob_zero, args.noise_sd)

	if args.rgb:
		side_length = int(sqrt(n/3))
	else:
		side_length = int(sqrt(n))

	plt.figure()

	examples = random.sample(range(m), args.nexamples)

	for i in examples:
		if args.rgb:
			image = np.zeros((side_length, side_length, 3))
			image[:,:,0] = X[i,0:1024].reshape((side_length, side_length))
			image[:,:,1] = X[i,1024:2048].reshape((side_length, side_length))
			image[:,:,2] = X[i,2048:].reshape((side_length, side_length))

			image = image / image.max()
			plt.imshow(image)
		else:
			image = X[i,:].reshape((side_length, side_length))
			plt.imshow(image, cmap=plt.cm.gray)
		plt.savefig('%s/%d.jpg' % (args.directory, i))
		plt.clf()
