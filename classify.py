#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

""" cross-validated classification """

import numpy as np
from datasets import get_dataset, get_dataset_names
import argparse
from numpy.linalg import norm
from common import get_max_iter, get_best_iter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from models import encode
from sklearn import preprocessing

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Classify')
	parser.add_argument('directory', type=str)
	parser.add_argument('iter', type=int, help='which iter of optimization to load model from')
	parser.add_argument('dataset', choices=get_dataset_names())
	parser.add_argument('algorithm', choices=['pca', 'pca_centered', 'kpca', 'kpca_gaussian', 'kpca_poly', 'pesd', 'ds_pesd'])
	parser.add_argument('k', type=str, help='number of components in model')
	parser.add_argument('nexamples_per_class', type=int, help='number of examples per class')
	parser.add_argument('--encoding_train_dataset', help='dataset used to train the model.  required for algorithm="kpca" or algorithm="ds_pesd"')
	parser.add_argument('--encoding_train_half', choices=['full', 'first', 'second'], default='full', help="which half of the encoding train dataset was used")
	parser.add_argument('--output_file', type=str, help="file to output results to")
	parser.add_argument('--classifier', choices=['logistic_sag','logistic', 'svm', 'multinomial'], default='logistic_sag', help="which classifier to use")
	parser.add_argument('--C', type=float, default=1.0, help='if classifier=svm, the level of regularization')
	parser.add_argument('--scale', action='store_true', help='if true, preprocess the data by scaling it so that each dimension has unit variance over the dataset')
	parser.add_argument('--half', choices=['full', 'first', 'second'], default='full', help="which half of the dataset to use")
	parser.add_argument('--note', type=str)
	parser.add_argument('--degree', type=int, default=2, help='if algorithm=kpca_poly, then the degree of the polynomial')
	parser.add_argument('--sigma2', type=float, default=1.0, help='if algorithm=kpca_gaussian, then the parameter for the gaussian kernel')
	args = parser.parse_args()

	X, labels = get_dataset(args.dataset, return_labels=True, half=args.half)

	n, d = X.shape

	if args.iter == -1:
		iter = get_best_iter(args.directory)
	else:
		iter = args.iter

	if args.classifier == 'logistic':
		cla = LogisticRegression(penalty='l2', tol=1e-4, max_iter=50000)
	elif args.classifier == 'multinomial':
		cla = LogisticRegression(penalty='l2', tol=1e-4, solver='lbfgs', multi_class='multinomial')
	elif args.classifier == 'logistic_sag':
		cla = LogisticRegression(penalty='l2', solver='sag', tol=1e-4, max_iter=50000, multi_class='ovr')
	elif args.classifier == 'svm':
		cla = LinearSVC(C=args.C)

	if args.encoding_train_dataset != None:
		Xencode_train = get_dataset(args.encoding_train_dataset, half=args.encoding_train_half, return_labels=False)
	else:
		Xencode_train = None

	# encode the data using the model
	Y = encode(X, args.directory, iter, args.algorithm, Xencode_train=Xencode_train, degree=args.degree, sigma2=args.sigma2)

	if args.scale:
		Y = preprocessing.scale(Y)

	NCLASSES = 10
	nexamples = args.nexamples_per_class * NCLASSES
	n_total_splits = n / nexamples
	nsplits = min(n_total_splits, 100)

	scores = np.zeros(nsplits)

	for j in range(nsplits): 
		ind = np.arange(n)
		ind_train = (ind % n_total_splits == j)  # training fold
		ind_test = (ind % n_total_splits != j) # testing fold

		print (ind_train).sum(), " training examples"

		Y_train = Y[ind_train, :]
		labels_train = labels[ind_train]

		Y_test = Y[ind_test, :]
		labels_test = labels[ind_test]

		cla.fit(Y_train, labels_train) # train the classifier on the training fold
		scores[j] = cla.score(Y_test, labels_test) # evaluate the classifier on the testing fold

	score = scores.mean()

	if args.output_file: # write results to file
		f = open(args.output_file, 'a')
		f.write('%s\t%s\t%d\t%f\n' % ('%s_%s' % (args.algorithm, args.note) if args.note else args.algorithm, args.k, iter, score))
		f.close()
	else: # print results
		print score




