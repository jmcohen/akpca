""" This file contains functions for encoding and reconstructing data according to saved models for 
	PCA, KPCA, AKPCA, and DS-AKPCA.
"""


import numpy as np
from common import poly2_preimage, poly2_preimage_kernel, poly2_kernel, poly_kernel, gaussian_kernel, match_sign, tensor_product, to_matrix, expand_poly

def reconstruct_pca(X, directory, iter):
	""" Reconstruct data according to a learned PCA model.

	xhat = A' A x

	Parameters
	----------
	X : ndarray, shape (n, d)
		data to reconstruct
	directory : string
		directory where the PCA model is located
	iter: int
		iter of optimization to load model from

	"""
	A = np.load('%s/iter%d_A.npy' % (directory, iter))
	return X.dot(A.T).dot(A)


def reconstruct_pca_centered(Xtrain, X, directory, iter):
	""" Reconstruct data according to a learned centered PCA model.

	xhat = A' A (x - x0) + x0    where x0 is the mean

	Parameters
	----------
	Xtrain : ndarray, shape (n, d)
		data that was used to learn the centered PCA model on
	X : ndarray, shape (n, d)
		data to reconstruct
	directory : string
		directory where the PCA model is located
	iter: int
		iter of optimization to load model from

	"""

	A = np.load('%s/iter%d_A.npy' % (directory, iter))

	# the mean of the data
	center = Xtrain.mean(0)

	# subtract the mean
	X_centered = X - center

	return (X_centered.dot(A.T).dot(A) + center)


def reconstruct_kpca_direct_explicit(Xtrain, Xtest, A, B):
	""" Reconstruct data according to a learned KPCA model.

	Form A Xtrain^2 explicitly 

	Parameters
	----------
	Xtrain : ndarray, shape (n, d)
		data that was used to learn the kernel PCA model
	Xtest: ndarray, shape (m, d)
		data to reconstruct
	A : ndarray, shape (k, n)
		encoding matrix
	B : ndarray, shape (n, k)
		decoding matrix

	"""

	m, d = Xtest.shape
	recon = np.zeros((m, d))
	k, n = A.shape
	d2 = d**2
	AX2 = np.zeros((k, d2)) # AX2
	X2tB = np.zeros((d2, k)) # (X2)' 

	block_size = 5000
	for i in range((n / block_size) + 1):
		ind = np.arange(i*block_size, min((i+1)*block_size, n))
		X2 = expand_poly(Xtrain[ind, :])
		AX2 += A[:,ind].dot(X2)
		X2tB += X2.T.dot(B[ind,:])

	for i, x in enumerate(Xtest):
		x2 = tensor_product(x)
		recon[i,:] = match_sign(poly2_preimage( X2tB.dot(AX2.dot(x2)).reshape((d, d)) ), x)[0]

	return recon


def reconstruct_kpca_direct(Xtrain, Xtest, A, B):
	""" Reconstruct data according to a learned KPCA model.

	Parameters
	----------
	Xtrain : ndarray, shape (n, d)
		data that was used to learn the kernel PCA model
	Xtest: ndarray, shape (m, d)
		data to reconstruct
	A : ndarray, shape (k, m)
		encoding matrix
	B : ndarray, shape (m, k)
		decoding matrix

	"""

	m, d = Xtest.shape
	recon = np.zeros((m, d))
	for i, x in enumerate(Xtest):
		k = poly2_kernel(Xtrain, x)
		recon[i,:] = match_sign(poly2_preimage_kernel(Xtrain, B.dot(A.dot(k))), x)[0]
	return recon

def reconstruct_kpca(Xtrain, Xtest, directory, iter):
	""" Reconstruct data according to a learned KPCA model.

	Parameters
	----------
	Xtrain : ndarray, shape (n, d)
		data that was used to learn the kernel PCA model
	Xtest: ndarray, shape (m, d)
		data to reconstruct
	directory : string
		directory where the KPCA model is located
	iter: int
		iter of optimization to load model from

	"""
	A = np.load('%s/iter%d_A.npy' % (directory, iter))
	n, d = Xtrain.shape
	if d < 300:
		return reconstruct_kpca_direct_explicit(Xtrain, Xtest, A, A.T)
	else:
		return reconstruct_kpca_direct(Xtrain, Xtest, A, A.T)

def reconstruct_pesd(X, directory, iter):
	""" Reconstruct data according to a learned PESD model.

	Parameters
	----------
	X: ndarray, shape (n, d)
		data to reconstruct
	directory : string
		directory where the PESD model is located
	iter: int
		iter of optimization to load model from


	"""
	A = np.load('%s/iter%d_A.npy' % (directory, iter))
	B = np.load('%s/iter%d_B.npy' % (directory, iter))
	n, d = X.shape
	recon = np.zeros((n, d))
	for i, x in enumerate(X):
		x2 = tensor_product(x)
		recon[i,:] = match_sign(poly2_preimage(to_matrix(B.dot(A.dot(x2)))), x)[0]
	return recon

def reconstruct_pesd_ds(Xtrain, Xtest, directory, iter):
	""" Reconstruct data according to a learned DS-PESD model.

	Parameters
	----------
	Xtrain : ndarray, shape (n, d)
		data that was used to learn the DS-PESD model
	Xtest: ndarray, shape (m, d)
		data to reconstruct
	directory : string
		directory where the DS-PESD model is located
	iter: int
		iter of optimization to load model from

	"""
	A = np.load('%s/iter%d_A.npy' % (directory, iter))
	B = np.load('%s/iter%d_B.npy' % (directory, iter))
	n, d = Xtrain.shape
	if d < 300:
		return reconstruct_kpca_direct_explicit(Xtrain, Xtest, A, B)
	else:
		return reconstruct_kpca_direct(Xtrain, Xtest, A, B)

def reconstruct(X, directory, iter, algorithm, Xtrain=None):
	""" Reconstruct data according to saved model

	Parameters
	----------
	Xtest: ndarray, shape (m, d)
		data to reconstruct
	directory : string
		directory where the model is located
	iter: int
		iter of optimization to load model from
	algorithm : 'pca' | 'pca_centered' | 'kpca' | 'pesd' | 'ds_pesd'
	Xtrain : ndarray, shape (n, d)  [optional]
		data used to train the model
		only necessary if algorithm = kpca or ds_pesd

	"""
	if algorithm == 'pca':
		return reconstruct_pca(X, directory, iter)
	if algorithm == 'pca_centered':
		return reconstruct_pca_centered(Xtrain, X, directory, iter)
	elif algorithm == 'kpca':
		return reconstruct_kpca(Xtrain, X, directory, iter)
	elif algorithm == 'pesd':
		return reconstruct_pesd(X, directory, iter)
	elif algorithm == 'ds_pesd':
		return reconstruct_pesd_ds(Xtrain, X, directory, iter)


def encode_pca(X, directory, iter):
	""" Encode data according to saved PCA model

	Parameters
	----------
	X : ndarray, shape (n, d)
		data to encode 
	directory : string
		directory where model is saved
	iter : int
		iteration of optimization to load model from
	
	Returns 
	-------
	Y : ndarray, shape (n, k)
		encoded data

	"""
	A = np.load('%s/iter%d_A.npy' % (directory, iter))
	return X.dot(A.T)

def encode_pca_centered(Xtrain, X, directory, iter):
	""" Encode data according to saved centered PCA model

	Parameters
	----------
	X : ndarray, shape (n, d)
		data to encode
	directory : string
		directory where model is saved
	iter : int
		iteration of optimization to load model from
	
	Returns 
	-------
	Y : ndarray, shape (n, k)
		encoded data

	"""
	A = np.load('%s/iter%d_A.npy' % (directory, iter))
	center = X.mean(0)
	X_centered = X - center
	return X_centered.dot(A.T)

def encode_kpca(Xtrain, X, directory, iter):
	""" Encode data according to saved kernel PCA model

	Parameters
	----------
	Xtrain : ndarray, shape (m, d)
		data that was used to train model 
	X : ndarray, shape (n, d)
		data to encode
	directory : string
		directory where model is saved
	iter : int
		iteration of optimization to load model from
	
	Returns 
	-------
	Y : ndarray, shape (n, k)
		encoded data

	"""
	A = np.load('%s/iter%d_A.npy' % (directory, iter))
	K = poly2_kernel(Xtrain, X)
	return A.dot(K).T

def encode_kpca_poly(Xtrain, X, directory, iter, degree):
	""" Encode data according to saved kernel PCA model

	Parameters
	----------
	Xtrain : ndarray, shape (m, d)
		data that was used to train model 
	X : ndarray, shape (n, d)
		data to encode
	directory : string
		directory where model is saved
	iter : int
		iteration of optimization to load model from
	degree : int
		degree of the polynomial kernel
	
	Returns 
	-------
	Y : ndarray, shape (n, k)
		encoded data

	"""
	A = np.load('%s/iter%d_A.npy' % (directory, iter))
	K = poly_kernel(Xtrain, X, degree)
	return A.dot(K).T

def encode_kpca_gaussian(Xtrain, X, directory, iter, sigma2):
	""" Encode data according to saved kernel PCA model

	Parameters
	----------
	Xtrain : ndarray, shape (m, d)
		data that was used to train model 
	X : ndarray, shape (n, d)
		data to encode
	directory : string
		directory where model is saved
	iter : int
		iteration of optimization to load model from
	sigma2 : float
		sigma^2 of the gaussian kernel

	Returns 
	-------
	Y : ndarray, shape (n, k)
		encoded data

	"""
	A = np.load('%s/iter%d_A.npy' % (directory, iter))
	K = gaussian_kernel(Xtrain, X, sigma2)
	return A.dot(K).T


def encode_pesd(X, directory, iter):
	""" Encode data according to saved PESD model

	Parameters
	----------
	X : ndarray, shape (n, d)
		data to encode
	directory : string
		directory where model is saved
	iter : int
		iteration of optimization to load model from
	
	Returns 
	-------
	Y : ndarray, shape (n, k)
		encoded data

	"""
	A = np.load('%s/iter%d_A.npy' % (directory, iter))
	n, p = X.shape
	k = A.shape[0]
	Y = np.zeros((n, k))
	for i, x in enumerate(X):
		Y[i,:] = A.dot(tensor_product(x))
	return Y

	# TOO MEMORY INTENSIVE:
	# X2 = expand_poly(X)
	# return X2.dot(A.T)

def encode_pesd_ds(Xtrain, X, directory, iter):
	""" Encode data according to saved DS-PESD model

	Parameters
	----------
	Xtrain : ndarray, shape (m, d)
		data that was used to train model 
	X : ndarray, shape (n, d)
		data to encode
	directory : string
		directory where model is saved
	iter : int
		iteration of optimization to load model from
	
	Returns 
	-------
	Y : ndarray, shape (n, k)
		encoded data

	"""
	A = np.load('%s/iter%d_A.npy' % (directory, iter))
	K = poly2_kernel(Xtrain, X)
	return A.dot(K).T

def encode(X, directory, iter, algorithm, Xencode_train=None, degree=2, sigma2=1.0):
	""" Encode data according to saved model

	Parameters
	---------- 
	X : ndarray, shape (n, d)
		data to encode
	directory : string
		directory where model is saved
	iter : int
		iteration of optimization to load model from
	algorithm : string
		which algorithm 
	Xencode_train : ndarray, shape (m, d)
		for algorithm='kpca' or algorithm='ds_pesd', the dataset
		that was used to train the model on
	
	Returns 
	-------
	Y : ndarray, shape (n, k)
		encoded data

	"""
	if algorithm == 'pca':
		return encode_pca(X, directory, iter)
	if algorithm == 'pca_centered':
		return encode_pca_centered(Xencode_train, X, directory, iter)
	elif algorithm == 'kpca':
		return encode_kpca(Xencode_train, X, directory, iter)
	elif algorithm == 'kpca_poly':
		return encode_kpca_poly(Xencode_train, X, directory, iter, degree)
	elif algorithm == 'kpca_gaussian':
		return encode_kpca_gaussian(Xencode_train, X, directory, iter, sigma2)
	elif algorithm == 'pesd':
		return encode_pesd(X, directory, iter)
	elif algorithm == 'ds_pesd':
		return encode_pesd_ds(Xencode_train, X, directory, iter)






