""" common functions """

from glob import glob
import re
import numpy as np
from math import sqrt
from scipy.sparse.linalg import eigs, svds, LinearOperator
from numpy.linalg import norm, svd
from datetime import datetime
from scipy.spatial.distance import cdist

def gaussian_kernel(X, Y, sigma2):
    """ Computes the Gaussian kernel function k(x, y) = exp( -||x - y||^2 / (2 sigma^2) ) 
    between each (x, y) pair

    Parameters
    ----------
    X : ndarray, shape (n1, d)
    Y : ndarray, shape (n2, d)

    Returns
    -------
    K : ndarray, shape (n1, n2)

    """
    dist = cdist(X, Y, 'sqeuclidean')
    return np.exp(-dist / (2*sigma2))


def compute_eigenvector(X, z):
    """ Computes the leading eigenvector and eigenvalue of

    \sum_i z[i] X[i,:] X[i,:]'

    using the Lanczos method


    Parameters
    ----------
    X : ndarray, shape (n, d)
    z : ndarray, shape (n, 1)

    Returns
    -------
    v : ndarray, shape (d, 1)
        leading eigenvector
    w : float
        leading eigenvalue
    """

    # product of M and a vector v
    def matvec(v):
        v = v.squeeze()
        return X.T.dot(z * X.dot(v))

    # # product of M' and a vector v
    def rmatvec(v):
        return matvec(v) # the operator is symmetric

    n, d = X.shape

    M = LinearOperator((d, d), matvec=matvec, rmatvec=rmatvec)

    W, V = eigs(M, k=1, tol=1e-3)

    return np.real(V[:,0]), W[0]

def poly2_preimage(Y):
    """ Given a point Y in R^(d^2), finds the point x in R^d which minimizes || x^(\otimes 2) - Y ||

    The solution is the leading eigenvector of Y times the leading eigenvalue of Y.

    Parameters
    ----------
    Y : ndarray, shape (d, d)

    Returns
    -------
    x : ndarray, shape (d, 1)

    """

    w, V = eigs(0.5 * (Y + Y.T), k=1, tol=1e-4, maxiter=10000)
    eigenvalue = w[0]
    eigenvector = V[:,0]
    if eigenvalue > 0:
        return eigenvector * sqrt(eigenvalue)
    else:
        return eigenvector * 0 # if an eigenvalue is negative, just return the zero vector

def poly2_preimage_kernel(X, z):
    """ Given a point Y in R^(d^2), finds the point x in R^d which minimizes || x^(\otimes 2) - Y ||

    Y is represented as \sum_i z[i] X[i,:] X[i,:]'

    The solution is the leading eigenvector of Y times the square root of the leading eigenvalue of Y.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    z : ndarray, shape (n, 1)

    Returns
    -------
    x : ndarray, shape (d, 1)
        preimage of  \sum_i z[i] X[i,:] X[i,:]'
    """

    eigenvector, eigenvalue = compute_eigenvector(X, z)
    if eigenvalue > 0:
        return eigenvector * sqrt(eigenvalue)
    else:
        return eigenvector * 0 # if an eigenvalue is negative, just return the zero vector

def match_sign(x, y):
    """ Figure out whether x or (-x) is closest to y

    Parameters
    ----------
    x : ndarray, shape (d,)
    y : ndarray, shape (d,)

    Returns
    -------
    xhat : ndarray, shape (d,)
    distance : float

    """
    pos_norm = norm(x - y, 2) ** 2
    neg_norm = norm(x + y, 2) ** 2
    if pos_norm < neg_norm:
        return x, pos_norm
    else:
        return -x, neg_norm

def unsigned_distance(x, y):
    """ min( || x - y ||^2, ||x + y||^2  ) 

    Parameters
    ----------
    x : ndarray, shape (d,)
    y : ndarray, shape (d,)

    Returns
    -------
    distance : float

    """ 
    _, dist = match_sign(x, y)
    return dist


def poly_kernel(X, Y, d):
    """ Computes the d-degree polynomial kernel function between two sets of points.

    The kernel function is f(x, y) = (x dot y)^2

    Parameters
    ----------
    X : ndarray, shape (n, d)
        data points are rows
    Y : ndarray, shape (m, d)
        data points are rows
    d : int
        degree of the polynomial kernel

    Returns
    -------
    K : ndarray, shape (n, m)
        the kernel matrix

    """
    return (X.dot(Y.T)) ** d

def poly2_kernel(X, Y):
    """ Computes the 2-degree polynomial kernel function between two sets of points.

    The kernel function is f(x, y) = (x dot y)^2

    Parameters
    ----------
    X : ndarray, shape (n, d)
        data points are rows
    Y : ndarray, shape (m, d)
        data points are rows

    Returns
    -------
    K : ndarray, shape (n, m)
        the kernel matrix

    """
    return poly_kernel(X, Y, 2)

def spectral_norm(X):
    """ Computes the spectral norm of a matrix 

    Parameters
    ----------
    X : ndarray, shape (n, d)

    Returns
    -------
    float

    """
    S = svds(X, k=1, return_singular_vectors=False, tol=1e-4, maxiter=10000)
    return S[0]

def pad(X):
    """ Reshapes a matrix into a long vector

    Parameters
    ----------
    X : ndarray, shape (m, n)

    Returns
    -------
    x : ndarray, shape (m*n, 1)

    """
    n = X.size
    return X.reshape((n, 1))

def tensor_product(x):
    """ Computes the degree-2 polynomial expansion of a vector

    Parameters
    ----------
    x : ndarray, shape (d, 1)

    Returns
    -------
    x2 : ndarray, shape (d^2, 1)

    """
    n = x.size
    x = x.reshape((n, 1))
    return x.dot(x.T).reshape((n ** 2,))

def expand_poly(X):
    """ Computes the degree-2 polynomial expansion of every row in a matrix 

    Parameters
    ----------
    X : ndarray, shape (n, d)

    Returns
    -------
    X2 : ndarray, shape (n, d^2)

    """
    m, n = X.shape
    n2 = n ** 2
    X2 = np.zeros((m, n2))
    for i in range(m):
        X2[i,:] = tensor_product(X[i,:])
    return X2

def to_matrix(x):
    """ Reshapes an (d^2 x 1) vector into an (d x d) matrix 

    Parameters
    ----------
    x : ndarray, shape (d^2, 1)

    Returns
    -------
    X : ndarray, shape (d, d)

    """
    n = int(sqrt(x.shape[0]))
    return x.reshape((n, n))


def get_max_iter(directory):
	""" Get the last iteration 

	Parameters
	----------
	directory : string
		the directory where the model is saved
	
	Returns
	-------
	max_iter : int
		the last iteration 

	"""
	iters = [int(re.match('.*iter([0-9]*)_recon\.txt', fname).group(1)) for fname in glob('%s/iter*_recon.txt' % directory)]
	return max(iters)

def get_best_iter(directory):
	""" Get the "best" iteration -- the one with the lowest reconstruction error

	Parameters
	----------
	directory : string
		the directory where the model is saved
	
	Returns
	-------
	best_iter : int
		the iteration with the lowest reconstruction error

	"""
	maxiter = get_max_iter(directory)
	recons = np.loadtxt('%s/iter%d_recon.txt' % (directory, maxiter))
	best = np.argsort(recons)[0]
	return best + 1 # to deal with the weirdness of when A is saved
