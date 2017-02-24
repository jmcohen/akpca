#!/usr/people/jmcohen/.conda/envs/pybench27/bin/python2.7 -u

import numpy as np 
from math import sqrt
from numpy.linalg import norm
from datetime import datetime
import argparse
import os
from datasets import load_dataset, get_dataset_names
from common import tensor_product, spectral_norm, poly2_preimage, unsigned_distance, to_matrix, pad
from scipy.sparse.linalg import svds, eigs, LinearOperator

REPORT_PROGRESS_EVERY = 1000
# REPORT_PROGRESS_EVERY = 1

def random_matrix(k, d):
    """ Return a (k x d2) matrix filled with random positive and negative values """
    A = np.random.rand(k, d) - 0.5
    A /= norm(A, 'fro')
    return A

def initialize_random(X, k, r):
    n, d = X.shape
    As = [random_matrix(r, d) for i in range(k)]
    Bs = [random_matrix(d, r) for i in range(k)]
    return As, Bs

def pca(X, k):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[0:k, :]

# def pesd_loss_approx(X, As, Bs, cheat_factor=1.0):
#     """ Approximates the objective function and reconstruction error (three ways)

#     Parameters
#     ----------
#     X : ndarray, shape (n, d)
#     A : ndarray, shape (k, d^2)
#         the current encoding matrix
#     B : ndarray, shape (d^2, k)
#         the current decoding matrix

#     Returns
#     -------
#     metrics: ndarray, shape (4, 0)
#         [loss, recon1, recon2, recon3]

#     """
#     n, d = X.shape
#     sample = np.random.choice(range(n), size=int(n / cheat_factor))
#     return pesd_loss_exact(X[sample, :], As, Bs)

# def pesd_loss_exact(X, As, Bs):
#     """ Computes the objective function and reconstruction error (three ways)

#     Parameters
#     ----------
#     X : ndarray, shape (n, d)
#     A : ndarray, shape (k, d^2)
#         the current encoding matrix
#     B : ndarray, shape (d^2, k)
#         the current decoding matrix

#     Returns
#     -------
#     metrics: ndarray, shape (4, 0)
#         [loss, recon1, recon2, recon3]

#     """
#     return sum([pesd_loss_term(x, As, Bs) for x in X]) / X.shape[0]

# def pesd_loss_term(x, As, Bs):
#     """ Computes the value of PESD objective function and the reconstruction error over one data point:

#     f(x) = ||B A x2 - x2||_2

#     Also computes the reconstruction error over this data point.

#     Parameters
#     ----------
#     x : ndarray, shape (d,)
#     A : ndarray, shape (k, d^2)
#         the current encoding matrix
#     B : ndarray, shape (d^2, k)
#         the current decoding matrix

#     Returns
#     -------
#     metrics: ndarray, shape (4,)
#         [loss, recon1, recon2, recon3]

#     """
#     k = len(As)
#     Axs = [As[i].dot(x) for i in range(k)]
#     Ax_norms = np.array([norm(Ax) ** 2 for Ax in Axs])
#     loss, u, v = compute_singular_vectors(x, Bs, Ax_norms)
#     eigenvalue, eigenvector = compute_eigenvector(Bs, Ax_norms)
#     recon = sqrt(eigenvalue) * eigenvector
#     recon_err = unsigned_distance(recon, x)
#     return np.array([loss, recon_err])

def compute_singular_vectors(x, y, Bs, D):
    """ computes the leading left and right singular vectors of the objective

    Returns
    -------
    s : float
        leading singular value
    u : ndarray, shape (d, 1)
        leading left singular vector
    v : ndarray, shape (d, 1)
        leading right singular vector

    """
    # product of M and a vector v
    def matvec(v):
        k = len(Bs)
        v = v.squeeze()

        # PCA residual
        z = x - D.dot(y)

        return sum([y[i] * Bs[i].dot(Bs[i].T.dot(v)) for i in range(k)]) - z * z.dot(v) 


    # product of M' and a vector v
    def rmatvec(v):
        return matvec(v) # the operator is symmetric

    d = x.size

    M = LinearOperator((d, d), matvec=matvec, rmatvec=rmatvec)
    U, S, Vt = svds(M, k=1, tol=1e-4, maxiter=10000)

    return S[0], U[:,0], Vt[0,:]


def compute_eigenvector(Bs, z):
    """ computes the eigenvector of

    \sum_i z[i] X[i,:] X[i,:]'

    Parameters
    ----------
    x : ndarray, shape (d, 1)
    X : ndarray, shape (n, d)
    z : ndarray, shape (n, 1)

    Returns
    -------
    s : float
        leading singular value
    u : ndarray, shape (d, 1)
        leading left singular vector
    v : ndarray, shape (d, 1)
        leading right singular vector

    """
    # product of M and a vector v
    def matvec(v):
        k = len(Bs)
        v = v.squeeze()
        return sum([z[i] * Bs[i].dot(Bs[i].T.dot(v)) for i in range(k)])


    # product of M' and a vector v
    def rmatvec(v):
        return matvec(v) # the operator is symmetric

    d = Bs[0].shape[0]

    Z = LinearOperator((d, d), matvec=matvec, rmatvec=rmatvec)
    w, v = eigs(Z, k=1, tol=1e-4, maxiter=10000)

    return w[0], v[:,0]

def check_gradient_A(x, As, Bs, C, D, i=0, j=0):
    eps = 1e-6
    As[0][i, j] += eps/2
    upper = pesd_loss(x, As, Bs, C, D)
    As[0][i, j] -= eps
    lower = pesd_loss(x, As, Bs, C, D)
    As[0][i, j] += eps/2
    return (upper - lower) / eps

def check_gradient_B(x, As, Bs, C, D, i=0, j=0):
    eps = 1e-6
    Bs[0][i, j] += eps/2
    upper = pesd_loss(x, As, Bs, C, D)
    Bs[0][i, j] -= eps
    lower = pesd_loss(x, As, Bs, C, D)
    Bs[0][i, j] += eps/2
    return (upper - lower) / eps


def pesd_loss(x, As, Bs, C, D):
    t0 = datetime.now()
    k = len(As)

    # y is the encoding
    y = C.dot(x)
    for i in range(k):
        y[i] += norm(As[i].dot(x)) ** 2
    # print("y: " + str(y))

    loss, u, v = compute_singular_vectors(x, y, Bs, D)
    return loss

def pesd_gradient(x, As, Bs, C, D):
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

    Returns
    -------
    grad_A : ndarray, shape (k, d^2)
        the derivative of the objective function wrt A
    grad_B : ndarray, shape (d^2, k)
        the derivative of the objective function wrt B
    metrics: ndarray, shape (4, 0)
        [loss, recon1, recon2, recon3]

    """
    t0 = datetime.now()
    k = len(As)

    # y is the encoding
    y = C.dot(x)
    for i in range(k):
        y[i] += norm(As[i].dot(x)) ** 2
    # print("y: " + str(y))

    loss, u, v = compute_singular_vectors(x, y, Bs, D)

    # pca residual
    z = x - D.dot(y)

    dloss_dy = np.zeros(k)
    for i in range(k):
        dloss_dy[i] = (Bs[i].T.dot(u).dot(Bs[i].T.dot(v))) + (u.dot(D[:,i]) * z.dot(v)) + (v.dot(D[:,i]) * z.dot(u))

    # print("dloss_dy: " + str(dloss_dy))

    gradC = np.outer(dloss_dy, x)
    gradD = (np.outer(u, v) + np.outer(v, u)).dot(np.outer(z, y))
    # tmp = np.outer(y, z)
    # gradD = np.outer(tmp.dot(u), v) + np.outer(tmp.dot(v), u)

    # print("C: " + str(norm(gradC)))
    # print("D: " + str(norm(gradD)))

    gradAs = []
    gradBs = []
    for i in range(k):
        gradAs.append(2* dloss_dy[i] * np.outer(As[i].dot(x), x) )
        gradBs.append( y[i] * (np.outer(u, Bs[i].T.dot(v)) + np.outer(v, Bs[i].T.dot(u))) )

    eigenvalue, eigenvector = compute_eigenvector(Bs, y)
    eigenvalue = max(eigenvalue, 0)
    recon1 = sqrt(eigenvalue) * eigenvector
    recon2 = D.dot(y)
    recon_err = min(norm(recon1 + recon2 - x) ** 2, norm(-recon1 + recon2 - x)  ** 2)
    # recon_err = 0

    # t1 = datetime.now()

    # metrics = np.array([loss])
    metrics = np.array([loss, recon_err])

    t2 = datetime.now()
    # print(t1 - t0, t2 - t1)
    return gradAs, gradBs, gradC, gradD, metrics

def sgd(X_train, X_test, k, r, save_dir, initial_step_size, minibatch_size, epoch_size, nepochs, start_after, cheat_factor, initialize):
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
    exp_dir = '%s/sgd_k_%s_r_%s_step_%s_epoch_%s_%s' % (save_dir, k, r, initial_step_size, epoch_size, initialize)
    if save_dir and not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    NUM_METRICS = 2

    n, d = X_train.shape

    if start_after == -1:
        As, Bs = initialize_pca(X_train, k, r) if initialize == 'pca' else initialize_random(X_train, k, r)
        all_metrics = np.zeros((nepochs, 2*NUM_METRICS))
    else:
        As = [np.load('%s/iter%d_A_%d.npy' % (exp_dir, start_after, i)) for i in range(k)]
        Bs = [np.load('%s/iter%d_B_%d.npy' % (exp_dir, start_after, i)) for i in range(k)]
        all_metrics = np.zeros((nepochs, 2*NUM_METRICS))
        all_metrics[:start_after, :] = np.loadtxt('%s/iter%d_metrics.txt' % (exp_dir, start_after))

    As = [random_matrix(r, d) for i in range(k)]
    Bs = [random_matrix(d, r) for i in range(k)]
    # As = [np.zeros((r, d)) for i in range(k)]
    # Bs = [np.zeros((d, r)) for i in range(k)]

    # C = np.zeros((k, d))
    # D = np.zeros((d, k))
    C = pca(X_train, k)
    D = C.T.copy()

    print(exp_dir)
    print("epoch\tloss\trecon err")


    num_steps_per_epoch = int(epoch_size * n)
    start_time = datetime.now()

    examples = range(n)

    for epoch in range(start_after + 1, nepochs):

        step_size = initial_step_size / (sqrt(epoch + 1))

        total_metrics = np.zeros(NUM_METRICS)

        for j in range(num_steps_per_epoch):
            index = np.random.choice(examples, size=(minibatch_size))[0]

            t1 = datetime.now()
            (gradAs, gradBs, gradC, gradD, metrics) = pesd_gradient(X_train[index,:], As, Bs, C, D)
            t2 = datetime.now()

            total_metrics += metrics

            for i in range(k):
                As[i] = As[i] - (step_size / minibatch_size) * gradAs[i]
                Bs[i] = Bs[i] - (step_size / minibatch_size) * gradBs[i]
                # C = C - (step_size / minibatch_size) * gradC
                # D = D - (step_size / minibatch_size) * gradD

            if j % REPORT_PROGRESS_EVERY == 0 and j > 0:
                average_metrics = total_metrics / REPORT_PROGRESS_EVERY
                total_metrics = np.zeros(NUM_METRICS)

                time_elapsed = datetime.now() - start_time

                print ("\t{}\t{}\t{}".format(epoch * num_steps_per_epoch + j, time_elapsed, "\t".join(map(str, list(average_metrics)))))

        # metrics_train = pesd_loss_approx(X_train, As, Bs, cheat_factor=cheat_factor)
        # metrics_test = pesd_loss_exact(X_test, As, Bs)
        # metrics = np.hstack((metrics_train, metrics_test))
        # all_metrics[epoch, :] = metrics
        # time_elapsed = datetime.now() - start_time
        # print ("{}\t{}\t{}".format(epoch, time_elapsed, "\t".join(map(str, list(metrics)))))

        if save_dir:
            for i in range(k):
                np.save('%s/iter%d_A_%d.npy' % (exp_dir, epoch, i), As[i])
                np.save('%s/iter%d_B_%d.npy' % (exp_dir, epoch, i), Bs[i])
                np.savetxt('%s/iter%s_metrics.txt' % (exp_dir, epoch), all_metrics[:epoch, :])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGD for PSED')
    parser.add_argument('dataset', type=str)
    parser.add_argument('directory', type=str, help="directory in which to save results")
    parser.add_argument('k', type=int, help="number of components to learn")
    parser.add_argument('rank', type=int, help="rank of each encoding / decoding matrix")
    parser.add_argument('step_size', type=float, help="initial step size")
    parser.add_argument('--epoch_size', type=float, default=1.0, help="epoch size, as a percentage of the dataset")
    parser.add_argument('--nepochs', type=int, default=10, help="run for this many epochs")
    parser.add_argument('--start_after', type=int, default=-1, help="continue a previous run, starting at this iteration")
    parser.add_argument('--cheat_factor', type=int, default=1, help="when computing the loss and reconstruction error, subsample by this factor")
    parser.add_argument('--minibatch_size', type=int, default=1, help="the size of each minibatch.  only used in SGD.")
    parser.add_argument('--initialize', type=str, choices=['random', 'pca'], default='random')
    args = parser.parse_args()

    X_train, X_test = load_dataset(args.dataset)

    sgd(X_train, X_test, args.k, args.rank, args.directory, args.step_size, args.minibatch_size, args.epoch_size, args.nepochs, args.start_after, args.cheat_factor, args.initialize)

