#!/usr/bin/env python3

import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from typing import List, Tuple


def get_cardinality(data_frame: pd.DataFrame=None) -> List[int]:
    '''
    Determine the cardinality for each nominal column of a dataframe

    :param pd.DataFrame data_frame: dataframe of nominal features
    :return: a vector of cardinalities for all columns of the dataframe
    :rtype: list
    :raises TypeError: if the data_frame is not a pandas dataframe
    '''

    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError('Expected an input of type pd.DataFrame')

    cardinality = []
    # 'Category' isn't a proper data type (that is, it's implemented
    # as a Python object rather than a low-level numpy array dtype),
    # so we need to coerce all of the dtypes to strings before
    # doing any comparison
    col_types = [str(col) for col in data_frame.dtypes]

    # If the column is nominal, we'll get the number of unique
    # elements. If it's ordinal, we set the cardinality to 0
    for col in range(len(col_types)):
        if col_types[col] in ['object', 'bool', 'category']:
            cardinality.append(len(data_frame.iloc[:, col].unique()))
        else:
            cardinality.append(0)

    return cardinality


def create_hamming_kernel(data_frame: pd.DataFrame=None,
                          lam: float=0.6) -> np.ndarray:
    '''
    Compute the  Hamming distance Kernel Matrix

    Based on Couto, J. (2005). Kernel k-means for categorical data.
    In Advances in Intelligent Data Analysis VI, pp. 46-56. Springer.

    :param pd.DataFrame data_frame: dataframe of nominal features
    :param lam float: the damping parameter in the range [0, 1]
    :return: the Hamming distance Kernel Matrix
    :rtype: np.ndarray
    '''

    cardinality = get_cardinality(data_frame=data_frame)

    # The hamming kernel is an N-by-N matrix where N
    # is equal to the number of observations in the dataset
    n = data_frame.shape[0]

    # Create an n-by-n all-ones matrix
    k_j = np.ones(shape=(n, n))

    # For each column of the data frame, compute the
    # recursive step to get the kernel
    for col in range(len(cardinality)):
        dhamming_j = np.ndarray(shape=(n, n))
        for row in range(n):
            # This is probably executing with O(n^2) runtime complexity
            # (we're comparing every element in a column of the input
            # dataset against every other element of that column). That's
            # very not good and could probably be improved???
            dhamming_j[row] = data_frame.iloc[:, col] != data_frame.iloc[row, col]

        k_j = (
            (lam**2) * (cardinality[col]-1-dhamming_j) +
            (2*lam-1) * dhamming_j + 1) * k_j

    # Normalize the kernel matrix
    d = 1/np.sqrt(np.outer(np.diag(k_j), np.diag(k_j)))
    k_j = k_j * d

    return k_j


def compute_sra(W: np.ndarray=None,
                X_i: float=None) -> Tuple(List[float], int, np.ndarray):
    '''
    Spectral Ranking for Abnormality (SRA)

    Based on K. Nian, H. Zhang, A. Tayal, T. F. Coelman,
    Y. Li, (2014). Auto Insurnace Fraud Detection Using
    Unsupervised Spectral Ranking for Anomaly.

    :param W np.ndarray: Hamming distance Kernel Matrix
    :param X_i float: Upper bound of the ratio of an anomaly
    :return f: A ranking vector representing degree of observation abnormality
    :rtype: list
    :return mFLAG: A flag indicating whether ranking is with respect to a
                   single global pattern or multiple major patterns
    :rtype: int
    :return z: ndarray of first two non-principal eigenvectors
    :rtype: np.ndarray
    '''
    n = W.shape[0]
    ident = np.zeros((n, n), float)
    np.fill_diagonal(ident, 1.)

    # Let D be the degree matrix of each vertex corresponding
    # the the row sum of the similarity matrix W
    d = np.sum(W, axis=1)
    d_sqrt = np.sqrt(d)

    # Let L be the symmetric normalized Laplacian
    L = ident - (1./d_sqrt) * np.matmul(W, np.diag(1./d_sqrt))

    # Extract the 2 non-principal eigenvectors.
    #
    # The eig() function returns 2 arrays: the
    # first represents eigenvalues, the second
    # eigenvectors. The eigenvectors are arranged
    # in descending order of magnitude. We sample
    # the second and third vectors here.
    npeigen = np.linalg.eig(L)[1][:, [1, 2]]

    # Components of the first non-principal
    # eigenvectors in the feature space.
    #
    # Note that here we're just duplicating
    # D_sqrt column-wise so that its dimensionality
    # is equal to that of npeigen
    z = np.transpose(np.array([d_sqrt, ]*2)) * npeigen

    # Let C_p be the positive class and and C_n
    # the negative class assigned based on the
    # sign of the 1st non-principal eigenvector
    # component of z
    C = np.sign(z[:, 0])
    C_cnt = np.unique(C, return_counts=True)
    C_cnt = pd.DataFrame(
        data=np.array(C_cnt[1])[np.newaxis],
        columns=C_cnt[0])

    if C_cnt.iloc[[0]].min(axis=1)[0]/n >= X_i:
        mFLAG = 1
        f = np.max(np.abs(z[:, 0])) - np.abs(z[:, 0])
    else:
        if C_cnt[1][0] > C_cnt[-1][0]:
            mFLAG = 0
            f = -z[:, 0]
        else:
            mFLAG = 0
            f = z[:, 0]

    return f, mFLAG, z


def plot_eigens(x=None, y=None, color=None):
    p = plt.scatter(
        x=x,
        y=y,
        c=color,
        cmap='seismic'
    )
    p = plt.colorbar()

    return p


def main():
    data = datasets.load_iris()

    data = pd.DataFrame(
        data=np.c_[data['data'], data['target']],
        columns=data['feature_names'] + ['target'])
    data['target'] = data['target'].apply(str)

    out = create_hamming_kernel(data_frame=data)
    f, _, z = compute_sra(W=out, X_i=0.4)

    plot_eigens(x=z[:, 0], y=z[:, 1], color=f)


if __name__ == '__main__':
    main()
