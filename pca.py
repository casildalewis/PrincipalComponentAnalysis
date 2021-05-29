from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):

    dataCent = np.load(filename)
    dataCent = dataCent - np.mean(dataCent, axis=0)

    return dataCent

def get_covariance(dataset):
    
    #S = np.cov(np.transpose(dataset))
    S = np.dot(np.transpose(dataset), dataset)
    S = np.divide(S, len(dataset) - 1)

    return S

def get_eig(S, m):
    
    n = len(S)
    values, vectors = eigh(S, subset_by_index = [n-m, n-1])

    values = np.diagflat(np.flip(values))
    vectors = np.flip(vectors, 1)

    return values, vectors

def get_eig_perc(S, perc):

    variance = np.sum(eigh(S, eigvals_only = True))
    variance *= perc

    values, vectors = eigh(S, subset_by_value = [variance, np.inf])

    values = np.diagflat(np.flip(values))
    vectors = np.flip(vectors, 1)

    return values, vectors


def project_image(img, U):

    alpha = np.dot(np.transpose(U), img)
    xPro = np.dot(U, alpha)

    return xPro


def display_image(orig, proj):

    orig = np.reshape(orig, (32,32), order = 'F')
    proj = np.reshape(proj, (32,32), order = 'F')
    
    fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols= 2)

    ax1.set_title('Original')
    ax2.set_title('Projection')

    col1 = ax1.imshow(orig)
    col2 = ax2.imshow(proj)

    fig.colorbar(col1, ax=ax1, fraction = 0.046, pad = 0.04)
    fig.colorbar(col2, ax=ax2, fraction = 0.046, pad = 0.04)

    plt.subplots_adjust(wspace = 0.5)
    plt.show()

