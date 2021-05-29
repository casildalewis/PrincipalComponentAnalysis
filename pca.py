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

def main():
    # x = load_and_center_dataset('YaleB_32x32.npy')
    # print(x)

    # print(len(x))
    # print(len(x[0]))
    # print(np.average(x))

    # print(x)

    # S = get_covariance(x)
    
    # print(S)
    # print(len(S))
    # print(len(S[0]))

    # Lambda, U = get_eig(S, 2)

    # Lambda, U = get_eig_perc(S, 0.07)

    # print(Lambda)
    # print(U)

    # projection = project_image(x[100], U)
    # print(projection)

    # display_image(x[100], projection)

    data = np.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 3], [2, 1, 3]])
    # print(data)
    mean = np.mean(data, axis=0)
    # print(mean)
    dataCent = data - mean
    # print(dataCent)
    S = np.cov(np.transpose(dataCent))
    # print(S)
    values, vectors = eigh(S)
    print(values)
    print(vectors)





if __name__=="__main__": 
    main() 

