import numpy as np
import numpy.linalg as la
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
################################################################################
def MixGauss(means, sigmas, n):
    """Generates a 2D dataset of two classes with gaussian noise
    
    usage: X, Y = MixGauss(means, sigmas, n)

    Arguments:
    means: (size [pxd]) and should be of the form [m1, ... ,mp] (each mi is
    d-dimensional)

    sigmas: (size px1) should be in the form [sigma_1, ..., sigma_p]

    n: number of points per class

    X: obtained input data matrix (size [p*nxd])
    Y: obtained output data vector (size [p*n])
    
    Returns:
    X: A [p*nxd] matrix with the coordinates of each point
    Y: A [n] array with the labels of each point, the label is an integer
        from the interval [0,p-1]

    EXAMPLE: X, Y = MixGauss([[0,1],[0,1]],[0.5, 0.25],1000)
        generates a 2D dataset with two classes, the first one centered on (0,0)
        with standard deviation 0.5, the second one centered on (1,1)
        with standard deviation 0.25.
        Each class will contain 1000 points.

    to visualize: plt.scatter(X[:,1],X[:,2],s=25,c=Y)"""
    
    means = np.array(means)
    sigmas = np.array(sigmas)

    d = means.shape[1]
    num_classes = sigmas.size
    data = np.full((n*num_classes, d), np.inf)
    labels = np.zeros(n*num_classes)

    for idx,sigma in enumerate(sigmas):
        data[idx*n:(idx+1)*n] = np.random.multivariate_normal(mean=means[idx], cov=np.eye(d)*sigmas[idx]**2, size=n)
        labels[idx*n:(idx+1)*n] = idx
    
    return data, labels
################################################################################
def flipLabels(Y, perc):
    """Flips randomly selected labels of a binary classification problem with labels +1,-1
    
    Arguments:
    Y: array of labels
    perc: percentage of labels to be flipped
    
    Returns:
    Y: array with flipped labels
    """
    if perc < 1 or perc > 100:
        print("p should be a percentage value between 0 and 100.")
        return -1
        
    if any(np.abs(Y) != 1):
        print("The values of Ytr should be +1 or -1.")
        return -1
    
    Y_noisy = np.copy(np.squeeze(Y))
    if Y_noisy.ndim > 1:
        print("Please supply a label array with only one dimension")
        return -1
    
    n = Y_noisy.size
    n_flips = int(np.floor(n * perc / 100))
    print("n_flips:",n_flips)
    idx_to_flip = np.random.choice(n, size=n_flips, replace=False)
    Y_noisy[idx_to_flip] = -Y_noisy[idx_to_flip] 
    
    return Y_noisy
################################################################################
def two_moons(npoints, pflip):
    mat_contents = sio.loadmat('./datasets/moons_dataset.mat')
    Xtr = mat_contents['Xtr']
    Ytr = mat_contents['Ytr']
    Xts = mat_contents['Xts']
    Yts = mat_contents['Yts']
    npoints = min([100, npoints])
    i = np.random.permutation(100)
    sel = i[0:npoints]
    Xtr = Xtr[sel, :]
    if pflip > 1:
        Ytrn = flipLabels(Ytr[sel], pflip)
        Ytsn = flipLabels(Yts, pflip)
    else:
        Ytrn = np.squeeze(Ytr[sel])
        Ytsn = np.squeeze(Yts)
    return Xtr, Ytrn, Xts, Ytsn
################################################################################"
def sqDist(X1, X2):
    """Computes all the distances between two set of points stored in two matrices
    
    usage: D = sqDist(X1, X2)
    
    Arguments:
    X1: a matrix of size [n1xd], where each row is a d-dimensional point
    
    X2: a matrix of size [n2xd], where each row is a d-dimensional point
    
    Returns:
    D: a [n1xn2] matrix where each element (D)_ij is the distance between points (X_i, X_j)
    """
    sqx = np.sum(np.multiply(X1,X1), 1)
    sqy = np.sum(np.multiply(X2,X2), 1)
    return np.outer(sqx, np.ones(sqy.shape[0])) + np.outer(np.ones(sqx.shape[0]), sqy.T) - 2 * np.dot(X1,X2.T)
################################################################################
def calcErr(T, Y):
    """Computes the average error given a true set of labels and computed labels
    
    usage: error = calcErr(T, Y)
    
    T: True labels of the test set
    Y: labels computed by the user, must belong to {-1,+1}
    """
    err = np.mean(np.sign(T)!=np.sign(Y))
    return err
################################################################################
def LLoyd(X, centers, maxiter):
    
    centers = np.array(centers)
    k = centers.shape[0]   # number of centroids
    n = X.shape[0]         # number of points
    d = X.shape[1]         # space dimensionality
    
    # vector for storing cluster assignments
    idx_prev = np.zeros(n)
    
    for idx_iter in range(maxiter):
        # Computes distances between centroids and point, find nearest centroid for each point
        distances = sqDist(centers, X)
        c_idx = np.argmin(distances, axis=0)
        
        #Update cluster centroids
        for idx_centroid in range(k):
            if not (c_idx == idx_centroid).any():    # Checks the number of points assigned to this cluster
                print("Cluster #", idx_centroid, "is empty")
            else:                                    # Updates centroids
                centers[idx_centroid] = np.mean( X[c_idx == idx_centroid])
        #Check for convergence
        if ( np.sum(c_idx - idx_prev) == 0):
            print("LLoyd's algorithm: convergence reached")
            break
        idx_prev = c_idx
       
    return idx_prev, centers

      
################################################################################
def PCA(X, k):
    """Computes the first k eigenvectors, eigenvalues and projections of the 
    data matrix X
    usage: V, d, X_proj = PCA(X, k)

    X: is the dataset
    k: is the number of components
     
    V:      is a matrix of the form [v_1, ..., v_k] where v_i is the i-th
    eigenvector
    d:      is the list of the first k eigenvalues
    X_proj: is the projection of X on the linear space spanned by the
    eigenvectors in V"""
    
    mean= X.mean(axis=0)
    X_z = X - mean
    cov_mat = X_z.T @ X_z   
    U, d, V = la.svd(cov_mat)
    X_proj = X_z @ V[:,:k]
    
    return V, d, X_proj 
################################################################################

def normLaplacian(X, sigma=0.1):
    """Generates the normalized Laplacian matrix of a dataset X using the Gaussian kernel
    
    usage: L = NormLaplacian(X=X, sigma=0.1)
    
    X:       the dataset
    sigma:   the bandwidth of the Gaussian kernel
    
    L:       the normalized Laplacian, computed as 
    L = D^(1/2) (D-W) D^(1/2)
    """
    W = np.exp(-sqDist(X, X)/(2 * sigma**2))
    d = np.sum(W, axis=0)
    D = np.diag(d)
    msqrtD = np.diag(d**(-1/2))
    L = D - W
    L = msqrtD @ L @ msqrtD
    return L

