import numpy as np
# for finding k eigen values
import scipy.sparse.linalg
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
def kNNClassify(Xtr, Ytr, k, Xte):
    """Classifies a set o test points given a training set and the number of neighbours to use
    
    usage:
    Ypred = kNNClassify(Xtr, Ytr, k=5, Xte)
    
    Arguments:
    Xtr: Training set matrix [nxd], each row is a point
    Ytr: Training set label array [n], values must be +1,-1
    k: Number of neighbours
    Xte: Test points
    
    Return 
    Ypred: estimated test output
    """
    n_train = Xtr.shape[0]
    n_test = Xte.shape[0]
    
    if any(np.abs(Ytr) != 1):
        print("The values of Ytr should be +1 or -1.")
        return -1
    
    if k > n_train:
        print("k is greater than the number of points, setting k=n_train")
        k = n_train
        
    Ypred = np.zeros(n_test)
    
    dist = sqDist(Xte, Xtr)
    
    for idx in range(n_test):
        neigh_indexes = np.argsort(dist[idx,:])[:k]
        #print(Ytr[neigh_indexes])
        avg_neigh = np.mean(Ytr[neigh_indexes])
        #print(avg_neigh)        
        Ypred[idx] = np.sign(avg_neigh)
        
    return Ypred
################################################################################"
def holdoutCVkNN(Xtr, Ytr, perc, n_rep, k_list):
    """Performs holdout cross-validation for k Nearest Neighbour algorithm
    
    Arguments:
    Xtr: Training set matrix [nxd], each row is a point
    Ytr: Training set label array [n], values must be +1,-1
    perc: percentage of training set ot be used for validation
    n_rep: number of repetitions of the test for each couple of parameters
    k_list: list/array of regularization parameters to try
       
    Returns:
    k: the value k in k_list that minimizes the mean of the validation error
    Vm, Vs: mean and variance of the validation error for each couple of parameters
    Tm, Tx: mean and variance of the error computed on the training set for each couple of parameters
    """
      
    if perc < 1 or perc > 100:
        print("p should be a percentage value between 0 and 100.")
        return -1
    
    if n_rep <= 0:
        print("Please supply a positive number of repetitions")
        return -1
    
    # Ensures that k_list is a numpy array
    k_list = np.array(k_list)
    num_k = k_list.size
    
    n_tot = Xtr.shape[0]
    n_train = int(np.ceil(n_tot * (1 - perc/100)))
    
    Tm = np.zeros(num_k)
    Ts = np.zeros(num_k)
    Vm = np.zeros(num_k)
    Vs = np.zeros(num_k)
    
    for kdx,k in enumerate(k_list):
        for rip in range(n_rep):
            
            # Randombly select a subset of the training set
            rand_idx = np.random.choice(n_tot, size=n_tot, replace=False)
            
            X     = Xtr[rand_idx[:n_train]]
            Y     = Ytr[rand_idx[:n_train]]
            X_val = Xtr[rand_idx[n_train:]]
            Y_val = Ytr[rand_idx[n_train:]]
            
            # Compute the training error of the kNN classifier for the given value of k
            trError = calcErr(kNNClassify(X, Y, k, X), Y)
            Tm[kdx] = Tm[kdx] + trError
            Ts[kdx] = Ts[kdx] + trError**2
            
            # Compute the validation error of the kNN classifier for the given value of k
            valError = calcErr(kNNClassify(X, Y, k, X_val), Y_val)
            Vm[kdx] = Vm[kdx] + valError
            Vs[kdx] = Vs[kdx] + valError**2
    
    Tm = Tm/n_rep
    Ts = Ts/n_rep - Tm**2
    
    Vm = Vm/n_rep
    Vs = Vs/n_rep - Vm**2
    
    best_k_idx = np.argmin(Vm)
    k = k_list[best_k_idx]
    
    return k, Vm, Vs, Tm, Ts     
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
def separatingFkNN(Xtr, Ytr, k):
    """Plots seprating function of a kNN classifier
    
    usage: separatingFkNN(Xtr=X, Ytr=Y, k=3)
    
    Xtr: The training points
    Ytr: The labels of the training points
    k : How many neighbours to use for classification
    
    Returns:
    nothing
    """
    
    Ypred = kNNClassify(Xtr=Xtr, Ytr=Ytr, k=k, Xte=Xtr)

    x = Xtr[:,0]
    y = Xtr[:,1]
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    zi = griddata(x, y, Ypred, xi, yi, interp='linear')

    CS = plt.contour(xi, yi, zi, 15, linewidths=2, colors='k', levels=[0])
    # plot data points.
    plt.scatter(x, y, c=Ytr, marker='o', s=20, zorder=10)
    plt.xlim(x.min(), x.max())
    plt.ylim(x.min(), x.max())
    plt.title('Separating function')
    plt.show()
