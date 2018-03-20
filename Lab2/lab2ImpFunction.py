import numpy as np
import scipy.io as sio
import scipy.sparse.linalg
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
import matplotlib.patches as mpatches

################################################################################
################################################################################
################################################################################
################################################################################
def holdoutCVkNN(Xtr, Ytr, perc, n_rep, kernel, lam_list, kerpar_list=[]):
    """Performs holdout cross-validation for Kernel Ridge regression
    
    Arguments:
    Xtr: Training set matrix [nxd], each row is a point
    Ytr: Training set label array [n], values must be +1,-1
    perc: percentage of training set ot be used for validation
    n_rep: number of repetitions of the test for each couple of parameters
    kernel: the chosen kernel
    lam_list: list/array of regularization parameters to try
    kerpar_list: st/array of regularization parameters to try
       
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
    
    # Ensures that parameters are in a numpy array
    lam_list = np.array(k_list)
    num_lam = k_list.size
    
    if not kerpar_list:
        kerpar_list = [0]
    kerpar_list = np.array(kerpar_list)
    num_kerpar = kerpar_list.size
    
    n_tot = Xtr.shape[0]
    n_train = int(np.ceil(n_tot * (1 - perc/100)))
    
    Tm = np.zeros(num_k)
    Ts = np.zeros(num_k)
    Vm = np.zeros(num_k)
    Vs = np.zeros(num_k)
    
    for lamdx,lam in enumerate(lam_list):
        for kerpardx, kerpar in enumerate(kerpar_list):
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
################################################################################
def holdoutCVKernRLS(x, y, perc, nrip, kernel, lam_list, kerpar_list):
    '''     
    Input:
    xtr: the training examples
    ytr: the training labels
    kernel: the kernel function (see KernelMatrix documentation).
    perc: percentage of the dataset to be used for validation, must be in range [1,100]
    nrip: number of repetitions of the test for each couple of parameters
    lam_list: list of regularization parameters
        for example intlambda = np.array([5,2,1,0.7,0.5,0.3,0.2,0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001])
    kerpar_list: list of kernel parameters
        for example intkerpar = np.array([10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01])
    
    Returns:
    l, s: the couple of lambda and kernel parameter that minimize the median of the validation error
    vm, vs: median and variance of the validation error for each couple of parameters
    tm, ts: median and variance of the error computed on the training set for each couple of parameters
    
    Example of usage:
    
    from regularizationNetworks import MixGauss
    from regularizationNetworks import holdoutCVKernRLS
    import matplotlib.pyplot as plt
    import numpy as np
    
    lam_list = np.array([5,2,1,0.7,0.5,0.3,0.2,0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001])
    kerpar_list = np.array([10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01])
    xtr, ytr = MixGauss.mixgauss([[0;0],[1;1]],[0.5,0.25],100);
    l, s, Vm, Vs, Tm, Ts = holdoutCVKernRLS.holdoutcvkernrls(xtr, ytr,'gaussian', 0.5, 5, lam_list, kerpar_list);
    plt.plot(lam_list, vm, 'b')
    plt.plot(lam_list, tm, 'r')
    plt.show()
    '''
    
    if perc < 1 or perc > 100:
        print("p should be a percentage value between 0 and 100.")
        return -1

    if isinstance(kerpar_list, int):
        kerpar_list = np.array([kerpar_list])
    else:
        kerpar_list = np.array(kerpar_list)
    nkerpar = kerpar_list.size
                               
                            
    if isinstance(lam_list, int):
        lam_list = np.array([lam_list])
    else:
        lam_list = np.array(lam_list)
    nlambda = lam_list.size
    
    n = x.shape[0]
    ntr = int(np.ceil(n*(1-perc/100)))

    tm = np.zeros((nlambda, nkerpar))
    ts = np.zeros((nlambda, nkerpar))
    vm = np.zeros((nlambda, nkerpar))
    vs = np.zeros((nlambda, nkerpar))

    ym = float(y.max() + y.min())/float(2)

    il = 0
    for l in lam_list:
        iss = 0
        for s in kerpar_list:
            trerr = np.zeros((nrip, 1))
            vlerr = np.zeros((nrip, 1))
            for rip in range(nrip):
                i = np.random.permutation(n)
                xtr = x[i[:ntr]]
                ytr = y[i[:ntr]]
                xvl = x[i[ntr:]]
                yvl = y[i[ntr:]]

                w = regularizedKernLSTrain(xtr, ytr, kernel, s, l)
                trerr[rip] = calcErr(regularizedKernLSTest(w, xtr, kernel, s, xtr), ytr, ym)
                vlerr[rip] = calcErr(regularizedKernLSTest(w, xtr, kernel, s, xvl), yvl, ym)
                print('l: ', l, ' s: ', s, ' valErr: ', vlerr[rip], ' trErr: ', trerr[rip])
            tm[il, iss] = np.median(trerr)
            ts[il, iss] = np.std(trerr)
            vm[il, iss] = np.median(vlerr)
            vs[il, iss] = np.std(vlerr)
            iss = iss + 1
        il = il + 1
    row, col = np.where(vm == np.amin(vm))
    l = lam_list[row]
    s = kerpar_list[col]

    return [l, s, vm, vs, tm, ts]
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
################################################################################
def regularizedKernLSTest(c, Xtr, kernel, sigma, Xte):
    '''
    Arguments:
    c: model weights
    Xtr: training input
    kernel: type of kernel ('linear', 'polynomial', 'gaussian')
    sigma: width of the gaussian kernel, if used
    Xts: test points
    
    Returns:
    y: predicted model values
    
    Example of usage:
    
    from regularizationNetworks import regularizedKernLSTest
    y =  regularizedKernLSTest.regularizedkernlstest(c, Xtr, 'gaussian', 1, Xte)
    '''

    Ktest = kernelMatrix(Xte, Xtr, sigma, kernel)
    y = np.dot(Ktest, c)

    return y
################################################################################
def regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, lam):
    '''
    Arguments:
    Xtr: training input
    Ytr: training output
    kernel: type of kernel ('linear', 'polynomial', 'gaussian')
    lam: regularization parameter
    
    Returns:
    c: model weights
    
    Example of usage:
    
    from regularizationNetworks import regularizedKernLSTrain
    c =  regularizedKernLSTrain.regularizedKernLSTrain(Xtr, Ytr, 'gaussian', 1, 1E-1);
    '''
    n = Xtr.shape[0]
    K = kernelMatrix(Xtr, Xtr, sigma, kernel)
    c = np.dot(np.linalg.pinv(K + lam * n * np.identity(n)), Ytr)

    return c
################################################################################
def separatingFKernRLS(c, Xtr, Ytr, kernel, sigma, Xte):

    '''The function classifies points evenly sampled in a visualization area,
    according to the classifier Regularized Least Squares
    
    Arguments:
    c: model weights
    Xtr: training input
    Ytr: training output
    kernel: type of kernel ('linear', 'polynomial', 'gaussian')
    sigma: width of the gaussian kernel, if used
    Xte: test points
    
    Example of usage:
    
    from regularizationNetworks import MixGauss
    from regularizationNetworks import separatingFKernRLS
    from regularizationNetworks import regularizedKernLSTrain
    import numpy as np
    
    lam = 0.01
    kernel = 'gaussian'
    sigma = 1
    
    Xtr, Ytr = MixGauss.mixgauss(np.matrix('0 1; 0 1'), np.matrix('0.5 0.25'), 100)
    Xts, Yts = MixGauss.mixgauss(np.matrix('0 1; 0 1'), np.matrix('0.5 0.3'), 100)
    
    c = regularizedKernLSTrain.regularizedkernlstrain(Xtr, Ytr, 'gaussian', sigma, lam)
    separatingFKernRLS.separatingfkernrls(c, Xtr, Ytr, 'gaussian', sigma, Xte)
    '''

    step = 0.05

    x = np.arange(Xte[:, 0].min(), Xte[:, 0].max(), step)
    y = np.arange(Xte[:, 1].min(), Xte[:, 1].max(), step)

    xv, yv = np.meshgrid(x, y)

    xv = xv.flatten('F')
    xv = np.reshape(xv, (xv.shape[0], 1))

    yv = yv.flatten('F')
    yv = np.reshape(yv, (yv.shape[0], 1))

    xgrid = np.concatenate((xv, yv), axis=1)

    ygrid = regularizedKernLSTest(c, Xtr, kernel, sigma, xgrid)

    '''cc = []
    for item in ytr: cc.append(colors[(int(item)+1)/2])
    plt.scatter(xtr[:, 0], xtr[:, 1], c=cc, s=50)'''
    af = plotdataset(Xtr, Ytr, 'separation')

    z = np.asarray(np.reshape(ygrid, (y.shape[0], x.shape[0]), 'F'))
    af.contour(x, y, z, 1)
    plt.show()
################################################################################
def plotdataset(x, y, name):
    '''
    Input:
    
    x: 2D points coordinate
    y: label of each point, only two labels are accepted
    name: a string containing the title of the plot
    '''
    #colors = ['b', 'y']
    colors = [-1, +1]
    cc = []
    for item in y: cc.append(colors[int((item + 1) / 2)])
    f = plt.figure()
    af = f.add_subplot(111)
    af.set_title(name)
    af.scatter(x[:, 0], x[:, 1], c=cc, s=50)
    plt.draw()

    return af
################################################################################
def kernelMatrix(x1, x2, param, kernel='linear'):
    '''
    Input:
    x1, x2: collections of points on which to compute the Gram matrix
    kernel: can be 'linear', 'polynomial' or 'gaussian'
    param: is [] for the linear kernel, the exponent of the polynomial kernel, 
           or the variance for the gaussian kernel
           
    Output:
    k: Gram matrix
    '''
    if kernel == 'linear':
        k = np.dot(x1, np.transpose(x2))
    elif kernel == 'polynomial':
        k = np.power((1 + np.dot(x1, np.transpose(x2))), param)
    elif kernel == 'gaussian':
        k = np.exp(float(-1)/float((2*param**2))*sqDist(x1, x2))
    return k
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
def calcErr(t, y, m):
    vt = (t >= m).astype(int)
    vy = (y >= m).astype(int)

    err = float(np.sum(vt != vy))/float(y.shape[0])
    return err