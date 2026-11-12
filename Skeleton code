##
## plotContours.py
##
##
## Simplified data and contour plotting routine specialised to work with CVXOPT Intro.ipynb
## notebook.
##
##
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as cvx
import scipy.spatial


def activation(testXvec, allXvecs, Ts, Ls, b, K):
    ks = K(allXvecs,testXvec)
    return b + sum(cvx.mul(cvx.mul(Ls,Ts),ks))


def plotContours(Xs,Ts,Ls,b,K,minRange=-0.6,maxRange=1.6,step=0.05,
                 filled=False,
                 boundcol="#208020", posmarcol="red", negmarcol="blue",
                 fillposcol="#ffe0e0", fillnegcol="#e0f0ff"):
    """Plot contours of activation function for a 2-d classifier, e.g. 2-input XOR.
       Xs - cvxopt Nx2 matrix, data points, one [x,y] pair per row.
       Ts - cvxopt Nx1 matrix, class labels as a cvxopt column vector.
       Ls - cvxopt Nx1 matrix, Lagrange multipliers as a cvxopt column vector.
       b  - bias, scalar
       K  - Kernel function taking 2 scalar arguments, x & y co-ordinates.
       minRange, maxRange - range of x,y to be plotted.
       step - delta used to sample SVM activations across the range.
    """

    # Build activation level array.
    xs = np.arange(minRange,maxRange+step/2.0,step)
    ys = np.arange(minRange,maxRange+step/2.0,step)
    als = np.array([activation(cvx.matrix([y,x]).T,Xs,Ts,Ls,b,K) 
                    for y in ys for x in xs])
    als = als.reshape((len(xs),len(ys)))
    if filled:
        plt.contourf(xs,ys,als.T,levels=(-1e6,0.0,1e6),\
                     colors=(fillnegcol, fillposcol))
    plt.contour(xs,ys,als.T,levels=(-1.0,0.0,1.0),\
                linewidths=(1,3,1),
                colors=(negmarcol,boundcol,posmarcol))
    pts, labs = np.array(Xs).T, np.array(Ts).ravel()
#    print(pts.shape, labs.shape)
    plt.scatter(pts[0], pts[1], c=labs, cmap='bwr')
    plt.grid()
#    plt.axis('equal');
    return xs, ys, als
