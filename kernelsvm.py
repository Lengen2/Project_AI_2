"""
=============================================================================
Soft-Margin Kernel SVM (RBF kernel) using the CVXOPT QP solver
=============================================================================
Team Members:
  - Conor Knight [23383542]
  - Oisin Dillon O' Rourke [23372109]
  - Adam Lengen Kelly [23360879]

Usage (C is given on the command line):
    python softmarginsvm.py 1          # C = 1
    python softmarginsvm.py 1000000    # C = 1,000,000

Input files (same directory as this script, three columns: x1, x2, label):
    Training_set.txt
    Testing_set.txt

What the program does:
    1. Loads the training and testing data.
    2. Solves the soft-margin SVM DUAL problem with cvxopt.solvers.qp to get
       the Lagrange multipliers (alphas), one per training point.
    3. Works out the support vectors and the bias term b.
    4. Classifies the training and testing sets, reporting accuracy,
       false positives and false negatives.
    5. Plots the decision boundary and margins.

Only numpy, cvxopt and matplotlib.pyplot are used for the classifier
(sys is just the standard library, for reading the command line).
=============================================================================
"""
import os
import sys                              # reading the command-line argument (C)
import numpy as np                      # array maths
import matplotlib.pyplot as plt         # plotting only
import cvxopt as cvx                    # matrix type used by the plot utility
from cvxopt import matrix, solvers      # matrix() builds QP inputs, solvers.qp solves

# Plotting utility provided with the project (must be in the same directory)
from plotContours import plotContours

# Stop CVXOPT printing every solver iteration to the console
solvers.options['show_progress'] = False

# If ye are using my code the Testing set and Training_set datasets are in the same directory as this script, and they should be named "Testing_set.txt" 
# and "Training_set.txt". But you can change them to whatever you want just make sure they match the file names  
# The Training_set shows how well the model memorized the 1,000 training points
# While the Testing_set gives an score of how accuratly the model performs on the 1,000 testing points it hasn't encountered before
# To run in POWERSHELL all you have to do is first run this code
# Then type cd then the directory where the project is for example mine is C:\Users\adaml\Documents\GitHub\Project_AI_2
#
# At least I think thats what happening tbh tho i'm a bit lost 

# NEW UPDATED VARIABE LEGEND 
# Tweaked a few things and changed some variable names so yeah em good luck soldiers 
# 1. Kernel Configuration
# SIGMA_SQUARED: defines the kernel variance parameter (0.25), fixed by the project specs
# GAMMA: the scale parameter used inside the RBF kernel, 1 / (2 * sigma^2) = 2.0
# X1, X2: the two sets of input points passed into the kernel function (N1 x D and N2 x D)
# dists: holds the squared Euclidean distance between every point in X1 and every point in X2
# all_x_vecs, test_x_vec: inputs to custom_kernel, all training points and one grid point from plotContours

# 2. Data loading & Preprocessing
# here: absolute path of the folder the script is in, so the data files are always found
# train: stores the raw array loaded from Training_set.txt (columns: x1, x2, label)
# test: stores the raw array loaded from Testing_set.txt (columns: x1, x2, label)
# x_train: extracts the 2D coordinates (x1, x2) from the training data
# y_train: extracts the labels (+1 or -1) corresponding to each training point
# x_test: extracts the 2D coordinates (x1, x2) from the testing data
# y_test: extracts the true class labels corresponding to each testing point
# Xs_cvx: converts the training features into a CVXOPT matrix (N x 2) for plotContours
# Ts_cvx: converts the training labels into a CVXOPT column vector (N x 1)
# Ls_cvx: converts the learned alphas into a CVXOPT column vector (N x 1)

# 3. SVM Training
# Xs, Ts: function parameters representing the training features and labels passed into train_svm
# C: soft-margin penalty from the command line, balances margin width against training error
# N: total number of training samples (also the number of QP variables)
# K: the RBF kernel (Gram) matrix evaluated across all training samples (N x N)
# P: quadratic coefficient matrix for the QP, C * y_i * y_j * K_ij plus a tiny 1e-8 diagonal jitter
# q: linear coefficient vector of negative ones for the objective
# G: constraint matrix (-I stacked on I) enforcing the box limits on beta
# h: right-hand side vector for G (N zeros then N ones), so 0 <= beta_i <= 1
# A: equality constraint matrix (row of labels) for the condition sum(beta_i * y_i) = 0
# b: scalar target (0.0) for the equality constraint (not the same as the SVM bias b_val)
# solution: dictionary returned by the CVXOPT QP solver (contains x, status, iterations)
# alphas: solved Lagrange multipliers for every training sample, converted back from beta with alpha = C * beta
# sv_thresh: small tolerance (1e-5) below which an alpha is treated as zero
# sv_idx: indices of the support vectors (alpha > sv_thresh)
# alphas_clean: alphas with the tiny noise values snapped to exactly 0
# bound_tol: how close to C an alpha can be before it counts as sitting at the upper bound
# free_idx: indices of free support vectors (0 < alpha < C), which lie exactly on the margin
# use_idx: the points used to estimate the bias (free SVs, or all SVs if there are no free ones)
# b_vals: one bias estimate per point in use_idx
# b_val: the final scalar bias term of the trained classifier (median of b_vals)
# margins: y * f(x) for every training point (diagnostic only, about 1 for free SVs)

# 4. Evaluation function
# X_eval, y_eval: function parameters representing the features and labels being evaluated
# alphas, b_val: the trained multipliers and bias passed into evaluate_model
# X_train, y_train: training points and labels, needed because f(x) sums over training points
# K_eval: kernel matrix between the evaluation points and the training points (N_eval x N_train)
# activations: raw decision function scores f(x) for every evaluation point
# preds: final binary class predictions (+1 or -1) based on the sign of the activations
# accuracy: percentage of correct predictions on the evaluation set
# fp: false positives, true label is -1 but the model predicted +1
# fn: false negatives, true label is +1 but the model predicted -1

# 5. Main & Plotting
# C: soft-margin penalty parsed from sys.argv[1] (for example 1 or 1000000)
# alphas, b_val, sv_idx, n_free: outputs of train_svm (multipliers, bias, support vector indices, number of free SVs)
# tr_acc, tr_fp, tr_fn: accuracy, false positives and false negatives on the training set
# te_acc, te_fp, te_fn: accuracy, false positives and false negatives on the unseen testing set
# minRange, maxRange, step: edges and spacing of the grid of points plotContours evaluates
# filled: False draws contour lines only, True fills the background
# boundcol, posmarcol, negmarcol: colours of the f(x) = 0 boundary, the +1 margin and the -1 margin
# custom_kernel: wraps rbf_kernel so its output is a CVXOPT matrix, as the plotting utility requires

# =============================================================================
# 1. KERNEL
# =============================================================================
# The assignment fixes the kernel variance: sigma^2 = 0.25.
# RBF kernel:  K(x, z) = exp( -||x - z||^2 / (2 * sigma^2) ) = exp(-GAMMA * ||x - z||^2)
SIGMA_SQUARED = 0.25
GAMMA = 1.0 / (2.0 * SIGMA_SQUARED)     # = 2.0


def rbf_kernel(X1, X2):
    """Compute the RBF kernel matrix between two sets of points.

    Args:
        X1: array-like of shape (N1, D), or a single point of shape (D,).
        X2: array-like of shape (N2, D), or a single point of shape (D,).

    Returns:
        numpy array of shape (N1, N2) where entry [i, j] = K(X1[i], X2[j]).
    """
    # Convert whatever we were given (list, cvxopt matrix, ndarray) to a float array
    X1 = np.array(X1, dtype=float)
    X2 = np.array(X2, dtype=float)

    # A single point arrives as a 1D vector; reshape it to one row so the
    # matrix maths below works the same way
    if X1.ndim == 1:
        X1 = X1.reshape(1, -1)
    if X2.ndim == 1:
        X2 = X2.reshape(1, -1)

    # Squared Euclidean distances for ALL pairs at once (no Python loops), using
    #     ||x - z||^2 = ||x||^2 + ||z||^2 - 2 x.z
    #   np.sum(X1**2, axis=1)  -> squared length of every row of X1
    #   .reshape(-1, 1)        -> column vector so it broadcasts down the rows
    #   X1 @ X2.T              -> matrix of dot products between every pair
    dists = (np.sum(X1 ** 2, axis=1).reshape(-1, 1)
             + np.sum(X2 ** 2, axis=1)
             - 2.0 * X1 @ X2.T)

    # Rounding error can make tiny distances slightly negative, so clip at 0
    # before applying the exponential
    return np.exp(-GAMMA * np.maximum(dists, 0.0))


def custom_kernel(all_x_vecs, test_x_vec):
    """Adapter so plotContours gets the kernel in the format it expects.

    plotContours calls kernel(all training points, one grid point) and
    needs a cvxopt matrix back, so we wrap rbf_kernel's numpy output.
    """
    return cvx.matrix(rbf_kernel(np.array(all_x_vecs),
                                 np.array(test_x_vec).flatten()), tc='d')


# =============================================================================
# 2. TRAINING
# =============================================================================
def train_svm(Xs, Ts, C):
    """Train a soft-margin SVM by solving the dual problem with cvxopt.solvers.qp.

    Dual: minimise 1/2 a^T P a - sum(a)  s.t. 0 <= a_i <= C, sum a_i y_i = 0,
    with P_ij = y_i y_j K(x_i, x_j).

    For numerical stability we solve in scaled variables beta = alpha / C,
    so the bounds are always 0..1 whatever C is. Substituting alpha = C*beta
    and dividing the objective by C gives:
        minimise 1/2 beta^T (C*P) beta - sum(beta)  s.t. 0 <= beta_i <= 1
    and we convert back with alpha = C * beta afterwards.
    """
    # Standard inputs: 2D feature matrix (N, d) and 1D label vector  
    Xs = np.array(Xs, dtype=float)
    Ts = np.array(Ts, dtype=float).flatten()
    N = Xs.shape[0]

    # Kernel (Gram) matrix
    K = rbf_kernel(Xs, Xs)

    # Quadratic term P includes C from beta-scaling and a tiny diagonal jitter (1e-8)
    # to keep P positive definite for the interior-point QP solver
    # editted to scaled by C
    P = matrix(C * np.outer(Ts, Ts) * K + 1e-8 * np.eye(N), tc='d')   # editted to scaled by C
    q = matrix(-np.ones((N, 1)), tc='d')

    # --- Inequality constraints: 0 <= beta_i <= 1 ---
    # cvxopt enforces G @ x <= h; stacking -I and I encodes -beta <= 0 and beta <= 1
    G = matrix(np.vstack((-np.eye(N), np.eye(N))), tc='d')
    # slightly tweaked it so the bounds are 0..1
    h = matrix(np.hstack((np.zeros(N), np.ones(N))), tc='d')         

    # --- Equality constraint: sum beta_i y_i = 0 ---
    # Row vector Ts (1 x N) inner-multiplied with beta (N x 1) equals sum(beta_i * y_i)
    A = matrix(Ts.reshape(1, -1), tc='d')
    b = matrix(0.0, tc='d')

    # --- Solver tolerances ---
    solvers.options['abstol'] = 1e-7
    solvers.options['reltol'] = 1e-7
    solvers.options['feastol'] = 1e-7
    solvers.options['maxiters'] = 500

    # --- Solve ---
    solution = solvers.qp(P, q, G, h, A, b)
    print(f"Solver status: {solution['status']}, iterations: {solution['iterations']}")
    if solution['status'] != 'optimal':
        print(f"Warning: solver status is '{solution['status']}'")
    alphas = C * np.array(solution['x']).flatten()                    # CHANGED: convert back to alpha

    # --- Support vectors ---
    sv_thresh = 1e-5
    sv_idx = np.where(alphas > sv_thresh)[0]
    alphas_clean = np.where(alphas > sv_thresh, alphas, 0.0)

    # --- Bias: median over free support vectors ---
    bound_tol = max(1e-5, 1e-4 * C)
    free_idx = np.where((alphas_clean > sv_thresh) &
                        (alphas_clean < C - bound_tol))[0]
    use_idx = free_idx if len(free_idx) > 0 else sv_idx
    b_vals = Ts[use_idx] - K[use_idx] @ (alphas_clean * Ts)
    b_val = float(np.median(b_vals))

    # --- Diagnostics (delete before submitting) ---
    print(f"max alpha: {alphas_clean.max():.3e}")
    print(f"b spread over free SVs: min {b_vals.min():.3f}, max {b_vals.max():.3f}, std {b_vals.std():.3f}")
    margins = Ts * (K @ (alphas_clean * Ts) + b_val)
    print(f"margin y*f(x) on SVs: min {margins[sv_idx].min():.3f}, max {margins[sv_idx].max():.3f}")

    return alphas_clean, b_val, sv_idx, len(free_idx)

# =============================================================================
# 3. EVALUATION
# =============================================================================
def evaluate_model(X_eval, y_eval, alphas, b_val, X_train, y_train):
    """Classify a dataset with the trained SVM and measure its errors.

    The decision function is
        f(x) = b + sum_i a_i y_i K(x_i, x)
    where the sum runs over the TRAINING points (only support vectors have
    non-zero alpha, so only they matter). Predict +1 if f(x) >= 0, else -1.

    Returns:
        accuracy (percent), false positives, false negatives
    """
    # Kernel between every evaluation point and every training point: (N_eval, N_train)
    K_eval = rbf_kernel(X_eval, X_train)

    # One matrix-vector product gives f(x) for all evaluation points at once
    activations = b_val + K_eval @ (alphas * y_train)

    # Class prediction from the sign of the activation
    preds = np.where(activations >= 0, 1.0, -1.0)

    # Percentage of predictions that match the true labels
    accuracy = np.mean(preds == y_eval) * 100.0

    # False positive: true label is -1 but we predicted +1
    fp = int(np.sum((y_eval == -1) & (preds == 1)))
    # False negative: true label is +1 but we predicted -1
    fn = int(np.sum((y_eval == 1) & (preds == -1)))

    return accuracy, fp, fn


# =============================================================================
# 4. MAIN PROGRAM
# =============================================================================
def main():  # Define the main execution function for training, evaluating, and plotting the SVM
    # Check that exactly one CLI argument (C) is provided besides the script name
    
    if len(sys.argv) != 2:  # sys.argv[0] is the script filename
        print("Usage: python softmarginsvm.py <C>   e.g. python softmarginsvm.py 1")  # Show expected CLI usage
        sys.exit(1)  # Exit with non-zero status to indicate invalid arguments
    try:  # Attempt to parse and validate the C hyperparameter
        C = float(sys.argv[1])  # Convert CLI argument string to floating-point C
        if C <= 0:  # Soft-margin penalty C must be strictly positive
            raise ValueError  # Trigger ValueError if C is zero or negative
    except ValueError:  # Catch non-numeric inputs or rejected values from above
        print("Error: C must be a positive number.")  # Inform user that C must be > 0
        sys.exit(1)  # Exit with error status code 1

    here = os.path.dirname(os.path.abspath(__file__))  # Get absolute path of the directory containing this script
    train = np.loadtxt(os.path.join(here, 'Training_set.txt'))  # Load training data rows (x1, x2, label)
    test = np.loadtxt(os.path.join(here, 'Testing_set.txt'))  # Load testing data rows (x1, x2, label)
   
    x_train, y_train = train[:, :2], train[:, 2]  # Split training data into 2D features and 1D labels
    x_test, y_test = test[:, :2], test[:, 2]  # Split testing data into 2D features and 1D labels
    print(f"Loaded {len(x_train)} training and {len(x_test)} testing samples.")  # Print dataset sizes

    print(f"\n=== Soft-margin RBF SVM, C = {C:g} ===")  # Print header with compactly formatted C
    alphas, b_val, sv_idx, n_free = train_svm(x_train, y_train, C)  # Train SVM and retrieve dual weights, bias, and SVs
    print(f"Support vectors: {len(sv_idx)} (free: {n_free})")  # Print total and free (0 < alpha < C) support vectors
    print(f"Bias b: {b_val:.4f}")  # Print learned bias term rounded to 4 decimal places

    tr_acc, tr_fp, tr_fn = evaluate_model(x_train, y_train, alphas, b_val, x_train, y_train)  # Evaluate on training set
    te_acc, te_fp, te_fn = evaluate_model(x_test, y_test, alphas, b_val, x_train, y_train)  # Evaluate on unseen test set
    print(f"Training: accuracy {tr_acc:.2f}%, FP {tr_fp}, FN {tr_fn}")  # Report training accuracy, false positives, false negatives
    print(f"Testing:  accuracy {te_acc:.2f}%, FP {te_fp}, FN {te_fn}")  # Report testing accuracy, false positives, false negatives

    Xs_cvx = cvx.matrix(x_train, tc='d')  # Convert training features to double-precision cvxopt matrix
    Ts_cvx = cvx.matrix(y_train.reshape(-1, 1), tc='d')  # Convert training labels to (N, 1) cvxopt column matrix
    Ls_cvx = cvx.matrix(alphas.reshape(-1, 1), tc='d')  # Convert learned alphas to (N, 1) cvxopt column matrix

    # Plot contours using the provided utility function, which visualizes the decision boundary and margins
    # So the midRange and maxRange generates a 2D grid of coordinate points across the graph using the step function
    # Filled just fills the background (crazy I know), 
    # Then boundcol is the main decision boundary when the activation score is 0, posmarcol draws the positive margin boundary when its +1
    # Negmarcol draws the negative margin boundary when its -1 
    # I think the reason there is no green line in C = 1 is because f(x) isn't in the visiable plot range?
    plt.figure(figsize=(8, 6))  # Create a new Matplotlib figure of size 8x6 inches
    plotContours(Xs_cvx, Ts_cvx, Ls_cvx, b_val, custom_kernel,  # Plot decision boundary (f=0) and margins (f=+-1)
                 minRange=-6, maxRange=6.5, step=0.1, filled=False,  # Define evaluation grid and line-only contour style
                 boundcol='green', posmarcol='red', negmarcol='blue')  # Set contour colors for boundary and +/- margins

    plt.xlim(-5.5, 6.0)  # Crop x-axis view to avoid edge artifacts near grid borders
    plt.ylim(-5.5, 5.5)  # Crop y-axis view to avoid edge artifacts near grid borders
   
    plt.title(f"RBF Kernel SVM Decision Boundary & Margins (C = {C:g})")  # Set plot title showing current C value
    plt.xlabel("Feature 1 (x1)")  # Label horizontal axis as first feature
    plt.ylabel("Feature 2 (x2)")  # Label vertical axis as second feature
    plt.show()  # Render and display the plot window


if __name__ == '__main__':  # Execute main() only when script is run directly, not on import
    main()  # Call main function