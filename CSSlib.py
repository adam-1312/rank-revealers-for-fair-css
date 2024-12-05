from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import matlab.engine

import numpy as np
import pandas as pd

import time
import signal
import traceback

def perform_CSS(M, k, alg='RRQR'):
    '''
    Given a matrix and an integer k, returns indices of the k columns
    that solve the Column Subset Selection problem found by a 
    specified algorithm. Available algorithms are:
    'RRQR', 'CPQR', 'lowQR'.
    
    in:
        - M: m by n matrix
        - k: number of columns to be selected
        - alg: string specifying algorithm to be used. Default: 'RRQR'
    out:
        - cols_order: list of the k selected column indices starting from 0
        - P: n by n permutation matrix s.t. first k columns  of MP
             correspond to cols_order 
    '''
    eng = matlab.engine.start_matlab()
    eng.cd(r'..', nargout=0)
    
    M_mat = eng.double(M)
    if alg == 'RRQR':
        _,_,P = eng.RRQR(M_mat, k, 2, nargout=3)
    elif alg == 'CPQR':
        _,_,P = eng.CPQR(M_mat, k, nargout=3)
    elif alg == 'lowQR':
        _,_,P = eng.lowQRforCSS(M_mat, k, nargout=3)
    
    P = np.asarray(P)
    
    n = M.shape[1]
    cols_order = np.arange(n).dot(P)
    cols_order = list(map(int, cols_order))
    
    return cols_order[:k], P

def perform_fair_CSS(A, B, k, alg='lowQR'):
    '''
    Given 2 matrices A and B with same number of columns
    and an integer k, returns indices of the k columns
    that solve the Fair Column Subset Selection problem found by a 
    specified algorithm. Available algorithms are:
    'CPQR','lowQR'.
    
    in:
        - A: m_1 by n matrix
        - B: m_2 by n matrix
        - k: number of columns to be selected
        - alg: string specifying algorithm to be used. Default: 'fairCPQR'
    out:
        - cols_order: list of the k selected column indices starting from 0
        - P: n by n permutation matrix s.t. first k columns  of MP
             correspond to cols_order 
    '''
    eng = matlab.engine.start_matlab()
    eng.cd(r'..', nargout=0)
    
    A_mat = eng.double(A)
    B_mat = eng.double(B)
    if alg == 'CPQR':
        P = eng.fairCPQR(A_mat, B_mat, k, nargout=1)
    elif alg == 'lowQR':
       P = eng.fairLowQRforCSS(A_mat, B_mat, k, nargout=1)
    elif alg == 'RRQR':
        _,_,P_A = eng.RRQR(A_mat, k, 2, nargout=3)
        _,_,P_B = eng.RRQR(A_mat, k, 2, nargout=3)
        
        P_A = np.asarray(P_A)
        P_B = np.asarray(P_B)
        n = A.shape[1]
        cols_A = np.arange(n).dot(P_A).astype(int)
        cols_B = np.arange(n).dot(P_B).astype(int)
        cols = np.intersect1d(cols_A, cols_B)
        diff_A = np.setdiff1d(cols_A, cols)
        diff_B = np.setdiff1d(cols_B, cols)
        diff_both = np.union1d(diff_A, diff_B)
        print('number of intersecting columns found: ', cols.size)
        if diff_both.size > 0:
            M = np.vstack((A[:,diff_both], B[:,diff_both]))
            print(M.shape)
            M_mat = eng.double(M)
            _,_,P = eng.RRQR(M_mat, diff_A.size, 2, nargout=3)
        else:
            P = P_A
    
    P = np.asarray(P)
    
    n = A.shape[1]
    cols_order = np.arange(n).dot(P)
    cols_order = list(map(int, cols_order))
    
    return cols_order[:k], P

def best_proj_CSS(X_test, filename):
    '''
    Saves the left singular vectors of X_test into a file.
    
    in:
        X_test: The matrix for which left s.v. to be computed
        filename: file name without extension
    '''
    U,_,_= np.linalg.svd(X_test, full_matrices=False)
    with open(f'{filename}.npy', 'wb') as f:
        np.save(f, U)

def reconstruction_err(A, cols, rel='mtx'):
    '''
    Computes error between A and A projected onto column space of cols.
    The error can be returned as relative error w.r.t. 
    'mtx': norm(A) or 
    'best': norm(A-A_k) where A_k the best k-rank approximation of A is.
    
    in:
        A: matrix to be projected
        cols: indices of columns defining the projection
        rel: type of relative error 'mtx' or 'best', see above
    out:
        rel_err: relative reconstruction error (relative to A)
    '''
    C = A[:,cols]
    pinvC=np.linalg.pinv(C)
    abs_err = np.linalg.norm(A-C.dot(pinvC.dot(A)), ord='fro')
    
    if rel == 'mtx':
        rel_err = abs_err/np.linalg.norm(A, "fro")
    elif rel == 'best':
        U,S,Vt= np.linalg.svd(A, full_matrices=False)
        S[len(cols):] = 0
        A_k = U.dot(np.diag(S).dot(Vt))
        rel_err = abs_err/np.linalg.norm(A-A_k, "fro")
    return rel_err

def get_mnist(test_size=1/7.0,rnd=22):
    '''
    Fetch MNIST dataset and preprocess it into disjoint training and 
    test subsets. There is an option to invert some of the images to
    generate two exclusive sample groups.
    
    in:
        - test_size: number between 0 and 1 indicating percentage of test set
        - rnd: random seed
        - unfair: boolean value on whether to invert some images from training or not
        - inverted: percentage of inverted images
    out:
        - X_train: feature matrix each row is an image of a digit, 
                   and each column represents an image pixel
        - X_test : same as X_train
        - y_train: labels vector for each row in X_train
        - y_test: labels vector for each row in X_test
    '''
    # ~ digits = load_digits()
    # ~ X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=test_size, random_state=rnd)
    
    mnist = fetch_openml(data_id=554)
    X_train, X_test, y_train, y_test = train_test_split(mnist.data, 
                                                        mnist.target.astype('int'),
                                                        test_size=test_size,
                                                        random_state=rnd)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    return X_train, X_test, y_train, y_test

def get_classifier(features, labels, rnd=22):
    '''
    Train and return a classifier trained on a dataset.
    
    in:
        - features: feature matrix each row is a sample, 
                    and each column represents a feature
        - labels: labels vector for each row in features
        - rnd: random seed
    out:
        - clf: scikit-learn's RandomForestClassifier object
    '''
    clf = RandomForestClassifier(n_estimators=100, random_state=rnd)
    clf.fit(features, labels)
    
    return clf

def unfair_mnist(X, inverted=1/3.0):
    '''
    Given image data from the MNIST dataset, invert the last images
    in the data w.r.t. a percentage of inverted images.
    
    in:
        - X: MNIST data before inverting
        - inverted: percentage of images to be inverted in X
    out:
        - new_X: MNIST data after inverting the last images in the dataset
    '''
    invert = lambda im : (np.full(im.shape, 255) - im).astype(np.uint8)
    rotate = lambda im : np.rot90(im.reshape(28,28)).flatten()
    
    n_inverted = int(inverted*X.shape[0])
    
    new_X = X.copy()
    # invert the last n_inverted images in the data
    # ~ new_X[-n_inverted:] = np.apply_along_axis(invert,1,new_X[-n_inverted:])
    new_X[-n_inverted:] = np.apply_along_axis(rotate,1,new_X[-n_inverted:])
    
    return new_X

def timeout_handler(signum, frame):
    raise Exception("Maximum computation time reached!")

###
# Script
###
print("Importing MNIST dataset...")
'''
Import new MNIST training/test sets
'''
X_train, X_test, y_train, y_test = get_mnist()
# ~ X_train = unfair_mnist(X_train, inverted=1) # rotated training set
# ~ pd.DataFrame(X_train).to_csv('X_train.csv')
# ~ pd.DataFrame(X_test).to_csv('X_test.csv')
# ~ pd.DataFrame(y_train).to_csv('y_train.csv')
# ~ pd.DataFrame(y_test).to_csv('y_test.csv')

'''
Import local MNIST training/test sets
'''
# ~ X_train = pd.read_csv('X_train.csv').to_numpy()
# ~ X_test = pd.read_csv('X_test.csv').to_numpy()
# ~ y_train = pd.read_csv('y_train.csv').to_numpy()
# ~ y_test = pd.read_csv('y_test.csv').to_numpy()

inverted_perc = 0.25
unfair_X_train = unfair_mnist(X_train, inverted=inverted_perc)
n_inverted = int(inverted_perc*X_train.shape[0])
unfair_X_test = unfair_mnist(X_test, inverted=1)

acc_noninverted = []
acc_inverted = []
err_noninverted = []
err_inverted = []
err_noninverted_test = []
err_inverted_test = []
all_cols = []

'''
Import local cols
'''
# ~ cols_df = pd.read_csv('RRQRcols_per_k_rotated.csv',index_col=0)

k_range = range(1,37)
for i, k in enumerate(k_range):

    # ~ print()#################### CSS on data with 1 group
    # ~ print(f'Performing CSS on data with 1 group for k={k}...')
    
    # ~ t = time.perf_counter()
    # ~ signal.signal(signal.SIGALRM, timeout_handler)
    # ~ signal.alarm(480) # Maximum computation time in seconds (8 mins)
    # ~ try:
        # ~ cols,_ = perform_CSS(X_train, k, alg='RRQR')
        # ~ cols = list(cols_df.loc[k][0:k].to_numpy().astype(int))
    # ~ except Exception:
        # ~ print('Timeout...')
        # ~ print(traceback.format_exc())
        # ~ continue
    # ~ css_time = time.perf_counter() - t
    # ~ print(f'It took {css_time} seconds for CSS...')
    
    # ~ all_cols.append(cols)
    
    # ~ '''Update result'''
    # ~ cols_per_k = pd.DataFrame(data=all_cols, index=k_range[:i+1])
    # ~ cols_per_k.to_csv('cols_per_k.csv')
    
    # ~ err = reconstruction_err(X_test, cols)
    # ~ print(f'Relative reconstruction error: {err}')
    # ~ err_noninverted.append(err)
    
    # ~ n_inverted = int(0.25*X_train.shape[0])
    
    # ~ print()#################### 1 group classifier
    
    # ~ print(f'Training 1 group classifier on data with 1 group for k={k}...')
    # ~ clf = get_classifier(X_train[:,cols], y_train)

    # ~ score = clf.score(X_test[:,cols], y_test)
    # ~ acc_noninverted.append(score)
    # ~ print(f'1 group classifier accuracy on noninverted group for k={k}:', score)
    # ~ score = clf.score(unfair_X_test[:,cols], y_test)
    # ~ acc_inverted.append(score)
    # ~ print(f'1 group classifier accuracy on inverted group for k={k}:', score)
    
    # ~ '''Update result'''
    # ~ results_one_group = pd.DataFrame(data=[err_noninverted, acc_noninverted, acc_inverted], index=['err','acc_noninverted', 'acc_inverted'], columns=k_range[:i+1])
    # ~ results_one_group.to_csv('results_one_group.csv')
    
    print()#################### CSS on data with 2 groups
    print(f'Performing CSS on data with both classes for k={k}...')
    t = time.perf_counter()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(480) # Maximum computation time in seconds (8 mins)
    try:
        cols,_ = perform_CSS(unfair_X_train, k, alg='lowQR')
        # ~ cols,_ = perform_fair_CSS(unfair_X_train[:-n_inverted], unfair_X_train[-n_inverted:], k, alg='lowQR')
    except Exception:
        print('Timeout...')
        print(traceback.format_exc())
        continue
    css_time = time.perf_counter() - t
    print(f'It took {css_time} seconds for CSS...')
    
    all_cols.append(cols)
    
    err = reconstruction_err(unfair_X_test, cols)
    print(f'Test relative reconstruction error: {err}')
    
    err = reconstruction_err(unfair_X_train[:-n_inverted], cols, rel='best')
    print(f'Noninverted group  training reconstruction error: {err}')
    err_noninverted.append(err)
    
    err = reconstruction_err(unfair_X_train[-n_inverted:], cols, rel='best')
    err_inverted.append(err)
    print(f'Inverted group training reconstruction error: {err}')
    
    err = reconstruction_err(X_test, cols, rel='best')
    print(f'Noninverted group test reconstruction error: {err}')
    err_noninverted_test.append(err)
    
    err = reconstruction_err(unfair_X_test, cols, rel='best')
    err_inverted_test.append(err)
    print(f'Inverted group test reconstruction error: {err}')
    
    '''Update result'''
    cols_per_k = pd.DataFrame(data=all_cols, index=k_range[:i+1])
    cols_per_k.to_csv('cols_per_k.csv')
    
    print()#################### 2 group classifier
    
    print(f'Training classifier on 2 groups for k={k}...')
    clf = get_classifier(unfair_X_train[:,cols], y_train)

    score = clf.score(X_test[:,cols], y_test)
    acc_noninverted.append(score)
    print(f'2 group classifier accuracy on noninverted for k={k}:', score)
    score = clf.score(unfair_X_test[:,cols], y_test)
    acc_inverted.append(score)
    print(f'2 group classifier accuracy on inverted for k={k}:', score)
    print()####################
    
    results_two_groups = pd.DataFrame(data=[err_noninverted, err_inverted, err_noninverted_test, err_inverted_test, acc_noninverted, acc_inverted], index=['err_noninverted','err_inverted','err_noninverted_test','err_inverted_test','acc_noninverted', 'acc_inverted'], columns=k_range[:i+1])
    results_two_groups.to_csv('results_two_groups.csv')

