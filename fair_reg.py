from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.linalg import svd

class MP_Fair_regression:
    '''
    Input:
    x: (n_sample, n_feature) //x contains s.
    s: (n_sample, n_protect_feature)
    y: (n_sample, n_label)
    kernel_xs: kernel function for (x,s)
    kernel_s: kernel function for s
    lmd: regularization parameter
    '''

    def __init__(self, x, s, y, kernel_xs, kernel_s = None, lmd = 0):
        self.x = x
        self.s = s
        self.y = y
        self.n = x.shape[0]
        self.P = None
        self.A = None

        self.kernel_s = kernel_s
        self.kernel_xs = kernel_xs

        self.K = self.kernel_xs(self.x, self.x)
        self.lmd=lmd

    def construct_P(self):

        # Optimized version with less computation complexity, more numerical stable, and removing assumption/rely on kernerl_s.
        # If you hope to see the initial implementation, please refer to the initial commit (https://github.com/shawkui/MP_Fair_Regression/commit/4b0a5d57a1669096b115a7b3a971d7518c065c50) in GitHub.

        s_flatten = self.s.flatten()  # ensure s is 1D
        n = len(s_flatten)
        unique_values, inverse = np.unique(s_flatten, return_inverse=True)
        m = len(unique_values)

        B = np.full((m, n), -1 / n)  # initialize all entries to -1/n
        self.B = B

        for j in range(m):
            indices = (inverse == j)
            B[j, indices] += 1  
        K_eigen = B.dot(self.K)

        U, S, Vt = svd(K_eigen, full_matrices=True)
        rank = (S > 1e-10).sum()
        V_null = Vt[rank:].T
        P = V_null @ V_null.T

        self.m = rank
        self.P = P

        return P

    def fit(self):
        
        P = self.construct_P()

        self.w_ = P.T.dot(self.K).dot(self.y)
        self.w_ = np.linalg.pinv(P.T.dot(self.K).dot(self.K.T).dot(P)+self.lmd*P.T.dot(self.K).dot(P)).dot(self.w_)
        self.w_ = P.dot(self.w_)
        if self.kernel_xs.__name__=='linear_kernel':
            self.w_=self.x.T.dot(self.w_)
        return self.w_

    def pred(self, x_):
        if self.kernel_xs.__name__=='linear_kernel':
            y_=x_.dot(self.w_)
        else:
            y_ =  self.kernel_xs(x_, self.x).dot(self.w_)
        return y_
    
    def validate(self):
        # Centralized Kernel Matrix
        disparity = self.B.dot(self.pred(self.x))
        print('E(y|s)-E(y):', disparity)
        return disparity


class MP_Penalty_regression:
    '''
    Input:
    x: (n_sample, n_feature) //x contains s.
    s: (n_sample, n_protect_feature)
    y: (n_sample, n_label)    
    kernel_xs: kernel function for (x,s)
    kernel_s: kernel function for s
    lmd: regularization parameter
    eta: penalty parameter
    '''

    def __init__(self, x, s, y, kernel_xs, kernel_s = None, lmd = 0, eta = 0):
        self.x = x
        self.s = s
        self.y = y
        self.n = x.shape[0]
        self.P = None
        self.A = None

        self.kernel_s = kernel_s
        self.kernel_xs = kernel_xs

        self.K = self.kernel_xs(self.x, self.x)
        self.lmd = lmd
        self.eta = eta

    def construct_P(self):

        # Optimized version with less computation complexity, more numerical stable, and removing assumption/rely on kernerl_s.
        # If you hope to see the initial implementation, please refer to the initial commit (https://github.com/shawkui/MP_Fair_Regression/commit/4b0a5d57a1669096b115a7b3a971d7518c065c50) in GitHub.

        s_flatten = self.s.flatten()  # ensure s is 1D
        n = len(s_flatten)
        unique_values, inverse = np.unique(s_flatten, return_inverse=True)
        m = len(unique_values)

        B = np.full((m, n), -1 / n)  # initialize all entries to -1/n
        self.B = B

        for j in range(m):
            indices = (inverse == j)
            B[j, indices] += 1  
        K_eigen = B.dot(self.K)

        U, S, Vt = svd(K_eigen, full_matrices=True)
        rank = (S > 1e-10).sum()
        V_null = Vt[rank:].T
        P = V_null @ V_null.T

        self.m = rank
        self.P = P

        return P

    def fit(self):
        P = self.construct_P()
        A = np.eye(P.shape[0])-P

        self.w_ =np.linalg.pinv(self.K.dot(self.K)+self.lmd*self.K+self.eta*(self.K.dot(A).dot(self.K))).dot(self.K).dot(self.y)
        self.A=A
        if self.kernel_xs.__name__=='linear_kernel':
            self.w_=self.x.T.dot(self.w_)
        return self.w_

    def pred(self, x_):
        if self.kernel_xs.__name__=='linear_kernel':
            y_=x_.dot(self.w_)
        else:
            y_ =  self.kernel_xs(x_, self.x).dot(self.w_)
        return y_

class Fair_kernel_learning:
    '''
    Fair Kernel Learning (https://arxiv.org/pdf/1710.05578.pdf): a regularizer-based method aims to eliminate the covariance between the predicted value and the sensitive attributes. 
    The implementation is borrowed from https://isp.uv.es/soft_regression.html.
        
    Input:
    x: (n_sample, n_feature) //x contains s.
    s: (n_sample, n_protect_feature)
    y: (n_sample, n_label)
    kernel_xs: kernel function for (x,s)
    lmd: regularization parameter
    eta: penalty parameter
    '''

    def __init__(self, x, s, y, kernel_xs, lmd = 0, eta = 0):
        self.x = x
        self.s = s
        self.y = y
        self.n = x.shape[0]

        self.kernel_xs = kernel_xs

        self.K = self.kernel_xs(self.x, self.x)
        self.K_s = self.kernel_xs(self.s, self.s)
        self.lmd=lmd
        self.eta = eta

    def fit(self):
        # Centralized Kernel Matrix
        H = np.eye(self.n)-1/self.n*np.ones((self.n, self.n))

        K_sb = H.dot(self.K_s.dot(H))
        # wd  = (la*eye(ntr) + K + mus(k)*HKqH*K)\(ytr);
        self.w_ =np.linalg.pinv(self.K+self.lmd*np.eye(self.n)+self.eta/self.n**2*K_sb.dot(self.K)).dot(self.y)
        if self.kernel_xs.__name__=='linear_kernel':
            self.w_=self.x.T.dot(self.w_)
        return self.w_

    def pred(self, x_):
        if self.kernel_xs.__name__=='linear_kernel':
            y_=x_.dot(self.w_)
        else:
            y_ =  self.kernel_xs(x_, self.x).dot(self.w_)
        return y_

