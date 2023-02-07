from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

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

    def __init__(self, x, s, y, kernel_xs, kernel_s, lmd = 0):
        self.x = x
        self.s = s
        self.y = y
        self.n = x.shape[0]
        self.P = None
        self.A = None

        self.kernel_s = kernel_s
        self.kernel_xs = kernel_xs

        self.K = self.kernel_xs(self.x, self.x)
        self.K_s = self.kernel_s(self.s, self.s)
        self.lmd=lmd

    def fit(self):

        # Centralized Kernel Matrix
        H = np.eye(self.n)-1/self.n*np.ones((self.n, self.n))

        K_b = H.dot(self.K.dot(H))
        K_sb = H.dot(self.K_s.dot(H))

        # Eigenvector Computation
        K_eigen = K_sb.dot(K_b)
        self.m=np.linalg.matrix_rank(K_eigen)
        # print('New m: ', self.m)

        eigvals, eigvecs = np.linalg.eig(K_eigen)
        A = eigvecs[:, 0:self.m].real

        # Uncentralization
        A = H.dot(A)

        # Gram-Schmidt Process
        for i in range(self.m):
            a_i = A[:, i:i+1]
            for j in range(i):
                a_j = A[:, j:j+1]
                a_i = a_i-a_j*(a_j.T.dot(self.K).dot(a_i))
            # Normalization
            a_i = a_i/np.sqrt(a_i.T.dot(self.K).dot(a_i))
            A[:, i:i+1] = a_i

        # # The off-diagonal elements should be 0 and diagonal elements should be 1
        # print(A.T.dot(self.K.dot(A)))

        # Projection
        P = np.eye(self.n)-A.dot(A.T).dot(self.K)

        self.w_ = P.T.dot(self.K).dot(self.y)
        self.w_ = np.linalg.pinv(P.T.dot(self.K).dot(self.K).dot(P)+self.lmd*P.T.dot(self.K).dot(P)).dot(self.w_)
        self.w_ = P.dot(self.w_)
        self.A=A
        self.P=P
        if self.kernel_xs.__name__=='linear_kernel':
            self.w_=self.x.T.dot(self.w_)
        return self.w_

    def pred(self, x_):
        if self.kernel_xs.__name__=='linear_kernel':
            y_=x_.dot(self.w_)
        else:
            y_ =  self.kernel_xs(x_, self.x).dot(self.w_)
        return y_

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

    def __init__(self, x, s, y, kernel_xs, kernel_s, lmd = 0, eta = 0):
        self.x = x
        self.s = s
        self.y = y
        self.n = x.shape[0]
        self.P = None
        self.A = None

        self.kernel_s = kernel_s
        self.kernel_xs = kernel_xs

        self.K = self.kernel_xs(self.x, self.x)
        self.K_s = self.kernel_s(self.s, self.s)
        self.lmd = lmd
        self.eta = eta

    def fit(self):

        # Centralized Kernel Matrix
        H = np.eye(self.n)-1/self.n*np.ones((self.n, self.n))

        K_b = H.dot(self.K.dot(H))
        K_sb = H.dot(self.K_s.dot(H))

        # Eigenvector Computation
        K_eigen = K_sb.dot(K_b)
        self.m=np.linalg.matrix_rank(K_eigen)

        eigvals, eigvecs = np.linalg.eig(K_eigen)
        A = eigvecs[:, 0:self.m].real

        # Uncentralization
        A = H.dot(A)

        # Gram-Schmidt Process
        for i in range(self.m):
            a_i = A[:, i:i+1]
            for j in range(i):
                a_j = A[:, j:j+1]
                a_i = a_i-a_j*(a_j.T.dot(self.K).dot(a_i))
            # Normalization
            a_i = a_i/np.sqrt(a_i.T.dot(self.K).dot(a_i))
            A[:, i:i+1] = a_i

        # The off-diagonal elements should be 0 and diagonal elements should be 1
        # print(A.T.dot(self.K.dot(A)))
        
        self.w_ =np.linalg.pinv(self.K.dot(self.K)+self.lmd*self.K+self.eta*(self.K.dot(A.dot(A.T)).dot(self.K))).dot(self.K).dot(self.y)
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

