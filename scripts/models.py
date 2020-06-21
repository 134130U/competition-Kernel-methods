#import numpy as np
#import KernelMethodBase

import numpy as np
import Kernels
from Kernels import linear_kernel,quadratic_kernel, rbf_kernel

class KernelMethodBase(object):
    '''
    Base class for kernel methods models

    Methods
    ----
    fit
    predict
    '''
    kernels_ = {
        'linear': linear_kernel,
        'quadratic': quadratic_kernel,
        'rbf': rbf_kernel
    }
    def __init__(self, kernel='linear', **kwargs):
        self.kernel_name = kernel
        self.kernel_function_ = self.kernels_[kernel]
        self.kernel_parameters = self.get_kernel_parameters(**kwargs)

    def get_kernel_parameters(self, **kwargs):
        params = {}
        if self.kernel_name == 'rbf':
            params['sigma'] = kwargs.get('sigma', None)
        return params

    def fit(self, X, y, **kwargs):
        return self

    def decision_function(self, X):
        pass

    def predict(self, X):
        pass

class KernelRidgeRegression(KernelMethodBase):
    '''
    Kernel Ridge Regression
    '''
    def __init__(self, lambd=0.001, **kwargs):
        self.lambd = lambd
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelRidgeRegression, self).__init__(**kwargs)

    def fit(self, X, y, sample_weights=None):
        n, p = X.shape
        assert (n == len(y))

        self.X_train = X
        self.y_train = y

        if sample_weights is not None:
            w_sqrt = np.sqrt(sample_weights)
            self.X_train = self.X_train * w_sqrt[:, None]
            self.y_train = self.y_train * w_sqrt

        A = self.kernel_function_(X, X, **self.kernel_parameters)
        A[np.diag_indices_from(A)] += n*self.lambd
        # self.alpha = (K + n lambda I)^-1 y
        self.alpha = np.linalg.solve(A , self.y_train)

        return self

    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)
        return K_x.dot(self.alpha)

    def predict(self, X):
        return np.sign(self.decision_function(X))

import cvxopt

def cvxopt_qp(P, q, G, h, A, b):
    A = A.astype('float')
    P = .5 * (P + P.T)
    cvx_matrices = [
        cvxopt.matrix(M) if M is not None else None for M in [P, q, G, h, A, b]
    ]
    #cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(*cvx_matrices, options={'show_progress': False})
    return np.array(solution['x']).flatten()

solve_qp = cvxopt_qp

def svm_dual_soft_to_qp_kernel(K, y, C=1):
    n = K.shape[0]
    assert (len(y) == n)

    # Dual formulation, soft margin
    P = np.diag(y).dot(K).dot(np.diag(y))
    # As a regularization, we add epsilon * identity to P
    eps = 1e-12
    P += eps * np.eye(n)
    q = - np.ones(n)
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.hstack([np.zeros(n), C * np.ones(n)])
    A = y[np.newaxis, :]
    b = np.array([0.])
    return P, q, G, h, A, b

class KernelSVM(KernelMethodBase):
    '''
    Kernel SVM Classification

    Methods
    ----
    fit
    predict
    '''
    def __init__(self, C=0.1, **kwargs):
        self.C = C
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelSVM, self).__init__(**kwargs)

    def fit(self, X, y, tol=1e-1):
        n, p = X.shape
        assert (n == len(y))

        self.X_train = X
        self.y_train = y

        # Kernel matrix
        K = self.kernel_function_(self.X_train,self.X_train, **self.kernel_parameters)

        # Solve dual problem
        self.alpha = solve_qp(*svm_dual_soft_to_qp_kernel(K, y, C=self.C))

        # Compute support vectors and bias b
        sv = np.logical_and((self.alpha > tol) ,(self.C - self.alpha > tol) )
        self.bias = y[sv] - K[sv].dot(self.alpha*y)
        #print(self.bias)
        self.bias = self.bias.mean()


        self.support_vector_indices = np.nonzero(sv)[0]

        return self

    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)
        return K_x.dot(self.alpha * self.y_train) + self.bias

    def predict(self, X):
        return np.sign(self.decision_function(X))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class KernelLogisticRegression(KernelMethodBase):
    '''
    Kernel Logistic Regression
    '''
    def __init__(self, lambd=0.001, **kwargs):
        self.lambd = lambd
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelLogisticRegression, self).__init__(**kwargs)

    def fit(self, X, y, max_iter=100, tol=1e-5):
        n, p = X.shape
        assert (n == len(y))

        self.X_train = X
        self.y_train = y

        K = self.kernel_function_(X, X, **self.kernel_parameters)

        # IRLS
        KRR = KernelRidgeRegression(
            lambd=2*self.lambd,
            kernel=self.kernel_name,
            **self.kernel_parameters
        )
        # Initialize
        alpha = np.zeros(n)
        # Iterate until convergence or max iterations
        for n_iter in range(max_iter):
            alpha_old = alpha
            m = K.dot(alpha_old)
            w = sigmoid(m) * sigmoid(-m)
            z = m + self.y_train / sigmoid(self.y_train * m)
            alpha = KRR.fit(self.X_train, z, sample_weights=w).alpha
            # Break condition (achieved convergence)
            if np.sum((alpha-alpha_old)**2) < tol:
                break

        self.n_iter = n_iter
        self.alpha = alpha

        return self

    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)
        return sigmoid(K_x.dot(self.alpha))

    def predict(self, X):
        proba = self.decision_function(X)
        predicted_classes = np.where(proba < 0.5, -1, 1)
        return predicted_classes