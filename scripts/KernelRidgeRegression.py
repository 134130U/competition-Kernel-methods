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