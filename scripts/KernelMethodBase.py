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