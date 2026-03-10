### Implementation of HMM AR(1) model ###

# Estimation with forward probabilites #

import numpy as np

def simulate_rs_ar1(T, beta, sigma, P, pi=None, seed = None):
    pass


def transform_params (theta: np.ndarray) -> np.ndarray:
    pass

def obs_density (y_t, y_tm1, beta, sigma):
    pass

def forward_algorithm(y, beta1, beta2, sigma1, sigma2, p11, p22, pi1):
    pass

def neg_loglik (theta, y):
    pass

def fit_model(y, theta):
    pass

def filtered_probs(alpha):
    pass

