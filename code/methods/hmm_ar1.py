### Implementation of HMM AR(1) model ###
# Two states only #
# Estimation with forward probabilites #

import numpy as np

def simulate_rs_ar1(T: int, 
                    beta: np.ndarray, 
                    sigma: np.ndarray, 
                    P: np.ndarray, 
                    seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a two-state regime-switching AR(1) process. Based on code 
    provided by advisor.

    Args:
        T (int): Number of observations.
        beta (np.ndarray): AR(1) coefficients for each state (length 2).
        sigma (np.ndarray): Innovation standard deviations for each state (length 2).
        P (np.ndarray): 2x2 transition matrix for hidden states.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray]: Simulated observations y and hidden states.
    """
    # Set seed
    np.random.seed(seed)
    
    states = np.zeros(T, dtype=int)
    y = np.zeros(T)
    
    # Initial values:
    states[0] = np.random.choice([0, 1])
    y[0] = np.random.normal()
    
    for t in range(1, T):
        
        # Simulate next state
        states[t] = np.random.choice([0, 1], p=P[states[t-1]])
        
        # Simulate observation
        s = states[t]
        y[t] = beta[s]*y[t-1] + np.random.normal(scale=sigma[s])
    
    return y, states
    


def transform_params(theta: np.ndarray) -> tuple:
    """Transform unconstrained parameters.

    Args:
        theta (np.ndarray): Unconstrained parameter vector
            [b1, b2, eta1, eta2, a1, a2]

    Returns:
        tuple: Transformed parameters
            (beta1, beta2, sigma1, sigma2, p11, p22)
    """
    
    b1, b2, eta1, eta2, a1, a2 = theta
    
    # AR coefficients (-1, 1)
    beta1 = (1-np.exp(-b1)) / (1+np.exp(-b1))
    beta2 = (1-np.exp(-b2)) / (1+np.exp(-b2))
    
    # Volatilities (>0)
    sigma1 = np.exp(eta1)
    sigma2 = np.exp(eta2)
    
    # Transition probabilites (0, 1)
    p11 = 1 / (1 + np.exp(-a1))
    p22 = 1 / (1 + np.exp(-a2))
    
    return beta1, beta2, sigma1, sigma2, p11, p22

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

