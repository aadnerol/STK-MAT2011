### HMM AR(1) code generalized to K states. ###

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
        beta (np.ndarray): AR(1) coefficients for each state (length K).
        sigma (np.ndarray): Innovation standard deviations for each state (length K).
        P (np.ndarray): KxK transition matrix for hidden states.
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

def obs_density(y_t: float, 
                 y_tm1: float, 
                 beta: float, 
                 sigma: float) -> float:
    """Compute the state-dependent Gaussian observation density

    Args:
        y_t (float): Observation at time t.
        y_tm1 (float): Observation at time t-1.
        beta (float): AR(1) coefficient for current state.
        sigma (float): Innovation standard deviation for the current state.

    Returns:
        float: Value of f(y_t | s_t, y_t-1)
    """
    return 1 / (np.sqrt(2 * np.pi * sigma**2)) * \
        np.exp(- (y_t - beta*y_tm1)**2 / (2 * sigma**2))

def forward_algorithm(y: np.ndarray, 
                      beta1: float, 
                      beta2: float, 
                      sigma1: float, 
                      sigma2: float, 
                      p11: float, 
                      p22: float, 
                      pi1: float = 0.5):
    """Compute scaled forward probabilities and log-likelihood

    Args:
        y (np.ndarray): Observed time series.
        beta1 (float): AR coefficient in state 0.
        beta2 (float): AR coefficient in state 1.
        sigma1 (float): Innovation std. deviation in state 0.
        sigma2 (float): Innovation std. deviation in state 1.
        p11 (float): Probability of staying in state 0.
        p22 (float): Probability of staying in state 1. 
        pi1 (float, optional): Initial probability of state 0. Defaults to 0.5.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: Scaled forward probabilites, 
        scaling factors, log-likelihood.
    """
    T = len(y)
    
    p12 = 1 - p11
    p21 = 1 - p22
    pi2 = 1 - pi1
    
    alpha = np.zeros((T, 2))
    c = np.zeros(T)
    
    # Initial step
    f1 = obs_density(y[0], 0.0, beta1, sigma1)
    f2 = obs_density(y[0], 0.0, beta2, sigma2)
    
    alpha[0, 0] = pi1 * f1
    alpha[0, 1] = pi2 * f2
    
    c[0] = alpha[0, 0] + alpha[0, 1]
    alpha[0, :] /= c[0]
    
    # Recursion
    for t in range(1, T):
        f1 = obs_density(y[t], y[t-1], beta1, sigma1)
        f2 = obs_density(y[t], y[t-1], beta2, sigma2)
        
        alpha[t, 0] = (alpha[t-1, 0] * p11 + alpha[t-1, 1] * p21) * f1
        alpha[t, 1] = (alpha[t-1, 0] * p12 + alpha[t-1, 1] * p22) * f2
        
        c[t] = alpha[t, 0] + alpha[t, 1]
        alpha[t, :] /= c[t]
    
    loglik = np.sum(np.log(c))
    
    return alpha, c, loglik
 
def neg_loglik (theta: np.ndarray, y: np.ndarray) -> float:
    """Compute negative log-likelihood for optimization.

    Args:
        theta (np.ndarray): Unconstrained parameter vector.
        y (np.ndarray): Observed time series.

    Returns:
        float: Negative log-likelihood.
    """
    beta1, beta2, sigma1, sigma2, p11, p22 = transform_params(theta)
    
    alpha_local, c_local, loglik = forward_algorithm(
        y=y,
        beta1=beta1,
        beta2=beta2,
        sigma1=sigma1,
        sigma2=sigma2,
        p11=p11,
        p22=p22,
        pi1=0.5
    )
    return -loglik

from scipy.optimize import minimize
def fit_model(y: np.ndarray, theta0: np.ndarray, method: str = "L-BFGS-B"):
    """Estimate model parameters by maximum likelihood.

    Args:
        y (np.ndarray): Observed time series.
        theta0 (np.ndarray): Initial guess for unconstrained parameters.
        method (str, optional): Optimization method. Defaults to "L-BFGS-B".

    Returns:
        tuple: Optimization result object and transformed parameter estimates.
    """
    
    result = minimize(
        fun=neg_loglik,
        x0 = theta0,
        args = (y,),
        method=method
    )
    
    beta1, beta2, sigma1, sigma2, p11, p22 = transform_params(result.x)
    
    params_hat = {
        "beta1": beta1,
        "beta2": beta2,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "p11": p11,
        "p22": p22,
        "p12": 1 - p11,
        "p21": 1 - p22
    }
    return result, params_hat

def filtered_probs(alpha: np.ndarray) -> np.ndarray:
    return alpha

