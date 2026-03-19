### HMM AR(1) code generalized to K states. ###

### Implementation of HMM AR(1) model ###
# K hidden states #
# Estimation with forward probabilites #

import numpy as np

def simulate_rs_ar1(T: int, 
                    beta: np.ndarray, 
                    sigma: np.ndarray, 
                    P: np.ndarray, 
                    seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a K-state regime-switching AR(1) process. Based on code 
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
    # Assert correct dimensions:
    assert beta.shape == sigma.shape, "Model parameter vectors have different length."
    n, m = P.shape
    assert n==m, "Transition matrix is not square"
    assert n==beta.shape[0], "Transition matrix dimensions does not correspond with parameter vector length."
    assert np.allclose(P.sum(axis=1), 1.0), "Each row of the transition matrix must sum to 1."
    # Set seed
    np.random.seed(seed)
    
    # Set number of hidden states:
    K = len(beta)
    
    # Initialize states and y 
    states = np.zeros(T, dtype=int)
    y = np.zeros(T)
    
    # Initial values:
    states[0] = np.random.choice(K, p=np.ones(K)/K)
    y[0] = np.random.normal()
    
    for t in range(1, T):
        
        # Simulate next state
        states[t] = np.random.choice(K, p=P[states[t-1]])
        
        # Simulate observation
        s = states[t]
        y[t] = beta[s]*y[t-1] + np.random.normal(scale=sigma[s])
    
    return y, states
    

def transform_params(beta_raw: np.ndarray, 
                     sigma_raw: np.ndarray,
                     P_raw: np.ndarray) -> tuple:
    """Transform unconstrained parameters.

    Args:
        beta_raw (np.ndarray): Unconstrained beta vector
        sigma_raw (np.ndarray): Unconstrained sigma vector
        P_raw (np.ndarray): Unconstrained transition matrix

    Returns:
        tuple: 
            beta: Transformed beta vector.
            sigma: Transformed sigma vector.
            P: Transformed transition matrix.
    """
    # Assertions
    assert beta_raw.shape == sigma_raw.shape, "Model parameter vectors have different length."
    n, m = P_raw.shape
    assert n==m, "Transition matrix is not square"
    assert n==beta_raw.shape[0], "Transition matrix dimensions does not correspond with parameter vector length." 
    
    K = len(beta_raw)   
    
    
    # AR coefficients (-1, 1)
    beta = (1-np.exp(-beta_raw)) / (1 + np.exp(-beta_raw)) 
    
    # Volatilities (>0)
    sigma = np.exp(sigma_raw)
    
    # Transition probabilites 
    P = np.zeros((K, K))
    for i in range(K):
        row = np.exp(P_raw[i])
        P[i] = row / np.sum(row)

    
    return beta, sigma, P

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
                      beta: np.ndarray, 
                      sigma: np.ndarray,                    
                      P: np.ndarray, 
                      pi: np.ndarray | None = None):
    """Compute scaled forward probabilities and log-likelihood

    Args:
        y (np.ndarray): Observed time series
        beta (np.ndarray): AR coefficients for each state (length K)
        sigma (np.ndarray): Innovation standard deviations for each state (length K)
        P (np.ndarray): KxK transition matrix
        pi (np.ndarray | None, optional): Initial state distribution. 
            Defaults to None. If None, a uniform distribution is used.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: Scaled forward probabilities, 
            scaling factors and log-likelihood.
    """
    # Assertions:
    assert len(y) >= 1, "Observed time series y must have length at least 1"
    assert beta.shape == sigma.shape, "Model parameter vectors have different length."
    n, m = P.shape
    assert n==m, "Transition matrix is not square"
    assert n==beta.shape[0], "Transition matrix dimensions does not correspond with parameter vector length."
    
    K = len(beta)
    
    if pi is None:
        pi = np.ones(K) / K
    else:
        assert pi.shape == (K,), "Initial distribution pi must have length K."
        assert np.allclose(pi.sum(), 1.0), "Initial distribution pi must sum to 1."
        
    T = len(y)
    alpha = np.zeros((T, K))
    c = np.zeros(T)
    
    # Initial step
    for i in range(K):
        alpha[0, i] = pi[i] * obs_density(y[0], 0.0, beta[i], sigma[i])
    
    
    c[0] = np.sum(alpha[0, :])
    alpha[0, :] /= c[0]
    
    # Recursion
    for t in range(1, T):
        for j in range(K):
            f_j = obs_density(y[t], y[t-1], beta[j], sigma[j])
            alpha[t, j] = np.sum(alpha[t-1, :] * P[:, j]) * f_j
        
        c[t] = np.sum(alpha[t, :])
        alpha[t, :] /= c[t]
    
    loglik = np.sum(np.log(c))
    
    return alpha, c, loglik

def neg_loglik(beta: np.ndarray, 
                sigma: np.ndarray, 
                P: np.ndarray,
                y: np.ndarray) -> float:
    """Compute negative log-likelihood for optimization.

    Args:
        beta (np.ndarray): Unconstrained beta vector. 
        sigma (np.ndarray): Unconstrained sigma vector.
        P (np.ndarray): Unconstrained transition matrix.
        y (np.ndarray): Observed time series. 

    Returns:
        float: Negative log-likelihood. 
    """
    beta, sigma, P = transform_params(beta, sigma, P)
    
    alpha_local, c_local, loglik = forward_algorithm(
        y=y,
        beta=beta,
        sigma=sigma,
        P = P,
        pi = None
    )
    return -loglik

from scipy.optimize import minimize

#moderat oppdatering
def fit_model(y: np.ndarray, 
              beta0: np.ndarray, 
              sigma0: np.ndarray,
              P0: np.ndarray,
              method: str = "L-BFGS-B"):
    """Estimate model parameters by maximum likelihood.

    Args:
        y (np.ndarray): Observed time series.
        beta0 (np.ndarray): Initial guess for unconstrained beta.
        sigma0 (np.ndarray): Initial guess for unconstrained sigma.
        P0 (np.ndarray): Initial guess for unconstrained transition matrix.
        method (str, optional): Optimization method. Defaults to "L-BFGS-B".

    Returns:
        tuple: Optimization result object and transformed parameter estimates.
    """
    K = len(beta0)
    
    theta0 = np.concatenate([beta0, sigma0, P0.ravel()])
    
    def objective(theta):
        beta_raw = theta[:K]
        sigma_raw = theta[K:2*K]
        P_raw = theta[2*K:].reshape(K, K)

        return neg_loglik(beta_raw, sigma_raw, P_raw, y)
    
    result = minimize(
        fun=objective,
        x0 = theta0,
        method=method
    )
    
    beta_raw_hat = result.x[:K]
    sigma_raw_hat = result.x[K:2*K]
    P_raw_hat = result.x[2*K:].reshape(K, K)

    beta_hat, sigma_hat, P_hat = transform_params(beta_raw_hat, sigma_raw_hat, P_raw_hat)

    params_hat = {
        "beta": beta_hat,
        "sigma": sigma_hat,
        "P": P_hat
    }

    return result, params_hat

# Kan stå men er en teit funksjon
def filtered_probs(alpha: np.ndarray) -> np.ndarray:
    return alpha

