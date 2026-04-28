### HMM AR(1) code generalized to K states. ###

### Implementation of HMM AR(1) model ###
# K hidden states #
# Estimation with forward probabilites #

import numpy as np
import numba

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

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
def log_obs_density(y_t, y_tm1, beta, sigma):
    residual = y_t - beta * y_tm1
    return -0.5 * np.log(2.0 * np.pi * sigma ** 2) - 0.5 * (residual / sigma) ** 2


@numba.jit(nopython=True)
def forward_algorithm(y: np.ndarray,
                      beta: np.ndarray,
                      sigma: np.ndarray,
                      P: np.ndarray,
                      pi=None):
    """Compute forward probabilities and log-likelihood in log-space.

    Numerically stable: avoids underflow for extreme observation scales.
    All log-sum-exp operations are inlined as scalar loops (no numpy array
    temporaries inside the hot path) and log(P) is precomputed once.
    The second return value (c) is a dummy kept for API compatibility.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: Normalized forward probabilities (T×K),
            dummy ones array, and log-likelihood.
    """
    K = len(beta)
    T = len(y)

    if pi is None:
        pi = np.ones(K) / K

    # Precompute log(P) once — avoids K*K*T log() calls inside the hot loop
    log_P = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            log_P[i, j] = np.log(P[i, j] + 1e-300)

    log_alpha = np.empty((T, K))
    loglik = 0.0

    # Initial step
    for i in range(K):
        log_alpha[0, i] = np.log(pi[i]) + log_obs_density(y[0], 0.0, beta[i], sigma[i])

    # Inline log-sum-exp (scalar, no array temporaries)
    max_val = log_alpha[0, 0]
    for i in range(1, K):
        if log_alpha[0, i] > max_val:
            max_val = log_alpha[0, i]
    s = 0.0
    for i in range(K):
        s += np.exp(log_alpha[0, i] - max_val)
    log_c = max_val + np.log(s)
    loglik += log_c
    for i in range(K):
        log_alpha[0, i] -= log_c

    # Recursion
    for t in range(1, T):
        for j in range(K):
            # Inline log-sum-exp over incoming transitions
            max_val = log_alpha[t-1, 0] + log_P[0, j]
            for i in range(1, K):
                v = log_alpha[t-1, i] + log_P[i, j]
                if v > max_val:
                    max_val = v
            s = 0.0
            for i in range(K):
                s += np.exp(log_alpha[t-1, i] + log_P[i, j] - max_val)
            log_alpha[t, j] = max_val + np.log(s) + log_obs_density(y[t], y[t-1], beta[j], sigma[j])

        # Inline log-sum-exp for normalisation
        max_val = log_alpha[t, 0]
        for j in range(1, K):
            if log_alpha[t, j] > max_val:
                max_val = log_alpha[t, j]
        s = 0.0
        for j in range(K):
            s += np.exp(log_alpha[t, j] - max_val)
        log_c = max_val + np.log(s)
        loglik += log_c
        for j in range(K):
            log_alpha[t, j] -= log_c

    alpha = np.exp(log_alpha)
    c = np.ones(T)
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
    assert len(y) >= 1, "Observed time series y must have length at least 1"
    
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
    
    # Bounds to prevent numerical issues: sigma_raw > -10 to avoid exp(sigma_raw) ≈ 0
    bounds = [(-np.inf, np.inf)] * K + [(-10, np.inf)] * K + [(-np.inf, np.inf)] * (K * K)
    
    def objective(theta):
        beta_raw = theta[:K]
        sigma_raw = theta[K:2*K]
        P_raw = theta[2*K:].reshape(K, K)
        try:
            val = neg_loglik(beta_raw, sigma_raw, P_raw, y)
            return val if np.isfinite(val) else 1e10
        except Exception:
            return 1e10
    
    result = minimize(
        fun=objective,
        x0 = theta0,
        bounds=bounds,
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


