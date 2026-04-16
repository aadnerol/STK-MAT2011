import numpy as np
from scipy.stats import norm
from methods.hmm_ar_1_k_states import fit_model, forward_algorithm, \
    transform_params


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_rmse(y_true, y_pred):
    """Compute root mean squared error, ignoring non-finite values.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def empirical_coverage(y_true, lower, upper):
    """Compute fraction of observations falling inside the prediction interval.

    Parameters
    ----------
    y_true : array-like
    lower : array-like
    upper : array-like

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    return np.mean((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask]))


def average_interval_width(lower, upper):
    """Compute average width of prediction intervals.

    Parameters
    ----------
    lower : array-like
    upper : array-like

    Returns
    -------
    float
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    mask = np.isfinite(lower) & np.isfinite(upper)
    return np.mean(upper[mask] - lower[mask])


def interval_score(y_true, lower, upper, alpha=0.05):
    """Compute the mean interval score (Gneiting & Raftery, 2007).

    A strictly proper scoring rule for prediction intervals. Lower is better.
    Penalizes wide intervals and observations outside the interval.

    IS = (u - l) + (2/alpha) * (l - y) * 1(y < l)
                 + (2/alpha) * (y - u) * 1(y > u)

    Parameters
    ----------
    y_true : array-like
    lower : array-like
    upper : array-like
    alpha : float
        Significance level matching the interval. Default 0.05 for 95% intervals.

    Returns
    -------
    float
        Mean interval score across all observations.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_low = (2 / alpha) * np.maximum(lower - y_true, 0)
    penalty_high = (2 / alpha) * np.maximum(y_true - upper, 0)

    return np.mean(width + penalty_low + penalty_high)


def evaluate_predictions(y_true, pred_mean, lower, upper, alpha=0.5):
    """Compute RMSE, empirical coverage, and average interval width.

    Parameters
    ----------
    y_true : array-like
    pred_mean : array-like
    lower : array-like
    upper : array-like

    Returns
    -------
    dict with keys 'rmse', 'coverage', 'avg_width'
    """
    return {
        "rmse": compute_rmse(y_true, pred_mean),
        "coverage": empirical_coverage(y_true, lower, upper),
        "avg_width": average_interval_width(lower, upper),
        "interval_score": interval_score(y_true, lower, upper, alpha=alpha),
    }


# ---------------------------------------------------------------------------
# Single AR(1)
# ---------------------------------------------------------------------------

def fit_single_ar1(y):
    """Fit a single AR(1) model by OLS.

    Estimates rho and sigma from y_t = rho * y_{t-1} + eps_t.

    Parameters
    ----------
    y : array-like

    Returns
    -------
    rho_hat : float
    sigma_hat : float
    """
    y = np.asarray(y)
    x = y[:-1]
    z = y[1:]
    rho_hat = np.sum(x * z) / np.sum(x ** 2)
    sigma_hat = np.sqrt(np.mean((z - rho_hat * x) ** 2))
    return rho_hat, sigma_hat


def predict_single_ar1(y_train, y_test, rho_hat, sigma_hat, alpha=0.05):
    """Produce one-step-ahead predictions and prediction intervals for AR(1).

    Prediction interval: rho * y_t +/- z_{alpha/2} * sigma

    Parameters
    ----------
    y_train : array-like
        Training data; last value used as starting point.
    y_test : array-like
        Test data.
    rho_hat : float
    sigma_hat : float
    alpha : float
        Significance level. Default 0.05 gives 95% intervals.

    Returns
    -------
    pred_mean : ndarray
    lower : ndarray
    upper : ndarray
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    z_alpha = norm.ppf(1 - alpha / 2)

    pred_mean = np.zeros(len(y_test))
    lower = np.zeros(len(y_test))
    upper = np.zeros(len(y_test))
    prev_y = y_train[-1]

    for t in range(len(y_test)):
        mu = rho_hat * prev_y
        pred_mean[t] = mu
        lower[t] = mu - z_alpha * sigma_hat
        upper[t] = mu + z_alpha * sigma_hat
        prev_y = y_test[t]

    return pred_mean, lower, upper


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

def fit_hmm_robust(y, K, n_starts=10, seed=123):
    """Fit a K-state HMM-AR(1) model with multiple random initializations.

    Runs the optimizer n_starts times with random initial values and returns
    the result with the highest log-likelihood.

    Parameters
    ----------
    y : array-like
    K : int
        Number of hidden states.
    n_starts : int
        Number of random initializations. Default 10.
    seed : int
        Random seed for reproducibility. Default 123.

    Returns
    -------
    result : OptimizeResult
    params_hat : dict with keys 'beta', 'sigma', 'P'

    Raises
    ------
    RuntimeError
        If all optimization runs fail.
    """
    rng = np.random.default_rng(seed)
    best_result = None
    best_params = None
    best_loglik = -np.inf

    for _ in range(n_starts):
        beta0 = rng.normal(0.0, 0.5, size=K)
        sigma0 = rng.normal(0.0, 0.3, size=K)
        P0 = rng.normal(0.0, 0.5, size=(K, K))

        try:
            result, params_hat = fit_model(y, beta0, sigma0, P0)
            loglik = -result.fun
            if result.success and np.isfinite(loglik) and loglik > best_loglik:
                best_result = result
                best_params = params_hat
                best_loglik = loglik
        except Exception:
            pass

    if best_params is None:
        raise RuntimeError("HMM fitting failed for all random starts.")

    return best_result, best_params


def get_filtered_probs(y, params_hat):
    """Run the forward algorithm and return filtered state probabilities.

    Parameters
    ----------
    y : array-like
    params_hat : dict with keys 'beta', 'sigma', 'P'

    Returns
    -------
    alpha : ndarray of shape (T, K)
        Filtered probabilities P(S_t = k | y_1, ..., y_t).
    """
    alpha, _, _ = forward_algorithm(
        y=np.asarray(y),
        beta=np.asarray(params_hat["beta"]),
        sigma=np.asarray(params_hat["sigma"]),
        P=np.asarray(params_hat["P"]),
        pi=None,
    )
    return alpha


def predict_hmm_hard(y_train, y_test, params_hat, alpha=0.05):
    """One-step-ahead predictions using the most likely next state (hard switch).

    At each step, predicts using the single state with the highest predicted
    probability. Ignores state uncertainty in the interval width.

    Parameters
    ----------
    y_train : array-like
    y_test : array-like
    params_hat : dict with keys 'beta', 'sigma', 'P'
    alpha : float
        Significance level. Default 0.05 gives 95% intervals.

    Returns
    -------
    pred_mean : ndarray
    lower : ndarray
    upper : ndarray
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    beta = np.asarray(params_hat["beta"])
    sigma = np.asarray(params_hat["sigma"])
    P = np.asarray(params_hat["P"])
    z_alpha = norm.ppf(1 - alpha / 2)

    state_probs_t = get_filtered_probs(y_train, params_hat)[-1]
    pred_mean = np.zeros(len(y_test))
    lower = np.zeros(len(y_test))
    upper = np.zeros(len(y_test))
    prev_y = y_train[-1]

    for t in range(len(y_test)):
        state_probs_next = state_probs_t @ P
        k = np.argmax(state_probs_next)
        mu = beta[k] * prev_y
        sd = sigma[k]

        pred_mean[t] = mu
        lower[t] = mu - z_alpha * sd
        upper[t] = mu + z_alpha * sd

        y_obs = y_test[t]
        emission = np.array([
            norm.pdf(y_obs, loc=beta[j] * prev_y, scale=sigma[j])
            for j in range(len(beta))
        ])
        numer = state_probs_next * emission
        denom = numer.sum()
        state_probs_t = numer / denom if denom > 0 and np.isfinite(denom) else state_probs_next / state_probs_next.sum()
        prev_y = y_obs

    return pred_mean, lower, upper


def predict_hmm_mixture(y_train, y_test, params_hat, alpha=0.05):
    """One-step-ahead predictions using a Gaussian mixture over all states.

    Combines predictions from all states weighted by their predicted
    probabilities. The interval is based on the mean and variance of the
    implied Gaussian mixture (law of total variance), accounting for
    state uncertainty.

    Parameters
    ----------
    y_train : array-like
    y_test : array-like
    params_hat : dict with keys 'beta', 'sigma', 'P'
    alpha : float
        Significance level. Default 0.05 gives 95% intervals.

    Returns
    -------
    pred_mean : ndarray
    lower : ndarray
    upper : ndarray
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    beta = np.asarray(params_hat["beta"])
    sigma = np.asarray(params_hat["sigma"])
    P = np.asarray(params_hat["P"])
    z_alpha = norm.ppf(1 - alpha / 2)

    state_probs_t = get_filtered_probs(y_train, params_hat)[-1]
    pred_mean = np.zeros(len(y_test))
    lower = np.zeros(len(y_test))
    upper = np.zeros(len(y_test))
    prev_y = y_train[-1]

    for t in range(len(y_test)):
        w = state_probs_t @ P
        means = beta * prev_y
        mix_mean = np.sum(w * means)
        mix_var = np.sum(w * (sigma ** 2 + means ** 2)) - mix_mean ** 2
        mix_sd = np.sqrt(max(mix_var, 0.0))

        pred_mean[t] = mix_mean
        lower[t] = mix_mean - z_alpha * mix_sd
        upper[t] = mix_mean + z_alpha * mix_sd

        y_obs = y_test[t]
        emission = np.array([
            norm.pdf(y_obs, loc=beta[j] * prev_y, scale=sigma[j])
            for j in range(len(beta))
        ])
        numer = w * emission
        denom = numer.sum()
        state_probs_t = numer / denom if denom > 0 and np.isfinite(denom) else w / w.sum()
        prev_y = y_obs

    return pred_mean, lower, upper


def fit_hmm_multiday(segments, K, n_starts=10, seed=123):
    """Fit a K-state HMM-AR(1) by summing log-likelihoods across independent segments.

    Each segment is treated as independent (e.g. one trading day).
    Total log-likelihood = sum of per-segment log-likelihoods.

    Parameters
    ----------
    segments : list of array-like
    K : int
    n_starts : int
    seed : int

    Returns
    -------
    result : OptimizeResult
    params_hat : dict with keys 'beta', 'sigma', 'P'
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    segs = [np.asarray(s, dtype=float) for s in segments]
    best_result = None
    best_params = None
    best_loglik = -np.inf

    def objective(theta):
        beta_raw = theta[:K]
        sigma_raw = theta[K:2 * K]
        P_raw = theta[2 * K:].reshape(K, K)
        beta, sigma, P = transform_params(beta_raw, sigma_raw, P_raw)
        total = 0.0
        for seg in segs:
            _, _, ll = forward_algorithm(seg, beta, sigma, P, pi=None)
            total += ll
        return -total

    bounds = (
        [(-np.inf, np.inf)] * K
        + [(-10, np.inf)] * K
        + [(-np.inf, np.inf)] * (K * K)
    )

    for _ in range(n_starts):
        beta0 = rng.normal(0.0, 0.5, size=K)
        sigma0 = rng.normal(0.0, 0.3, size=K)
        P0 = rng.normal(0.0, 0.5, size=(K, K))
        theta0 = np.concatenate([beta0, sigma0, P0.ravel()])
        try:
            result = minimize(objective, theta0, bounds=bounds, method="L-BFGS-B")
            loglik = -result.fun
            if result.success and np.isfinite(loglik) and loglik > best_loglik:
                best_result = result
                best_loglik = loglik
                b_hat, s_hat, P_hat = transform_params(
                    result.x[:K], result.x[K:2*K], result.x[2*K:].reshape(K, K)
                )
                best_params = {"beta": b_hat, "sigma": s_hat, "P": P_hat}
        except Exception:
            pass

    if best_params is None:
        raise RuntimeError("HMM fitting failed for all random starts.")

    return best_result, best_params
