import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from joblib import Parallel, delayed

from methods.hmm_ar_1_k_states import fit_model, forward_algorithm, transform_params


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def empirical_coverage(y_true, lower, upper):
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    return np.mean((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask]))


def average_interval_width(lower, upper):
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    mask = np.isfinite(lower) & np.isfinite(upper)
    return np.mean(upper[mask] - lower[mask])


def interval_score(y_true, lower, upper, alpha=0.05):
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    width = upper - lower
    penalty_low = (2 / alpha) * np.maximum(lower - y_true, 0)
    penalty_high = (2 / alpha) * np.maximum(y_true - upper, 0)
    return np.mean(width + penalty_low + penalty_high)


def evaluate_predictions(y_true, pred_mean, lower, upper, alpha=0.05):
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
    y = np.asarray(y)
    x = y[:-1]
    z = y[1:]
    rho_hat = np.sum(x * z) / np.sum(x ** 2)
    sigma_hat = np.sqrt(np.mean((z - rho_hat * x) ** 2))
    return rho_hat, sigma_hat


def predict_single_ar1(y_train, y_test, rho_hat, sigma_hat, alpha=0.05):
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

def _single_hmm_start(y_arr, K, sigma0_center, seed):
    rng = np.random.default_rng(seed)
    beta0 = rng.normal(0.0, 0.5, size=K)
    sigma0 = sigma0_center + rng.normal(0.0, 0.5, size=K)
    P0 = rng.normal(0.0, 0.5, size=(K, K))
    try:
        result, params_hat = fit_model(y_arr, beta0, sigma0, P0)
        loglik = -result.fun
        if np.isfinite(loglik) and result.fun < 1e9:
            return result, params_hat, loglik
    except Exception:
        pass
    return None, None, -np.inf


def fit_hmm_robust(y, K, n_starts=10, seed=123, n_jobs=-1):
    """Fit a K-state HMM-AR(1) with parallel random initializations.

    Runs n_starts optimizations in parallel and returns the result with
    the highest log-likelihood.
    """
    y_arr = np.asarray(y, dtype=float)
    sigma0_center = np.log(np.std(y_arr) + 1e-6)
    seeds = [seed + i for i in range(n_starts)]

    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_single_hmm_start)(y_arr, K, sigma0_center, s)
        for s in seeds
    )

    best_result, best_params, best_loglik = None, None, -np.inf
    for result, params_hat, loglik in outputs:
        if result is not None and loglik > best_loglik:
            best_result = result
            best_params = params_hat
            best_loglik = loglik

    if best_params is None:
        raise RuntimeError("HMM fitting failed for all random starts.")

    return best_result, best_params


def get_filtered_probs(y, params_hat):
    alpha, _, _ = forward_algorithm(
        y=np.asarray(y),
        beta=np.asarray(params_hat["beta"]),
        sigma=np.asarray(params_hat["sigma"]),
        P=np.asarray(params_hat["P"]),
        pi=None,
    )
    return alpha


def predict_hmm_mixture(y_train, y_test, params_hat, alpha=0.05):
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
