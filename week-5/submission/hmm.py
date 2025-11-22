import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from scipy.stats import multivariate_normal
import matplotlib.dates as mdates

class CustomGaussianMixture:
    def __init__(self, n_components=3, max_iter=200, tol=1e-6, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights = None
        self.means = None
        self.covariances = None

    def _initialize_parameters(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        if n_samples < self.n_components:
            raise ValueError("n_components must be <= n_samples")
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_indices].astype(float)
        self.weights = np.ones(self.n_components, dtype=float) / self.n_components
        emp_cov = np.cov(X.T) if n_samples > 1 else np.eye(n_features)
        self.covariances = [emp_cov.copy() + np.eye(n_features) * 1e-6 for _ in range(self.n_components)]

    def _compute_log_likelihood(self, X, means, covariances, weights):
        n_samples, _ = X.shape
        log_likelihood = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            mvn = multivariate_normal(mean=means[k].ravel(), cov=covariances[k], allow_singular=True)
            log_likelihood[:, k] = np.log(weights[k] + 1e-12) + mvn.logpdf(X)
        return log_likelihood

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        self._initialize_parameters(X)
        for iteration in range(self.max_iter):
            log_likelihood = self._compute_log_likelihood(X, self.means, self.covariances, self.weights)
            log_sum = np.logaddexp.reduce(log_likelihood, axis=1)
            log_responsibilities = log_likelihood - log_sum[:, np.newaxis]
            responsibilities = np.exp(log_responsibilities)
            N_k = responsibilities.sum(axis=0) + 1e-12
            new_weights = N_k / N_k.sum()
            new_means = (responsibilities.T @ X) / N_k[:, np.newaxis]
            new_covariances = []
            for k in range(self.n_components):
                diff = X - new_means[k]
                weighted = (responsibilities[:, k][:, np.newaxis] * diff).T @ diff
                cov = weighted / N_k[k]
                cov += np.eye(n_features) * 1e-6
                new_covariances.append(cov)
            weight_diff = np.max(np.abs(new_weights - self.weights))
            mean_diff = np.max(np.abs(new_means - self.means))
            self.weights = new_weights
            self.means = new_means
            self.covariances = new_covariances
            if weight_diff < self.tol and mean_diff < self.tol:
                break
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        log_likelihood = self._compute_log_likelihood(X, self.means, self.covariances, self.weights)
        return np.argmax(log_likelihood, axis=1)

def download_stock_data(ticker, start_date, end_date, use_adj=False):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data is None or data.empty:
        raise ValueError(f"No data found for ticker {ticker} between {start_date} and {end_date}.")
    col = 'Adj Close' if (use_adj and 'Adj Close' in data.columns) else 'Close'
    data = data[[col]].rename(columns={col: 'Price'})
    data['Returns'] = data['Price'].pct_change()
    data = data.dropna()
    return data

def compute_transition_matrix(states, n_states):
    T = np.zeros((n_states, n_states), dtype=float)
    for (s_from, s_to) in zip(states[:-1], states[1:]):
        T[s_from, s_to] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        T_norm = np.divide(T, row_sums, where=(row_sums != 0))
    T_norm[np.isnan(T_norm)] = 0.0
    return T_norm

def plot_price_by_state(data, states, n_states, ticker):
    fig, ax = plt.subplots(figsize=(14,5))
    cmap = plt.get_cmap('tab10')
    for s in range(n_states):
        mask = (states == s)
        ax.plot(data.index[mask], data['Price'][mask], '.', color=cmap(s % 10), label=f'State {s}')
    ax.set_title(f'{ticker} Price colored by inferred GMM state')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.tight_layout()
    plt.show()

def plot_returns_by_state(data, states, n_states, ticker):
    fig, ax = plt.subplots(figsize=(14,5))
    cmap = plt.get_cmap('tab10')
    for s in range(n_states):
        mask = (states == s)
        ax.plot(data.index[mask], data['Returns'][mask], '.', color=cmap(s % 10), label=f'State {s}')
    ax.set_title(f'{ticker} Daily Returns colored by inferred GMM state')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Return')
    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_transition_matrix(T):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(T, cmap='viridis', interpolation='nearest')
    ax.set_xlabel('To state')
    ax.set_ylabel('From state')
    ax.set_title('Empirical state transition matrix')
    plt.colorbar(im, ax=ax, label='Transition probability')
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            ax.text(j, i, f'{T[i,j]:.2f}', ha='center', va='center', color='white' if T[i,j]>0.5 else 'black')
    fig.tight_layout()
    plt.show()

def compute_state_statistics(states, returns, data):
    n_states = int(states.max())+1
    stats = []
    # returns is expected as numpy array shape (n,1) or (n,)
    returns_flat = returns.ravel()
    for s in range(n_states):
        mask = (states == s)
        count = int(mask.sum())
        if count > 0:
            mean_ret = float(np.mean(returns_flat[mask]))
            vol = float(np.std(returns_flat[mask], ddof=0))
            avg_price = float(data['Price'].loc[mask].mean())
        else:
            mean_ret = float('nan')
            vol = float('nan')
            avg_price = float('nan')
        stats.append((count, mean_ret, vol, avg_price))
    return stats

def main():
    TICKER = "AAPL"
    START_DATE = "2010-01-01"
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    N_STATES = 3

    data = download_stock_data(TICKER, START_DATE, END_DATE, use_adj=False)
    returns = data['Returns'].to_numpy().reshape(-1, 1)

    gmm = CustomGaussianMixture(n_components=N_STATES, max_iter=300, tol=1e-6, random_state=0)
    gmm.fit(returns)
    states = gmm.predict(returns)

    weights = getattr(gmm, "weights", None)
    if weights is None:
        print("GMM Weights: None")
    else:
        weights_arr = np.asarray(weights, dtype=float)
        print("GMM Weights:", np.round(weights_arr, 4))
    means = getattr(gmm, "means", None)
    if means is None:
        print("GMM Means: None")
    else:
        try:
            means_array = np.asarray(means, dtype=float)
            print("GMM Means:", np.round(means_array.ravel(), 6))
        except Exception as e:
            print("GMM Means: (error formatting means)", e)
    covs_raw = getattr(gmm, "covariances", None)
    if covs_raw is None:
        print("GMM Covariances (showing trace/shape): None")
    else:
        covs = [np.round(cov, 6) for cov in covs_raw]
        print("GMM Covariances (showing trace/shape):", [np.trace(cov) for cov in covs])

    T = compute_transition_matrix(states, N_STATES)
    print("\nEmpirical Transition Matrix (rows from -> cols to):\n", np.round(T, 4))

    stats = compute_state_statistics(states, returns, data)
    for i, (count, mean_ret, vol, avg_price) in enumerate(stats):
        # all values are Python floats / ints now so formatting is safe
        print(f"State {i}: count={count}, mean_return={mean_ret:.6f}, vol={vol:.6f}, avg_price={avg_price:.4f}")

    plot_price_by_state(data, states, N_STATES, TICKER)
    plot_returns_by_state(data, states, N_STATES, TICKER)
    plot_transition_matrix(T)

if __name__ == "__main__":
    main()
