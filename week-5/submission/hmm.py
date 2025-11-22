import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from scipy.stats import multivariate_normal

class CustomGaussianMixture:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, random_state=42):
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
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_indices]
        self.weights = np.ones(self.n_components) / self.n_components
        self.covariances = [np.cov(X.T) for _ in range(self.n_components)]

    def _compute_log_likelihood(self, X, means, covariances, weights):
        n_samples, _ = X.shape
        log_likelihood = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            mvn = multivariate_normal(mean=means[k], cov=covariances[k], allow_singular=True)
            log_likelihood[:, k] = np.log(weights[k] + 1e-10) + mvn.logpdf(X)
        return log_likelihood

    def fit(self, X):
        X = np.atleast_2d(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] == 1:
            X = X.reshape(-1, 1)
        self._initialize_parameters(X)
        for iteration in range(self.max_iter):
            log_likelihood = self._compute_log_likelihood(X, self.means, self.covariances, self.weights)
            log_sum = np.logaddexp.reduce(log_likelihood, axis=1)
            log_responsibilities = log_likelihood - log_sum[:, np.newaxis]
            responsibilities = np.exp(log_responsibilities)
            N_k = responsibilities.sum(axis=0)
            new_weights = N_k / N_k.sum()
            new_means = responsibilities.T @ X / N_k[:, np.newaxis]
            new_covariances = []
            for k in range(self.n_components):
                diff = X - new_means[k]
                cov = (responsibilities[:, k][:, np.newaxis] * diff).T @ diff / N_k[k]
                cov += np.eye(cov.shape[0]) * 1e-6
                new_covariances.append(cov)
            weight_diff = np.abs(new_weights - self.weights).max()
            mean_diff = np.abs(new_means - self.means).max()
            self.weights = new_weights
            self.means = new_means
            self.covariances = new_covariances
            if weight_diff < self.tol and mean_diff < self.tol:
                break
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        log_likelihood = self._compute_log_likelihood(X, self.means, self.covariances, self.weights)
        return np.argmax(log_likelihood, axis=1)

def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data is None or data.empty:
        raise ValueError(f"No data found for ticker {ticker} between {start_date} and {end_date}.")
    data = data[['Close']]
    data['Daily Returns'] = data['Close'].pct_change()
    data = data.dropna()
    return data

def plot_market_states(data, gmm, ticker, returns):
    plt.figure(figsize=(15, 10))
    states = gmm.predict(returns)
    for i in range(gmm.n_components):
        mask = states == i
        plt.plot(data.index[mask], data['Close'][mask], '.', label=f'State {i}')
    plt.title(f'Market States for {ticker} - Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return states

def analyze_market_states(states, gmm, returns, data):
    for i in range(gmm.n_components):
        mask = states == i
        _ = returns[mask]
        _ = data['Close'][mask]

def main():
    TICKER = "AAPL"
    START_DATE = "2010-01-01"
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    N_STATES = 3
    data = download_stock_data(TICKER, START_DATE, END_DATE)
    returns = data['Daily Returns'].to_numpy().reshape(-1, 1)
    gmm = CustomGaussianMixture(n_components=N_STATES)
    gmm.fit(returns)
    states = plot_market_states(data, gmm, TICKER, returns)
    analyze_market_states(states, gmm, returns, data)

if __name__ == "__main__":
    main()
