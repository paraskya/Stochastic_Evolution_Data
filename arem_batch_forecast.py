

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import warnings
import yfinance as yf
import os

warnings.filterwarnings('ignore')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define save paths
base_dir = "/content/drive/MyDrive/Stochastic_Evolution_Data"
script_dir = os.path.join(base_dir, "Python_Scripts")
plot_dir = os.path.join(base_dir, "arem_forecasts")
os.makedirs(script_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

tickers = {
    "^GSPC": "S&P 500",
    "CL=F": "Crude Oil",
    "^TNX": "10Y Treasury",
    "BTC-USD": "Bitcoin",
    "EURUSD=X": "EUR/USD"
}

start_date = "2002-01-01"
split_date = "2019-01-01"
end_date = "2025-03-01"

summary = []

def fetch_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty or 'Close' not in df:
        raise ValueError("Data download failed or 'Close' missing.")
    df = df[['Close']].dropna()
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    return df.dropna()

def rolling_ewma_log_returns(prices, alpha=0.1):
    prices = np.asarray(prices).flatten()  # Ensure 1D input
    log_prices = np.log(prices)
    log_returns = np.diff(log_prices, prepend=log_prices[0])
    return pd.Series(log_returns).ewm(alpha=alpha, adjust=False).mean().values

def simulate_process(S0, days, lambda_param, alpha_param, r, sigma, prices):
    dt = 1 / 252
    S = np.zeros(days)
    S[0] = S0
    np.random.seed(42)
    dW = np.random.normal(0, np.sqrt(dt), days - 1)
    
    # Ensure prices is 1D and has sufficient length
    prices = np.asarray(prices).flatten()
    if len(prices) < days - 1:
        raise ValueError("Insufficient price data for simulation")
        
    drift_series = rolling_ewma_log_returns(prices, alpha=alpha_param)[:days-1]
    
    for i in range(days - 1):
        drift = np.clip(drift_series[i], -0.1, 0.1)
        S[i + 1] = S[i] * np.exp(drift + sigma * dW[i])
    return S

def objective_function(params, prices):
    lambda_, alpha_, r, sigma = params
    if not (0.01 <= lambda_ <= 1.0 and 0.01 <= alpha_ <= 1.0 and -0.1 <= r <= 0.1 and 0.05 <= sigma <= 0.5):
        return 1e6
    
    try:
        simulated = simulate_process(prices[0], len(prices), lambda_, alpha_, r, sigma, prices)
    except ValueError as e:
        return 1e6  # Penalize invalid parameters
    
    norm_actual = prices / prices[0]
    norm_sim = simulated / simulated[0]
    return np.sqrt(np.mean((norm_actual - norm_sim) ** 2))

def calibrate_parameters(prices, returns):
    prices = np.asarray(prices).flatten()  # Ensure 1D
    init_guess = [0.1, 0.02, np.mean(returns), np.std(returns) * np.sqrt(252)]
    bounds = [(0.01, 1.0), (0.01, 1.0), (-0.1, 0.1), (0.05, 0.5)]
    result = optimize.minimize(objective_function, init_guess, args=(prices,), bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x

def run_forecast(ticker, label):
    try:
        print(f"Downloading {label} data...")
        df = fetch_data(ticker)
        df = df.loc[df.index >= start_date]
        train_df = df.loc[df.index < split_date]
        test_df = df.loc[df.index >= split_date]

        # Explicitly flatten arrays
        train_prices = train_df['Close'].values.flatten()
        train_returns = train_df['Log_Return'].values.flatten()
        test_prices = test_df['Close'].values.flatten()
        test_index = test_df.index

        if len(train_prices) < 2 or len(test_prices) < 2:
            raise ValueError("Insufficient data for calibration")

        lambda_, alpha_, r, sigma = calibrate_parameters(train_prices, train_returns)
        simulated = simulate_process(train_prices[-1], len(test_prices), lambda_, alpha_, r, sigma, test_prices)

        norm_actual = (test_prices / test_prices[0]).flatten()
        norm_sim = (simulated / simulated[0]).flatten()

        if len(norm_actual) != len(norm_sim):
            raise ValueError("Length mismatch between actual and simulated")

        corr = np.corrcoef(norm_actual, norm_sim)[0, 1]
        rmse = np.sqrt(np.mean((norm_actual - norm_sim) ** 2))

        # Save results
        forecast_df = pd.DataFrame({
            'Date': test_index,
            'Actual': norm_actual,
            'Forecast': norm_sim
        })
        forecast_df.to_csv(os.path.join(plot_dir, f"{ticker}_forecast.csv"), index=False)

        plt.figure(figsize=(12, 6))
        plt.plot(test_index, norm_actual, label='Actual')
        plt.plot(test_index, norm_sim, label='Forecast', alpha=0.7)
        plt.title(f'{label}: Corr={corr:.4f}, RMSE={rmse:.4f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f"{ticker}_plot.png"))
        plt.close()

        summary.append({
            'Ticker': ticker,
            'Label': label,
            'Lambda': lambda_,
            'Alpha': alpha_,
            'r': r,
            'Sigma': sigma,
            'Correlation': corr,
            'RMSE': rmse
        })
        print(f"Completed {label} successfully")

    except Exception as e:
        print(f"Failed for {label}: {str(e)}")

# Execute forecasts
for ticker, label in tickers.items():
    run_forecast(ticker, label)

# Save summary
summary_df = pd.DataFrame(summary)
summary_path = os.path.join(plot_dir, "summary.csv")
summary_df.to_csv(summary_path, index=False)

print("✅ All processes completed")
print(f"Results saved to: {plot_dir}")
