import datetime

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from skfolio.datasets import load_sp500_dataset, load_sp500_implied_vol_dataset


def plot_volatility_and_variance(ticker, start_date, end_date, window=30):
    """
    This function downloads stock data for a given ticker and date range,
    calculates the realized volatility and variance, and plots them.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.
    window (int): The rolling window size for calculating volatility and variance.

    Returns:
    None
    """
    # Download stock data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate daily returns and squared returns
    data["Return"] = data["Close"].pct_change()
    data["Squared_Return"] = data["Return"] ** 2

    # Calculate realized volatility and variance
    data["Realized_Volatility"] = (
        data["Squared_Return"].rolling(window=window).sum() ** 0.5
    )

    data["Realized_Variance"] = data["Squared_Return"].rolling(window=window).sum()

    data["rolling_return"] = data["Return"].rolling(window=window).mean()

    # Calculate mean return
    mean_return = data["Return"].mean()

    # Calculate annualized realized volatility
    realized_volatility = data["Return"].std() * (252**0.5)

    # print(
    #     f"Realized Volatility for {ticker} from {start_date} to {end_date}: {realized_volatility:.2%}"
    # )
    # print(
    #     f"Mean Return for {ticker} from {start_date} to {end_date}: {mean_return:.2%}"
    # )

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(
        data.index,
        data["Realized_Volatility"],
        label="Realized Volatility (30-day rolling)",
    )
    axs[0].set_title(f"Realized Volatility for {ticker}")
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("Realized Volatility")
    axs[0].legend()

    axs[1].plot(
        data.index,
        data["rolling_return"],
        label="Return (30-day rolling)",
    )
    axs[1].set_title(f"Return for {ticker}")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Return")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# Plotting Realized Volatility and Variance for AAPL
# Uncomment the line below to run the function
# plot_volatility_and_variance("AAPL", "2010-01-01", "2023-01-01")

ticker = "AAPL"
start_date = "2001-01-01"
end_date = datetime.datetime.today().strftime("%Y-%m-%d")
df = yf.download(ticker, start=start_date, end=end_date)
df.reset_index(inplace=True)
df.sort_values("Date", inplace=True)

df["Time_Years"] = (df["Date"] - df["Date"].iloc[0]).dt.days / 365.0
df["Return"] = df["Close"].pct_change()
df["delta_t"] = df["Time_Years"].diff()
# print(df.columns)

# print(df.head())

df.dropna(subset=[("Return", ""), ("delta_t", "")], inplace=True)

# Historical trend estimator - Equation 9 in assignment notes
df["Term"] = df["Return"] / df["delta_t"]

mu_hat = df["Term"].mean()

print(f"Classic historical mean (mu_hat): {mu_hat:.6f} over the period 2010-2023")

N = len(df)

# Historical volatility estimator - Equation 14 in assignment notes
df["residual"] = df["Return"] - mu_hat * df["delta_t"]
sum_of_squared_residuals = (df["residual"] ** 2).sum()
df["squared_residual"] = df["residual"] ** 2
sigma2_hat = sum_of_squared_residuals / (N - 1)
sigma_hat = np.sqrt(sigma2_hat)
# annualized_sigma = sigma_hat * np.sqrt(252)
print(
    f"Classic historical volatility estimator: {sigma_hat:.6f} over the period 2010-2023"
)

# Parkinson volatility estimator
sigma_parkinson = float(
    (1 / 4 * np.log(2) * (np.log(df["High"] / df["Low"]) ** 2).sum()) ** 0.5
) / np.sqrt(252)
print(
    f"Parkinson volatility estimator: {sigma_parkinson:.6f} over the period 2010-2023"
)

# Garman-Klass volatility estimator
sigma_garman_klass = float(
    (
        (1 / 2) * np.log(2) * (np.log(df["High"] / df["Low"]) ** 2).sum()
        - (2 * np.log(2) - 1) * (np.log(df["Close"] / df["Open"]) ** 2).sum()
    )
    ** 0.5
) / np.sqrt(252)
print(
    f"Garman-Klass volatility estimator: {sigma_garman_klass:.6f} over the period 2010-2023"
)

# 30-day rolling window
window = 30
df["rolling_variance_historical"] = (
    df["squared_residual"]
    .rolling(window=window)
    .apply(lambda x: np.sum(x) / (len(x) - 1), raw=False)
)
# print(df["rolling_variance_historical"].head(50))

# 30-day rolling window for Parkinson volatility
df["rolling_variance_parkinson"] = (
    1
    / 4
    * np.log(2)
    * (np.log(df["High"] / df["Low"]) ** 2).rolling(window=window).sum()
)
# print(df["rolling_variance_parkinson"].head(50))

# 30-day rolling window for Garman-Klass volatility
df["rolling_variance_garman_klass"] = (1 / 2) * np.log(2) * (
    np.log(df["High"] / df["Low"]) ** 2
).rolling(window=window).sum() - (2 * np.log(2) - 1) * (
    np.log(df["Close"] / df["Open"]) ** 2
).rolling(window=window).sum()
# print(df["rolling_variance_garman_klass"].head(50))


prices = load_sp500_dataset()
implied_vol = load_sp500_implied_vol_dataset()
ticker = "AAPL"
prices_single = prices[[ticker]]
print("Prices single:")
print(prices_single.head(5))
implied_vol_single = implied_vol[[ticker]]
implied_vol_single = implied_vol_single.loc["2010":"2023"]
print("Implied Volatility single:")
print(implied_vol_single.head(5))
implied_vol_single_rescaled = implied_vol_single / np.sqrt(252)
# print(implied_vol_single.head(50))


# Plotting the rolling variance
# plt.figure(figsize=(10, 6))
# plt.plot(df["Date"], df["rolling_variance_historical"], label="Historical")
# plt.plot(df["Date"], df["rolling_variance_parkinson"], label="Parkinson")
# plt.plot(df["Date"], df["rolling_variance_garman_klass"], label="Garman-Klass")
# plt.plot(
#     implied_vol_single.index, implied_vol_single_rescaled, label="Implied Volatility"
# )
# plt.plot()
# plt.title("30-day Rolling Variance")
# plt.xlabel("Date")
# plt.ylabel("Variance")
# plt.legend()
# plt.show()
