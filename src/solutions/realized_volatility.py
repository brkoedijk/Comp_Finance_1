import matplotlib.pyplot as plt
import yfinance as yf


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

    # Calculate mean return
    mean_return = data["Return"].mean()

    # Calculate annualized realized volatility
    realized_volatility = data["Return"].std() * (252**0.5)

    print(
        f"Realized Volatility for {ticker} from {start_date} to {end_date}: {realized_volatility:.2%}"
    )
    print(
        f"Mean Return for {ticker} from {start_date} to {end_date}: {mean_return:.2%}"
    )

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
        data["Realized_Variance"],
        label="Realized Variance (30-day rolling)",
    )
    axs[1].set_title(f"Realized Variance for {ticker}")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Realized Variance")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# Plotting Realized Volatility and Variance for AAPL
# Uncomment the line below to run the function
# plot_volatility_and_variance("AAPL", "2010-01-01", "2023-01-01")

ticker = "AAPL"
start_date = "2010-01-01"
end_date = "2023-01-01"
df = yf.download(ticker, start=start_date, end=end_date)
df.reset_index(inplace=True)
df.sort_values("Date", inplace=True)

# print(df.head())

df["Time_Years"] = (df["Date"] - df["Date"].iloc[0]).dt.days / 365.0
df["Return"] = df["Close"].pct_change()
df["delta_t"] = df["Time_Years"].diff()
# print(df.columns)

df.dropna(subset=[("Return", ""), ("delta_t", "")], inplace=True)

df["Term"] = df["Return"] / df["delta_t"]

mu_hat = df["Term"].mean()

print(f"Estimated drift (mu_hat): {mu_hat:.6f} per year")

df["Partial_Term"] = df["Term"].expanding().mean()

plt.figure(figsize=(10, 6))
plt.plot(df["Date"], df["Partial_Term"], label="Partial Average of Term")
plt.axhline(mu_hat, color="red", linestyle="--", label=f"Final Estimate: {mu_hat:.6f}")
plt.title(f"Drift Estimator Over Time ({ticker})")
plt.xlabel("Date")
plt.ylabel("Estimated Drift (per year)")
plt.legend()
plt.grid(True)
plt.show()
