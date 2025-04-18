import datetime

import numpy as np
import pandas as pd
import yfinance as yf

spx_symbol = "^SPX"
today_string = "2025-03-28"
today = datetime.datetime.strptime(today_string, "%Y-%m-%d")
end_date = today
start_date = end_date - datetime.timedelta(days=365)
spx_data = yf.download(spx_symbol, start=start_date, end=end_date)

lastBusDay = spx_data.index[-1]
vix_data = yf.download(
    "^VIX", start=lastBusDay, end=lastBusDay + datetime.timedelta(days=1)
)
print(vix_data.head(10))
# print(spx_data.head(10))

spx_ticker = yf.Ticker("^SPX")
expiry_date = "2025-04-28"
chain = spx_ticker.option_chain(expiry_date)

calls = pd.read_csv("src/solutions/Call_option_data_2025-04-03_final.csv")
# print("Calls - Head:")
calls = calls.reset_index(drop=True)
calls.drop(columns=["Unnamed: 0"], inplace=True)
# print(calls.head(10))


puts = pd.read_csv("src/solutions/Put_option_data_2025-04-03_final.csv")
print("Puts - Head:")
puts = puts.reset_index(drop=True)
puts.drop(columns=["Unnamed: 0"], inplace=True)
# print(puts.head(10))
# print(puts.columns)

# print("Calls - Head:")
# print(calls.head())
# print(calls.columns)

# puts = puts[~puts["inTheMoney"]]
# calls = calls[~calls["inTheMoney"]]


print(puts.head(20))
print(calls.head(20))

# Given by the assignment (CBOE)
tau = 30 / 365

# risk-free interest rate is set equal to US treasury yield for 30-days
r = 0.044

F = np.exp(r * tau) * spx_data["Close"].iloc[-1].values[0]
print(f"Forward Price F: {F}")

puts = puts[puts["strike"] < F]
calls = calls[calls["strike"] > F]

# Where the put price is denoted by the average of the bid and ask prices
puts_component = (
    ((puts["Last"]) * ((1 / puts["strike"]) - (1 / puts["strike"].shift(-1))))
    .dropna()
    .sum()
)
print("Puts component:", puts_component)

calls_component = (
    ((calls["Last"]) * ((1 / calls["strike"].shift(1)) - (1 / (calls["strike"]))))
    .dropna()
    .sum()
)
print("Calls component:", calls_component)
summation = puts_component + calls_component
VIX = np.sqrt((((2 * np.exp(r * tau)) / tau) * summation))
print("VIX:", VIX * 100)
