from model import predict
import pandas as pd
import numpy as np
import vectorbt as vbt

# Data for backtesting
prices = vbt.YFData.download("NVDA", start="2024-01-01 UTC", end="2025-01-01 UTC").get("Close")
closing_prices = prices.values.reshape(-1, 1)  # Shape (n_days, 1)

# Build input-output pairs: X = today
X = closing_prices[:-1]

# Normalize with training stats
mean = np.mean(X)
std = np.std(X)
X_norm = (X - mean) / std

preds = predict(x_test=X_norm.T)  #  shape must be (features, samples)

# Transform the predictions to a pandas Series, important to shift the indices
preds_series = pd.Series(preds.flatten(), index=prices.index[1:], name="Prediction")
aligned_prices = prices.iloc[1:]

# Go long if predicted price > current price, else short
entries = preds_series > aligned_prices  
exits = preds_series < aligned_prices   

# Short strategy
short_entries = preds_series < aligned_prices
short_exits = preds_series > aligned_prices

print(exits[exits].index)

# # Long-only portfolio
# long_pf = vbt.Portfolio.from_signals(
#     close=aligned_prices,
#     entries=entries,
#     exits=exits,
#     init_cash=10_000,
#     fees=0.001  
# )


# Long + Short 
combined_pf = vbt.Portfolio.from_signals(
    close=aligned_prices,
    entries=entries,
    exits=exits,
    short_entries=short_entries,
    short_exits=short_exits,
    init_cash=10_000,
    fees=0.001,
)

# Mostrar estadísticas y gráfico
print(combined_pf.stats())
combined_pf.plot().show()
