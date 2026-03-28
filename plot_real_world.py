import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import os

artifacts_dir = r"C:\Users\Leon\.gemini\antigravity\brain\734659b4-f37d-4f49-adca-4d2d30daf8a2\artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

print("Fetching real macroeconomic data from Yahoo Finance...")

# In internet data:
# ^IRX = 13 Week Treasury Bill (good proxy for short term policy rates)
# ^GSPC = S&P 500 (good proxy for general economic health / GDP growth)
tickers = ['^IRX', '^GSPC']
internet_data = yf.download(tickers, start="2015-01-01", end="2024-12-31")['Close']
internet_data = internet_data.resample('ME').last()

# Load Manifold Portfolio Metrics
metrics = pd.read_csv('portfolio_metrics.csv')
metrics['date'] = pd.to_datetime(metrics['date'])
metrics.set_index('date', inplace=True)

# Generate Plot
fig, ax1 = plt.subplots(figsize=(14, 7))

# Primary Axis: Rates
ax1.plot(internet_data.index, internet_data['^IRX'], label='Real US 13-Wk T-Bill (^IRX)', color='blue', linestyle='--', linewidth=2)
ax1.plot(metrics.index, metrics['policy_rate'], label='Manifold Policy Rate', color='lightblue', linewidth=3)
ax1.plot(metrics.index, metrics['unemployment'], label='Manifold Unemployment', color='pink', linewidth=3)
ax1.set_xlabel('Date')
ax1.set_ylabel('Percentage (%)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.5)

# Secondary Axis: S&P 500
ax2 = ax1.twinx()
ax2.plot(internet_data.index, internet_data['^GSPC'], label='Real S&P 500 Index (^GSPC)', color='green', linestyle=':', linewidth=2)
ax2.set_ylabel('S&P 500 Index Value', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')

plt.title('Baseline Manifold Macro Scenarios vs. Real World Historical Data (2015-2024)')
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(artifacts_dir, 'real_world_macro.png'), dpi=300)
plt.savefig('real_world_macro.png', dpi=300)
plt.close()

print("Internet data successfully fetched and plotted!")
