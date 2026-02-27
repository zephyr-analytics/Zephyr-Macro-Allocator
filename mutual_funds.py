import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def check_fund_trends():
    # Tickers for MDLOX and PGMAX
    tickers = ["MDLOX", "PGMAX"]
    sma_window = 168

    print(f"Fetching data for {tickers}...")
    # Fetch 2 years of data to ensure enough history for the 168 SMA
    data = yf.download(tickers, period="10y", interval="1d", auto_adjust=True)

    if data.empty:
        print("No data found. Please check your internet connection or ticker symbols.")
        return

    prices = data['Close']

    # Calculate SMAs
    smas = prices.rolling(window=sma_window).mean()

    # Setup Visualization
    fig, axes = plt.subplots(len(tickers), 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    for i, ticker in enumerate(tickers):
        ax = axes[i]

        # Plot Price and SMA
        ax.plot(prices.index, prices[ticker], label=f'{ticker} Price', color='#1f77b4', lw=1.5)
        ax.plot(smas.index, smas[ticker], label=f'168-day SMA', color='#ff7f0e', linestyle='--', lw=2)

        # Get latest values
        current_price = prices[ticker].iloc[-1]
        current_sma = smas[ticker].iloc[-1]
        status = "BULLISH (Above SMA)" if current_price > current_sma else "BEARISH (Below SMA)"
        color = "green" if current_price > current_sma else "red"

        # Formatting
        ax.set_title(f"{ticker} Trend: {status}", fontsize=14, fontweight='bold', color=color)
        ax.set_ylabel("Price ($)")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        print(f"{ticker}: Current Price = {current_price:.2f} | 168 SMA = {current_sma:.2f} -> {status}")

    plt.xlabel("Date")
    plt.suptitle(f"Trend Analysis: Price vs. {sma_window}-Day SMA", fontsize=16)

    # Save and show
    plt.savefig("fund_trend_analysis.png")
    print("\nVisual saved as 'fund_trend_analysis.png'")
    plt.show()

if __name__ == "__main__":
    check_fund_trends()
