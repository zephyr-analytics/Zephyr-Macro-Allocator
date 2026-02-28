import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def check_fund_trends():
    ticker = "MDLOX"
    sma_window = 168

    print(f"Fetching data for {ticker}...")
    # auto_adjust=True is good, but let's ensure we get a clean Series
    data = yf.download(ticker, period="10y", interval="1d", auto_adjust=True)

    if data.empty:
        print("No data found.")
        return

    # Force 'Close' to a Series even if yfinance returns a MultiIndex DataFrame
    prices = data['Close']
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    # Calculate SMA
    sma = prices.rolling(window=sma_window).mean()

    # Get latest values and force them to single floats using .item()
    # This prevents the "ambiguous truth value" error
    current_price = prices.iloc[-1].item()
    current_sma = sma.iloc[-1].item()

    # Now the boolean check will work perfectly
    is_bullish = current_price > current_sma
    status = "BULLISH (Above SMA)" if is_bullish else "BEARISH (Below SMA)"
    color = "green" if is_bullish else "red"

    # Setup Visualization
    plt.figure(figsize=(18, 10))
    plt.plot(prices.index, prices, label=f'{ticker} Price', color='#1f77b4', lw=1.5)
    plt.plot(sma.index, sma, label=f'{sma_window}-day SMA', color='#ff7f0e', linestyle='--', lw=2)

    plt.title(f"{ticker} Trend: {status}", fontsize=14, fontweight='bold', color=color)
    plt.ylabel("Price ($)")
    plt.xlabel("Date")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    print(f"{ticker}: Current Price = {current_price:.2f} | {sma_window} SMA = {current_sma:.2f} -> {status}")

    plt.tight_layout()
    plt.savefig("fund_trend_analysis.png")
    plt.show()

if __name__ == "__main__":
    check_fund_trends()
