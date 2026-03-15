import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime

class LocalMomentumMVO:
    def __init__(self):
        self.core_tickers = ["VTI", "VEA", "VWO", "BND", "BNDX", "EMB", "DBC", "GLD", "VGSH", "VGIT", "VGLT", "IBIT"]
        self.sector_tickers = ["IXP", "RXI", "KXI", "IXC", "IXG", "IXJ", "EXI", "MXI", "IXN", "JXI", "REET"]
        self.bond_tickers = ["BND", "BNDX", "VGSH", "VGIT"]
        self.cash_substitute = "SHV"
        # Combine all to a list of string tickers
        self.symbols = list(set(self.core_tickers + self.sector_tickers + [self.cash_substitute]))
        self.lookbacks = [21, 63, 126, 189, 252]

    def run(self):
        # 1. Fetch data
        data = yf.download(self.symbols, period="10y", auto_adjust=True, progress=False)['Close']
        returns_df = data.pct_change().dropna()

        # 2. Get Scores
        scores = self.get_momentum_scores(data)

        # 3. Calculate MVO Weights
        weights = self.calculate_mvo_weights(returns_df, scores)

        # 4. Log Output
        print(f"\n--- Local MVO Run: {datetime.now().strftime('%Y-%m-%d')} ---")
        for ticker, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True):
            if weight > 0.001:
                print(f"{ticker:<10}: {weight:.2%}")
        return weights

    def get_momentum_scores(self, prices):
        scores = {}
        for ticker in self.symbols:
            if ticker not in prices.columns: continue
            
            s_prices = prices[ticker].dropna()
            if len(s_prices) < 253: continue
            
            cur_price = s_prices.iloc[-1]
            
            # Simplified logic: compare ticker string directly
            sma_window = 126 if ticker in self.bond_tickers else 168
            sma = s_prices.iloc[-sma_window:].mean()

            if cur_price > sma:
                price_scores = [(cur_price / s_prices.iloc[-d-1]) - 1 for d in self.lookbacks]
                scores[ticker] = max(np.mean(price_scores), 0)
            else:
                scores[ticker] = 0
        return scores

    def calculate_mvo_weights(self, returns_df, scores, risk_aversion=10.0):
        # Filter symbols that have a calculated score
        valid_symbols = [s for s in returns_df.columns if s in scores and scores[s] > 0]
        if not valid_symbols: return {}
        
        cov = returns_df[valid_symbols].cov() * 252
        mu = np.array([scores[s] for s in valid_symbols])

        def objective(w):
            port_ret = np.sum(w * mu)
            port_var = w.T @ cov @ w
            return -1 * (port_ret - 0.5 * risk_aversion * port_var)

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.00, 0.20) for _ in valid_symbols)

        init_w = np.array([1.0/len(valid_symbols)] * len(valid_symbols))
        result = minimize(objective, init_w, bounds=bounds, constraints=cons, method='SLSQP')

        return dict(zip(valid_symbols, result.x))

if __name__ == "__main__":
    algo = LocalMomentumMVO()
    algo.run()
