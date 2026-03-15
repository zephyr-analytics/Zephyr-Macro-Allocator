from AlgorithmImports import *
import pandas as pd
import numpy as np
from scipy.optimize import minimize

class MomentumMVOAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_cash(100000)
        self.set_benchmark("ACWI")

        # Universe definition
        self.core_tickers = ["VTI", "VEA", "VWO", "BND", "BNDX", "EMB", "DBC", "GLD", "VGSH", "VGIT", "VGLT", "IBIT"]
        self.sector_tickers = ["IXP", "RXI", "KXI", "IXC", "IXG", "IXJ", "EXI", "MXI", "IXN", "JXI", "REET"]
        self.bond_tickers = ["BND", "BNDX", "VGSH", "VGIT",]
        self.all_tickers = self.core_tickers + self.sector_tickers
        
        self.symbols = [self.add_equity(ticker, Resolution.DAILY).Symbol for ticker in self.all_tickers]
        self.shv = self.add_equity("SHV", Resolution.DAILY).Symbol

        for symbol in self.symbols:
            self.securities[symbol].set_fee_model(ConstantFeeModel(0))

        self.lookback = 2520
        self.lookbacks = [21, 63, 126, 189, 252]
        self.set_warm_up(self.lookback, Resolution.DAILY)

        self.schedule.on(self.date_rules.month_start("VTI"), self.time_rules.after_market_open("VTI", 120), self.rebalance)

    def rebalance(self):
        if self.is_warming_up: return

        history = self.history(self.symbols, self.lookback, Resolution.DAILY, data_normalization_mode=DataNormalizationMode.TOTAL_RETURN)
        if history.empty: return

        prices = history['close'].unstack(level=0)
        returns_df = prices.pct_change().dropna()

        # 1. Momentum & Universe
        scores = self.get_momentum_scores(prices)
        valid_symbols = [s for s in self.symbols if scores.get(s, 0) > 0]

        # 2. Safety Check
        if len(valid_symbols) <= 4:
            self.apply_safety_switch()
            return

        # 3. MVO Weights
        weights = self.calculate_mvo_weights(valid_symbols, returns_df, scores)

        # --- DIAGNOSTIC LOG: WEIGHTS ONLY ---
        log_msg = f"\n--- {self.time.strftime('%Y-%m-%d')} ---\n"

        # Sort symbols by weight descending for better readability
        sorted_weights = sorted(weights.items(), key=lambda item: item[1], reverse=True)

        for symbol, weight in sorted_weights:
            if weight > 0.001:  # Only log active positions
                log_msg += f"{symbol.value:<10}: {weight:.2%},\n"

        self.debug(log_msg)

        # 4. Execution
        self.liquidate()
        for symbol, weight in weights.items():
            if weight > 0.001:
                self.set_holdings(symbol, weight)

    def calculate_mvo_weights(self, symbols, returns_df, scores, risk_aversion=10.0):
        cov = returns_df[symbols].cov() * 252
        # Extract momentum scores as a vector
        mu = np.array([scores[s] for s in symbols])
        
        # Objective: Maximize (Utility) = Mean - (1/2 * lambda * Variance)
        # Since we use 'minimize', we negate the Utility:
        def objective(w):
            port_ret = np.sum(w * mu)
            port_var = w.T @ cov @ w
            return -1 * (port_ret - 0.5 * risk_aversion * port_var)

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.00, 0.20) for _ in symbols)
        
        init_w = np.array([1.0/len(symbols)] * len(symbols))
        result = minimize(objective, init_w, bounds=bounds, constraints=cons, method='SLSQP')
        
        return dict(zip(symbols, result.x))

    def get_momentum_scores(self, prices):
        scores = {}
        for symbol in self.symbols:
            if symbol not in prices.columns: continue
            s_prices = prices[symbol].dropna()
            if len(s_prices) < 253: continue
            cur_price = s_prices.iloc[-1]
            sma = s_prices.iloc[- (126 if symbol.value in self.bond_tickers else 168):].mean()
            if cur_price > sma:
                price_scores = [(cur_price / s_prices.iloc[-d-1]) - 1 for d in self.lookbacks]
                scores[symbol] = max(np.mean(price_scores), 0)
            else:
                scores[symbol] = 0
        return scores

    def apply_safety_switch(self):
        self.debug("Breadth Failure")
        self.liquidate()
        self.set_holdings(self.shv, 1.0)
