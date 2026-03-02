import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

class LocalMomentumCVaR:
    def __init__(self):
        # --- CONFIGURATION (Synced with QC) ---
        self.core_tickers = ["VTI", "VEA", "VWO", "BND", "BNDX", "EMB", "DBC", "GLD", "VGIT", "VGLT"]
        self.sector_tickers = ["IXP", "RXI", "KXI", "IXC", "IXG", "IXJ", "EXI", "MXI", "IXN", "JXI", "REET"]
        self.bond_tickers = ["BND", "BNDX", "VGIT"]
        self.cash_substitute = "SHV"
        
        self.all_tickers = list(set(self.core_tickers + self.sector_tickers + [self.cash_substitute]))
        
        # Risk & Lookback Parameters
        self.target_cvar = 0.03
        self.max_cap = 0.25
        self.history_days = 1265 
        self.lookbacks = [21, 63, 126, 189, 252]
        self.confidence_level = 1  # 1% for asset-level (CVaR tail)
        self.port_confidence = 5   # 5% for portfolio-level (Scaling)

    def run(self):
        # 1. FETCH DATA
        print(f"Fetching data for {len(self.all_tickers)} symbols...")
        # Only Close prices needed now
        data = yf.download(self.all_tickers, period="6y", interval="1d", auto_adjust=True, progress=False)
        prices = data['Close']
        returns_df = prices.pct_change(fill_method=None)

        # 2. CALCULATE MOMENTUM SCORES (Price-Only + SMA Filter)
        all_mom_scores = self.get_momentum_scores(prices)

        # 3. SELECT UNIVERSE
        valid_symbols = self.select_universe(all_mom_scores)
        print(f"Assets passing filters: {valid_symbols}")

        # 4. SAFETY SWITCH
        if len(valid_symbols) < 4:
            print("Safety Switch Triggered: Insufficient signals. Moving to 100% Cash.")
            return {self.cash_substitute: 1.0}

        # 5. CALCULATE CVaR WEIGHTS
        weights_map = self.calculate_cvar_weights(valid_symbols, returns_df, all_mom_scores)

        # 6. NORMALIZATION & CAPPING
        capped_weights = self.apply_weight_caps(weights_map)

        # 7. FINAL SCALING & CASH ALLOCATION
        final_allocations = self.get_final_allocations(capped_weights, returns_df)
        
        return final_allocations

    def get_momentum_scores(self, prices):
        scores = {}
        for ticker in (self.core_tickers + self.sector_tickers):
            if ticker not in prices.columns: continue
            
            s_prices = prices[ticker].dropna()
            if len(s_prices) < 253: continue

            cur_price = s_prices.iloc[-1]
            
            # Trend Filter (SMA 126 for Bonds, 168 for Equities)
            sma_len = 126 if ticker in self.bond_tickers else 168
            sma = s_prices.iloc[-sma_len:].mean()

            # Logic: Price must be above SMA, and average momentum must be positive
            if cur_price > sma:
                m_rets = [(cur_price / s_prices.iloc[-d-1]) - 1 for d in self.lookbacks]
                avg_m = np.mean(m_rets)
                scores[ticker] = max(avg_m, 0)
            else:
                scores[ticker] = 0
        return scores

    def select_universe(self, scores):
        # Top 4 Sectors by Momentum
        sector_scores = {t: scores[t] for t in self.sector_tickers if scores.get(t, 0) > 0}
        top_4_sectors = sorted(sector_scores, key=sector_scores.get, reverse=True)[:4]
        
        # All Core assets with positive momentum
        core_valid = [t for t in self.core_tickers if scores.get(t, 0) > 0]
        
        return list(set(core_valid + top_4_sectors))

    def calculate_cvar_weights(self, symbols, returns_df, scores):
        weights = {}
        analysis_returns = returns_df.tail(self.history_days)
        
        for ticker in symbols:
            asset_rets = analysis_returns[ticker].dropna()
            if len(asset_rets) < 504: continue

            # Asset-level CVaR (Expected Shortfall at 1% percentile)
            var_limit = np.percentile(asset_rets, self.confidence_level)
            tail = asset_rets[asset_rets <= var_limit]
            cvar = -tail.mean() if not tail.empty else asset_rets.std()
            
            # Risk-Adjusted Weight = Momentum / CVaR
            weights[ticker] = scores[ticker] / max(cvar, 0.0001)
        return weights

    def apply_weight_caps(self, weights_map):
        total_raw = sum(weights_map.values())
        if total_raw == 0: return {}
        
        current_weights = {t: w / total_raw for t, w in weights_map.items()}

        # Iterative Capping to redistribute excess weight
        for _ in range(10):
            total_excess = sum(max(0, w - self.max_cap) for w in current_weights.values())
            if total_excess <= 1e-6: break
            
            eligible = [t for t, w in current_weights.items() if w < self.max_cap]
            if not eligible: break
            
            for t in current_weights:
                if current_weights[t] > self.max_cap:
                    current_weights[t] = self.max_cap
            
            rem_sum = sum(current_weights[t] for t in eligible)
            for t in eligible:
                current_weights[t] += total_excess * (current_weights[t] / rem_sum)
        return current_weights

    def get_final_allocations(self, weights, returns_df):
        if not weights: return {self.cash_substitute: 1.0}
        
        # Portfolio-level Risk Assessment
        weights_series = pd.Series(weights)
        sub_returns = returns_df[weights_series.index].tail(252).dropna()
        portfolio_rets = sub_returns.dot(weights_series)
        
        # Portfolio CVaR at 5% percentile
        p_var_threshold = np.percentile(portfolio_rets, self.port_confidence)
        portfolio_cvar = -portfolio_rets[portfolio_rets <= p_var_threshold].mean()

        # Scale down if portfolio risk exceeds TARGET_CVAR
        scaling_factor = min(1.0, self.target_cvar / portfolio_cvar) if portfolio_cvar > 0 else 1.0

        final_weights = {t: 0.0 for t in self.all_tickers}
        total_momentum_exposure = 0.0
        
        for t, w in weights.items():
            scaled_w = w * scaling_factor
            final_weights[t] = scaled_w
            total_momentum_exposure += scaled_w

        # Remainder goes to Cash (SHV)
        final_weights[self.cash_substitute] = max(0, 1.0 - total_momentum_exposure)
        return final_weights

if __name__ == "__main__":
    algo = LocalMomentumCVaR()
    signals = algo.run()
    
    print(f"\n--- TARGET ALLOCATIONS ({datetime.now().date()}) ---")
    active = sorted({k: v for k, v in signals.items() if v > 0.001}.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in active:
        tag = "(Cash/Safe)" if ticker == "SHV" else ""
        print(f"{ticker:5}: {weight:6.2%} {tag}")
