# region imports
from AlgorithmImports import *
import pandas as pd
import numpy as np
# endregion

class MomentumCVaRAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2013, 1, 1)
        self.set_cash(100000)
        self.set_benchmark("AOR")

        # Core Assets
        self.core_tickers = ["VTI", "VEA", "VWO", "BND", "BNDX", "EMB", "DBC", "GLD", "VGIT", "VGLT"]
        # Global Sector ETFs
        self.sector_tickers = ["IXP", "RXI", "KXI", "IXC", "IXG", "IXJ", "EXI", "MXI", "IXN", "IXU"]

        self.all_tickers = self.core_tickers + self.sector_tickers
        self.symbols = []

        for ticker in self.all_tickers:
            equity = self.add_equity(ticker, Resolution.DAILY)
            equity.SetFeeModel(ConstantFeeModel(0))
            self.symbols.append(equity.Symbol)

        self.lookbacks = [21, 63, 126, 189, 252]
        self.set_warm_up(1265, Resolution.DAILY)

        self.schedule.on(
            self.date_rules.month_start(self.symbols[0]),
            self.time_rules.after_market_open(self.symbols[0], 120),
            self.rebalance
        )

    def rebalance(self):
        if self.is_warming_up: return
        self.debug(f"\n--- Rebalance Heartbeat: {self.time} ---")

        # 1. Get History
        history = self.history(self.symbols, 1265, Resolution.DAILY)
        if history.empty:
            return

        prices = history['close'].unstack(level=0)
        returns_df = prices.pct_change().dropna()

        bond_tickers = ["BND", "BNDX", "VGIT"]
        all_mom_scores = {}

        # 2. Calculate Momentum safely
        for symbol in self.symbols:
            # SAFETY CHECK: Skip if symbol is not in the history dataframe columns
            if symbol not in prices.columns:
                continue

            s_prices = prices[symbol].dropna()
            if len(s_prices) < 253: # Ensure we have enough data for the longest lookback
                continue

            cur_price = s_prices.iloc[-1]
            
            # Trend Check
            sma_len = 126 if symbol.value in bond_tickers else 168
            sma = s_prices.iloc[-sma_len:].mean()
            
            if cur_price > sma:
                scores = [(cur_price / s_prices.iloc[-d-1]) - 1 for d in self.lookbacks]
                avg_m = np.mean(scores)
                if avg_m > 0:
                    all_mom_scores[symbol] = avg_m
                    continue
            all_mom_scores[symbol] = 0.0

        # 3. Select Top 4 Global Sectors
        sector_symbols = [s for s in self.symbols if s.value in self.sector_tickers and s in all_mom_scores]
        sorted_sectors = sorted(sector_symbols, key=lambda x: all_mom_scores.get(x, 0), reverse=True)
        top_4_sectors = [s for s in sorted_sectors[:4] if all_mom_scores.get(s, 0) > 0]

        # 4. Final Universe Construction
        core_available = [s for s in self.symbols if s.value in self.core_tickers and s in all_mom_scores]
        valid_symbols = [s for s in core_available if all_mom_scores.get(s, 0) > 0] + top_4_sectors

        if len(valid_symbols) < 4:
            self.debug(f"Safety Switch: Only {len(valid_symbols)} assets pass. Liquidating.")
            shv_symbol = self.AddEquity("SHV", Resolution.Daily).Symbol
            self.SetHoldings(shv_symbol, 1.0)
            self.liquidate()
            return

        # 5. CVaR Weighting (Optimized & Robust)
        weights_map = {}
        
        # Ensure returns_df only contains valid symbols to avoid alignment issues
        available_columns = returns_df.columns
        
        for symbol in valid_symbols:
            # SAFETY: Ensure symbol is actually in our historical data
            if symbol not in available_columns: 
                continue
            
            # Use .dropna() to handle cases where an asset started trading recently
            asset_rets = returns_df[symbol].dropna()
            
            # Ensure we have a statistically significant sample (e.g., at least 504 days)
            if len(asset_rets) < 504:
                continue

            # Calculate VaR (Value at Risk) at 95% confidence
            var_threshold = np.percentile(asset_rets, 5)
            
            # Filter returns below the VaR threshold
            tail_events = asset_rets[asset_rets <= var_threshold]

            if not tail_events.empty:
                # CVaR is the expected loss given we are in the tail
                cvar = -tail_events.mean()
            else:
                # Fallback to standard deviation if no tail events found (rare)
                cvar = asset_rets.std()

            # Avoid division by zero and ensure positive risk metric
            cvar = max(cvar, 0.0001) 

            # Momentum/Risk weighting
            weights_map[symbol] = float(all_mom_scores[symbol] / cvar)

        # 6. Normalization & Capping
        MAX_CAP = 0.25 
        total_raw_weight = sum(weights_map.values())
        if total_raw_weight == 0: return
        
        current_weights = {s: weights_map[s] / total_raw_weight for s in weights_map}

        for _ in range(10):
            total_excess = 0.0
            eligible_symbols = []
            for s, w in current_weights.items():
                if w > MAX_CAP:
                    total_excess += (w - MAX_CAP)
                    current_weights[s] = MAX_CAP
                elif w < MAX_CAP:
                    eligible_symbols.append(s)

            if total_excess <= 1e-6 or not eligible_symbols: break

            rem_sum = sum(current_weights[s] for s in eligible_symbols)
            for s in eligible_symbols:
                current_weights[s] += total_excess * (current_weights[s] / rem_sum)

        # --- NEW: Step 6.5: Portfolio CVaR Targeting ---
        TARGET_CVAR = 0.02
        CONFIDENCE_LEVEL = 0.05

        # 1. Calculate Portfolio Returns based on current_weights
        # Get returns for only the symbols we are actually holding
        portfolio_symbols = list(current_weights.keys())
        sub_returns = returns_df[portfolio_symbols]

        # Multiply daily asset returns by their weights to get daily portfolio returns
        weights_series = pd.Series(current_weights)
        portfolio_returns = sub_returns.dot(weights_series)

        # 2. Calculate Portfolio CVaR
        var_threshold = np.percentile(portfolio_returns, CONFIDENCE_LEVEL * 100)
        portfolio_cvar = -portfolio_returns[portfolio_returns <= var_threshold].mean()

        # 3. Calculate Scaling Factor
        # If portfolio_cvar is 0.04 and target is 0.02, we need to hold 50% cash
        scaling_factor = 1.0
        if portfolio_cvar > TARGET_CVAR:
            scaling_factor = TARGET_CVAR / portfolio_cvar

        # Apply scaling to weights
        final_weights = {s: w * scaling_factor for s, w in current_weights.items()}

        # 7. EXECUTION
        # First, ensure SHV is in your symbols list (Add it in Initialize if not already there)
        shv_symbol = self.AddEquity("SHV", Resolution.Daily).Symbol

        # Calculate how much total weight is taken by the momentum strategy
        total_momentum_weight = sum(final_weights.values())
        
        # Any remaining weight goes to SHV (Cash Substitute)
        # We use max(0, ...) to prevent negative numbers due to rounding
        shv_weight = max(0, 1.0 - total_momentum_weight)

        self.Liquidate() 
        
        weight_list = []
        
        # 1. Allocate Momentum Assets
        for symbol, weight in final_weights.items():
            if weight > 0.001:
                self.SetHoldings(symbol, float(weight))
                weight_list.append(f"{symbol.Value}: {weight:.1%}")
        
        # 2. Allocate Cash Substitute (SHV)
        if shv_weight > 0.01:
            self.SetHoldings(shv_symbol, float(shv_weight))
            weight_list.append(f"SHV(Cash): {shv_weight:.1%}")

        if weight_list:
            self.Debug(f"REBALANCE | {', '.join(weight_list)}")
