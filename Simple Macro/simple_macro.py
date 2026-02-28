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

        self.core_tickers = ["VTI", "VEA", "VWO", "BND", "BNDX", "EMB", "DBC", "GLD", "VGIT", "VGLT"]
        self.sector_tickers = ["IXP", "RXI", "KXI", "IXC", "IXG", "IXJ", "EXI", "MXI", "IXN", "JXI", "REET"]
        self.bond_tickers = ["BND", "BNDX", "VGIT"]
        
        self.all_tickers = self.core_tickers + self.sector_tickers
        self.symbols = [self.add_equity(ticker, Resolution.DAILY).Symbol for ticker in self.all_tickers]
        
        for symbol in self.symbols:
            self.securities[symbol].set_fee_model(ConstantFeeModel(0))

        self.target_cvar = 0.02
        self.vol_lookback_short = 21
        self.vol_lookback_long = 42
        self.lookbacks = [21, 63, 126, 189, 252]
        self.set_warm_up(1265, Resolution.DAILY)

        self.schedule.on(
            self.date_rules.month_start("VTI"),
            self.time_rules.after_market_open("VTI", 120),
            self.rebalance
        )

    def rebalance(self):
        """Main orchestrator for the monthly rebalancing logic."""
        if self.is_warming_up: return
        
        history = self.history(self.symbols, 1265, Resolution.DAILY, data_normalization_mode=DataNormalizationMode.TOTAL_RETURN)
        if history.empty: return
        
        prices = history['close'].unstack(level=0)
        returns_df = prices.pct_change()

        all_mom_scores = self.get_momentum_scores(prices)
        valid_symbols = self.select_universe(all_mom_scores)
        
        if len(valid_symbols) <= 4:
            self.apply_safety_switch()
            return

        weights_map = self.calculate_cvar_weights(valid_symbols, returns_df, all_mom_scores)
        if not weights_map: return

        capped_weights = self.apply_weight_caps(weights_map, max_cap=0.25)
        self.execute_orders(capped_weights, returns_df)

    def get_momentum_scores(self, prices):
        """
        Calculates momentum scores based on multiple lookbacks and a trend-following SMA filter.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame of historical close prices where columns are Symbols and rows are Timestamps.

        Returns
        -------
        dict
            A dictionary where keys are Symbol objects and values are the mean momentum score (float).
        """
        scores = {}
        for symbol in self.symbols:
            if symbol not in prices.columns: continue
            s_prices = prices[symbol].dropna()
            if len(s_prices) < 253: continue

            cur_price = s_prices.iloc[-1]
            sma_len = 126 if symbol.value in self.bond_tickers else 168
            sma = s_prices.iloc[-sma_len:].mean()

            if cur_price > sma:
                price_scores = [(cur_price / s_prices.iloc[-d-1]) - 1 for d in self.lookbacks]
                avg_m = np.mean(price_scores)
                scores[symbol] = max(avg_m, 0)
            else:
                scores[symbol] = 0
        return scores

    def select_universe(self, scores):
        """
        Filters assets based on momentum scores and limits exposure to the top 4 global sectors.

        Parameters
        ----------
        scores : dict
            Dictionary of {Symbol: float} containing calculated momentum scores.

        Returns
        -------
        list
            A list of Symbol objects that passed the momentum and sector-ranking filters.
        """
        sector_symbols = [s for s in self.symbols if s.value in self.sector_tickers and scores.get(s, 0) > 0]
        top_4_sectors = sorted(sector_symbols, key=lambda x: scores[x], reverse=True)[:4]
        
        core_available = [s for s in self.symbols if s.value in self.core_tickers and scores.get(s, 0) > 0]
        return core_available + top_4_sectors

    def calculate_cvar_weights(self, symbols, returns_df, scores):
        """
        Calculates raw weights using a Momentum-to-CVaR ratio (Risk-Adjusted Momentum).

        Parameters
        ----------
        symbols : list
            List of Symbol objects to include in the calculation.
        returns_df : pd.DataFrame
            DataFrame of historical daily percentage returns.
        scores : dict
            Dictionary of {Symbol: float} momentum scores.

        Returns
        -------
        dict
            A dictionary of {Symbol: float} representing unnormalized weights.
        """
        weights = {}
        for symbol in symbols:
            if symbol not in returns_df.columns: continue
            asset_rets = returns_df[symbol].dropna()
            if len(asset_rets) < 504: continue

            var_threshold = np.percentile(asset_rets, 5)
            tail_events = asset_rets[asset_rets <= var_threshold]
            cvar = -tail_events.mean() if not tail_events.empty else asset_rets.std()
            
            cvar = max(cvar, 0.0001)
            weights[symbol] = scores[symbol] / cvar
        return weights

    def apply_weight_caps(self, weights_map, max_cap=0.25):
        """
        Normalizes weights to sum to 1.0 and applies an iterative capping process.

        Parameters
        ----------
        weights_map : dict
            Dictionary of {Symbol: float} with raw, unnormalized weights.
        max_cap : float, optional
            The maximum weight allowed for any single asset (default is 0.25).

        Returns
        -------
        dict
            Dictionary of {Symbol: float} where values sum to 1.0 and none exceed max_cap.
        """
        total_raw = sum(weights_map.values())
        if total_raw == 0: return {}
        
        current_weights = {s: w / total_raw for s, w in weights_map.items()}
        
        for _ in range(10):
            total_excess = sum(max(0, w - max_cap) for w in current_weights.values())
            if total_excess <= 1e-6: break
            
            eligible = [s for s, w in current_weights.items() if w < max_cap]
            if not eligible: break
            
            for s in current_weights:
                if current_weights[s] > max_cap:
                    current_weights[s] = max_cap
            
            rem_sum = sum(current_weights[s] for s in eligible)
            for s in eligible:
                current_weights[s] += total_excess * (current_weights[s] / rem_sum)
        return current_weights

    def execute_orders(self, weights, returns_df):
        """
        Calculates portfolio CVaR, scales weights to meet the target, and executes trades.

        Parameters
        ----------
        weights : dict
            Dictionary of {Symbol: float} representing normalized, capped weights.
        returns_df : pd.DataFrame
            DataFrame of historical returns used to calculate aggregate portfolio risk.

        Returns
        -------
        None
        """
        if not weights: return
        
        # Calculate Portfolio-level CVaR
        sub_returns = returns_df[list(weights.keys())].dropna()
        portfolio_returns = sub_returns.dot(pd.Series(weights))
        var_threshold = np.percentile(portfolio_returns, 5)
        portfolio_cvar = -portfolio_returns[portfolio_returns <= var_threshold].mean()

        # Scale based on target CVaR (e.g., 0.03)
        scaling_factor = min(1.0, self.target_cvar / portfolio_cvar) if portfolio_cvar > 0 else 1.0
        
        self.liquidate()
        
        log_msg = f"\n--- REBALANCE: {self.time.strftime('%Y-%m-%d')} ---\n"
        log_msg += f"Portfolio CVaR: {portfolio_cvar:.4f} | Target: {self.target_cvar} \n"
        log_msg += f"{'Symbol':<10} | {'Weight':<10}\n" + "-"*5 + "\n"

        total_momentum_weight = 0
        for symbol, w in weights.items():
            final_w = w * scaling_factor
            if final_w > 0.001:
                self.set_holdings(symbol, final_w)
                total_momentum_weight += final_w
                log_msg += f"{symbol.value:<10} | {final_w:.2%}\n"

        # Handle SHV (Cash Substitute)
        shv_symbol = self.add_equity("SHV", Resolution.DAILY).Symbol
        shv_weight = max(0, 1.0 - total_momentum_weight)
        if shv_weight > 0.01:
            self.set_holdings(shv_symbol, shv_weight)
            log_msg += f"{'SHV (Cash)':<10} | {shv_weight:.2%}\n"

        self.debug(log_msg)

    def apply_safety_switch(self):
        """
        Liquidates all positions and moves 100% into SHV when momentum criteria aren't met.

        Returns
        -------
        None
        """
        self.debug(f"[{self.time}] Safety Switch: Market regimes unfavorable. Moving to Cash (SHV).")
        shv_symbol = self.add_equity("SHV", Resolution.DAILY).Symbol
        self.liquidate()
        self.set_holdings(shv_symbol, 1.0)
