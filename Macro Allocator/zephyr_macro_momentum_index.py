from AlgorithmImports import *
import numpy as np
import pandas as pd

class ZephyrFullHierarchyAllocator(QCAlgorithm):
    def initialize(self) -> None:
        self.set_start_date(2012, 1, 1)
        self.set_cash(100_000)

        # --- Strategy Parameters ---
        self.winrate_lookback = 21
        self.vol_lookback = 252
        self.stock_ema_period = 189
        self.vol_mom_avg = 10
        self.vol_mom_lookback = 42

        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        # --- Ticker Buckets ---
        self.buckets = {
            "sectors": ["VGT", "VFH", "VHT", "VOX", "VDE", "VIS", "VDC", "VCR", "VAW", "VPU", "VNQ"],
            "factors": ["MTUM", "IUSV", "IUSG", "USMV", "QUAL", "DGRO"],
            "cap_style": ["MGK", "MGV", "MGC", "VUG", "VTV", "VV", "VOT", "VOE", "VO", "VBK", "VBE", "VB"]
        }

        self.UniverseSettings.Resolution = Resolution.Daily
        self.etf_to_stocks = {}
        self.stock_emas = {}
        
        all_etfs = [t for b in self.buckets.values() for t in b]
        for ticker in all_etfs:
            symbol = self.AddEquity(ticker, Resolution.Daily).Symbol
            self.AddUniverse(self.Universe.ETF(symbol, self.UniverseSettings, 
                lambda const, e=symbol: self.filter_constituents(const, e)))

        self.SetWarmUp(max(self.max_lookback, self.vol_lookback) + 5)
        self.Schedule.On(self.DateRules.MonthEnd("VTI"), self.TimeRules.BeforeMarketClose("VTI", 15), self.rebalance)

    def filter_constituents(self, constituents, etf_symbol):
        selected = sorted([c for c in constituents if c.Weight], key=lambda x: x.Weight, reverse=True)
        # Liquid large-caps only
        symbols = [c.Symbol for c in selected if c.Symbol.Value not in ["AMC", "GME"]][:100]
        self.etf_to_stocks[etf_symbol] = symbols
        return symbols

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            if security.Symbol not in self.stock_emas:
                self.stock_emas[security.Symbol] = self.EMA(security.Symbol, self.stock_ema_period, Resolution.Daily)

    def compute_volume_momentum(self, symbol: Symbol, history: pd.DataFrame) -> float:
        """
        Compute volume momentum using EMA of volume changes.

        Calculates the exponential moving average of volume changes to identify
        sustained increases or decreases in trading activity. Positive momentum
        confirms price trends with growing participation.

        Parameters
        ----------
        symbol : Symbol
            Asset symbol for which volume momentum is computed.
        history : pandas.DataFrame
            Historical data containing volume information.

        Returns
        -------
        float
            Normalized volume momentum score. Returns 0.0 if insufficient data.
        """
        try:
            if symbol not in history.index.get_level_values(0):
                return 0.0

            volume_data = history.loc[symbol]['volume'].tail(self.volume_momentum_lookback)

            if len(volume_data) < self.volume_momentum_period + 1:
                return 0.0

            volume_changes = volume_data.pct_change().dropna()

            if len(volume_changes) < self.volume_momentum_period:
                return 0.0

            alpha = 2 / (self.volume_momentum_period + 1)
            volume_ema = volume_changes.ewm(alpha=alpha, adjust=False).mean()

            return float(volume_ema.iloc[-1])
        except:
            return 0.0

    def compute_momentum(self, symbol, closes):
        col = symbol if symbol in closes.columns else (symbol.Value if hasattr(symbol, 'Value') and symbol.Value in closes.columns else None)
        if col is None: return -1.0
        p = closes[col].dropna()
        if len(p) < self.max_lookback + 1: return -1.0
        return float(np.mean([p.iloc[-1] / p.iloc[-(lb + 1)] - 1 for lb in self.momentum_lookbacks]))

    def rebalance(self):
        if self.IsWarmingUp: return
        
        # 1. BUCKET COMPETITION
        all_etf_symbols = [s for b in self.buckets.values() for s in b]
        etf_history = self.History(all_etf_symbols, self.max_lookback + 5, Resolution.Daily)
        if etf_history.empty: return
        etf_closes = etf_history["close"].unstack(0).ffill()

        selected_etfs = []
        for bucket, tickers in self.buckets.items():
            moms = {t: self.compute_momentum(t, etf_closes) for t in tickers if t in etf_closes.columns}
            selected_etfs.extend(sorted(moms, key=moms.get, reverse=True)[:3])

        # 2. CONSTITUENT SELECTION WITH VOL_MOM TANH
        final_group_mapping = {}
        all_potential_stocks = []

        for etf_ticker in selected_etfs:
            candidates = self.etf_to_stocks.get(self.Symbol(etf_ticker), [])
            if not candidates: continue
            
            h_sub = self.History(candidates, self.max_lookback + 5, Resolution.Daily)
            if h_sub.empty: continue
            c_sub = h_sub["close"].unstack(0).ffill()
            
            valid_asset_edges = {}
            for s in candidates:
                if s not in c_sub.columns: continue
                
                # Trend Filter
                ema = self.stock_emas.get(s)
                if not (ema and ema.IsReady and self.Securities[s].Price > ema.Current.Value): continue
                
                # Momentum & Vol Momentum (tanh)
                mom = self.compute_momentum(s, c_sub)
                if mom <= 0: continue
                
                v_mom_score = self.compute_volume_momentum(s, h_sub)
                # Combine using tanh to amplify/penalize based on volume
                combined_edge = mom * (1.0 + np.tanh(v_mom_score))
                valid_asset_edges[s] = combined_edge
            
            # Select top 4 stocks per ETF based on Volume-Adjusted Edge
            top_4 = sorted(valid_asset_edges, key=valid_asset_edges.get, reverse=True)[:4]
            if top_4:
                final_group_mapping[etf_ticker] = {s: valid_asset_edges[s] for s in top_4}
                all_potential_stocks.extend(top_4)

        if not final_group_mapping:
            self.Liquidate(); return

        # 3. CONVICTION WEIGHTING (9-Group Competition)
        stock_history = self.History(list(set(all_potential_stocks)), self.vol_lookback + 5, Resolution.Daily)
        stock_closes = stock_history["close"].unstack(0).ffill()

        group_convictions = {}
        total_conviction = 0.0

        for etf_ticker, stock_edges in final_group_mapping.items():
            stocks = list(stock_edges.keys())
            edge_sum = sum(stock_edges.values())
            weights = pd.Series(stock_edges) / edge_sum

            group_rets = stock_closes[stocks].pct_change().dropna().dot(weights)
            if len(group_rets) < self.winrate_lookback: continue

            vol = max(float(group_rets.tail(self.vol_lookback).std() * np.sqrt(252)), 0.0001)
            win_rate = float((group_rets.tail(self.winrate_lookback) > 0).mean())
            group_edge = (pd.Series(stock_edges) * weights).sum()

            # Conviction calculation consistent with Macro script
            conviction = (group_edge / vol) * win_rate
            group_convictions[etf_ticker] = conviction
            total_conviction += conviction

        # 4. FINAL ALLOCATION & AGGREGATED DEBUGGING
        if total_conviction <= 0:
            self.Liquidate()
            return

        final_targets = {}
        group_weights = {}

        for etf_ticker, conviction in group_convictions.items():
            group_share = conviction / total_conviction
            group_weights[etf_ticker] = group_share

            stock_edges = final_group_mapping[etf_ticker]
            edge_sum = sum(stock_edges.values())

            for s, edge in stock_edges.items():
                # Aggregate weights for stocks appearing in multiple buckets
                total_asset_weight = group_share * (edge / edge_sum)
                final_targets[s] = final_targets.get(s, 0) + total_asset_weight

        # Execute Holdings
        self.Liquidate()
        # Sort targets by weight descending for easier reading
        sorted_assets = sorted(final_targets.items(), key=lambda x: x[1], reverse=True)
        sorted_groups = sorted(group_weights.items(), key=lambda x: x[1], reverse=True)

        for s, w in sorted_assets:
            if w > 0.001: # Filter out negligible weights
                self.SetHoldings(s, w)

        # Build Two-Line Output
        groups_line = " | ".join([f"{ticker}: {w:.1%}" for ticker, w in sorted_groups])
        assets_line = ", ".join([f"{s.Value}: {w:.1%}" for s, w in sorted_assets if w > 0])

        self.Debug(f"REBALANCE | GROUPS: [{groups_line}]")
        self.Debug(f"REBALANCE | ASSETS: [{assets_line}] | TOTAL: {sum(final_targets.values()):.1%}")
