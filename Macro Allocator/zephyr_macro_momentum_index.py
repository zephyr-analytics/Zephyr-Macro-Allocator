from AlgorithmImports import *
from typing import Dict, List
import numpy as np
import pandas as pd
import numpy as np


class ZephyrMacroAllocator(QCAlgorithm):
    """
    A multi-asset macro allocation algorithm that uses dual-momentum, 
    trend-following filters, and volatility-adjusted weighting across 
    sectors, income equities, and alternative assets.
    """

    def initialize(self) -> None:
        """
        Initializes the algorithm state, symbols, universes, and 
        scheduling for the rebalancing logic.
        """
        self.set_start_date(2012, 1, 1)
        self.set_cash(100_000)

        # Strategy Parameters
        self.enable_sma_filter = True
        self.min_required_groups = 3
        self.winrate_lookback = 126
        self.vol_lookback = 126
        self.sma_period = 147
        self.bond_sma_period = 126
        self.stock_ema_period = 189 
        self.crypto_cap = 0.10

        self.cash_weight = 0.25
        self.cash_safety_scale = 0.50
        self.bond_safety_scale = 0.50

        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        # 1. Global Sector Config
        self.sector_tickers = [
            "VGT", "VFH", "VHT", "VOX", "VDE", 
            "VIS", "VDC", "VCR", "VAW", "VPU", "VNQ"
        ]
        self.blacklist_tickers = ["AMC", "GME"]
        self.sector_etfs = [
            self.AddEquity(t, Resolution.Daily).Symbol for t in self.sector_tickers
        ]

        # 2. Equity Income Config
        self.factor_etf_tickers = ["MTUM", "IUSV", "IUSG", "USMV", "QUAL", "DGRO"]
        self.factor_etfs = [
            self.AddEquity(t, Resolution.Daily).Symbol for t in self.factor_etf_tickers
        ]

        self.max_sector_etfs = 3
        self._top_sector_stocks = 4
        self.max_factor_etfs = 3
        self.top_factor_stocks = 4

        self.etf_to_stocks: Dict[Symbol, List[Symbol]] = {}
        self.stock_emas = {}

        # 1. Update the settings BEFORE the loop
        # This ensures newly added securities get historical data for indicators automatically
        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.Leverage = 1.0

        # 2. Universe Selection for ETFs
        all_parent_etfs = self.sector_etfs + self.factor_etfs
        for etf in all_parent_etfs:
            self.AddUniverse(
                self.Universe.ETF(
                    etf, 
                    self.UniverseSettings, # Now uses the updated settings above
                    lambda constituents, e=etf: self.filter_constituents(constituents, e)
                )
            )

        # 3. Macro Asset Groups
        self.group_tickers = {
            "real": ["GLD", "DBC"],
            "corp_bonds": ["VCSH", "VCIT", "VCLT"],
            "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
            "high_yield_bonds": ["SHYG", "HYG"],
            "crypto": ["BTCUSD", "ETHUSD"],
            "cash": ["SHV"]
        }

        self.symbols: Dict[str, List[Symbol]] = {}
        for group, tickers in self.group_tickers.items():
            self.symbols[group] = []
            for ticker in tickers:
                if ticker.endswith("USD"):
                    security = self.AddCrypto(ticker, Resolution.Daily)
                else:
                    security = self.AddEquity(ticker, Resolution.Daily)
                
                security.SetFeeModel(ConstantFeeModel(0))
                self.symbols[group].append(security.Symbol)

        self.all_symbols = [s for v in self.symbols.values() for s in v]
        etf_parents = self.sector_etfs + self.factor_etfs
        self.bond_symbols = {
            s for g in ["corp_bonds", "treasury_bonds", "high_yield_bonds"] 
            for s in self.symbols[g]
        }

        # Indicators
        self.smas = {}
        for s in (self.all_symbols + etf_parents):
            if s not in self.bond_symbols:
                self.smas[s] = self.SMA(s, self.sma_period, Resolution.Daily)

        self.bond_smas = {
            s: self.SMA(s, self.bond_sma_period, Resolution.Daily) 
            for s in self.bond_symbols
        }

        # Warm-up and Scheduling
        self.SetWarmUp(
            max(self.winrate_lookback, self.vol_lookback, 
                self.max_lookback, self.stock_ema_period)
        )

        self.Schedule.On(
            self.DateRules.MonthEnd("VTI"), 
            self.TimeRules.BeforeMarketClose("VTI", 45), 
            self.rebalance
        )


    def filter_constituents(self, constituents: List[ETFConstituentUniverse], etf_symbol: Symbol) -> List[Symbol]:
        # 1. Filter out None weights first to prevent sorted() from crashing
        valid_constituents = [c for c in constituents if c.Weight is not None and c.Weight > 0]

        # 2. Sort by weight and filter out blacklisted tickers
        sorted_consts = sorted(valid_constituents, key=lambda x: x.Weight, reverse=True)

        # 3. Extract Symbols (limiting to top 30 for performance)
        selected = [
            c.Symbol for c in sorted_consts 
            if c.Symbol.Value not in self.blacklist_tickers
        ][:75]

        # 4. Update the map for your rebalance logic
        self.etf_to_stocks[etf_symbol] = selected
        return selected


    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            security.SetFeeModel(ConstantFeeModel(0))
            
            # Only register indicators for new stocks
            if symbol not in self.stock_emas and symbol not in self.bond_symbols:
                self.stock_emas[symbol] = self.EMA(symbol, self.stock_ema_period, Resolution.Daily)

        for security in changes.RemovedSecurities:
            self.symbols.pop(security.Symbol, None)
            self.stock_emas.pop(security.Symbol, None)


    def passes_trend(self, symbol: Symbol) -> bool:
        """
        Determines if an asset is in an uptrend based on SMA/EMA filters.

        Parameters
        ----------
        symbol : Symbol
            The asset symbol to check.

        Returns
        -------
        bool
            True if the asset price is above its moving average or filters are disabled.
        """
        if not self.enable_sma_filter: 
            return True
            
        indicator = (self.stock_emas.get(symbol) or 
                     self.bond_smas.get(symbol) or 
                     self.smas.get(symbol))
        
        return (symbol in self.Securities and 
                self.Securities[symbol].Price > 0 and 
                indicator and indicator.IsReady and 
                self.Securities[symbol].Price > indicator.Current.Value)


    def rebalance(self) -> None:
        """
        Main execution logic for monthly rebalancing including gatekeeper filters,
        momentum scoring, and portfolio construction.
        """
        if self.IsWarmingUp: 
            return
        investable_limit = 1.0 - self.cash_weight
        self.Liquidate()
        edges, group_assets, group_asset_edges = {}, {}, {}
        intra_group_weights = {}

        def get_valid_subset(all_symbols):
            """Internal helper to filter for Market Cap and Liquidity."""
            valid = []
            for s in all_symbols:
                if s.Value in self.blacklist_tickers: 
                    continue
                if s in self.Securities and self.Securities[s].Fundamentals:
                    m_cap = self.Securities[s].Fundamentals.MarketCap
                    dollar_vol = self.Securities[s].Price * self.Securities[s].Volume

                    if m_cap >= 5e9 and dollar_vol > 1e6: 
                        if s not in self.stock_emas:
                            self.stock_emas[s] = self.EMA(s, self.stock_ema_period, Resolution.Daily)
                        valid.append(s)
            return valid

        # Sector Selection
        current_sector_symbols = []
        history_sect = self.History(self.sector_etfs, self.max_lookback + 2, Resolution.Daily)

        if not history_sect.empty:
            sector_closes = history_sect["close"].unstack(0)
            qualified_sector_etfs = {
                s: self.compute_asset_momentum(s, sector_closes) 
                for s in self.sector_etfs 
                if s in sector_closes.columns and self.passes_trend(s)
            }

            qualified_sector_etfs = {s: m for s, m in qualified_sector_etfs.items() if m > 0}
            top_sectors = sorted(qualified_sector_etfs, key=qualified_sector_etfs.get, reverse=True)[:self.max_sector_etfs]

            if top_sectors:
                raw_sector_candidates = [s for etf in top_sectors for s in self.etf_to_stocks.get(etf, [])]
                valid_sector_candidates = get_valid_subset(raw_sector_candidates)

                if valid_sector_candidates:
                    h_stocks = self.History(valid_sector_candidates, self.max_lookback + 2, Resolution.Daily)
                    if not h_stocks.empty:
                        s_closes = h_stocks["close"].unstack(0)
                        for etf in top_sectors:
                            candidates = [s for s in self.etf_to_stocks.get(etf, []) if s in valid_sector_candidates]
                            m_scores = {s: self.compute_asset_momentum(s, s_closes) 
                                       for s in candidates if s in s_closes.columns}
                            if m_scores:
                                picks = sorted(m_scores, key=m_scores.get, reverse=True)[:self._top_sector_stocks]
                                current_sector_symbols.extend(picks)

        current_sector_symbols = list(set(current_sector_symbols))

        # Income Selection
        current_income_symbols = []
        h_etf_income = self.History(self.factor_etfs, self.max_lookback + 2, Resolution.Daily)
        qualified_income_etfs = []

        if not h_etf_income.empty:
            ie_closes = h_etf_income["close"].unstack(0)

            # 1. Create a list of tuples (etf, momentum_score) for those that pass the trend
            scored_etfs = []
            for etf in self.factor_etfs:
                if etf in ie_closes.columns:
                    m_score = self.compute_asset_momentum(etf, ie_closes)
                    if self.passes_trend(etf) and m_score > 0:
                        scored_etfs.append((etf, m_score))
            
            # 2. Sort by momentum score (index 1 of the tuple) descending
            scored_etfs.sort(key=lambda x: x[1], reverse=True)
            
            # 3. Slice the list to your desired cap
            qualified_income_etfs = [etf for etf, score in scored_etfs[:self.max_factor_etfs]]

        if qualified_income_etfs:
            raw_income_pool = list(set([s for etf in qualified_income_etfs for s in self.etf_to_stocks.get(etf, [])]))
            valid_income_pool = get_valid_subset(raw_income_pool)
            if valid_income_pool:
                h_income = self.History(valid_income_pool, self.max_lookback + 2, Resolution.Daily)
                if not h_income.empty:
                    i_closes = h_income["close"].unstack(0)
                    for etf in qualified_income_etfs:
                        etf_specific = [s for s in self.etf_to_stocks.get(etf, []) if s in valid_income_pool]
                        m_scores = {s: self.compute_asset_momentum(s, i_closes) 
                                   for s in etf_specific if s in i_closes.columns}
                        if m_scores:
                            top_stocks = sorted(m_scores, key=m_scores.get, reverse=True)[:self.top_factor_stocks]
                            current_income_symbols.extend(top_stocks)

        current_income_symbols = list(set(current_income_symbols))

        # Unified Hurdle & Allocation
        active_pool = list(set(self.all_symbols + current_sector_symbols + current_income_symbols))

        history = self.History(active_pool, self.max_lookback + 2, Resolution.Daily)
        if history.empty: return
        closes = history["close"].unstack(0).loc[:, ~history["close"].unstack(0).columns.duplicated()]

        cash_symbol = self.symbols["cash"][0]
        bil_6m = self.six_month_return(cash_symbol, closes)

        t_cand = self.get_duration_regime_for_group(closes, self.symbols["treasury_bonds"])
        t_mom = {s: self.compute_asset_momentum(s, closes) for s in t_cand if s in closes.columns and self.passes_trend(s)}
        t_mom = {s: m for s, m in t_mom.items() if m > 0}
        t_hurdle = (pd.Series(t_mom).dot(pd.Series(t_mom) / sum(t_mom.values()))) if t_mom else 0.0
        
        # self.debug(f"T-Hudle: {t_hurdle}, C-Hurdle: {bil_6m}")

        risk_groups = {
            "real": self.symbols["real"], 
            "corp_bonds": self.get_duration_regime_for_group(closes, self.symbols["corp_bonds"]),
            "treasury_bonds": t_cand, 
            "high_yield_bonds": self.get_duration_regime_for_group(closes, self.symbols["high_yield_bonds"]),
            "equity_factor": current_income_symbols, 
            "crypto": self.symbols["crypto"], 
            "global_sectors": current_sector_symbols
        }

        # 1. GROUP CONVICTION ANALYSIS (Pass 1)
        group_weights = {}
        group_assets = {}
        potential_group_meta = {}
        total_conviction = 0.0

        for group, symbols in risk_groups.items():
            unique_symbols = list(dict.fromkeys(symbols))
            if not unique_symbols: continue
            
            # Gather momentum for all group assets to establish internal proportions
            # Even if an asset fails later, it must be part of the initial 'denominator'
            m_values = {s: max(self.compute_asset_momentum(s, closes), 0) for s in unique_symbols if s in closes.columns}
            total_m = sum(m_values.values())
            
            if total_m <= 0: continue

            # Calculate Conviction for the group (Internal Momentum Weighting)
            edge_series = pd.Series(m_values)
            asset_w = edge_series / total_m
            
            # Risk-adjusting the edge: Volatility and Win Rate
            p_rets = closes[list(m_values.keys())].pct_change().dropna().dot(asset_w)
            vol = max(float(np.log1p(p_rets).tail(self.vol_lookback).std() * np.sqrt(252)), 0.0001)
            win_rate = float((p_rets.tail(self.winrate_lookback) > 0).mean())
            
            # Conviction = (Avg Edge / Vol) * Win Rate
            conviction = ((edge_series * asset_w).sum() / vol) * win_rate
            
            potential_group_meta[group] = {
                'conviction': conviction,
                'm_values': m_values,
                'total_m': total_m
            }
            total_conviction += conviction

        # 2. FILTERING & LEAKAGE ATTRIBUTION (Pass 2)
        failed_to_cash_weight = 0.0
        failed_to_treasury_weight = 0.0
        
        final_group_weights = {} # Use a new dict to track groups that ACTUALLY have assets

        for group, meta in potential_group_meta.items():
            group_potential_weight = (meta['conviction'] / total_conviction) * investable_limit
            is_treasury = (group == "treasury_bonds")
            
            for s, m in meta['m_values'].items():
                asset_share = (m / meta['total_m']) * group_potential_weight
                
                # Filter Checks
                if s not in closes.columns or not self.passes_trend(s):
                    failed_to_cash_weight += asset_share
                    continue
                
                s_6m = self.six_month_return(s, closes)
                if not is_treasury:
                    if s_6m < bil_6m:
                        failed_to_cash_weight += asset_share
                        continue
                    if m < t_hurdle:
                        failed_to_treasury_weight += asset_share
                        continue

                # Asset PASSED: Now we can safely say the group is active
                if group not in group_assets: 
                    group_assets[group] = {}
                group_assets[group][s] = m
            
            # CRITICAL FIX: Only add to group_weights if the group actually has assets surviving
            if group in group_assets and len(group_assets[group]) > 0:
                final_group_weights[group] = meta['conviction']

        # Update the reference for the Fail-Safe checks
        group_weights = final_group_weights 

        # 3. FINAL ALLOCATION & REDISTRIBUTION
        final_targets = {cash_symbol: self.cash_weight + failed_to_cash_weight}

        # CONDITION 1: Extreme Failure (Min groups check)
        if len(group_weights) < self.min_required_groups and "treasury_bonds" not in group_weights:
            self.Debug(f"FAIL-SAFE: {len(group_weights)} groups active. Total Cash.")
            self.SetHoldings(cash_symbol, 1.0)
            return

        # CONDITION 2: Treasury Safety Net
        elif len(group_weights) < self.min_required_groups and "treasury_bonds" in group_weights:
            self.Debug(f"FAIL-SAFE: {len(group_weights)} groups active. Moving to Treasuries.")
            
            # 1. Isolate the Treasury group assets first
            t_assets = group_assets["treasury_bonds"]
            t_sum = sum(t_assets.values())
            
            # 2. POP ALL OTHER GROUPS: Overwrite group_weights to contain ONLY treasury_bonds
            # We set it to 1.0 to ensure 100% of the 'investable_limit' goes here
            group_weights = {"treasury_bonds": 1.0}

            # 3. Clear the final_targets so old equity/crypto targets don't persist
            final_targets = {} 
            
            # 4. Map the treasury sub-assets
            for s, m in t_assets.items():
                final_targets[s] = (m / t_sum) * investable_limit
                
            # 5. Execute with the cleaned dictionaries
            self.ExecuteTargets(final_targets, group_weights, investable_limit)
            return

        # CONDITION 3: Normal Operation
        else:
            # Normalize active group conviction scores for redistribution (e.g. Crypto Cap)
            total_active_conviction = sum(group_weights.values())
            active_group_allocs = {g: (c / total_active_conviction) for g, c in group_weights.items()}

            # Apply Crypto Cap Logic
            if "crypto" in active_group_allocs and active_group_allocs["crypto"] > self.crypto_cap:
                excess = active_group_allocs["crypto"] - self.crypto_cap
                active_group_allocs["crypto"] = self.crypto_cap
                others = [g for g in active_group_allocs if g != "crypto"]
                if others:
                    denom = sum(active_group_allocs[g] for g in others)
                    for g in others:
                        active_group_allocs[g] += excess * (active_group_allocs[g] / denom)

            # Assign failures to Treasury pool
            if failed_to_treasury_weight > 0 and "treasury_bonds" in group_assets:
                t_assets = group_assets["treasury_bonds"]
                t_sum = sum(t_assets.values())
                for s, m in t_assets.items():
                    final_targets[s] = final_targets.get(s, 0) + (m / t_sum) * failed_to_treasury_weight

            # Distribute surviving capital (investable - leaks)
            remaining_pie = investable_limit - failed_to_cash_weight - failed_to_treasury_weight
            
            for group, g_share in active_group_allocs.items():
                actual_g_weight = g_share * remaining_pie
                assets = group_assets[group]
                a_sum = sum(assets.values())
                for s, m in assets.items():
                    final_targets[s] = final_targets.get(s, 0) + (m / a_sum) * actual_g_weight

            self.ExecuteTargets(final_targets, group_weights, investable_limit)


    def ExecuteTargets(self, final_targets: Dict[Symbol, float], group_weights: Dict[str, float], investable_limit: float) -> None:
            """
            Executes holdings and provides detailed portfolio attribution logs.
            """
            # 1. Actual Execution
            for s, w in final_targets.items():
                if w > 0:
                    self.SetHoldings(s, w)

            # 2. Corrected Debugs
            actual_total = sum(final_targets.values())
            
            # Calculate normalized group weights for logging (showing their relative conviction)
            total_g_score = sum(group_weights.values()) if group_weights else 1
            group_log = ", ".join([
                f"{g}: {(w/total_g_score):.2%}" 
                for g, w in sorted(group_weights.items(), key=lambda x: x[1], reverse=True)
            ])

            self.Debug(f"--- Conviction Weights: {group_log} ---")
            # Note: actual_total should be ~1.00 because it includes baseline cash + attributed failures
            self.Debug(f"TOTAL PORTFOLIO LOAD: {actual_total:.4f} (Baseline Cash: {self.cash_weight})")

            # 3. Map for display (Assets)
            display_targets = { 
                (s.Value if hasattr(s, 'Value') else str(s)): round(w, 4) 
                for s, w in final_targets.items() 
            }
            sorted_targets = sorted(display_targets.items(), key=lambda x: x[1], reverse=True)
            self.Debug(f"Final Allocations: {', '.join([f'{k}: {v}' for k, v in sorted_targets])}")

    def six_month_return(self, symbol: Symbol, closes: pd.DataFrame) -> float:
        """
        Calculates the 6-month (126-day) return for an asset.

        Parameters
        ----------
        symbol : Symbol
            Asset symbol.
        closes : pd.DataFrame
            DataFrame containing historical close prices.

        Returns
        -------
        float
            The trailing 6-month return.
        """
        if symbol not in closes.columns: 
            return -np.inf

        # Drop NaNs for this specific asset to ignore days it didn't trade
        series = closes[symbol].dropna()

        if len(series) < 127: 
            return -np.inf

        # Now iloc targets the 127th actual price point
        return float(series.iloc[-1] / series.iloc[-127] - 1)


    def get_duration_regime_for_group(self, closes: pd.DataFrame, symbols: List[Symbol]) -> List[Symbol]:
        """
        Selects the appropriate bond duration based on momentum hierarchy.

        Parameters
        ----------
        closes : pd.DataFrame
            Historical prices.
        symbols : List[Symbol]
            List of symbols representing short, intermediate, and long durations.

        Returns
        -------
        List[Symbol]
            The subset of symbols that meet the duration regime criteria.
        """
        def mom(symbol):
            if symbol not in closes.columns: return -np.inf
            p = closes[symbol].dropna()
            return np.mean(
                [p.iloc[-1]/p.iloc[-22]-1, p.iloc[-1]/p.iloc[-64]-1, p.iloc[-1]/p.iloc[-127]-1]
            ) if len(p) >= 127 else -np.inf

        valid = [s for s in symbols if s in closes.columns]
        if len(valid) == 3:
            m_s, m_i, m_l = mom(valid[0]), mom(valid[1]), mom(valid[2])
            if m_l > m_i > m_s: return valid
            return valid[:2] if m_i > m_s else [valid[0]]
        return valid


    def compute_asset_momentum(self, symbol: Symbol, closes: pd.DataFrame) -> float:
        """
        Computes multi-lookback average momentum for an asset.

        Parameters
        ----------
        symbol : Symbol
            Asset symbol.
        closes : pd.DataFrame
            Historical price DataFrame.

        Returns
        -------
        float
            The average return across all defined lookback periods.
        """
        if symbol not in closes.columns: 
            return -np.inf
        p = closes[symbol].replace(0, np.nan).dropna()
        if len(p) < self.max_lookback + 1: 
            return -np.inf

        return float(np.mean([p.iloc[-1] / p.iloc[-(lb + 1)] - 1 for lb in self.momentum_lookbacks]))
