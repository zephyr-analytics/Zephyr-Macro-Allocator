from AlgorithmImports import *
from typing import Dict, List
import numpy as np
import pandas as pd

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
        # self.set_start_date(2012, 1, 1)
        # self.set_cash(100_000)

        # Strategy Parameters
        self.enable_sma_filter = True
        self.min_required_groups = 3
        self.winrate_lookback = 126
        self.vol_lookback = 126
        self.sma_period = 147
        self.bond_sma_period = 126
        self.stock_ema_period = 189 
        self.crypto_cap = 0.10
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
        self.income_etf_tickers = ["VTV", "VUG"]
        self.income_etfs = [
            self.AddEquity(t, Resolution.Daily).Symbol for t in self.income_etf_tickers
        ]

        self._top_sector_stocks = 3
        self.income_top_count = 4
        self.etf_to_stocks: Dict[Symbol, List[Symbol]] = {}
        self.stock_emas = {}

        # Universe Selection for ETFs
        all_parent_etfs = self.sector_etfs + self.income_etfs
        for etf in all_parent_etfs:
            self.AddUniverse(
                self.Universe.ETF(
                    etf, 
                    self.UniverseSettings, 
                    lambda constituents, e=etf: self.filter_etf_constituents(constituents, e)
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
        etf_parents = self.sector_etfs + self.income_etfs
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
            self.TimeRules.BeforeMarketClose("VTI", 5), 
            self.rebalance
        )


    def filter_etf_constituents(self, constituents: List[ETFConstituentUniverse], 
                                etf_symbol: Symbol) -> List[Symbol]:
        """
        Filters ETF constituents based on weight, market listing, and blacklist.

        Parameters
        ----------
        constituents : List[ETFConstituentUniverse]
            List of constituent objects from the ETF.
        etf_symbol : Symbol
            The symbol of the parent ETF.

        Returns
        -------
        List[Symbol]
            The top 30 filtered stock symbols.
        """
        selected = []
        sorted_consts = sorted(
            [c for c in constituents if c.Weight], 
            key=lambda x: x.Weight, 
            reverse=True
        )

        for c in sorted_consts:
            if len(selected) >= 30: 
                break
            
            if c.Symbol.Value in self.blacklist_tickers:
                continue 

            if c.Symbol.ID.Market == Market.USA:
                symbol = c.Symbol
                if not self.Securities.ContainsKey(symbol):
                    res = self.AddEquity(symbol, Resolution.Daily)
                    res.SetFeeModel(ConstantFeeModel(0))
                
                if symbol not in self.stock_emas:
                    self.stock_emas[symbol] = self.EMA(symbol, self.stock_ema_period, Resolution.Daily)
                    self.WarmUpIndicator(symbol, self.stock_emas[symbol], Resolution.Daily)
                    
                selected.append(symbol)

        self.etf_to_stocks[etf_symbol] = selected
        return selected


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
            top_sectors = sorted(qualified_sector_etfs, key=qualified_sector_etfs.get, reverse=True)[:3]

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
        h_etf_income = self.History(self.income_etfs, self.max_lookback + 2, Resolution.Daily)
        qualified_income_etfs = []

        if not h_etf_income.empty:
            ie_closes = h_etf_income["close"].unstack(0)
            for etf in self.income_etfs:
                if etf in ie_closes.columns:
                    if self.passes_trend(etf) and self.compute_asset_momentum(etf, ie_closes) > 0:
                        qualified_income_etfs.append(etf)

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
                            top_stocks = sorted(m_scores, key=m_scores.get, reverse=True)[:self.income_top_count]
                            current_income_symbols.extend(top_stocks)

        current_income_symbols = list(set(current_income_symbols))

        # Unified Hurdle & Allocation
        active_pool = list(set(self.all_symbols + current_sector_symbols + current_income_symbols))
        self.Liquidate()

        history = self.History(active_pool, self.max_lookback + 2, Resolution.Daily)
        if history.empty: return
        closes = history["close"].unstack(0).loc[:, ~history["close"].unstack(0).columns.duplicated()]

        cash_symbol = self.symbols["cash"][0]
        bil_6m = self.six_month_return(cash_symbol, closes)

        t_cand = self.get_duration_regime_for_group(closes, self.symbols["treasury_bonds"])
        t_mom = {s: self.compute_asset_momentum(s, closes) for s in t_cand if s in closes.columns and self.passes_trend(s)}
        t_mom = {s: m for s, m in t_mom.items() if m > 0}
        t_hurdle = (pd.Series(t_mom).dot(pd.Series(t_mom) / sum(t_mom.values()))) if t_mom else 0.0

        risk_groups = {
            "real": self.symbols["real"], 
            "corp_bonds": self.get_duration_regime_for_group(closes, self.symbols["corp_bonds"]),
            "treasury_bonds": t_cand, 
            "high_yield_bonds": self.get_duration_regime_for_group(closes, self.symbols["high_yield_bonds"]),
            "equity_income": current_income_symbols, 
            "crypto": self.symbols["crypto"], 
            "global_sectors": current_sector_symbols
        }

        edges, group_assets, group_asset_edges = {}, {}, {}
        for group, symbols in risk_groups.items():
            eligible, asset_edges = [], {}
            for s in list(dict.fromkeys(symbols)): 
                if s not in closes.columns or not self.passes_trend(s): continue
                m = self.compute_asset_momentum(s, closes)
                if m <= 0 or m < t_hurdle or self.six_month_return(s, closes) < bil_6m: continue
                eligible.append(s)
                asset_edges[s] = m

            if not eligible: continue
            edge_series = pd.Series(asset_edges)
            asset_weights = edge_series / edge_series.sum()
            weighted_rets = (closes[eligible].pct_change() * asset_weights).sum(axis=1).dropna()

            if len(weighted_rets) < self.vol_lookback: continue
            win_rate = float(np.mean(np.log1p(weighted_rets).tail(self.winrate_lookback) > 0))
            vol = float(np.std(np.log1p(weighted_rets).tail(self.vol_lookback)) * np.sqrt(252))
            if vol > 0:
                edges[group] = win_rate * (1.0 + ((edge_series * asset_weights).sum() / vol))
                group_assets[group] = eligible
                group_asset_edges[group] = asset_edges

        is_only_treasuries = (len(edges) == 1 and "treasury_bonds" in edges)
        if len(edges) < self.min_required_groups and not is_only_treasuries:
            self.SetHoldings(cash_symbol, 1.0)
            return

        total_edge = sum(edges.values())
        weights = {g: (e / total_edge) for g, e in edges.items()}

        if "crypto" in weights and weights["crypto"] > self.crypto_cap:
            excess = weights["crypto"] - self.crypto_cap
            weights["crypto"] = self.crypto_cap
            others = [g for g in weights if g != "crypto"]
            if others:
                total_o = sum(weights[g] for g in others)
                for g in others: weights[g] += excess * (weights[g] / total_o)

        for group, g_weight in weights.items():
            e_sum = sum(group_asset_edges[group].values())
            for s in group_assets[group]:
                w = g_weight * (group_asset_edges[group][s] / e_sum)
                self.SetHoldings(s, w)

        if weights:
            final_allocs = {
                s.Value: round(self.Portfolio[s].Quantity * self.Securities[s].Price / self.Portfolio.TotalPortfolioValue, 4) 
                for s in self.Securities.Keys if self.Portfolio[s].Invested
            }
            group_log = ", ".join([f"{g}: {w:.2%}" for g, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)])
            self.Debug(f"--- Group Weights: {group_log} ---")
            self.Debug(f"Rebalance: {', '.join([f'{k}: {v}' for k, v in sorted(final_allocs.items(), key=lambda x: x[1], reverse=True)])}")


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
        if symbol not in closes.columns or len(closes[symbol]) < 127: 
            return -np.inf
        return float(closes[symbol].iloc[-1] / closes[symbol].iloc[-127] - 1)


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
