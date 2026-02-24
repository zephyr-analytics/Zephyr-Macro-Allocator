from AlgorithmImports import *
from typing import Dict, List
import numpy as np


class ZephyrMacroAllocator(QCAlgorithm):
    """
    Multi-asset, regime-aware allocation strategy using momentum,
    trend filtering, and volatility-adjusted win-rate confidence.

    Volatility is used ONLY to penalize noisy signals in edge construction.
    There is NO volatility targeting or exposure scaling.
    """

    # ====================================================
    # initialize
    # ====================================================
    def initialize(self) -> None:
        """
        Initialize the Zephyr Macro Allocator algorithm.

        Sets global algorithm parameters, user-configurable options, asset
        universes, indicators, warm-up period, and the scheduled rebalance event.

        The strategy constructs a multi-asset portfolio using:
        - Multi-horizon momentum
        - SMA-based trend filtering
        - Win-rate and volatility-adjusted confidence scoring
        - Optional sector and treasury regime logic

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method configures the algorithm state and does not return a value.
        """
        self.set_start_date(2012, 1, 1)
        self.set_cash(100_000)
        self.set_benchmark("AOR")
        # ============================
        # user options
        # ============================
        self.enable_sma_filter = True
        self.enable_treasury_kill_switch = True
        self.enable_volume_momentum = True

        self.winrate_lookback = 21
        self.vol_lookback = 252

        self.sma_period = 147
        self.bond_sma_period = 126
        self.crypto_cap = 0.10

        self.volume_momentum_period = 20
        self.volume_momentum_lookback = 50

        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        # ============================
        # asset groups
        # ============================
        self.group_tickers = {
            "real": ["GLD", "DBC"],
            "corp_bonds": ["VCSH", "VCIT", "VCLT"],
            "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
            "high_yield_bonds": ["SHYG", "HYG"],
            "sectors": ["IXN", "IXJ", "IXC", "IXG", "RXI", "KXI", "EXI", "IXP", "MXI", "JXI", "REET"],
            "us_factor": ["MTUM", "ITOT", "VLUE", "USMV", "QUAL", "DGRO", "SIZE"],
            "int_factor": ["IMTM", "EFA", "IVLU", "EFAV", "IQLT", "IGRO", "SCZ"],
            "em_factor": ["EEM", "EVLU", "EEMV", "EQLT", "DGRE", "EEMS"],
            "crypto": ["BTCUSD", "ETHUSD"],
            "cash": ["SHV"]
        }

        self.symbols: Dict[str, List[Symbol]] = {}

        for group, tickers in self.group_tickers.items():
            self.symbols[group] = []
            for ticker in tickers:
                symbol = (
                    self.add_crypto(ticker, Resolution.Daily).Symbol
                    if ticker.endswith("USD")
                    else self.add_equity(ticker, Resolution.Daily).Symbol
                )
                self.symbols[group].append(symbol)
                self.Securities[symbol].FeeModel = ConstantFeeModel(0)

        self.all_symbols = [s for v in self.symbols.values() for s in v]

        # ----------------------------
        # bond universe
        # ----------------------------
        self.bond_symbols = {
            s for g in ["corp_bonds", "treasury_bonds", "high_yield_bonds"]
            for s in self.symbols[g]
        }

        # ============================
        # indicators
        # ============================
        self.smas = {
            s: self.SMA(s, self.sma_period, Resolution.DAILY)
            for s in self.all_symbols if s not in self.bond_symbols
        }

        self.bond_smas = {
            s: self.SMA(s, self.bond_sma_period, Resolution.DAILY)
            for s in self.bond_symbols
        }

        self.set_warmup(
            max(
                self.winrate_lookback,
                self.vol_lookback,
                self.max_lookback,
                self.bond_sma_period
            )
        )

        self.Schedule.On(
            self.DateRules.MonthEnd(self.all_symbols[0]),
            self.TimeRules.BeforeMarketClose(self.all_symbols[0], 5),
            self.rebalance
        )

    # ====================================================
    # trend filter
    # ====================================================
    def passes_trend(self, symbol: Symbol) -> bool:
        """
        Determine whether an asset passes the trend filter.

        Uses a simple moving average (SMA) trend filter. Equity-like assets
        use `self.sma_period`, while bond assets use `self.bond_sma_period`.
        If the SMA filter is disabled, all assets automatically pass.

        Parameters
        ----------
        symbol : Symbol
            The QuantConnect symbol to evaluate.

        Returns
        -------
        bool
            True if the asset price is above its SMA or if trend filtering
            is disabled; False otherwise.
        """
        if not self.enable_sma_filter:
            return True

        sma = (
            self.bond_smas.get(symbol)
            if symbol in self.bond_symbols
            else self.smas.get(symbol)
        )

        return sma and sma.IsReady and self.Securities[symbol].Price > sma.Current.Value

    # ====================================================
    # NEW: treasury trend-only kill helper
    # ====================================================
    def all_treasuries_fail_trend(self) -> bool:
        """
        Check whether all treasury assets fail the trend filter.

        This helper supports the treasury-only kill switch. Risk is disabled
        only if *every* treasury bond asset is trading below its SMA.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if all treasury assets fail the trend filter; False otherwise.
            Always returns False if the SMA filter is disabled.
        """
        if not self.enable_sma_filter:
            return False

        for s in self.symbols["treasury_bonds"]:
            sma = self.bond_smas.get(s)
            if sma and sma.IsReady and self.Securities[s].Price > sma.Current.Value:
                return False 

        return True

    # ====================================================
    # rebalance
    # ====================================================
    def rebalance(self) -> None:
        """
        Perform a full portfolio rebalance using momentum and confidence scoring.

        This method liquidates existing positions, determines regime-qualified assets,
        constructs group-level edges, handles the treasury kill switch, and 
        allocates capital via SetHoldings.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Portfolio holdings are updated via SetHoldings; no value is returned.
        """
        if self.IsWarmingUp:
            return

        self.Liquidate()

        history = self.history(
            self.all_symbols,
            self.max_lookback + 1,
            Resolution.Daily,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )

        closes = history["close"].unstack(0)
        cash_symbol = self.symbols["cash"][0]

        bil_6m = self.six_month_return(cash_symbol, closes)
        self.debug(f"Cash Hurdle: {bil_6m}")
        corp_bonds = self.get_duration_regime_for_group(
            closes, self.symbols["corp_bonds"]
        )
        treasury_bonds = self.get_duration_regime_for_group(
            closes, self.symbols["treasury_bonds"]
        )
        high_yield_bonds = self.get_duration_regime_for_group(
            closes, self.symbols["high_yield_bonds"]
        )

        risk_groups = {
            "real": self.symbols["real"],
            "corp_bonds": corp_bonds,
            "treasury_bonds": treasury_bonds,
            "high_yield_bonds": high_yield_bonds,
            "sectors": self.symbols["sectors"],
            "us_factor": self.symbols["us_factor"],
            "int_factor": self.symbols["int_factor"],
            "crypto": self.symbols["crypto"],
            "em_factor": self.symbols["em_factor"]
        }

        vols = {}
        edges = {}
        group_assets = {}
        group_asset_edges = {}

        # ============================
        # group + asset edge construction
        # ============================
        for group, symbols in risk_groups.items():
            eligible_candidates = {} # Temporary store for all trend-passing assets

            for s in symbols:
                if s not in closes.columns: continue
                if not self.passes_trend(s): continue

                asset_momentum = self.compute_asset_momentum(s, closes)
                asset_6m = self.six_month_return(s, closes)

                # Basic inclusion filters
                if asset_6m < bil_6m or asset_momentum <= 0:
                    continue

                # Volume Momentum Confirmation
                vol_mom = 1.0
                if self.enable_volume_momentum:
                    vol_mom = self.compute_volume_momentum(s, history)

                combined_edge = asset_momentum * (1.0 + np.tanh(vol_mom))
                eligible_candidates[s] = combined_edge

            if not eligible_candidates:
                continue

            # --- TOP 3 SELECTION LOGIC ---
            if group in ["int_factor", "us_factor", "em_factor"]:
                # Logic specifically for Factor groups
                top_3_keys = sorted(eligible_candidates, key=eligible_candidates.get, reverse=True)[:3]
                eligible = top_3_keys
                asset_edges = {s: eligible_candidates[s] for s in top_3_keys}

            elif group == "sectors":
                # Separate logic for Global Sectors
                # This is now independent so you can adjust the count or sorting metric later
                top_3_keys = sorted(eligible_candidates, key=eligible_candidates.get, reverse=True)[:3]
                eligible = top_3_keys
                asset_edges = {s: eligible_candidates[s] for s in top_3_keys}

            else:
                # Default for real, bonds, crypto, etc.
                eligible = list(eligible_candidates.keys())
                asset_edges = eligible_candidates

            # 1. Create weights based only on the Top 3 (or all if not a factor/sector group)
            edge_series = pd.Series(asset_edges)
            asset_weights = edge_series / edge_series.sum()

            # 2. Construct the Weighted Return Series (Representative of the Top 3)
            group_returns_df = closes[eligible].pct_change()
            weighted_group_returns = (group_returns_df * asset_weights).sum(axis=1).dropna()

            if len(weighted_group_returns) < max(self.winrate_lookback, self.vol_lookback):
                continue

            # 3. Group Momentum (Weighted Average of Top 3)
            group_momentum = (edge_series * asset_weights).sum()

            # 4. Calculate Weighted Win Rate & Volatility for the Top 3 basket
            log_group = np.log1p(weighted_group_returns)
            win_rate = float(np.mean(log_group.tail(self.winrate_lookback) > 0))
            group_vol = float(np.std(log_group.tail(self.vol_lookback)) * np.sqrt(252))

            if not np.isfinite(group_vol) or group_vol <= 0:
                continue

            vols[group] = group_vol

            # 5. Volume Momentum Confirmation
            volume_multiplier = 1.0
            if self.enable_volume_momentum:
                asset_vol_momentums = {}
                for s in eligible:
                    vol_mom = self.compute_volume_momentum(s, history)
                    asset_vol_momentums[s] = vol_mom
                
                group_vol_momentum = (pd.Series(asset_vol_momentums) * asset_weights).sum()
                volume_multiplier = 1.0 + np.tanh(group_vol_momentum)

            # 6. Final Confidence & Edge Construction
            confidence = (group_momentum) / (group_vol + 1e-6)

            edges[group] = win_rate * confidence * volume_multiplier

            group_assets[group] = eligible
            group_asset_edges[group] = asset_edges

        if not edges:
            self.debug("No groups moving to cash.")
            self.SetHoldings(cash_symbol, 1.0)
            return

        # ====================================================
        # Lack of groups stop trading.
        # ====================================================
        if len(edges) < 4:
            self.SetHoldings(cash_symbol, 1.0)
            self.Debug("Less than X groups moving to cash.")
            return

        # ============================
        # group weights
        # ============================
        effective_edges = {g: e for g, e in edges.items()}
        total_edge = sum(effective_edges.values())

        weights = {
            g: (effective_edges[g] / total_edge)
            for g in effective_edges
        }

        # crypto cap
        if "crypto" in weights and weights["crypto"] > self.crypto_cap:
            excess = weights["crypto"] - self.crypto_cap
            weights["crypto"] = self.crypto_cap
            others = [g for g in weights if g != "crypto"]
            total_other = sum(weights[g] for g in others)
            for g in others:
                weights[g] += excess * (weights[g] / total_other)

        cash_weight = max(0.0, 1.0 - sum(weights.values()))

        # ============================
        # intra-group allocation (EDGE ONLY)
        # ============================
        for group, group_weight in weights.items():
            symbols = group_assets.get(group, [])
            if not symbols:
                continue

            edges_i = {
                s: group_asset_edges[group][s]
                for s in symbols
            }

            edge_sum = sum(edges_i.values())
            if edge_sum <= 0:
                continue

            for s, edge in edges_i.items():
                self.SetHoldings(s, group_weight * edge / edge_sum)

        self.SetHoldings(cash_symbol, cash_weight)

        self.Debug(
            "GROUPS | "
            + " | ".join(f"{g}:{w:.2f}" for g, w in sorted(weights.items()))
            + f" | cash:{cash_weight:.2f}"
        )


    # ====================================================
    # helpers
    # ====================================================
    def six_month_return(self, symbol: Symbol, closes: pd.DataFrame) -> float:
        """
        Compute the trailing six-month simple return for a single asset.

        Parameters
        ----------
        symbol : Symbol
            The asset symbol to evaluate.
        closes : pandas.DataFrame
            DataFrame of historical close prices indexed by date and symbol.

        Returns
        -------
        float
            Six-month simple return. Returns -inf if insufficient data is available.
        """
        if symbol not in closes.columns or len(closes[symbol]) < 127:
            return -np.inf
        prices = closes[symbol].dropna()
        return float(prices.iloc[-1] / prices.iloc[-127] - 1)


    def get_duration_regime_for_group(self, closes: pd.DataFrame, symbols: List[Symbol]) -> List[Symbol]:
        """
        Select bond assets based on a duration momentum regime.

        Uses short-, intermediate-, and long-duration momentum ordering
        to dynamically select which bonds are eligible for allocation.

        Parameters
        ----------
        closes : pandas.DataFrame
            DataFrame of historical close prices indexed by date and symbol.
        symbols : list[Symbol]
            Ordered list of bond symbols (short to long duration).

        Returns
        -------
        list[Symbol]
            Subset of input symbols representing the active duration regime.
        """
        def momentum(symbol):
            if symbol not in closes.columns:
                return -np.inf
            prices = closes[symbol].replace(0, np.nan).dropna()
            if len(prices) < 127:
                return -np.inf
            return np.mean([
                prices.iloc[-1] / prices.iloc[-22] - 1,
                prices.iloc[-1] / prices.iloc[-64] - 1,
                prices.iloc[-1] / prices.iloc[-127] - 1,
            ])

        valid_symbols = [s for s in symbols if s in closes.columns]
        
        if len(valid_symbols) == 3:
            short, intermediate, long = valid_symbols
            m_long = momentum(long)
            m_int = momentum(intermediate)
            m_short = momentum(short)
            
            if m_long > m_int > m_short:
                return [short, intermediate, long]
            elif m_int > m_short:
                return [short, intermediate]
            else:
                return [short]

        if len(valid_symbols) == 2:
            short, long = valid_symbols
            return [short, long] if momentum(long) > momentum(short) else [short]

        return valid_symbols


    def compute_group_momentum(self, symbols: List[Symbol], closes: pd.DataFrame) -> float:
        """
        Compute aggregate momentum for an asset group.

        Parameters
        ----------
        symbols : list[Symbol]
            Assets belonging to the group.
        closes : pandas.DataFrame
            DataFrame of historical close prices indexed by date and symbol.

        Returns
        -------
        float
            Average group momentum. Returns 0.0 if no valid assets exist.
        """
        values = []
        for symbol in symbols:
            mom = self.compute_asset_momentum(symbol, closes)

            if np.isfinite(mom):
                values.append(mom)

        return float(np.mean(values)) if values else 0.0


    def compute_asset_momentum(self, symbol: Symbol, closes: pd.DataFrame) -> float:
        """
        Compute multi-horizon momentum for a single asset.

        Parameters
        ----------
        symbol : Symbol
            Asset symbol for which momentum is computed.
        closes : pandas.DataFrame
            DataFrame of historical close prices indexed by date and symbol.

        Returns
        -------
        float
            Momentum score for the asset. Returns -inf if insufficient
            historical data is available.
        """
        if symbol not in closes.columns:
            return -np.inf
            
        prices = closes[symbol].replace(0, np.nan).dropna()
        
        if len(prices) < self.max_lookback + 1:
            return -np.inf

        return float(np.mean([
            prices.iloc[-1] / prices.iloc[-(lb + 1)] - 1
            for lb in self.momentum_lookbacks
        ]))


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
