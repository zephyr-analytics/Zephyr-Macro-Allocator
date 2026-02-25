from AlgorithmImports import *
from typing import Dict, List
import numpy as np
import pandas as pd

class ZephyrMacroAllocator(QCAlgorithm):
    def initialize(self) -> None:
        self.set_start_date(2012, 1, 1)
        self.set_cash(100_000)
        self.set_benchmark("AOR")

        # Configuration
        self.enable_sma_filter = True
        self.enable_volume_momentum = True
        self.use_vol_adjusted = True

        self.vol_lookback = 252
        self.sma_period = 147
        self.bond_sma_period = 126
        self.crypto_cap = 0.05
        self.volume_momentum_period = 10
        self.volume_momentum_lookback = 63

        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        # Asset Definitions
        self.group_tickers = {
            "real": ["GLD", "DBC"],
            "corp_bonds": ["VCSH", "VCIT", "VCLT"],
            "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
            "sectors": ["VGT", "VCR", "VDC", "VHT", "VPU", "VFH", "VDE", "VIS", "VAW", "VOX", "VNQ"],
            "us_factor": ["MTUM", "IUSV", "IUSG", "USMV", "QUAL", "DGRO", "SIZE"],
            "us_cap_style": ["MGK", "MGV", "MGC", "VUG", "VTV", "VV", "VOT", "VOE", "VO", "VBK", "VBR", "VB"],
            "crypto": ["BTCUSD", "ETHUSD"],
            "cash": ["SHV"]
        }

        self.symbols = {}
        for group, tickers in self.group_tickers.items():
            self.symbols[group] = []
            for ticker in tickers:
                if group == "crypto":
                    # Use AddCrypto for BTCUSD/ETHUSD
                    symbol = self.add_crypto(ticker, Resolution.Daily).Symbol
                else:
                    symbol = self.add_equity(ticker, Resolution.Daily).Symbol

                self.symbols[group].append(symbol)
                self.Securities[symbol].FeeModel = ConstantFeeModel(0)

        self.all_symbols = [s for v in self.symbols.values() for s in v]
        self.bond_symbols = {s for g in ["corp_bonds", "treasury_bonds"] for s in self.symbols[g]}

        self.smas = {s: self.SMA(s, self.sma_period, Resolution.DAILY) for s in self.all_symbols if s not in self.bond_symbols}
        self.bond_smas = {s: self.SMA(s, self.bond_sma_period, Resolution.DAILY) for s in self.bond_symbols}

        self.set_warmup(max(self.vol_lookback, self.max_lookback))

        self.Schedule.On(
            self.DateRules.MonthEnd(self.all_symbols[0]),
            self.TimeRules.BeforeMarketClose(self.all_symbols[0], 5),
            self.rebalance
        )

    def passes_trend(self, symbol: Symbol) -> bool:
        if not self.enable_sma_filter: return True
        sma = self.bond_smas.get(symbol) if symbol in self.bond_symbols else self.smas.get(symbol)
        return sma and sma.IsReady and self.Securities[symbol].Price > sma.Current.Value

    def rebalance(self) -> None:
        if self.IsWarmingUp: return
        
        history = self.history(self.all_symbols, self.max_lookback + 1, Resolution.Daily, data_normalization_mode=DataNormalizationMode.TOTAL_RETURN)
        if history.empty: return
        
        closes = history["close"].unstack(0)
        cash_symbol = self.symbols["cash"][0]
        
        risk_groups = {
            "real": self.symbols["real"],
            "corp_bonds": self.get_duration_regime_for_group(closes, self.symbols["corp_bonds"]),
            "treasury_bonds": self.get_duration_regime_for_group(closes, self.symbols["treasury_bonds"]),
            "sectors": self.symbols["sectors"],
            "us_factor": self.symbols["us_factor"],
            "crypto": self.symbols["crypto"],
            "us_cap_style": self.symbols["us_cap_style"],
            "cash": self.symbols["cash"]
        }

        final_asset_scores = {}
        group_totals = {}

        # 1. Process Assets and calculate scores
        for group, symbols in risk_groups.items():
            eligible_candidates = {} 
            for s in symbols:
                if s not in closes.columns: continue
                if group != "cash" and not self.passes_trend(s): continue

                asset_mom = self.compute_asset_momentum(s, closes)
                if asset_mom <= 0: continue

                vol_mom = self.compute_volume_momentum(s, history) if self.enable_volume_momentum else 0.0
                asset_edge = asset_mom * (1.0 + np.tanh(vol_mom))

                if asset_edge <= 0:
                    continue

                eligible_candidates[s] = asset_edge

            if not eligible_candidates: continue

            selected_assets = list(eligible_candidates.keys())
            if group in ["us_factor", "sectors", "us_cap_style"]:
                selected_assets = sorted(eligible_candidates, key=eligible_candidates.get, reverse=True)[:3]

            group_returns = closes[selected_assets].pct_change().mean(axis=1).dropna()
            if len(group_returns) < self.vol_lookback: continue

            log_returns = np.log1p(group_returns)
            group_vol = float(np.std(log_returns.tail(self.vol_lookback)) * np.sqrt(252))

            confidence = (np.mean([eligible_candidates[s] for s in selected_assets])) / (group_vol + 1e-6) if self.use_vol_adjusted else 1.0
            volume_multiplier = np.tanh(np.mean([self.compute_volume_momentum(s, history) for s in selected_assets])) if self.enable_volume_momentum else 1.0

            group_multiplier = (confidence * volume_multiplier)

            if group_multiplier <= 0:
                continue

            current_group_total = 0
            for s in selected_assets:
                score = group_multiplier * eligible_candidates[s]
                final_asset_scores[s] = score
                current_group_total += score

            group_totals[group] = current_group_total

        # 2. Breadth Check
        active_risk_groups = [g for g in group_totals.keys() if g != "cash"]
        if len(active_risk_groups) <= 3:
            self.Debug(f"Breadth Low ({len(active_risk_groups)} risk groups). Moving to Cash.")
            self.Liquidate()
            self.SetHoldings(cash_symbol, 1.0)
            return

        # 3. Final Allocation & Group Capping
        total_score = sum(final_asset_scores.values())
        if total_score <= 0:
            self.SetHoldings(cash_symbol, 1.0)
            return

        # Calculate initial group weights
        group_weights = {g: group_totals[g] / total_score for g in group_totals}
        exempt_groups = ["cash", "corp_bonds", "treasury_bonds"]

        # Iterative redistribution to handle the 25% group cap AND 10% crypto cap
        weights_to_redistribute = group_weights.copy()
        for _ in range(10): # Increased iterations for stability
            excess = 0.0

            for g, w in weights_to_redistribute.items():
                # Apply specific 10% cap to crypto group
                if g == "crypto" and w > self.crypto_cap:
                    excess += (w - self.crypto_cap)
                    weights_to_redistribute[g] = self.crypto_cap
                # Apply 25% cap to other risk groups
                elif g not in exempt_groups and g != "crypto" and w > 0.25:
                    excess += (w - 0.25)
                    weights_to_redistribute[g] = 0.25

            if excess <= 1e-6: break

            # Redistribute to groups that aren't already capped
            eligible_groups = [
                g for g in weights_to_redistribute 
                if (g in exempt_groups) or 
                   (g == "crypto" and weights_to_redistribute[g] < self.crypto_cap) or 
                   (g not in exempt_groups and g != "crypto" and weights_to_redistribute[g] < 0.25)
            ]
            
            if not eligible_groups: break

            denom = sum(weights_to_redistribute[g] for g in eligible_groups)
            if denom > 0:
                for g in eligible_groups:
                    weights_to_redistribute[g] += excess * (weights_to_redistribute[g] / denom)

        self.Liquidate()

        # 4. Asset Allocation based on Adjusted Group Weights
        for group, g_weight in weights_to_redistribute.items():
            group_symbols = [s for s in risk_groups[group] if s in final_asset_scores]
            group_score_sum = sum(final_asset_scores[s] for s in group_symbols)
            
            if group_score_sum <= 0: continue
            
            for s in group_symbols:
                # Proportional weight within the group
                # Normalization is now inherent to the group-level redistribution
                s_weight = (final_asset_scores[s] / group_score_sum) * g_weight
                self.SetHoldings(s, s_weight)

        # Logging
        group_logs = [f"{g}: {weights_to_redistribute[g]:.2%}" for g in sorted(weights_to_redistribute)]
        self.Debug(f"GROUPS (Capped) | {' | '.join(group_logs)}")

    # --- Momentum and Helper Methods remain the same ---
    def compute_asset_momentum(self, symbol: Symbol, closes: pd.DataFrame) -> float:
        prices = closes[symbol].replace(0, np.nan).dropna()
        if len(prices) < self.max_lookback + 1: return -np.inf
        return float(np.mean([prices.iloc[-1] / prices.iloc[-(lb + 1)] - 1 for lb in self.momentum_lookbacks]))

    def compute_volume_momentum(self, symbol: Symbol, history: pd.DataFrame) -> float:
        try:
            volume_data = history.loc[symbol]['volume'].tail(self.volume_momentum_lookback)
            volume_changes = volume_data.pct_change().dropna()
            return float(volume_changes.ewm(span=self.volume_momentum_period, adjust=False).mean().iloc[-1])
        except: return 0.0

    def get_duration_regime_for_group(self, closes: pd.DataFrame, symbols: List[Symbol]) -> List[Symbol]:
        def mom(s):
            p = closes[s].dropna()
            return (p.iloc[-1]/p.iloc[-22]-1 + p.iloc[-1]/p.iloc[-64]-1 + p.iloc[-1]/p.iloc[-127]-1)/3 if len(p) > 127 else -1
        valid = [s for s in symbols if s in closes.columns]
        if len(valid) == 3:
            s, i, l = valid
            ms, mi, ml = mom(s), mom(i), mom(l)
            if ml > mi > ms: return [s, i, l]
            if mi > ms: return [s, i]
            return [s]
        return valid
