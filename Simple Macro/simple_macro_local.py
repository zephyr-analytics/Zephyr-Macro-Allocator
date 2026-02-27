import yfinance as yf
import pandas as pd
import numpy as np

def get_momentum_cvar_signals():
    # --- CONFIGURATION (Synced with QC) ---
    core_tickers = ["VTI", "VEA", "VWO", "BND", "BNDX", "EMB", "DBC", "GLD", "VGIT", "VGLT"]
    sector_tickers = ["IXP", "RXI", "KXI", "IXC", "IXG", "IXJ", "EXI", "MXI", "IXN", "IXU"]
    cash_substitute = "SHV"
    
    # We use a set to avoid duplicates, then list for indexing
    all_tickers = list(set(core_tickers + sector_tickers + [cash_substitute]))
    
    bond_tickers = ["BND", "BNDX", "VGIT"]
    lookbacks = [21, 63, 126, 189, 252]
    
    # Constraints
    MAX_CAP = 0.25      
    TARGET_CVAR = 0.015 
    CONFIDENCE = 0.05

    # 1. FETCH DATA 
    # 1265 trading days is ~5 years. We fetch 8 years to be safe for returns calculation.
    data = yf.download(all_tickers, period="8y", interval="1d", auto_adjust=True, progress=False)
    prices = data['Close']
    returns_df = prices.pct_change(fill_method=None).dropna()
    
    all_mom_scores = {}

    # 2. CALCULATE MOMENTUM & TREND
    for ticker in (core_tickers + sector_tickers):
        if ticker not in prices.columns: continue
        
        # Get the slice of prices for this ticker
        s_prices = prices[ticker].dropna()
        if len(s_prices) < 253: continue

        cur_price = s_prices.iloc[-1]
        
        # Trend Check (126 for bonds, 168 for others)
        sma_len = 126 if ticker in bond_tickers else 168
        sma = s_prices.iloc[-sma_len:].mean()
        
        # Must be above SMA and have positive average momentum
        if cur_price > sma:
            m_rets = [(cur_price / s_prices.iloc[-d-1]) - 1 for d in lookbacks]
            avg_m = np.mean(m_rets)
            if avg_m > 0:
                all_mom_scores[ticker] = avg_m
                continue
        all_mom_scores[ticker] = 0

    # 3. SELECT TOP 4 SECTORS
    sector_scores = {t: all_mom_scores[t] for t in sector_tickers if t in all_mom_scores and all_mom_scores[t] > 0}
    top_4_sectors = sorted(sector_scores, key=sector_scores.get, reverse=True)[:4]

    # 4. FINAL UNIVERSE 
    core_valid = [t for t in core_tickers if all_mom_scores.get(t, 0) > 0]
    valid_symbols = core_valid + top_4_sectors

    # Safety Switch Logic (Matches QC Step 4)
    if len(valid_symbols) < 4:
        print(f"SAFETY SWITCH: Only {len(valid_symbols)} assets passed. 100% SHV.")
        final_allocs = {t: 0.0 for t in all_tickers}
        final_allocs[cash_substitute] = 1.0
        return final_allocs

    # 5. ASSET-LEVEL CVaR WEIGHTING
    weights_map = {}
    # We use 1265 days for the CVaR calculation to match QC warm-up
    sub_returns = returns_df[valid_symbols].tail(1265)

    for ticker in valid_symbols:
        asset_rets = sub_returns[ticker].dropna()

        # Need enough data for a stable CVaR
        if len(asset_rets) < 504: continue

        var_limit = np.percentile(asset_rets, CONFIDENCE * 100)
        tail = asset_rets[asset_rets <= var_limit]

        # CVaR is the negative mean of the tail (expected loss)
        cvar = -tail.mean() if not tail.empty else asset_rets.std()

        # Momentum / CVaR ratio (Risk-adjusted momentum)
        weights_map[ticker] = all_mom_scores[ticker] / max(cvar, 0.0001)

    # 6. NORMALIZATION & CAPPING (Iterative Redistribution)
    total_raw = sum(weights_map.values())
    if total_raw == 0:
        return {cash_substitute: 1.0}

    current_weights = {t: weights_map[t] / total_raw for t in weights_map}

    for _ in range(10):
        total_excess = 0.0
        eligible = []
        for t, w in current_weights.items():
            if w > MAX_CAP:
                total_excess += (w - MAX_CAP)
                current_weights[t] = MAX_CAP
            elif w < MAX_CAP:
                eligible.append(t)

        if total_excess <= 1e-6 or not eligible: break

        rem_sum = sum(current_weights[t] for t in eligible)
        for t in eligible:
            current_weights[t] += total_excess * (current_weights[t] / rem_sum)

    # 7. PORTFOLIO CVaR TARGETING
    weights_series = pd.Series(current_weights)
    portfolio_rets = sub_returns[list(current_weights.keys())].dot(weights_series)

    p_var_threshold = np.percentile(portfolio_rets, CONFIDENCE * 100)
    portfolio_cvar = -portfolio_rets[portfolio_rets <= p_var_threshold].mean()

    scaling_factor = 1.0
    if portfolio_cvar > TARGET_CVAR:
        scaling_factor = TARGET_CVAR / portfolio_cvar
        print(f"Target CVaR Exceeded: {portfolio_cvar:.2%} > {TARGET_CVAR:.2%}. Scaling: {scaling_factor:.2f}")

    # 8. FINAL ALLOCATION
    final_weights = {t: 0.0 for t in all_tickers}
    total_momentum_exposure = 0.0
    
    for t, w in current_weights.items():
        scaled_w = w * scaling_factor
        if scaled_w > 0.001:
            final_weights[t] = scaled_w
            total_momentum_exposure += scaled_w

    # Remaining weight goes to SHV (Cash Substitute)
    final_weights[cash_substitute] = max(0, 1.0 - total_momentum_exposure)
    
    return final_weights

if __name__ == "__main__":
    signals = get_momentum_cvar_signals()
    print("\n--- FINAL TARGET ALLOCATIONS ---")
    active_signals = {k: v for k, v in signals.items() if v > 0.001}
    
    # Sort by weight
    sorted_signals = sorted(active_signals.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_signals:
        is_cash = "(Cash Substitute)" if ticker == "SHV" else ""
        print(f"{ticker:5}: {weight:6.2%} {is_cash}")
        
    print("-" * 35)
    print(f"TOTAL EXPOSURE: {sum(signals.values()):.2%}")
