import yfinance as yf
import pandas as pd
import numpy as np

def get_momentum_cvar_signals():
    # --- CONFIGURATION (Aligned with QC) ---
    core_tickers = ["VTI", "VEA", "VWO", "BND", "BNDX", "EMB", "DBC", "GLD", "VGIT", "VGLT"]
    sector_tickers = ["IXP", "RXI", "KXI", "IXC", "IXG", "IXJ", "EXI", "MXI", "IXN", "IXU"]
    bond_tickers = ["BND", "BNDX", "VGIT"]
    cash_substitute = "SHV"
    
    all_tickers = list(set(core_tickers + sector_tickers + [cash_substitute]))
    lookbacks = [21, 63, 126, 189, 252]
    
    # Risk/Portfolio Constraints
    MAX_CAP = 0.25      
    TARGET_CVAR = 0.02 # 2% Daily CVaR Target
    CONFIDENCE = 0.05  # 95% Confidence Level (5th percentile)
    HISTORY_DAYS = 1265 # QC standard ~5 years

    # 1. FETCH DATA
    # Fetching 6 years to ensure enough data for the 1265-day returns window + SMA calculation
    data = yf.download(all_tickers, period="6y", interval="1d", auto_adjust=True, progress=False)
    prices = data['Close']
    volumes = data['Volume']
    # Use fill_method=None to avoid warnings in newer pandas
    returns_df = prices.pct_change(fill_method=None)
    
    all_mom_scores = {}

    # 2. CALCULATE VOLUME-ADJUSTED MOMENTUM
    for ticker in (core_tickers + sector_tickers):
        if ticker not in prices.columns: continue

        s_prices = prices[ticker].dropna()
        s_vols = volumes[ticker].dropna()
        
        if len(s_prices) < 253: continue

        cur_price = s_prices.iloc[-1]

        # Volume Acceleration Ratio (21d avg / 42d avg)
        short_vol = s_vols.iloc[-21:].mean()
        long_vol = s_vols.iloc[-42:].mean()
        vol_ratio = short_vol / long_vol if long_vol > 0 else 0.5

        # Trend Filter (SMA length varies by asset class)
        sma_len = 126 if ticker in bond_tickers else 168
        sma = s_prices.iloc[-sma_len:].mean()
        
        if cur_price > sma:
            # Multi-lookback Momentum
            m_rets = [(cur_price / s_prices.iloc[-d-1]) - 1 for d in lookbacks]
            avg_m = np.mean(m_rets)
            
            if avg_m > 0:
                all_mom_scores[ticker] = avg_m * vol_ratio
            else:
                all_mom_scores[ticker] = 0
        else:
            all_mom_scores[ticker] = 0

    # 3. SELECT UNIVERSE (Top 4 Sectors + All positive Core)
    sector_scores = {t: all_mom_scores[t] for t in sector_tickers if all_mom_scores.get(t, 0) > 0}
    top_4_sectors = sorted(sector_scores, key=sector_scores.get, reverse=True)[:4]
    core_valid = [t for t in core_tickers if all_mom_scores.get(t, 0) > 0]
    
    valid_symbols = core_valid + top_4_sectors

    # 4. SAFETY SWITCH (If too few signals, go to cash)
    if len(valid_symbols) < 4:
        return {ticker: (1.0 if ticker == cash_substitute else 0.0) for ticker in all_tickers}

    # 5. ASSET-LEVEL CVaR WEIGHTING
    weights_map = {}
    analysis_returns = returns_df.tail(HISTORY_DAYS)

    for ticker in valid_symbols:
        asset_rets = analysis_returns[ticker].dropna()
        if len(asset_rets) < 504: continue

        # Calculate CVaR (Expected Shortfall)
        var_limit = np.percentile(asset_rets, CONFIDENCE * 100)
        tail = asset_rets[asset_rets <= var_limit]
        cvar = -tail.mean() if not tail.empty else asset_rets.std()

        # Weight = Momentum Score / Risk
        weights_map[ticker] = all_mom_scores[ticker] / max(cvar, 0.0001)

    # 6. NORMALIZATION & CAPPING (Iterative)
    total_raw = sum(weights_map.values())
    if total_raw == 0: return {cash_substitute: 1.0}
    
    current_weights = {t: w / total_raw for t, w in weights_map.items()}

    for _ in range(10):
        total_excess = sum(max(0, w - MAX_CAP) for w in current_weights.values())
        if total_excess <= 1e-6: break
        
        eligible = [t for t, w in current_weights.items() if w < MAX_CAP]
        if not eligible: break
        
        for t in current_weights:
            if current_weights[t] > MAX_CAP:
                current_weights[t] = MAX_CAP
        
        rem_sum = sum(current_weights[t] for t in eligible)
        for t in eligible:
            current_weights[t] += total_excess * (current_weights[t] / rem_sum)

    # 7. PORTFOLIO CVaR TARGETING (Scaling)
    weights_series = pd.Series(current_weights)
    # Ensure columns match the weights
    portfolio_rets = analysis_returns[weights_series.index].dot(weights_series)
    p_var_threshold = np.percentile(portfolio_rets, CONFIDENCE * 100)
    portfolio_cvar = -portfolio_rets[portfolio_rets <= p_var_threshold].mean()

    # Scaling factor ensures we don't exceed target risk
    scaling_factor = min(1.0, TARGET_CVAR / portfolio_cvar) if portfolio_cvar > 0 else 1.0

    # 8. FINAL ALLOCATION
    final_weights = {t: 0.0 for t in all_tickers}
    total_momentum_exposure = 0.0
    
    for t, w in current_weights.items():
        scaled_w = w * scaling_factor
        final_weights[t] = scaled_w
        total_momentum_exposure += scaled_w

    final_weights[cash_substitute] = max(0, 1.0 - total_momentum_exposure)
    return final_weights

if __name__ == "__main__":
    signals = get_momentum_cvar_signals()
    print(f"\n--- TARGET ALLOCATIONS ({pd.Timestamp.now().date()}) ---")
    active = sorted({k: v for k, v in signals.items() if v > 0.001}.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in active:
        tag = "(Cash/Safe)" if ticker == "SHV" else ""
        print(f"{ticker:5}: {weight:6.2%} {tag}")
