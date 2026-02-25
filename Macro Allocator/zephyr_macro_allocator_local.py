import numpy as np
import pandas as pd
import yfinance as yf

# ============================
# CONFIG (STRICT QC ALIGNMENT)
# ============================
CRYPTO_CAP = 0.05       
GROUP_CAP = 0.25        
ENABLE_SMA_FILTER = True
ENABLE_VOLUME_MOMENTUM = True
USE_VOL_ADJUSTED = True

VOL_LOOKBACK = 252      
SMA_PERIOD = 147
BOND_SMA_PERIOD = 126
VOLUME_MOMENTUM_PERIOD = 10
VOLUME_MOMENTUM_LOOKBACK = 63 

MOMENTUM_LOOKBACKS = [21, 63, 126, 189, 252]
MAX_LOOKBACK = max(MOMENTUM_LOOKBACKS)

GROUPS = {
    "real": ["GLD", "DBC"],
    "corp_bonds": ["VCSH", "VCIT", "VCLT"],
    "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
    "sectors": ["VGT", "VCR", "VDC", "VHT", "VPU", "VFH", "VDE", "VIS", "VAW", "VOX", "VNQ"],
    "us_factor": ["MTUM", "IUSV", "IUSG", "USMV", "QUAL", "DGRO", "SIZE"],
    "us_cap_style": ["MGK", "MGV", "MGC", "VUG", "VTV", "VV", "VOT", "VOE", "VO", "VBK", "VBR", "VB"],
    "crypto": ["IBIT", "ETHA"], # Proxies for BTC/ETH
    "cash": ["SHV"]
}

EXEMPT_GROUPS = ["cash", "corp_bonds", "treasury_bonds"]

# ============================
# DATA FETCHING & PREP
# ============================
all_tickers = sorted(set(sum(GROUPS.values(), [])))
# Request extra data to ensure indicators (SMA) are ready
data = yf.download(all_tickers, period="2y", auto_adjust=True, progress=False)
closes = data["Close"].ffill()
volumes = data["Volume"].ffill()

# ============================
# ALIGNED HELPER METHODS
# ============================

def get_duration_regime(group_name, current_closes):
    symbols = GROUPS[group_name]
    def mom(t):
        p = current_closes[t].dropna()
        if len(p) < 128: return -1
        # QC Style: (Price/Price_t-n) - 1
        return (p.iloc[-1]/p.iloc[-22]-1 + p.iloc[-1]/p.iloc[-64]-1 + p.iloc[-1]/p.iloc[-127]-1)/3
    
    s, i, l = symbols[0], symbols[1], symbols[2]
    ms, mi, ml = mom(s), mom(i), mom(l)
    
    if ml > mi > ms: return [s, i, l]
    if mi > ms: return [s, i]
    return [s]

def compute_volume_momentum(ticker, current_vols):
    try:
        # QC: volume_changes.ewm(span=10).mean()
        vol_series = current_vols[ticker].tail(VOLUME_MOMENTUM_LOOKBACK)
        vol_changes = vol_series.pct_change().dropna()
        return float(vol_changes.ewm(span=VOLUME_MOMENTUM_PERIOD, adjust=False).mean().iloc[-1])
    except: return 0.0

def passes_trend(t, current_closes):
    if not ENABLE_SMA_FILTER: return True
    px = current_closes[t]
    period = BOND_SMA_PERIOD if any(t in GROUPS[g] for g in ["corp_bonds", "treasury_bonds"]) else SMA_PERIOD
    if len(px) < period: return False
    ma = px.rolling(period).mean().iloc[-1]
    return px.iloc[-1] > ma

def compute_asset_momentum(t, current_closes):
    px = current_closes[t]
    if len(px) < MAX_LOOKBACK + 1: return -np.inf
    # Exact QC Logic: Average of (Price_now / Price_lookback) - 1
    return np.mean([px.iloc[-1] / px.iloc[-(lb + 1)] - 1 for lb in MOMENTUM_LOOKBACKS])

# ============================
# REBALANCE LOGIC (MONTH END SIM)
# ============================
final_asset_scores = {}
group_totals = {}

# Filter risk groups for duration regimes
risk_groups = {g: GROUPS[g] for g in GROUPS}
risk_groups["corp_bonds"] = get_duration_regime("corp_bonds", closes)
risk_groups["treasury_bonds"] = get_duration_regime("treasury_bonds", closes)

for group, tickers in risk_groups.items():
    eligible_candidates = {}
    for s in tickers:
        if group != "cash" and not passes_trend(s, closes): continue
        
        asset_mom = compute_asset_momentum(s, closes)
        if asset_mom <= 0: continue

        vol_mom = compute_volume_momentum(s, volumes) if ENABLE_VOLUME_MOMENTUM else 0.0
        # QC: asset_mom * (1 + tanh(vol_mom))
        asset_edge = asset_mom * (1.0 + np.tanh(vol_mom))
        if asset_edge > 0:
            eligible_candidates[s] = asset_edge

    if not eligible_candidates: continue

    # Selection Logic
    selected_assets = list(eligible_candidates.keys())
    if group in ["us_factor", "sectors", "us_cap_style"]:
        selected_assets = sorted(eligible_candidates, key=eligible_candidates.get, reverse=True)[:3]

    # Volatility Adjustment
    group_returns = closes[selected_assets].pct_change().mean(axis=1).dropna()
    if len(group_returns) < VOL_LOOKBACK: continue
    
    # QC: Annualized StDev of log returns
    group_vol = float(np.std(np.log1p(group_returns).tail(VOL_LOOKBACK)) * np.sqrt(252))

    confidence = (np.mean([eligible_candidates[s] for s in selected_assets])) / (group_vol + 1e-6) if USE_VOL_ADJUSTED else 1.0
    
    vol_moms = [compute_volume_momentum(s, volumes) for s in selected_assets]
    volume_multiplier = np.tanh(np.mean(vol_moms)) if ENABLE_VOLUME_MOMENTUM else 1.0

    group_multiplier = confidence * volume_multiplier
    if group_multiplier <= 0: continue

    current_group_total = 0
    for s in selected_assets:
        score = group_multiplier * eligible_candidates[s]
        final_asset_scores[s] = score
        current_group_total += score
    
    group_totals[group] = current_group_total

# Breadth Check & Iterative Capping
active_risk_groups = [g for g in group_totals.keys() if g != "cash"]

if len(active_risk_groups) <= 3:
    final_alloc = {GROUPS["cash"][0]: 1.0}
else:
    total_score = sum(final_asset_scores.values())
    group_weights = {g: group_totals[g] / total_score for g in group_totals}

    # 10 Iterations for Cap Stability (Redistribution)
    for _ in range(10):
        excess = 0.0
        for g, w in group_weights.items():
            limit = CRYPTO_CAP if g == "crypto" else (GROUP_CAP if g not in EXEMPT_GROUPS else 1.0)
            if w > limit:
                excess += (w - limit)
                group_weights[g] = limit

        if excess <= 1e-7: break

        eligible = [g for g in group_weights if (g in EXEMPT_GROUPS) or 
                    (g == "crypto" and group_weights[g] < CRYPTO_CAP) or 
                    (g not in EXEMPT_GROUPS and group_weights[g] < GROUP_CAP)]

        if not eligible: break
        denom = sum(group_weights[g] for g in eligible)
        if denom > 0:
            for g in eligible:
                group_weights[g] += excess * (group_weights[g] / denom)

    # Final Map
    final_alloc = {}
    for group, g_weight in group_weights.items():
        g_symbols = [s for s in risk_groups[group] if s in final_asset_scores]
        g_score_sum = sum(final_asset_scores[s] for s in g_symbols)
        if g_score_sum > 0:
            for s in g_symbols:
                final_alloc[s] = (final_asset_scores[s] / g_score_sum) * g_weight

# --- RESULTS ---
print(f"\n{'SYMBOL':<10} | {'WEIGHT':<10}")
print("-" * 25)
for s, w in sorted(final_alloc.items(), key=lambda x: x[1], reverse=True):
    print(f"{s:<10} | {w:>10.2%}")
