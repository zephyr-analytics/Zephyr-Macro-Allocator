import numpy as np
import pandas as pd
import yfinance as yf

# ============================
# CONFIG (QC IDENTICAL)
# ============================
CRYPTO_CAP = 0.10
ENABLE_SMA_FILTER = True
ENABLE_TREASURY_KILL_SWITCH = True
ENABLE_VOLUME_MOMENTUM = True

WINRATE_LOOKBACK = 21   
VOL_LOOKBACK = 252      

SMA_PERIOD = 147
BOND_SMA_PERIOD = 126

VOLUME_MOMENTUM_PERIOD = 20
VOLUME_MOMENTUM_LOOKBACK = 50

MOMENTUM_LOOKBACKS = [21, 63, 126, 189, 252]
MAX_LOOKBACK = max(MOMENTUM_LOOKBACKS)

# ============================
# ASSET GROUPS
# ============================
GROUPS = {
    "real": ["GLD", "DBC"],
    "corp_bonds": ["VCSH", "VCIT", "VCLT"],
    "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
    "high_yield_bonds": ["SHYG", "HYG"],
    "sectors": ["IXN", "IXJ", "IXC", "IXG", "RXI", "KXI", "EXI", "IXP", "MXI", "JXI", "REET"],
    "us_factor": ["MTUM", "ITOT", "VLUE", "USMV", "QUAL", "DGRO", "SIZE"],
    "int_factor": ["IMTM", "EFA", "IVLU", "EFAV", "IQLT", "IGRO", "SCZ"],
    "em_factor": ["EEM", "EVLU", "EEMV", "EQLT", "DGRE", "EEMS"],
    "crypto": ["IBIT", "ETHA"], 
    "cash": ["SHV"]
}

BOND_GROUPS = {"corp_bonds", "treasury_bonds", "high_yield_bonds"}

# ============================
# DATA FETCHING
# ============================
tickers = sorted(set(sum(GROUPS.values(), [])))
# Download enough for SMA and Momentum lookbacks
raw_data = yf.download(tickers, period="2y", auto_adjust=True, progress=True)

closes = raw_data["Close"].dropna(how="all")
volumes = raw_data["Volume"].ffill()

# ============================
# CORE FUNCTIONS
# ============================

def compute_volume_momentum(ticker):
    """Matches QC compute_volume_momentum EMA logic"""
    try:
        vol_series = volumes[ticker].tail(VOLUME_MOMENTUM_LOOKBACK)
        if len(vol_series) < VOLUME_MOMENTUM_PERIOD + 1: return 0.0

        vol_changes = vol_series.pct_change().dropna()
        alpha = 2 / (VOLUME_MOMENTUM_PERIOD + 1)
        # Using EWM to mimic QuantConnect's EMA
        vol_ema = vol_changes.ewm(alpha=alpha, adjust=False).mean()
        return float(vol_ema.iloc[-1])
    except:
        return 0.0

def passes_trend(t):
    if not ENABLE_SMA_FILTER: return True
    px = closes[t]
    period = BOND_SMA_PERIOD if any(t in GROUPS[g] for g in BOND_GROUPS) else SMA_PERIOD
    ma = px.rolling(period).mean()
    return px.iloc[-1] > ma.iloc[-1]

def momentum_score(t):
    px = closes[t]
    if len(px) < MAX_LOOKBACK + 1: return -np.inf
    return np.mean([px.iloc[-1] / px.iloc[-(lb + 1)] - 1 for lb in MOMENTUM_LOOKBACKS])

def duration_regime(symbols):
    def m(t):
        px = closes[t]
        if len(px) < 127: return -np.inf
        return np.mean([px.iloc[-1]/px.iloc[-22]-1, px.iloc[-1]/px.iloc[-64]-1, px.iloc[-1]/px.iloc[-127]-1])
    valid = [s for s in symbols if s in closes.columns]
    if len(valid) == 3:
        s, i, l = valid
        m_s, m_i, m_l = m(s), m(i), m(l)
        return [s, i, l] if m_l > m_i > m_s else [s, i] if m_i > m_s else [s]
    return valid

# ============================
# EXECUTION
# ============================
shv_6m = closes["SHV"].iloc[-1] / closes["SHV"].iloc[-127] - 1
risk_groups = {g: (duration_regime(GROUPS[g]) if g in BOND_GROUPS else GROUPS[g]) for g in GROUPS if g != "cash"}

edges, group_assets, group_asset_edges = {}, {}, {}

for group, symbols in risk_groups.items():
    eligible_candidates = {}

    for s in symbols:
        if s not in closes.columns or not passes_trend(s): continue

        m_6m = closes[s].iloc[-1] / closes[s].iloc[-127] - 1
        mom = momentum_score(s)
        
        if m_6m < shv_6m or mom <= 0: continue
        
        # Asset-level Volume Momentum
        vol_mult = 1.0
        if ENABLE_VOLUME_MOMENTUM:
            v_mom = compute_volume_momentum(s)
            vol_mult = 1.0 + np.tanh(v_mom)
            
        eligible_candidates[s] = mom * vol_mult

    if not eligible_candidates: continue

    # --- TOP N SELECTION LOGIC ---
    if group in ["int_factor", "us_factor", "em_factor", "sectors"]:
        top_keys = sorted(eligible_candidates, key=eligible_candidates.get, reverse=True)[:3]
    else:
        top_keys = list(eligible_candidates.keys())

    asset_edges = {k: eligible_candidates[k] for k in top_keys}
    eligible = list(asset_edges.keys())
    edge_series = pd.Series(asset_edges)
    asset_weights = edge_series / edge_series.sum()

    # Group Statistics based on Top N Basket
    weighted_group_rets = (closes[eligible].pct_change() * asset_weights).sum(axis=1).dropna()
    log_group = np.log1p(weighted_group_rets)

    if len(log_group) < max(WINRATE_LOOKBACK, VOL_LOOKBACK): continue

    win_rate = float((log_group.tail(WINRATE_LOOKBACK) > 0).mean())
    group_vol = float(np.std(log_group.tail(VOL_LOOKBACK)) * np.sqrt(252))
    group_momentum = (edge_series * asset_weights).sum()

    if group_vol <= 0: continue

    # Volume Momentum Confirmation (Group Level)
    volume_multiplier = 1.0
    if ENABLE_VOLUME_MOMENTUM:
        g_vol_mom = sum(compute_volume_momentum(s) * asset_weights[s] for s in eligible)
        volume_multiplier = 1.0 + np.tanh(g_vol_mom)

    # Confidence Construction (Matches QC: Momentum / Vol)
    confidence = group_momentum / (group_vol + 1e-6)
    edges[group] = win_rate * confidence * volume_multiplier

    group_assets[group] = eligible
    group_asset_edges[group] = asset_edges

# ============================
# FINAL ALLOCATION
# ============================
if not edges:
    weights, cash_weight, alloc = {}, 1.0, {"SHV": 1.0}
elif len(edges) < 4:
    weights, cash_weight, alloc = {}, 1.0, {"SHV": 1.0}
else:
    total_edge = sum(edges.values())
    weights = {g: (edges[g] / total_edge) for g in edges}

    # Crypto Cap
    if "crypto" in weights and weights["crypto"] > CRYPTO_CAP:
        excess = weights["crypto"] - CRYPTO_CAP
        weights["crypto"] = CRYPTO_CAP
        others = [g for g in weights if g != "crypto"]
        total_other = sum(weights[g] for g in others)
        for g in others: weights[g] += excess * (weights[g] / total_other)

    cash_weight = max(0.0, 1.0 - sum(weights.values()))
    alloc = {}
    for g, w in weights.items():
        g_edges = pd.Series(group_asset_edges[g])
        ssum = g_edges.sum()
        for s, e in g_edges.items():
            alloc[s] = w * (e / ssum)
    
    if "SHV" in alloc: alloc["SHV"] += cash_weight
    else: alloc["SHV"] = cash_weight

# Summary Prints
print("-" * 30)
print("GROUPS | " + " | ".join(f"{g}:{w:.2f}" for g, w in sorted(weights.items())) + f" | cash:{cash_weight:.2f}")
print("-" * 30)
print("\nFINAL SIGNALS (%)\n")
out = pd.Series(alloc).sort_values(ascending=False)
print((out * 100).round(2))
