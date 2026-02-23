import numpy as np
import pandas as pd
import yfinance as yf

# ============================
# CONFIG (QC IDENTICAL)
# ============================
CRYPTO_CAP = 0.10
ENABLE_SMA_FILTER = True
ENABLE_TREASURY_KILL_SWITCH = True

WINRATE_LOOKBACK = 21   # Matched to your QC update
VOL_LOOKBACK = 252      # Matched to your QC update

SMA_PERIOD = 147
BOND_SMA_PERIOD = 126

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
    "us_factor": ["MTUM", "IUSV", "IUSG", "USMV", "QUAL", "DGRO"],
    "int_factor": ["IMTM", "EFV", "EFG", "EFAV", "IQLT", "IGRO"],
    "crypto": ["IBIT", "ETHA"], # yfinance format
    "cash": ["SHV"]
}

BOND_GROUPS = {"corp_bonds", "treasury_bonds", "high_yield_bonds"}

# ============================
# DATA FETCHING
# ============================
tickers = sorted(set(sum(GROUPS.values(), [])))
raw_data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)

closes = raw_data["Close"].ffill().dropna(how="all")
highs = raw_data["High"].ffill()
lows = raw_data["Low"].ffill()

# ============================
# CORE FUNCTIONS
# ============================

def compute_manual_adx(ticker, period=14):
    h, l, c = highs[ticker], lows[ticker], closes[ticker]
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    alpha = 1 / period
    s_plus_dm = pd.Series(plus_dm, index=h.index).ewm(alpha=alpha, adjust=False).mean()
    s_minus_dm = pd.Series(minus_dm, index=h.index).ewm(alpha=alpha, adjust=False).mean()
    s_tr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (s_plus_dm / s_tr)
    minus_di = 100 * (s_minus_dm / s_tr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    return dx.ewm(alpha=alpha, adjust=False).mean().iloc[-1]

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
        return np.mean([px.iloc[-1]/px.iloc[-22]-1, px.iloc[-1]/px.iloc[-64]-1, px.iloc[-1]/px.iloc[-127]-1])
    valid = [s for s in symbols if s in closes.columns]
    if len(valid) == 3:
        s, i, l = valid
        return [s, i, l] if m(l) > m(i) > m(s) else [s, i] if m(i) > m(s) else [s]
    return valid

# ============================
# EXECUTION
# ============================

# 1. Treasury Kill Switch
if ENABLE_TREASURY_KILL_SWITCH and all(not passes_trend(t) for t in GROUPS["treasury_bonds"]):
    print("TREASURY KILL SWITCH: all treasuries failed trend → 100% CASH")
    pd.Series({"SHV": 1.0}).to_csv("signals.csv")
    exit()

bil_6m = closes["SHV"].iloc[-1] / closes["SHV"].iloc[-127] - 1
risk_groups = {g: (duration_regime(GROUPS[g]) if g in BOND_GROUPS else GROUPS[g]) for g in GROUPS if g != "cash"}

edges, group_assets, group_asset_edges = {}, {}, {}

for group, symbols in risk_groups.items():
    eligible_candidates = {}

    for s in symbols:
        if s not in closes.columns or not passes_trend(s): continue
        m_6m = closes[s].iloc[-1] / closes[s].iloc[-127] - 1
        mom = momentum_score(s)
        if m_6m < bil_6m or mom <= 0: continue
        eligible_candidates[s] = mom

    if not eligible_candidates: continue

    # --- TOP N SELECTION LOGIC (NEW) ---
    if group in ["int_factor", "us_factor"]:
        top_keys = sorted(eligible_candidates, key=eligible_candidates.get, reverse=True)[:3]
    elif group == "sectors":
        top_keys = sorted(eligible_candidates, key=eligible_candidates.get, reverse=True)[:2]
    else:
        top_keys = list(eligible_candidates.keys())

    # Filter candidates to top selection
    asset_edges = {k: eligible_candidates[k] for k in top_keys}
    eligible = list(asset_edges.keys())

    # Weight construction for the representative basket
    edge_series = pd.Series(asset_edges)
    asset_weights = edge_series / edge_series.sum()
    
    # Group Statistics
    weighted_group_rets = (closes[eligible].pct_change() * asset_weights).sum(axis=1).dropna()
    log_group = np.log1p(weighted_group_rets)
    
    if len(log_group) < max(WINRATE_LOOKBACK, VOL_LOOKBACK): continue

    win_rate = float((log_group.tail(WINRATE_LOOKBACK) > 0).mean())
    group_vol = float(np.std(log_group.tail(VOL_LOOKBACK)) * np.sqrt(252))
    group_adx = (pd.Series({s: compute_manual_adx(s) for s in eligible}) * asset_weights).sum()
    group_momentum = (edge_series * asset_weights).sum()

    if group_vol <= 0: continue

    # Final Confidence & Edge (MATCHED TO QC)
    confidence = (group_momentum * group_adx) / (group_vol + 1e-6)
    edges[group] = win_rate * (confidence)
    
    group_assets[group] = eligible
    group_asset_edges[group] = asset_edges

# ============================
# FINAL ALLOCATION
# ============================
if not edges:
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
    alloc["SHV"] = cash_weight

# Summary Prints
print("-" * 30)
print("GROUPS | " + " | ".join(f"{g}:{w:.2f}" for g, w in sorted(weights.items())) + f" | cash:{cash_weight:.2f}")
print("-" * 30)
print("\nFINAL SIGNALS (%)\n")
out = pd.Series(alloc).sort_values(ascending=False)
print((out * 100).round(2))
