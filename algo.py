import pandas as pd
import numpy as np
import math

# Load ICE options data and stock prices
data1 = pd.read_csv("data/ice_options_2021.csv")
data2 = pd.read_csv("data/ice_options_2022.csv")
data3 = pd.read_csv("data/ice_options_2023.csv")
ice_options = pd.concat([data1, data2, data3], ignore_index=True)
ice_stock_df = pd.read_csv("data/ice_stock_2021_2023.csv",index_col="date", parse_dates=["date"])
#  Helpers
def bs_delta(S, K, T, sigma, r, cp_flag):

    # handle expiry / degenerate cases
    if T <= 0 or sigma <= 0:
        if cp_flag.upper() == "C":
            return float(S > K)   # 1 if in-the-money, else 0
        else:
            return -float(S < K)  # -1 if in-the-money, else 0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    # standard normal CDF via erf
    Nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))

    if cp_flag.upper() == "C":
        return Nd1
    else:  # put
        return Nd1 - 1.0
    
def compute_delta_row(row, r=0.01):
    """
    Compute BS delta using a dataframe row.

    Assumes:
      - row['S'] is spot
      - row['strike_price'] is encoded like 100000 -> 100.0
      - row['impl_volatility'] is annualized vol
      - row['date'], row['exdate'] are datetime64
      - row['cp_flag'] is 'C' or 'P'
    """
    S = float(row["S"])
    K = float(row["strike_price"]) / 1000.0   # Strikes are in 1000s
    sigma = float(row["impl_volatility"])
    T = (row["exdate"] - row["date"]).days / 252

    return bs_delta(S, K, T, sigma, r, row["cp_flag"])

# Data Processing:


call230120C100000 = ice_options[ice_options["symbol"] == "ICE 230120C100000"].sort_values(by="date").reset_index(drop=True)
call230120C100000["date"] = pd.to_datetime(call230120C100000["date"])
call230120C100000["exdate"] = pd.to_datetime(call230120C100000["exdate"])
call230120C100000["impl_volatility"] = call230120C100000["impl_volatility"].ffill().bfill()


# merge on calendar dates where we have BOTH option and stock
df = call230120C100000.merge(
    ice_stock_df,
    left_on="date",
    right_on=ice_stock_df.index,
    how="inner"
)

# keep only what we need for now
cols_keep = [
    "date", "exdate", "cp_flag", "strike_price",
    "best_bid", "best_offer", "delta",
    "contract_size", "S","impl_volatility"
]
df = df[cols_keep].copy()

#Filling NaN deltas from the ffill and bfill of the implied volatilities
mask_nan = df["delta"].isna()
df.loc[mask_nan, "delta"] = df.loc[mask_nan].apply(compute_delta_row, axis=1)

df["mid"] = 0.5 * (df["best_bid"] + df["best_offer"])

# Num of contracts and multiplier
n_contracts = 1
contract_size = df["contract_size"].iloc[0]  # should be 100

df["V"] = n_contracts * contract_size * df["mid"]
df["H"] = - df["delta"] * contract_size * n_contracts  # stock position for delta-hedging
def hedging_pnl_accounting(S, V, H, kappa):
    """
    Accounting P&L with proportional transaction costs.

    S, V, H : arrays of length n+1
        S_t : stock prices
        V_t : value of option position (here: long call)
        H_t : stock position held on [t_i, t_{i+1})
    kappa : float
        proportional trading cost (e.g. 0.001 = 10 bps)
    """
    S = np.asarray(S, float)
    V = np.asarray(V, float)
    H = np.asarray(H, float)

    assert len(S) == len(V) == len(H)
    n = len(S) - 1

    dV = V[1:] - V[:-1]        # option value changes
    dS = S[1:] - S[:-1]        # stock price changes
    dH = H[1:] - H[:-1]        # hedging rebalances

    trading_costs = kappa * np.abs(S[1:] * dH)   # κ | S_{i+1} (H_{i+1} - H_i) |

    # per-step reward (PnL increment)
    R = dV + H[:-1] * dS - trading_costs

    # initial + final liquidation trading costs
    init_cost = kappa * abs(S[0] * H[0])
    final_cost = kappa * abs(S[-1] * H[-1])

    pnl_steps = np.concatenate(([-init_cost], R, [-final_cost]))
    pnl_path = np.cumsum(pnl_steps)
    pnl_total = pnl_path[-1]

    return pnl_total, pnl_path

S = df["S"].values
V = df["V"].values
H = df["H"].values

for kappa in [0.0005, 0.001, 0.002]:  # 5, 10, 20 bps
    pnl_kappa, _ = hedging_pnl_accounting(S, V, H, kappa)
    print(f"kappa={kappa:.4f} → PnL={pnl_kappa:.2f}")