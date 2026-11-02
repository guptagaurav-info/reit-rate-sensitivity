"""Rate sensitivity of listed REITs: two-factor betas, rolling duration, and a walk-forward Treasury hedge."""
import numpy as np
import pandas as pd


def ew_index(wk, names):
    """Equal-weight weekly return across the REITs trading that week (rebalanced weekly)."""
    return wk[names].mean(axis=1, skipna=True)


def ols(y, X):
    """OLS with Newey-West (4 lags) standard errors. Returns coefficients, t-stats and R^2."""
    m = pd.concat([y, X], axis=1).dropna()
    yv = m.iloc[:, 0].values
    Xv = np.column_stack([np.ones(len(m)), m.iloc[:, 1:].values])
    b, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    e = yv - Xv @ b
    n, k = Xv.shape
    XtXi = np.linalg.inv(Xv.T @ Xv)
    S = (Xv * e[:, None]).T @ (Xv * e[:, None])
    for L in range(1, 5):
        w = 1 - L / 5
        G = (Xv[L:] * e[L:, None]).T @ (Xv[:-L] * e[:-L, None])
        S += w * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi))
    r2 = 1 - e.var() / yv.var()
    return pd.Series(b, index=["alpha"] + list(X.columns)), pd.Series(b / se, index=["alpha"] + list(X.columns)), r2, n


def duration_years(beta_per_bp):
    """Equity 'rate duration': % price change for a 100bp rise, expressed in years (sign flipped)."""
    return -beta_per_bp * 1e4


def two_factor(r, mkt, dy):
    X = pd.DataFrame({"market": mkt, "d10y_bp": dy})
    return ols(r, X)


def rolling_betas(r, mkt, dy, window):
    out = []
    df = pd.concat([r.rename("r"), mkt.rename("m"), dy.rename("dy")], axis=1).dropna()
    for i in range(window, len(df) + 1):
        w = df.iloc[i - window:i]
        Xv = np.column_stack([np.ones(window), w.m, w.dy])
        b, *_ = np.linalg.lstsq(Xv, w.r.values, rcond=None)
        out.append((df.index[i - 1], b[1], b[2]))
    return pd.DataFrame(out, columns=["date", "beta_mkt", "beta_dy"]).set_index("date")


def walk_forward_hedge(r, mkt, tlt, dy, window, cost_bp):
    """Each week, estimate on the prior `window` weeks only: REIT ~ SPY + d10y and TLT ~ d10y.
    Short TLT so the hedged book has zero estimated yield beta; hold for one week. Costs on hedge turnover."""
    df = pd.concat([r.rename("r"), mkt.rename("m"), tlt.rename("t"), dy.rename("dy")], axis=1).dropna()
    rows = []
    prev_h = 0.0
    for i in range(window, len(df)):
        w = df.iloc[i - window:i]
        Xv = np.column_stack([np.ones(window), w.m, w.dy])
        b, *_ = np.linalg.lstsq(Xv, w.r.values, rcond=None)
        bt = np.polyfit(w.dy.values, w.t.values, 1)[0]  # TLT return per bp
        h = b[2] / bt if bt < 0 else 0.0  # hedge notional per $1 of REITs (positive = short TLT)
        h = float(np.clip(h, 0.0, 1.5))
        nxt = df.iloc[i]
        cost = abs(h - prev_h) * cost_bp / 1e4
        rows.append((df.index[i], nxt.r, nxt.r - h * nxt.t - cost, h, b[2], b[1]))
        prev_h = h
    return pd.DataFrame(rows, columns=["date", "unhedged", "hedged", "hedge_ratio", "beta_dy", "beta_mkt"]).set_index("date")


def stats(r, periods=52):
    r = r.dropna()
    nav = (1 + r).cumprod()
    dd = nav / nav.cummax() - 1
    ann = nav.iloc[-1] ** (periods / len(r)) - 1
    vol = r.std() * np.sqrt(periods)
    return {"ann_return": ann, "ann_vol": vol, "max_drawdown": dd.min(), "weeks": len(r)}


def window_return(r, a, b):
    s = r.loc[a:b].dropna()
    return float((1 + s).prod() - 1)
