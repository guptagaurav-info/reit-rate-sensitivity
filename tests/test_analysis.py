import numpy as np
import pandas as pd
from rrs import analysis as A


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="W-FRI")
    m = pd.Series(rng.normal(0, 0.02, n), idx)
    dy = pd.Series(rng.normal(0, 8, n), idx)
    r = 0.9 * m - 0.0006 * dy + pd.Series(rng.normal(0, 0.01, n), idx)
    t = -0.0015 * dy + pd.Series(rng.normal(0, 0.003, n), idx)
    return r, m, dy, t


def test_two_factor_recovers_betas():
    r, m, dy, _ = _toy()
    c, tt, r2, n = A.two_factor(r, m, dy)
    assert abs(c.market - 0.9) < 0.08 and abs(A.duration_years(c.d10y_bp) - 6) < 1.0 and n == 400


def test_walk_forward_uses_only_past_data():
    r, m, dy, t = _toy()
    h1 = A.walk_forward_hedge(r, m, t, dy, 104, 5)
    r2 = r.copy(); r2.iloc[-1] = 5.0  # shock the last week: earlier hedge ratios must not change
    h2 = A.walk_forward_hedge(r2, m, t, dy, 104, 5)
    assert np.allclose(h1.hedge_ratio.values, h2.hedge_ratio.values)


def test_hedge_ratio_near_duration_ratio():
    r, m, dy, t = _toy()
    h = A.walk_forward_hedge(r, m, t, dy, 104, 0)
    assert abs(h.hedge_ratio.median() - 0.4) < 0.1


def test_stats_drawdown_negative():
    s = pd.Series([0.1, -0.5, 0.2])
    assert A.stats(s)["max_drawdown"] <= -0.5 + 1e-12
