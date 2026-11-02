"""Run everything: python -m rrs.pipeline  -> reports/metrics.json + reports/figures/*.png"""
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from . import data, analysis as A

NAVY, GREY, GOLD, RED = "#1F3A5F", "#8A8F98", "#9A7F2E", "#B5473A"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
PERIODS = {"2005-2026": ("2005", "2026"), "2005-2012": ("2005", "2012"), "2013-2021": ("2013", "2021"), "2022-2026": ("2022", "2026")}


def year_axis(ax):
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))


def main():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    wk, dy, names = data.weekly(cfg)
    reit = A.ew_index(wk, names)
    m, t = wk[cfg["market"]], wk[cfg["treasury"]]
    out = {"sample": {"start": str(wk.index[0].date()), "end": str(wk.index[-1].date()), "weeks": len(wk), "reits": len(names)}}

    # 1. two-factor betas by period
    per = {}
    for k, (a, b) in PERIODS.items():
        c, tt, r2, n = A.two_factor(reit.loc[a:b], m.loc[a:b], dy.loc[a:b])
        _, _, r2m, _ = A.ols(reit.loc[a:b], pd.DataFrame({"m": m}).loc[a:b])
        per[k] = {"market_beta": c.market, "rate_duration_yrs": A.duration_years(c.d10y_bp), "t_rate": tt.d10y_bp,
                  "r2_both": r2, "r2_market_only": r2m, "r2_added_by_rates": r2 - r2m, "weeks": n}
    out["periods"] = per
    ct, _, _, _ = A.ols(t, pd.DataFrame({"d": dy}))
    out["tlt_duration_yrs"] = A.duration_years(ct.d)

    yearly = {}
    for yr in range(2005, int(cfg["end"][:4]) + 1):
        s = str(yr)
        c, tt, _, n = A.two_factor(reit.loc[s], m.loc[s], dy.loc[s])
        yearly[yr] = {"rate_duration_yrs": A.duration_years(c.d10y_bp), "t": tt.d10y_bp, "weeks": n}
    out["yearly"] = yearly

    sec = {}
    for s, v in cfg["sectors"].items():
        r = wk[v].mean(axis=1)
        c1, t1, _, _ = A.two_factor(r, m, dy)
        c2, t2, _, _ = A.two_factor(r.loc["2022":], m.loc["2022":], dy.loc["2022":])
        sec[s] = {"names": v, "dur_full": A.duration_years(c1.d10y_bp), "t_full": t1.d10y_bp,
                  "dur_2022_26": A.duration_years(c2.d10y_bp), "t_2022_26": t2.d10y_bp, "market_beta_full": c1.market}
    out["sectors"] = sec

    # 2. walk-forward hedge (estimated on the prior 104 weeks only)
    h = A.walk_forward_hedge(reit, m, t, dy, cfg["window_weeks"], cfg["cost_bp"])
    hs = {}
    for k, (a, b) in {"all": (None, None), "2007-2012": ("2007", "2012"), "2013-2021": ("2013", "2021"), "2022-2026": ("2022", "2026")}.items():
        hs[k] = {c: A.stats(h[c].loc[a:b]) for c in ["unhedged", "hedged"]}
    hs["year_2022"] = {c: A.window_return(h[c], "2022-01-01", "2022-12-31") for c in ["unhedged", "hedged"]}
    hs["hedge_ratio"] = h.hedge_ratio.describe().to_dict()
    hs["start"] = str(h.index[0].date())
    # static policy hedge: ratio fixed from 2013-2021 data only, applied out of sample from 2022
    c, _, _, _ = A.two_factor(reit.loc["2013":"2021"], m.loc["2013":"2021"], dy.loc["2013":"2021"])
    ctl, _, _, _ = A.ols(t.loc["2013":"2021"], pd.DataFrame({"d": dy}).loc["2013":"2021"])
    hr = float(c.d10y_bp / ctl.d)
    oos = reit.loc["2022":] - hr * t.loc["2022":]
    ins = reit.loc["2013":"2021"] - hr * t.loc["2013":"2021"]
    hs["static_policy"] = {"hedge_ratio": hr, "oos_2022_2026": A.stats(oos), "unhedged_2022_2026": A.stats(reit.loc["2022":]),
                           "year_2022": A.window_return(oos, "2022-01-01", "2022-12-31"),
                           "same_hedge_2013_2021": A.stats(ins), "unhedged_2013_2021": A.stats(reit.loc["2013":"2021"])}
    hs["walk_forward_ratio_q1_2022"] = float(h.hedge_ratio.loc["2022-01":"2022-03"].mean())
    out["hedge"] = hs

    # 3. 2022 ex-ante attribution (betas estimated before each week)
    y22 = h.loc["2022"]
    mk = (y22.beta_mkt * m.reindex(y22.index)).sum()
    rt = (y22.beta_dy * dy.reindex(y22.index)).sum()
    tot = y22.unhedged.sum()
    out["attribution_2022"] = {"market": mk, "rates": rt, "residual": tot - mk - rt, "total_sum_weekly": tot,
                               "compound_total": A.window_return(reit, "2022-01-01", "2022-12-31"),
                               "spy_2022": A.window_return(m, "2022-01-01", "2022-12-31"),
                               "tlt_2022": A.window_return(t, "2022-01-01", "2022-12-31"), "d10y_bp_2022": float(dy.loc["2022"].sum())}
    # 4. DV01 per $1bn
    d = per["2022-2026"]["rate_duration_yrs"]
    out["dv01_per_bn"] = {"rate_duration_yrs": d, "dv01_usd_per_bp": d * 1e-4 * 1e9, "tlt_equivalent_usd": d / out["tlt_duration_yrs"] * 1e9,
                          "loss_per_100bp_usd": d / 100 * 1e9}
    json.dump(out, open("reports/metrics.json", "w"), indent=2, default=float)

    # ---- figures ----
    rb = A.rolling_betas(reit, m, dy, cfg["window_weeks"])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(rb.index, A.duration_years(rb.beta_dy), color=NAVY, lw=1.8)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axhline(out["tlt_duration_yrs"], color=GOLD, ls="--", lw=1, label=f"Long Treasury fund (TLT), about {out['tlt_duration_yrs']:.0f} years")
    ax.set_ylabel("Years of rate duration\n(after removing stock-market beta)")
    ax.set_title("How rate-sensitive are REITs? Rolling two-year estimate")
    year_axis(ax); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig("reports/figures/rolling_duration.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    yrs = list(yearly); vals = [yearly[y]["rate_duration_yrs"] for y in yrs]
    ax.bar(yrs, vals, color=[NAVY if v > 0 else RED for v in vals])
    ax.axhline(0, color=GREY, lw=0.8)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylabel("Years of rate duration"); ax.set_xlabel("Calendar year")
    ax.set_title("Rate duration of REITs, estimated year by year")
    fig.tight_layout(); fig.savefig("reports/figures/yearly_duration.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    order = sorted(sec, key=lambda s: sec[s]["dur_2022_26"])
    yv = np.arange(len(order))
    ax.barh(yv - 0.2, [sec[s]["dur_full"] for s in order], 0.4, color=GREY, label="2005-2026")
    ax.barh(yv + 0.2, [sec[s]["dur_2022_26"] for s in order], 0.4, color=NAVY, label="2022-2026")
    ax.set_yticks(yv, order); ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("Years of rate duration (after removing stock-market beta)")
    ax.set_title("Rate duration by REIT sector"); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig("reports/figures/sector_duration.png"); plt.close(fig)

    at = out["attribution_2022"]
    c22, _, _, _ = A.two_factor(reit.loc["2022"], m.loc["2022"], dy.loc["2022"])
    mk2, rt2 = c22.market * m.loc["2022"].sum(), c22.d10y_bp * dy.loc["2022"].sum()
    at.update({"inyear_market": mk2, "inyear_rates": rt2, "inyear_residual": at["total_sum_weekly"] - mk2 - rt2})
    json.dump(out, open("reports/metrics.json", "w"), indent=2, default=float)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labs = ["Stock market", "Rates", "Everything else"]
    ea = [at["market"], at["rates"], at["residual"]]; iy = [mk2, rt2, at["inyear_residual"]]
    x = np.arange(3)
    for off, vals, col, lab in [(-0.2, ea, GREY, "Rolling model, betas set before each week"), (0.2, iy, NAVY, "Measured within 2022 (hindsight)")]:
        ax.bar(x + off, [v * 100 for v in vals], 0.4, color=col, label=lab)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v * 100 + (0.6 if v > 0 else -1.6), f"{v * 100:.1f}", ha="center", fontsize=10)
    ax.set_xticks(x, labs); ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("Contribution to 2022 loss (percentage points)")
    ax.set_title(f"REITs lost {abs(at['compound_total']) * 100:.1f}% in 2022: the rolling model missed the rate risk")
    ax.legend(frameon=False, loc="lower right"); ax.set_ylim(-20, 4)
    fig.tight_layout(); fig.savefig("reports/figures/attribution_2022.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for c, col, lab in [("unhedged", NAVY, "REITs, unhedged"), ("hedged", GOLD, "REITs with walk-forward Treasury hedge (after costs)")]:
        ax.plot(h.index, (1 + h[c]).cumprod(), color=col, lw=1.6, label=lab)
    ax.set_yscale("log"); ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 4, 8])); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_ylabel("Growth of $1 (log scale)"); ax.set_title("Hedging the rate risk with Treasuries cost more than it saved")
    year_axis(ax); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig("reports/figures/hedge_nav.png"); plt.close(fig)
    print(json.dumps({k: out[k] for k in ["sample", "tlt_duration_yrs", "dv01_per_bn", "attribution_2022"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
