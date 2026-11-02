"""Weekly total returns (price + dividends) for listed REITs, SPY and TLT, plus the 10-year Treasury yield."""
import pandas as pd

RAW = "data/"


def daily_total_returns(syms):
    px = pd.read_parquet(RAW + "prices.parquet")
    px = px[px.symbol.isin(syms)].assign(d=lambda x: pd.to_datetime(x.report_date), close=lambda x: x.close.astype(float))
    px = px.drop_duplicates(["symbol", "d"]).pivot(index="d", columns="symbol", values="close").sort_index()
    dv = pd.read_parquet(RAW + "divs.parquet")
    dv = dv[dv.symbol.isin(syms)].assign(d=lambda x: pd.to_datetime(x.report_date), amount=lambda x: x.amount.astype(float))
    dv = dv.groupby(["d", "symbol"]).amount.sum().unstack().reindex(px.index).fillna(0.0)
    dv = dv.reindex(columns=px.columns, fill_value=0.0)
    # a name's series ends at its last price; no forward-filling of missing prices
    return (px + dv) / px.shift(1) - 1


def weekly(cfg):
    names = [s for v in cfg["sectors"].values() for s in v]
    d = daily_total_returns(names + [cfg["market"], cfg["treasury"]])
    wk = (1 + d).resample("W-FRI").prod(min_count=1) - 1
    y = pd.read_parquet(RAW + "daily_treasury_yield.parquet")
    y = y.assign(d=pd.to_datetime(y.report_date)).set_index("d")[cfg["yield_col"]].astype(float).sort_index()
    dy = y.resample("W-FRI").last().diff() * 1e4  # weekly change in basis points
    wk = wk.loc[cfg["start"]:cfg["end"]]
    return wk, dy.reindex(wk.index), names
