"""Download the raw files if missing, verify the pinned sha256, and cut the REIT/SPY/TLT subset into data/."""
import hashlib
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import yaml


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(cfg_path="configs/default.yaml", strict=True):
    cfg = yaml.safe_load(open(cfg_path))
    raw = Path("data/raw"); raw.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, src in cfg["sources"].items():
        p = raw / name
        if not p.exists():
            print(f"downloading {name}"); urllib.request.urlretrieve(src["url"], p)
        match = sha256(p) == src["sha256"]; ok &= match
        print(f"{name}: {'OK' if match else 'CHECKSUM CHANGED'}")
    if strict and not ok:
        sys.exit("Source data changed since the pinned download. Review, then update the sha256 in the config.")
    syms = [s for v in cfg["sectors"].values() for s in v] + [cfg["market"], cfg["treasury"]]
    flt = [("symbol", "in", syms)]
    pq.read_table(raw / "stock_prices.parquet", columns=["symbol", "report_date", "close"], filters=flt).to_pandas().to_parquet("data/prices.parquet")
    pq.read_table(raw / "stock_dividend_events.parquet", filters=flt).to_pandas().to_parquet("data/divs.parquet")
    pq.read_table(raw / "daily_treasury_yield.parquet").to_pandas().to_parquet("data/daily_treasury_yield.parquet")
    print("subset written to data/")


if __name__ == "__main__":
    main(*sys.argv[1:2])
