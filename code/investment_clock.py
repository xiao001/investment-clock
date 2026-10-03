"""Prototype v2: simple, real-time Merrill Lynch Investment Clock.

Changes vs. v1 (based on Greetham 2004 + real-time literature):
  * Growth = DIRECTION of a timely indicator (3m-smoothed YoY industrial production),
    not above/below a trailing mean and not the (revised, quarterly) output gap.
  * Inflation = direction of YoY CPI.
  * Signal lagged LAG months (publication delay) -> tradable, no hindsight turning points.
  * Real (CPI-deflated) total returns.
  * Long sample from 1973-04 (Greetham's start) with index-style series:
      Equity     Ken French Mkt-RF + RF (US total market, CRSP)
      Bond       10y Treasury total return approximated from FRED GS10
      Commodity  S&P GSCI (Yahoo ^SPGSCI, from 1984-01) spliced with WTI spot (FRED WTISPLC)
                 before that; T-bill return added to approximate a total-return series
  Growth options: INDPRO (industrial production), UNRATE (unemployment, sign-flipped) or CFNAI (Chicago Fed activity index;
  ISM PMI is no longer on FRED and no free full history was found)
      Cash       FRED TB3MS

Phases (Greetham 2004):
  Recovery    growth up   / inflation down -> Equity
  Overheat    growth up   / inflation up   -> Commodity
  Stagflation growth down / inflation up   -> Cash
  Reflation   growth down / inflation down -> Bond
"""
import io
import urllib.request
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

RESULTS = Path(__file__).resolve().parent.parent / "results" / "model_outputs"
PLOTS = Path(__file__).resolve().parent.parent / "results" / "plots"
RESULTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)
START = "1973-04"
LAG = 2          # months between period end and data availability
SMOOTH = 3       # months for smoothing growth / inflation before taking direction
DIR_WIN = 3      # direction = change over this many months
COMPOSITE_SRCS = ("INDPRO", "CFNAI", "UNRATE")
ASSETS = ["Equity", "Commodity", "Cash", "Bond"]
PHASES = {
    (True, False): "Recovery",
    (True, True): "Overheat",
    (False, True): "Stagflation",
    (False, False): "Reflation",
}
BEST = {"Recovery": "Equity", "Overheat": "Commodity", "Stagflation": "Cash", "Reflation": "Bond"}
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"


def _get(url, retries=4, timeout=30):
    for i in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except Exception as e:  # FRED is occasionally slow; retry
            if i == retries - 1:
                raise
            print(f"retry {url[-40:]}: {e}")


def fred(series):
    raw = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=1970-01-01").decode()
    df = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    s.index = s.index.to_period("M")
    return s


def french_monthly():
    z = zipfile.ZipFile(io.BytesIO(_get(FF_URL)))
    lines = z.read(z.namelist()[0]).decode().splitlines()
    rows = []
    for ln in lines:
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 5 and len(p[0]) == 6 and p[0].isdigit():
            rows.append((pd.Period(p[0], "M"), float(p[1]), float(p[4])))
        elif rows:
            break  # monthly block ended (annual block follows)
    df = pd.DataFrame(rows, columns=["m", "mkt_rf", "rf"]).set_index("m")
    return (df.mkt_rf + df.rf) / 100  # total market return


def equity_return():
    """S&P 500: ^SP500TR (total return, dividends included) from 1988; ^GSPC (price only,
    NO dividends) before that, since no free S&P 500 total-return history reaches back to 1973.
    Pre-1988 returns are therefore understated relative to Greetham's actual "S&P 500 Composite,
    with income" input -- flagged here rather than silently accepted."""
    px = yf.download(["^GSPC", "^SP500TR"], start="1973-01-01", auto_adjust=True, progress=False)["Close"]
    gspc = px["^GSPC"].resample("ME").last().pct_change()
    tr = px["^SP500TR"].resample("ME").last().pct_change()
    gspc.index = gspc.index.to_period("M")
    tr.index = tr.index.to_period("M")
    r = gspc.where(gspc.index < tr.dropna().index[0]).combine_first(tr.dropna())
    return r


def bond_return(y10, duration=7.0):
    """Approx. monthly 10y Treasury total return from yields (%): carry + price change."""
    y = y10 / 100
    return y.shift(1) / 12 - duration * (y - y.shift(1))


def growth_series(src):
    """Smoothed growth measure; higher = stronger growth."""
    if src == "INDPRO":
        return (fred("INDPRO").pct_change(12) * 100).rolling(SMOOTH).mean()
    if src == "CFNAI":
        return fred("CFNAI").rolling(SMOOTH).mean()  # already a growth-vs-trend index
    if src == "UNRATE":
        return -fred("UNRATE").rolling(SMOOTH).mean()  # falling unemployment = growth up
    raise ValueError(src)


def growth_direction(src):
    g = growth_series(src)
    d = g - g.shift(DIR_WIN)
    return (d > 0).where(d.notna())  # True / False / NaN


def load_macro(growth_src="INDPRO"):
    cpi = fred("CPIAUCSL")
    if growth_src == "COMPOSITE":  # majority vote: growth is "up" if >=2 of 3 sources point up
        votes = pd.concat([growth_direction(s) for s in COMPOSITE_SRCS], axis=1).dropna()
        growth_up = votes.astype(int).sum(axis=1) >= 2
    else:
        gd = growth_direction(growth_src).dropna()
        growth_up = gd.astype(bool)
    infl = (cpi.pct_change(12) * 100).rolling(SMOOTH).mean()
    di = (infl - infl.shift(DIR_WIN)).dropna()
    out = pd.DataFrame({"growth_up": growth_up, "infl_up": di > 0}).dropna()
    out["phase"] = [PHASES[(bool(g), bool(i))] for g, i in zip(out.growth_up, out.infl_up)]
    return out[["phase"]], cpi


def commodity_return(cash_m):
    """S&P GSCI (1984+) spliced with WTI spot before; add T-bill return as collateral yield."""
    px = yf.download("^SPGSCI", start="1983-12-01", auto_adjust=True, progress=False)["Close"].squeeze()
    gsci = px.resample("ME").last().pct_change()
    gsci.index = gsci.index.to_period("M")
    wti = fred("WTISPLC").pct_change()
    r = wti.where(wti.index < gsci.dropna().index[0]).combine_first(gsci.dropna())
    return r + cash_m


def load_assets(cpi):
    cash_m = fred("TB3MS") / 100 / 12
    nominal = pd.DataFrame({
        "Equity": equity_return(),
        "Commodity": commodity_return(cash_m),
        "Cash": cash_m,
        "Bond": bond_return(fred("GS10")),
    })
    infl_m = cpi.pct_change()
    real = (1 + nominal).div(1 + infl_m, axis=0) - 1
    return real[ASSETS].dropna()


def geo_annual(x):
    return ((1 + x).prod() ** (12 / len(x)) - 1) * 100


def run(growth_src, rets, save=False, start=START, end=None, verbose=True):
    macro, _ = load_macro(growth_src)
    signal = macro["phase"].copy()
    signal.index = signal.index + LAG  # known only LAG months after the period

    df = rets.join(signal, how="inner").dropna()
    df = df[df.index >= pd.Period(start, "M")]
    if end:
        df = df[df.index <= pd.Period(end, "M")]

    tbl = df.groupby("phase")[ASSETS].apply(lambda g: g.apply(geo_annual)).round(1)
    tbl["months"] = df.groupby("phase").size()
    tbl["clock_says"] = [BEST[p] for p in tbl.index]
    tbl["actual_best"] = tbl[ASSETS].idxmax(axis=1)
    if verbose:
        print(f"\n=== Growth = {growth_src} | sample {df.index[0]} to {df.index[-1]} ({len(df)} months) ===")
        print("\nReal geometric annual return (%) by phase:\n")
        print(tbl.to_string())
        print("\nAll months:", df[ASSETS].apply(geo_annual).round(1).to_dict())

    held = df["phase"].map(BEST)
    strat = pd.Series([df.loc[i, a] for i, a in zip(df.index, held)], index=df.index)
    curves = pd.DataFrame({"Clock": strat, "Equal weight": df[ASSETS].mean(axis=1), "Equity": df["Equity"]})
    stats = pd.DataFrame({
        "real_ann_ret_%": curves.apply(geo_annual),
        "ann_vol_%": curves.std() * np.sqrt(12) * 100,
    })
    stats["ret_over_vol"] = stats["real_ann_ret_%"] / stats["ann_vol_%"]
    if verbose:
        print("\nBacktest, real returns, no costs, lagged signal:\n")
        print(stats.round(2).to_string())

    if save:
        fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        (1 + curves).cumprod().plot(ax=ax[0], logy=True, title=f"Real cumulative growth of $1 (log), growth={growth_src}")
        colors = {"Recovery": "green", "Overheat": "red", "Stagflation": "orange", "Reflation": "blue"}
        ax[1].scatter(df.index.to_timestamp(), [list(colors).index(p) for p in df["phase"]],
                      c=[colors[p] for p in df["phase"]], s=8)
        ax[1].set_yticks(range(4))
        ax[1].set_yticklabels(list(colors))
        ax[1].set_title("Detected phase (lagged)")
        plt.tight_layout()
        plt.savefig(PLOTS / f"investment_clock_{growth_src}.png", dpi=130)
        df.to_csv(RESULTS / f"investment_clock_data_{growth_src}.csv")
    return tbl, stats


def main():
    _, cpi = load_macro("INDPRO")
    rets = load_assets(cpi)
    for src in ("INDPRO", "CFNAI", "UNRATE", "COMPOSITE"):
        run(src, rets, save=True, verbose=False)
    periods = (("1973-04 to 1999-12", "1973-04", "1999-12"), ("2000-01 to 2026-08", "2000-01", None))
    for label, a, b in periods:
        print(f"\n################ SUBPERIOD {label} ################")
        for src in ("INDPRO", "CFNAI", "UNRATE", "COMPOSITE"):
            tbl, stats = run(src, rets, start=a, end=b, verbose=(src == "COMPOSITE"))
            hits = [p for p in tbl.index if tbl.loc[p, "clock_says"] == tbl.loc[p, "actual_best"]]
            c, e, w = stats.loc["Clock"], stats.loc["Equity"], stats.loc["Equal weight"]
            print(f"[{src}] matches {len(hits)}/4 {hits} | clock {c['real_ann_ret_%']:.1f}% vol {c['ann_vol_%']:.1f}% "
                  f"| equity {e['real_ann_ret_%']:.1f}% vol {e['ann_vol_%']:.1f}% | EW {w['real_ann_ret_%']:.1f}%")
    print("\nSaved investment_clock_<growth>.png / investment_clock_data_<growth>.csv")


if __name__ == "__main__":
    main()
