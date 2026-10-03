"""Investment Clock dashboard. Reads the precomputed CSVs in results/model_outputs."""
import io
import urllib.request
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "results" / "model_outputs"
ASSETS = ["Equity", "Commodity", "Cash", "Bond"]
BEST = {"Recovery": "Equity", "Overheat": "Commodity", "Stagflation": "Cash", "Reflation": "Bond"}
PHASES = ["Recovery", "Overheat", "Stagflation", "Reflation"]
COLORS = {"Recovery": "#4fae6b", "Overheat": "#d9604c", "Stagflation": "#d9a93b", "Reflation": "#4f86c6"}
SOURCES = {"Composite (INDPRO + CFNAI + UNRATE vote)": "COMPOSITE", "INDPRO only": "INDPRO",
           "CFNAI only": "CFNAI", "UNRATE only": "UNRATE"}

st.set_page_config(page_title="Investment Clock", layout="wide")


@st.cache_data
def load(src):
    df = pd.read_csv(DATA / f"investment_clock_data_{src}.csv", index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


@st.cache_data
def load_nber():
    df = pd.read_csv(DATA / "investment_clock_vs_nber.csv", index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


def geo_annual(x):
    return ((1 + x).prod() ** (12 / len(x)) - 1) * 100


# ---- live charts: the app fetches FRED itself and computes everything here ----
@st.cache_data(ttl=86400, show_spinner="Fetching FRED data...")
def fred_live(series):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=1960-01-01"
    last = None
    for _ in range(3):
        try:
            raw = urllib.request.urlopen(url, timeout=25).read().decode()
            d = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True)
            return pd.to_numeric(d.iloc[:, 0], errors="coerce").dropna().rename(series)
        except Exception as e:  # FRED is occasionally slow
            last = e
    raise RuntimeError(f"Could not fetch {series} from FRED: {last}")


def spans(flag):
    """contiguous (start, end) date spans where flag is True; flag has a month-start DatetimeIndex"""
    out, start, prev = [], None, None
    for t, v in flag.items():
        if v and start is None:
            start = t
        elif not v and start is not None:
            out.append((start, t)); start = None
        prev = t
    if start is not None:
        out.append((start, prev + pd.offsets.MonthBegin(1)))
    return pd.DataFrame(out, columns=["start", "end"])


def live_phases():
    ip = (fred_live("INDPRO").pct_change(12) * 100).rolling(3).mean()
    cf = fred_live("CFNAI").rolling(3).mean()
    un = -fred_live("UNRATE").rolling(3).mean()
    infl = (fred_live("CPIAUCSL").pct_change(12) * 100).rolling(3).mean()
    up = lambda x: ((x - x.shift(3)) > 0).where((x - x.shift(3)).notna())
    votes = pd.concat([up(ip), up(cf), up(un)], axis=1).dropna().astype(int).sum(axis=1)
    g_up = votes >= 2
    i_up = up(infl).dropna().astype(bool)
    d = pd.DataFrame({"g_up": g_up, "i_up": i_up}).dropna()
    names = {(True, False): "Recovery", (True, True): "Overheat", (False, True): "Stagflation", (False, False): "Reflation"}
    d["phase"] = [names[(bool(g), bool(i))] for g, i in zip(d.g_up, d.i_up)]
    return d, cf, infl


st.title("Investment Clock")
st.caption("Growth direction x inflation direction gives four phases (Greetham, Merrill Lynch, 2004). "
           "Signals are lagged 2 months. Returns are real, monthly, 1973-04 to 2026-08.")

with st.sidebar:
    label = st.selectbox("Growth indicator", list(SOURCES))
    df_all = load(SOURCES[label])
    lo, hi = df_all.index[0].to_timestamp().date(), df_all.index[-1].to_timestamp().date()
    start, end = st.slider("Date range", min_value=lo, max_value=hi, value=(lo, hi), format="YYYY-MM")
    st.markdown("**Phase to asset (the clock's claim)**")
    for p in PHASES:
        st.markdown(f"- {p}: {BEST[p]}")

df = df_all[(df_all.index.to_timestamp().date >= start) & (df_all.index.to_timestamp().date <= end)]
if len(df) < 12:
    st.warning("Pick a longer date range.")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Current phase", "Assets by phase", "Phase timeline", "NBER recessions",
                                              "Indicators (live)", "Cycle chart (live)"])

with tab1:
    latest = df_all.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest month", str(df_all.index[-1]))
    c2.metric("Phase", latest["phase"])
    c3.metric("Clock says hold", BEST[latest["phase"]])
    st.subheader("Last 18 months")
    recent = df_all.tail(18)[["phase"] + ASSETS].copy()
    recent[ASSETS] = (recent[ASSETS] * 100).round(2)
    recent.index = recent.index.astype(str)
    st.dataframe(recent.rename(columns={a: f"{a} %" for a in ASSETS}), width="stretch")
    st.caption("Asset columns are real monthly returns in percent.")

with tab2:
    tbl = df.groupby("phase")[ASSETS].apply(lambda g: g.apply(geo_annual)).reindex(PHASES)
    out = tbl.round(1)
    out["months"] = df.groupby("phase").size().reindex(PHASES)
    out["clock says"] = [BEST[p] for p in PHASES]
    out["actual best"] = tbl.idxmax(axis=1)
    out["match"] = np.where(out["clock says"] == out["actual best"], "yes", "no")
    st.subheader("Real annualised return by phase (%)")
    st.dataframe(out, width="stretch")
    long = tbl.reset_index().melt("phase", var_name="asset", value_name="return")
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        x=alt.X("phase:N", sort=PHASES, title=None), y=alt.Y("return:Q", title="annualised real return %"),
        color=alt.Color("asset:N"), xOffset="asset:N", tooltip=["phase", "asset", alt.Tooltip("return:Q", format=".1f")]
    ).properties(height=340), width="stretch")
    held = df["phase"].map(BEST)
    strat = pd.Series([df.loc[i, a] for i, a in zip(df.index, held)], index=df.index)
    curves = pd.DataFrame({"Clock": strat, "Equal weight": df[ASSETS].mean(axis=1), "Equity": df["Equity"]})
    stats = pd.DataFrame({"real return %": curves.apply(geo_annual), "volatility %": curves.std() * np.sqrt(12) * 100})
    stats["return / vol"] = stats["real return %"] / stats["volatility %"]
    st.subheader("Backtest (no costs, lagged signal)")
    st.dataframe(stats.round(2), width="stretch")
    growth = (1 + curves).cumprod()
    growth.index = growth.index.to_timestamp()
    st.line_chart(growth)

with tab3:
    ph = df["phase"].rename_axis("month").reset_index()
    ph["start"] = ph["month"].dt.to_timestamp()
    ph["end"] = (ph["month"] + 1).dt.to_timestamp()
    st.altair_chart(alt.Chart(ph).mark_rect().encode(
        x=alt.X("start:T", title=None), x2="end:T",
        color=alt.Color("phase:N", scale=alt.Scale(domain=PHASES, range=[COLORS[p] for p in PHASES])),
        tooltip=["month:N", "phase:N"]).properties(height=120), width="stretch")
    flips = int((df["phase"] != df["phase"].shift()).sum() - 1)
    st.write(f"{flips} phase changes in {len(df)} months, about {flips / (len(df) / 12):.1f} per year, "
             f"average episode {len(df) / (flips + 1):.1f} months. The signal flips often.")
    share = (df["phase"].value_counts(normalize=True).reindex(PHASES).fillna(0) * 100).round(1)
    st.dataframe(share.rename("% of months"), width="stretch")

with tab4:
    n = load_nber()
    n = n[(n.index.to_timestamp().date >= start) & (n.index.to_timestamp().date <= end)]
    st.caption("Uses the composite phase. 'Recession-like' means Stagflation or Reflation.")
    pred = n["phase"].isin(["Stagflation", "Reflation"])
    act = n["nber_recession"] == "Recession"
    tp, fp, fn, tn = (pred & act).sum(), (pred & ~act).sum(), (~pred & act).sum(), (~pred & ~act).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Recall", f"{tp / max(tp + fn, 1) * 100:.1f}%", help="Share of NBER recession months flagged")
    c2.metric("Precision", f"{tp / max(tp + fp, 1) * 100:.1f}%", help="Share of flagged months that were recessions")
    c3.metric("Recession months", int(act.sum()))
    st.dataframe(pd.crosstab(n["phase"], n["nber_recession"]).reindex(PHASES), width="stretch")

with tab5:
    st.caption("Built live from FRED in this app. Grey bars are NBER recessions.")
    try:
        rec = fred_live("USREC")
        rs = spans(rec == 1)
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        rs = rs[(rs["end"] >= t0) & (rs["start"] <= t1)]
        meta = {"INDPRO": "Industrial production (index)", "CFNAI": "Chicago Fed National Activity Index",
                "UNRATE": "Unemployment rate (%)", "CPIAUCSL": "CPI, all urban consumers (index)"}
        for code_, title in meta.items():
            sr = fred_live(code_)
            sr = sr[(sr.index >= t0) & (sr.index <= t1)].rename("value").rename_axis("date").reset_index()
            bars = alt.Chart(rs).mark_rect(color="grey", opacity=0.3).encode(x="start:T", x2="end:T")
            line = alt.Chart(sr).mark_line(color="#1f5fa8").encode(
                x=alt.X("date:T", title=None), y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                tooltip=["date:T", alt.Tooltip("value:Q", format=".2f")])
            st.markdown(f"**{code_}**: {title}")
            st.altair_chart((bars + line).properties(height=190), width="stretch")
    except Exception as e:
        st.error(f"Live data unavailable: {e}")

with tab6:
    st.caption("Growth vs. trend (CFNAI) and CPI inflation. Background = phase from a 2-of-3 growth vote "
               "(INDPRO, CFNAI, UNRATE) plus CPI direction. No publication lag is applied here.")
    try:
        d, cf, infl = live_phases()
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        d = d[(d.index >= t0) & (d.index <= t1)]
        if len(d) < 3:
            st.warning("Pick a longer date range.")
        else:
            seg = (d["phase"] != d["phase"].shift()).cumsum()
            sp = d.reset_index(names="date").groupby(seg.values).agg(
                start=("date", "first"), end=("date", "last"), phase=("phase", "first"))
            sp["end"] = sp["end"] + pd.offsets.MonthBegin(1)
            rec = fred_live("USREC")
            rs = spans(rec == 1)
            rs = rs[(rs["end"] >= t0) & (rs["start"] <= t1)]
            lines = pd.DataFrame({"CFNAI (3m avg, left axis)": cf, "CPI YoY % (3m avg, right axis)": infl})
            lines = lines[(lines.index >= t0) & (lines.index <= t1)].rename_axis("date").reset_index()
            shade = alt.Chart(sp).mark_rect(opacity=0.55).encode(
                x="start:T", x2="end:T",
                color=alt.Color("phase:N", scale=alt.Scale(domain=PHASES, range=[COLORS[p] for p in PHASES])),
                tooltip=["phase:N", "start:T"])
            rec_bar = alt.Chart(rs).mark_rect(color="black", opacity=0.9, height=8, yOffset=0).encode(x="start:T", x2="end:T")
            lg = alt.Chart(lines).mark_line(color="black", point=alt.OverlayMarkDef(size=14)).encode(
                x=alt.X("date:T", title=None), y=alt.Y("CFNAI (3m avg, left axis):Q", title="CFNAI vs. trend"),
                tooltip=["date:T", alt.Tooltip("CFNAI (3m avg, left axis):Q", format=".2f")])
            li = alt.Chart(lines).mark_line(color="#b03060", point=alt.OverlayMarkDef(size=14)).encode(
                x="date:T", y=alt.Y("CPI YoY % (3m avg, right axis):Q", title="CPI YoY %",
                                    axis=alt.Axis(orient="right", titleColor="#b03060")),
                tooltip=["date:T", alt.Tooltip("CPI YoY % (3m avg, right axis):Q", format=".2f")])
            st.altair_chart(alt.layer(shade, lg, li).resolve_scale(y="independent").properties(height=420), width="stretch")
            st.caption("Black bar along the bottom edge of the shading marks NBER recession months." if len(rs) else
                       "No NBER recession in this window.")
            cur = d.iloc[-1]
            st.write(f"Latest month in view ({d.index[-1]:%Y-%m}): **{cur['phase']}**. "
                     f"Phases flip often when growth sits near trend, so read the label as a direction, not a regime.")
    except Exception as e:
        st.error(f"Live data unavailable: {e}")

st.divider()
st.caption("Based on revised FRED data with a flat 2-month lag, not real-time vintages. Educational research only, not investment advice.")
