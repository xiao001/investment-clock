"""Investment Clock dashboard. Reads the precomputed CSVs in results/model_outputs."""
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

tab1, tab2, tab3, tab4 = st.tabs(["Current phase", "Assets by phase", "Phase timeline", "NBER recessions"])

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

st.divider()
st.caption("Based on revised FRED data with a flat 2-month lag, not real-time vintages. Educational research only, not investment advice.")
