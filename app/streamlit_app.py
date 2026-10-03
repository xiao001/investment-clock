"""Investment Clock: one chart of the four business cycle phases, built live from FRED."""
import io
import urllib.request

import altair as alt
import pandas as pd
import streamlit as st

PHASES = ["Recovery", "Overheat", "Stagflation", "Reflation"]
COLORS = {"Recovery": "#4fae6b", "Overheat": "#d9604c", "Stagflation": "#d9a93b", "Reflation": "#4f86c6"}
NAMES = {(True, False): "Recovery", (True, True): "Overheat", (False, True): "Stagflation", (False, False): "Reflation"}

st.set_page_config(page_title="Investment Clock", layout="wide")


@st.cache_data(ttl=86400, show_spinner="Fetching FRED data...")
def fred(series):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=1960-01-01"
    last = None
    for _ in range(3):
        try:
            raw = urllib.request.urlopen(url, timeout=25).read().decode()
            d = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True)
            return pd.to_numeric(d.iloc[:, 0], errors="coerce").dropna().rename(series)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not fetch {series} from FRED: {last}")


def spans(flag):
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


@st.cache_data(ttl=86400)
def build():
    ip = (fred("INDPRO").pct_change(12) * 100).rolling(3).mean()
    cf = fred("CFNAI").rolling(3).mean()
    un = -fred("UNRATE").rolling(3).mean()
    infl = (fred("CPIAUCSL").pct_change(12) * 100).rolling(3).mean()
    up = lambda x: ((x - x.shift(3)) > 0).where((x - x.shift(3)).notna())
    ups = pd.concat({"ip_up": up(ip), "cf_up": up(cf), "un_up": up(un)}, axis=1).dropna().astype(bool)
    votes = ups.astype(int).sum(axis=1)
    d = pd.DataFrame({"g_up": votes >= 2, "votes": votes, "i_up": up(infl).dropna().astype(bool)}).join(ups).dropna()
    d["g_up"] = d["g_up"].astype(bool)
    d["i_up"] = d["i_up"].astype(bool)
    d["phase"] = [NAMES[(bool(g), bool(i))] for g, i in zip(d.g_up, d.i_up)]
    return d, cf, infl, fred("USREC")


st.title("Investment Clock: the four business cycle phases")
st.caption("Growth direction x inflation direction, after Greetham (Merrill Lynch, 2004). "
           "Growth rising = at least 2 of 3 indicators (industrial production, CFNAI, unemployment) are higher than 3 months ago. "
           "Inflation rising = smoothed CPI YoY is higher than 3 months ago.")

try:
    d, cf, infl, rec = build()
except Exception as e:
    st.error(f"Live data unavailable: {e}")
    st.stop()

lo, hi = d.index[0].date(), d.index[-1].date()
default_start = max(lo, (d.index[-1] - pd.DateOffset(years=5)).date())
start, end = st.slider("Date range", min_value=lo, max_value=hi, value=(default_start, hi), format="YYYY-MM")
t0, t1 = pd.Timestamp(start).to_period("M").to_timestamp(), pd.Timestamp(end).to_period("M").to_timestamp()

v = d[(d.index >= t0) & (d.index <= t1)]
if len(v) < 3:
    st.warning("Pick a longer date range.")
    st.stop()

gap = v.index.to_series().diff() > pd.Timedelta(days=40)   # a missing month breaks the segment
seg = ((v["phase"] != v["phase"].shift()) | gap.values).cumsum()
sp = v.reset_index(names="date").groupby(seg.values).agg(start=("date", "first"), end=("date", "last"), phase=("phase", "first"))
sp["end"] = sp["end"] + pd.offsets.MonthBegin(1)
rs = spans(rec == 1)
rs = rs[(rs["end"] >= t0) & (rs["start"] <= t1)]
lines = pd.DataFrame({"growth": cf, "inflation": infl})
lines = lines[(lines.index >= t0) & (lines.index <= t1)].rename_axis("date").reset_index()

months = list(pd.date_range(t0, t1, freq="MS"))
picked = st.select_slider("Check a month: which phase was it?", options=months, value=months[-1],
                          format_func=lambda t: f"{t:%Y-%m}")

shade = alt.Chart(sp).mark_rect(opacity=0.55).encode(
    x="start:T", x2="end:T",
    color=alt.Color("phase:N", title="Phase", scale=alt.Scale(domain=PHASES, range=[COLORS[p] for p in PHASES])),
    tooltip=["phase:N", "start:T"])
growth = alt.Chart(lines).mark_line(color="black", point=alt.OverlayMarkDef(size=14)).encode(
    x=alt.X("date:T", title=None), y=alt.Y("growth:Q", title="Growth vs. trend (CFNAI, 3m avg)"),
    tooltip=["date:T", alt.Tooltip("growth:Q", format=".2f")])
inflation = alt.Chart(lines).mark_line(color="#b03060", point=alt.OverlayMarkDef(size=14)).encode(
    x="date:T", y=alt.Y("inflation:Q", title="CPI YoY % (3m avg)", axis=alt.Axis(orient="right", titleColor="#b03060")),
    tooltip=["date:T", alt.Tooltip("inflation:Q", format=".2f")])
marker = alt.Chart(pd.DataFrame({"date": [picked]})).mark_rule(color="black", strokeDash=[5, 4], size=2).encode(x="date:T")
layers = [shade, growth, inflation, marker]
if len(rs):
    rs_plot = rs.assign(label="NBER recession")
    rec_band = alt.Chart(rs_plot).mark_rect(opacity=0.35).encode(
        x="start:T", x2="end:T",
        color=alt.Color("label:N", title=None, scale=alt.Scale(domain=["NBER recession"], range=["#222222"]),
                        legend=alt.Legend(orient="top-right")))
    layers.insert(1, rec_band)
st.altair_chart(alt.layer(*layers).resolve_scale(y="independent", color="independent").properties(height=460),
                width="stretch")

arrow = lambda b: "rising" if b else "falling"
c1, c2, c3, c4 = st.columns(4)
in_rec = bool(rec.reindex([picked]).iloc[0] == 1) if picked in rec.index else None
c4.metric("NBER recession", "Yes" if in_rec else ("No" if in_rec is not None else "Not dated yet"),
          help="NBER's official monthly recession indicator (FRED USREC)")
if picked in v.index:
    row = v.loc[picked]
    c1.metric(f"Phase in {picked:%Y-%m}", row["phase"])
    c2.metric("Growth", arrow(row["g_up"]), help="At least 2 of 3 indicators higher than 3 months ago")
    c3.metric("Inflation", arrow(row["i_up"]), help="Smoothed CPI YoY higher than 3 months ago")
    st.caption(f"Growth votes: industrial production {arrow(row['ip_up'])}, CFNAI {arrow(row['cf_up'])}, "
               f"unemployment (sign flipped) {arrow(row['un_up'])} ({int(row['votes'])} of 3 rising).")
else:
    c1.metric(f"Phase in {picked:%Y-%m}", "No data")
    c2.metric("Growth", "-")
    c3.metric("Inflation", "-")
    st.caption("No phase for this month: at least one input series has not been published (or was never released).")
st.caption(f"Latest month with data: {d.index[-1]:%Y-%m}, phase {d.iloc[-1]['phase']}.")
st.caption("Black line: growth vs. trend. Pink line: inflation. Dark grey bands mark NBER recessions. "
           "Phases flip often when growth sits near trend, so read each label as a direction, not a long regime. "
           "Uses revised FRED data and no publication lag. Educational research, not investment advice.")
