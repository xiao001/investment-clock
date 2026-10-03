"""Investment Clock: an interactive explorer for the four business cycle phases, built live from FRED."""
import io
import urllib.request

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PHASES = ["Recovery", "Overheat", "Stagflation", "Reflation"]
COLORS = {"Recovery": "#4fae6b", "Overheat": "#d9604c", "Stagflation": "#d9a93b", "Reflation": "#4f86c6"}
RGBA = {"Recovery": "rgba(79,174,107,0.45)", "Overheat": "rgba(217,96,76,0.45)",
        "Stagflation": "rgba(217,169,59,0.45)", "Reflation": "rgba(79,134,198,0.45)"}
NAMES = {(True, False): "Recovery", (True, True): "Overheat", (False, True): "Stagflation", (False, False): "Reflation"}
ASSET = {"Recovery": "Stocks", "Overheat": "Commodities", "Stagflation": "Cash", "Reflation": "Bonds"}
SOURCES = ["Composite vote (industrial production + CFNAI + unemployment)", "Industrial production only",
           "CFNAI only", "Unemployment only"]

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
    """contiguous (start, end) spans where flag is True; end is exclusive (next month start)"""
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
def build(source, smooth, win, i_smooth, i_win):
    ip = (fred("INDPRO").pct_change(12) * 100).rolling(smooth).mean()
    cf = fred("CFNAI").rolling(smooth).mean()
    un = -fred("UNRATE").rolling(smooth).mean()
    infl = (fred("CPIAUCSL").pct_change(12) * 100).rolling(i_smooth).mean()

    def up(x, w=win):
        c = x - x.shift(w)
        return (c > 0).where(c.notna())

    ups = pd.concat({"ip_up": up(ip), "cf_up": up(cf), "un_up": up(un)}, axis=1).dropna().astype(bool)
    votes = ups.sum(axis=1)
    if source == SOURCES[0]:
        g, gline, glabel = votes >= 2, cf, "Growth vs. trend (CFNAI)"
    elif source == SOURCES[1]:
        g, gline, glabel = ups["ip_up"], ip, "Industrial production YoY %"
    elif source == SOURCES[2]:
        g, gline, glabel = ups["cf_up"], cf, "Growth vs. trend (CFNAI)"
    else:
        g, gline, glabel = ups["un_up"], un, "Unemployment, sign flipped (%)"
    d = pd.DataFrame({"g_up": g.astype(bool), "votes": votes, "i_up": up(infl, i_win).dropna().astype(bool)}).join(ups).dropna()
    d["g_up"], d["i_up"] = d["g_up"].astype(bool), d["i_up"].astype(bool)
    d["phase"] = [NAMES[(bool(a), bool(b))] for a, b in zip(d.g_up, d.i_up)]
    return d, gline, glabel, infl, fred("USREC")


def arrow(b):
    return "rising" if b else "falling"


def make_figure(v, sp, rs, gline, glabel, infl, rec, show_phase, show_rec, show_lines, picked):
    lines = pd.DataFrame({"growth": gline, "inflation": infl})
    lines = lines[(lines.index >= v.index[0]) & (lines.index <= v.index[-1])]
    lo, hi = lines["growth"].min(), lines["growth"].max()
    pad = (hi - lo) * 0.08 or 0.1
    ymin, ymax = lo - pad, hi + pad

    H = pd.Timedelta(days=15)  # centre each band on its month so a month and its dot line up
    fig = go.Figure()
    if show_phase:
        for _, r in sp.iterrows():
            fig.add_vrect(x0=r["start"] - H, x1=r["end"] - H, fillcolor=RGBA[r["phase"]], line_width=0, layer="below")
    if show_rec:
        for i, (_, r) in enumerate(rs.iterrows()):
            fig.add_trace(go.Scatter(
                x=[r["start"] - H, r["start"] - H, r["end"] - H, r["end"] - H, r["start"] - H], y=[ymin, ymax, ymax, ymin, ymin],
                mode="lines", fill="toself", line=dict(width=0), hoverinfo="skip",
                fillpattern=dict(shape="/", fgcolor="black", bgcolor="rgba(0,0,0,0)", size=7, solidity=0.18),
                name="NBER recession", legendgroup="rec", showlegend=(i == 0)))
    if show_lines:
        fig.add_trace(go.Scatter(x=lines.index, y=lines["growth"], mode="lines+markers", name=glabel,
                                 line=dict(color="black", width=2), marker=dict(size=4), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=lines.index, y=lines["inflation"], mode="lines+markers", name="CPI YoY % (smoothed)",
                                 line=dict(color="#b03060", width=2), marker=dict(size=4), yaxis="y2", hoverinfo="skip"))
    # invisible points carry the hover card for every month
    recflag = rec.reindex(v.index).fillna(0) == 1
    card = [f"<b>{t:%b %Y}: {r.phase}</b><br>Growth {arrow(r.g_up)} ({int(r.votes)} of 3 indicators rising)"
            f"<br>Inflation {arrow(r.i_up)}<br>Clock favours: {ASSET[r.phase]}"
            f"<br>NBER recession: {'yes' if recflag[t] else 'no'}" for t, r in v.iterrows()]
    yy = lines["growth"].reindex(v.index)
    fig.add_trace(go.Scatter(x=v.index, y=yy, mode="markers", marker=dict(size=14, opacity=0), showlegend=False,
                             text=card, hovertemplate="%{text}<extra></extra>"))
    for p in PHASES:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=p, hoverinfo="skip",
                                 marker=dict(symbol="square", size=12, color=RGBA[p].replace("0.45", "0.8"))))
    fig.update_layout(
        template="plotly_white", height=560, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
        hoverlabel=dict(bgcolor="white", font_size=13),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.06), range=[v.index[0] - H, v.index[-1] + H]),  # stop at the end of the last month with data
        yaxis=dict(title=glabel, range=[ymin, ymax], zeroline=True, zerolinecolor="#888"),
        yaxis2=dict(title="CPI YoY % (3m avg)", overlaying="y", side="right", showgrid=False,
                    title_font=dict(color="#b03060"), tickfont=dict(color="#b03060")),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    return fig


st.title("Investment Clock explorer")
st.markdown("Growth direction crossed with inflation direction gives four phases, after Greetham (Merrill Lynch, 2004). "
            "**Hover** any month for its phase, **drag** the slider under the chart to zoom, **click legend items** to hide them, "
            "and **change the settings on the left** to see how fragile or stable the labels are.")

with st.sidebar:
    st.header("Settings")
    source = st.selectbox("Growth indicator", SOURCES,
                          help="The composite needs at least 2 of 3 indicators to be rising. The others use one series.")
    smooth = st.slider("Smoothing (months)", 1, 6, 3, help="Moving average applied to each series before testing direction.")
    win = st.slider("Direction window (months)", 1, 6, 3, help="Growth is rising if higher than this many months ago.")
    st.caption("Inflation can use its own rule. Smoothing 1 and window 1 means: is raw CPI YoY higher than last month?")
    i_smooth = st.slider("Inflation smoothing (months)", 1, 6, 3)
    i_win = st.slider("Inflation direction window (months)", 1, 6, 3)
    st.header("Show")
    show_phase = st.checkbox("Phase colours", True)
    show_rec = st.checkbox("NBER recessions (hatched)", True)
    show_lines = st.checkbox("Growth and inflation lines", True)
    st.header("Data")
    hide_latest = st.checkbox("Hide the latest month (provisional)", False,
                              help="The newest CFNAI and CPI readings are the ones most likely to be revised, "
                                   "and the newest month is never confirmed by later data. Untick to show it.")

try:
    d, gline, glabel, infl, rec = build(source, smooth, win, i_smooth, i_win)
except Exception as e:
    st.error(f"Live data unavailable: {e}")
    st.stop()

if hide_latest and len(d) > 12:
    d = d.iloc[:-1]
last = d.index[-1]
presets = {
    "Last 1 year": (last - pd.DateOffset(years=1), last),
    "Last 2 years": (last - pd.DateOffset(years=2), last),
    "Last 5 years": (last - pd.DateOffset(years=5), last),
    "Last 10 years": (last - pd.DateOffset(years=10), last),
    "Since 1973": (d.index[0], last),
    "2008 financial crisis": (pd.Timestamp("2007-01-01"), pd.Timestamp("2010-12-01")),
    "2020 COVID shock": (pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-01")),
    "Custom": None,
}
choice = st.radio("Window", list(presets), index=2, horizontal=True)
if presets[choice] is None:
    lo, hi = d.index[0].date(), last.date()
    s0, s1 = st.slider("Custom date range", min_value=lo, max_value=hi, value=(lo, hi), format="YYYY-MM")
    t0, t1 = pd.Timestamp(s0), pd.Timestamp(s1)
else:
    t0, t1 = presets[choice]
t0 = max(pd.Timestamp(t0).to_period("M").to_timestamp(), d.index[0])
t1 = min(pd.Timestamp(t1).to_period("M").to_timestamp(), last)

v = d[(d.index >= t0) & (d.index <= t1)]
if len(v) < 3:
    st.warning("Pick a longer window.")
    st.stop()

gap = v.index.to_series().diff() > pd.Timedelta(days=40)  # a missing month breaks the band, so it stays blank
seg = ((v["phase"] != v["phase"].shift()) | gap.values).cumsum()
sp = v.reset_index(names="date").groupby(seg.values).agg(start=("date", "first"), end=("date", "last"), phase=("phase", "first"))
sp["end"] = sp["end"] + pd.offsets.MonthBegin(1)
rs = spans(rec == 1)
rs = rs[(rs["end"] >= t0) & (rs["start"] <= t1 + pd.offsets.MonthBegin(1))]

months = list(pd.date_range(t0, t1, freq="MS"))
picked = st.select_slider("Pick a month: which phase was it?", options=months, value=months[-1], format_func=lambda t: f"{t:%Y-%m}")

st.plotly_chart(make_figure(v, sp, rs, gline, glabel, infl, rec, show_phase, show_rec, show_lines, picked), width="stretch")

c1, c2, c3, c4 = st.columns(4)
in_rec = None if picked not in rec.index else bool(rec.loc[picked] == 1)
c4.metric("NBER recession", "Not dated yet" if in_rec is None else ("Yes" if in_rec else "No"))
if picked in v.index:
    row = v.loc[picked]
    c1.metric(f"Phase in {picked:%Y-%m}", row["phase"])
    c2.metric("Growth", arrow(row["g_up"]))
    c3.metric("Inflation", arrow(row["i_up"]))
    st.caption(f"Clock favours **{ASSET[row['phase']]}**. Votes: industrial production {arrow(row['ip_up'])}, "
               f"CFNAI {arrow(row['cf_up'])}, unemployment (sign flipped) {arrow(row['un_up'])}.")
else:
    c1.metric(f"Phase in {picked:%Y-%m}", "No data")
    c2.metric("Growth", "-")
    c3.metric("Inflation", "-")
    st.caption("No phase for this month: at least one input has not been published (or was never released).")

st.subheader("How stable are the labels?")
flips = int((v["phase"] != v["phase"].shift()).sum() - 1)
years = len(v) / 12
m1, m2, m3 = st.columns(3)
m1.metric("Phase changes per year", f"{flips / years:.1f}")
m2.metric("Average episode (months)", f"{len(v) / (flips + 1):.1f}")
rec_months = int((rec.reindex(v.index) == 1).sum())
m3.metric("NBER recession months in view", rec_months)
share = (v["phase"].value_counts(normalize=True).reindex(PHASES).fillna(0) * 100).round(1)
share_df = pd.DataFrame({"% of months": share, "Clock favours": [ASSET[p] for p in PHASES]})
st.dataframe(share_df, width="stretch")
st.caption("Try Smoothing = 1 and then 6. Short smoothing flips the label constantly; long smoothing is steadier but slower to react.")

with st.expander("What do the four phases mean?"):
    st.markdown(
        "| Phase | Growth | Inflation | Clock favours |\n|---|---|---|---|\n"
        "| Recovery | rising | falling | Stocks |\n| Overheat | rising | rising | Commodities |\n"
        "| Stagflation | falling | rising | Cash |\n| Reflation | falling | falling | Bonds |\n\n"
        "In this data the Recovery and Overheat links hold up best. Reflation and Stagflation depend on the era."
    )

with st.expander("Method and limits"):
    st.markdown(
        "- Growth rising: for the composite, at least 2 of 3 indicators are higher than the chosen number of months ago. "
        "Unemployment is sign flipped, so falling unemployment counts as rising growth.\n"
        "- Inflation rising: smoothed CPI year-on-year is higher than the chosen number of months ago.\n"
        "- Data: FRED (INDPRO, CFNAI, UNRATE, CPIAUCSL, USREC), revised values, no publication lag, no real-time vintages.\n"
        "- A month with a missing input (for example October 2025, when the CPI release was cancelled) is left blank.\n"
        "- Educational research, not investment advice."
    )

st.download_button("Download phases for this window (CSV)",
                   v[["phase", "g_up", "i_up", "votes"]].rename(columns={"g_up": "growth_rising", "i_up": "inflation_rising"})
                   .to_csv().encode("utf-8"), file_name="investment_clock_phases.csv", mime="text/csv")
