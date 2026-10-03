"""Markov-switching generalization of investment_clock.py.

Replaces the hard 0/1 growth/inflation direction tests with two independent 2-state
Markov-switching models (Hamilton 1989 style), each fit with statsmodels'
MarkovRegression (mean- and variance-switching, EM/MLE estimated). The joint phase
probability is the product of the two filtered marginal probabilities, and the
backtest holds a probability-weighted blend of the four assets each month instead
of an all-or-nothing bet on one.

  Growth chain:    y_t^g = CFNAI_t                      (already a "vs. trend" index)
  Inflation chain: y_t^pi = Delta{smoothed YoY CPI}_t    (the same 3m-change already
                                                           used as the hard-threshold
                                                           signal in investment_clock.py,
                                                           now modeled as a persistent,
                                                           regime-switching mean instead
                                                           of thresholded at zero)

Only FILTERED probabilities are used for trading (Pr(S_t | F_t), t' <= t) -- the
smoothed probabilities statsmodels also reports are for parameter estimation only
and are never used to build the tradable signal, preserving the no-look-ahead
property used throughout the project.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from investment_clock import RESULTS, ASSETS, BEST, LAG, SMOOTH, fred, geo_annual, load_assets

DIR_WIN = 3
PHASES = {
    (True, False): "Recovery",
    (True, True): "Overheat",
    (False, True): "Stagflation",
    (False, False): "Reflation",
}


def fit_regime(y, name):
    """Fit a 2-state mean/variance-switching model; return filtered P(state=UP)."""
    y = y.dropna()
    mod = MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True)
    res = mod.fit()
    p = dict(zip(mod.param_names, res.params))  # look up by name, not position
    means = [p["const[0]"], p["const[1]"]]
    up_state = int(np.argmax(means))  # the regime with the higher mean = "up"
    persistence = p["p[0->0]"] if up_state == 0 else (1 - p["p[1->0]"])
    p_up = res.filtered_marginal_probabilities[up_state]
    p_up.index = y.index
    print(f"{name}: regime means = {[round(m, 3) for m in means]}, "
          f"'up' = regime {up_state}, persistence = {persistence:.3f}")
    return p_up


def markov_phase_probs():
    cfnai = fred("CFNAI")
    cpi = fred("CPIAUCSL")
    infl_sm = (cpi.pct_change(12) * 100).rolling(SMOOTH).mean()
    infl_chg = (infl_sm - infl_sm.shift(DIR_WIN)).dropna()

    p_growth_up = fit_regime(cfnai.dropna(), "Growth (CFNAI)")
    p_infl_up = fit_regime(infl_chg, "Inflation (3m change in smoothed CPI YoY)")

    idx = p_growth_up.index.intersection(p_infl_up.index)
    g, i = p_growth_up.loc[idx], p_infl_up.loc[idx]

    probs = pd.DataFrame({
        "Recovery": g * (1 - i),
        "Overheat": g * i,
        "Stagflation": (1 - g) * i,
        "Reflation": (1 - g) * (1 - i),
    }, index=idx)
    probs["hard_phase"] = probs.idxmax(axis=1)  # for comparison with the threshold version
    return probs, cpi


def main():
    probs, cpi = markov_phase_probs()
    probs_lag = probs.copy()
    probs_lag.index = probs_lag.index + LAG  # same publication lag as investment_clock.py

    rets = load_assets(cpi)
    df = rets.join(probs_lag, how="inner").dropna()

    # probability-weighted expected return each month: E[r_k] = sum_s Pr(s) * mu_{k,s}
    # (mu_{k,s} estimated once, in-sample, from the hard-threshold phase buckets --
    #  consistent with "estimate theta once, trade only on filtered probabilities")
    phase_means = df.groupby("hard_phase")[ASSETS].mean()
    print("\nIn-sample monthly mean real return by (hard) phase, for reference:\n")
    print((phase_means * 100).round(2).to_string())

    # probability-weighted portfolio: weight on each asset = sum of Pr(phase) over
    # phases whose clock-recommended asset is that asset
    weights = pd.DataFrame(0.0, index=df.index, columns=ASSETS)
    for phase, asset in BEST.items():
        weights[asset] += df[phase]
    weights = weights.div(weights.sum(axis=1), axis=0)  # normalize (should already sum to 1)

    r_markov = (weights[ASSETS] * df[ASSETS]).sum(axis=1)

    # hard-threshold comparison, using the same probability table's argmax phase
    held = df["hard_phase"].map(BEST)
    r_hard = pd.Series([df.loc[t, a] for t, a in zip(df.index, held)], index=df.index)

    curves = pd.DataFrame({
        "Markov (probability-weighted)": r_markov,
        "Hard threshold (100% one asset)": r_hard,
        "Equal weight": df[ASSETS].mean(axis=1),
        "Equity": df["Equity"],
    })
    stats = pd.DataFrame({
        "real_ann_ret_%": curves.apply(geo_annual),
        "ann_vol_%": curves.std() * np.sqrt(12) * 100,
    })
    stats["ret_over_vol"] = stats["real_ann_ret_%"] / stats["ann_vol_%"]
    print(f"\nBacktest, {df.index[0]} to {df.index[-1]} ({len(df)} months), real returns, no costs:\n")
    print(stats.round(2).to_string())

    print("\nAverage monthly phase probability (how confident/decisive the filter is):")
    print(probs[["Recovery", "Overheat", "Stagflation", "Reflation"]].mean().round(3).to_string())

    df.join(weights.add_suffix("_weight")).to_csv(RESULTS / "investment_clock_markov_data.csv")
    print("\nSaved investment_clock_markov_data.csv")


if __name__ == "__main__":
    main()
