"""
AQI forecasting - WEEKLY version (more data, more statistical power)
Run:  python aqi_weekly.py city_day.csv

Models: Naive, Seasonal naive (52 weeks ago), Seasonal mean (avg of same week
in all past years), AR, ARMA, ARIMA, Harmonic-ARIMAX (ARIMA errors + Fourier terms).
Plain SARIMA with s=52 is skipped on purpose: statsmodels fits it very slowly and
unstably for s=52. Keep SARIMA (s=12) in the monthly script.
All output files are prefixed 'weekly_' so nothing from the monthly run is overwritten.
"""
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ CONFIG
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "city_day.csv"
CITY = "Delhi"                      # same city as your monthly run
START, END = "2015-01-01", "2019-12-31"   # 2020 excluded (COVID lockdown)
TEST_WEEKS = 52                     # last year is the test set
S = 52                              # weeks per year
K = 3                               # number of Fourier sine/cosine pairs
USE_LOG = True


def fwd(x): return np.log1p(x) if USE_LOG else x
def inv(x): return np.expm1(x) if USE_LOG else x


# ------------------------------------------------------------------ DATA
def load_weekly(path, city):
    df = pd.read_csv(path, parse_dates=["Date"])
    d = df[df["City"] == city][["Date", "AQI"]].set_index("Date").sort_index().loc[START:END]
    print(f"[data] {city}: {len(d)} daily rows, {d['AQI'].isna().mean():.1%} missing AQI")
    w = d["AQI"].resample("W").mean()
    print(f"[data] weekly points: {len(w)}, missing weeks: {w.isna().sum()}")
    return w.interpolate(limit=2).dropna()


def fourier_terms(index, K=K):
    t = np.asarray(index.dayofyear)
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / 365.25)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / 365.25)
    return pd.DataFrame(cols, index=index)


# ------------------------------------------------------------------ MODELS
def fit(y, order, exog=None):
    trend = "c" if order[1] == 0 else "n"
    m = SARIMAX(y, exog=exog, order=order, trend=trend,
                enforce_stationarity=False, enforce_invertibility=False)
    return m.fit(disp=False, maxiter=200)


def grid_search(y, orders, exog=None):
    best = (np.inf, None)
    for o in orders:
        try:
            r = fit(y, o, exog)
            if np.isfinite(r.aic) and r.aic < best[0]:
                best = (r.aic, o)
        except Exception:
            continue
    return best


def select_models(train):
    pq = [(p, q) for p, q in itertools.product(range(4), range(4)) if p + q > 0]
    specs = {
        "AR": grid_search(train, [(p, 0, 0) for p in range(1, 6)]),
        "ARMA": grid_search(train, [(p, 0, q) for p, q in pq if p > 0 and q > 0]),
        "ARIMA": grid_search(train, [(p, 1, q) for p, q in itertools.product(range(4), range(4))]),
        "Harmonic-ARIMAX": grid_search(
            train, [(p, 0, q) for p, q in itertools.product(range(3), range(3))],
            exog=fourier_terms(train.index)),
    }
    print("\n[order selection by AIC on training data]")
    for k, (aic, o) in specs.items():
        print(f"  {k:<16} order={o}  AIC={aic:.1f}")
    return specs


def walk_forward(y_t, y_raw, n_test, specs):
    n = len(y_t)
    names = ["Naive", "Seasonal naive", "Seasonal mean"] + list(specs)
    preds = {k: [] for k in names}
    actual, dates = [], []
    for i in range(n - n_test, n):
        h_raw, h_t = y_raw.iloc[:i], y_t.iloc[:i]
        actual.append(y_raw.iloc[i]); dates.append(y_raw.index[i])
        preds["Naive"].append(h_raw.iloc[-1])
        preds["Seasonal naive"].append(h_raw.iloc[-S])
        lagged = [h_raw.iloc[-S * j] for j in range(1, 6) if len(h_raw) >= S * j]
        preds["Seasonal mean"].append(np.mean(lagged))
        for name, (_, o) in specs.items():
            try:
                if name == "Harmonic-ARIMAX":
                    r = fit(h_t, o, fourier_terms(h_t.index))
                    f = r.forecast(1, exog=fourier_terms(y_t.index[i:i + 1])).iloc[0]
                else:
                    f = fit(h_t, o).forecast(1).iloc[0]
                preds[name].append(float(inv(f)))
            except Exception:
                preds[name].append(np.nan)
    idx = pd.DatetimeIndex(dates)
    return pd.Series(actual, idx), {k: pd.Series(v, idx) for k, v in preds.items()}


def metrics(a, p):
    e = a - p
    return {"RMSE": np.sqrt(np.nanmean(e ** 2)), "MAE": np.nanmean(np.abs(e)),
            "MAPE %": np.nanmean(np.abs(e / a)) * 100}


def diebold_mariano(a, p1, p2):
    d = ((a - p1) ** 2 - (a - p2) ** 2).dropna()
    n = len(d)
    stat = d.mean() / np.sqrt(d.var(ddof=1) / n)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))


# ------------------------------------------------------------------ MAIN
def main():
    y = load_weekly(CSV_PATH, CITY)
    y_t = fwd(y)
    train_t = y_t.iloc[:-TEST_WEEKS]
    print(f"[split] train weeks: {len(train_t)}, test weeks: {TEST_WEEKS}")

    fig, ax = plt.subplots(figsize=(11, 4))
    y.plot(ax=ax, lw=1); ax.set_title(f"{CITY}: weekly mean AQI"); ax.set_ylabel("AQI")
    fig.tight_layout(); fig.savefig("weekly_01_series.png", dpi=150); plt.close(fig)

    print()
    for nm, s in [("level", train_t), ("first difference", train_t.diff().dropna())]:
        a = adfuller(s, autolag="AIC")[1]; k = kpss(s, regression="c", nlags="auto")[1]
        print(f"[stationarity] {nm:<18} ADF p={a:.4f} | KPSS p={k:.4f}")

    specs = select_models(train_t)
    actual, preds = walk_forward(y_t, y, TEST_WEEKS, specs)

    table = pd.DataFrame({k: metrics(actual, p) for k, p in preds.items()}).T.round(3)
    table = table.sort_values("RMSE")
    print("\n[weekly test results - walk-forward, 1-step ahead, 52 test weeks]\n", table.to_string())
    table.to_csv("weekly_results_table.csv")

    print("\n[Diebold-Mariano: Harmonic-ARIMAX vs others (negative = Harmonic better)]")
    for base in ["Naive", "Seasonal naive", "Seasonal mean", "ARIMA"]:
        s, p = diebold_mariano(actual, preds["Harmonic-ARIMAX"], preds[base])
        print(f"  vs {base:<15} DM={s:.3f}  p={p:.3f}")

    # residual diagnostics for the Harmonic model
    _, o = specs["Harmonic-ARIMAX"]
    best = fit(train_t, o, fourier_terms(train_t.index))
    resid = best.resid.iloc[1:]
    dfm = o[0] + o[2]
    lags = [l for l in (10, 20) if l > dfm]
    lb = acorr_ljungbox(resid, lags=lags, model_df=dfm, return_df=True)
    print("\n[Ljung-Box on Harmonic-ARIMAX residuals - p > 0.05 means white noise]\n", lb.round(4).to_string())
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(resid.index, resid.values); ax[0, 0].axhline(0, color="k", lw=0.8)
    ax[0, 0].set_title("Residuals over time")
    ax[0, 1].hist(resid.values, bins=20, edgecolor="k"); ax[0, 1].set_title("Histogram")
    stats.probplot(resid.values, dist="norm", plot=ax[1, 0]); ax[1, 0].set_title("Normal Q-Q")
    plot_acf(resid, lags=20, ax=ax[1, 1]); ax[1, 1].set_title("Residual ACF")
    fig.tight_layout(); fig.savefig("weekly_02_residuals.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    actual.plot(ax=ax, color="black", label="Actual", lw=1.5)
    for k, c in [("Harmonic-ARIMAX", "tab:purple"), ("Seasonal mean", "tab:orange"),
                 ("Seasonal naive", "tab:green")]:
        preds[k].plot(ax=ax, label=k, color=c, ls="--", lw=1)
    ax.set_title(f"{CITY}: weekly walk-forward 1-step forecasts (test year)")
    ax.set_ylabel("AQI"); ax.legend(); fig.tight_layout()
    fig.savefig("weekly_03_forecasts.png", dpi=150); plt.close(fig)

    best_name = table.index[0]
    err = (actual - preds[best_name]).abs().sort_values(ascending=False)
    print(f"\n[largest absolute errors of best model ({best_name})]\n", err.head(5).round(1).to_string())
    print("\nSaved: weekly_*.png and weekly_results_table.csv")


if __name__ == "__main__":
    main()
