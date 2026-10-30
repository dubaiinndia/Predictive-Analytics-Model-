"""
AQI forecasting with the ARIMA family (AR, MA, ARMA, ARIMA, SARIMA)
Predictive Analytics project - starter pipeline

Usage:
    pip install pandas numpy matplotlib statsmodels scipy
    python aqi_sarima_pipeline.py city_day.csv

Data: Kaggle "Air Quality Data in India (2015-2020)" -> city_day.csv
      (columns used: City, Date, AQI)
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
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ CONFIG
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "city_day.csv"
CITY = "Delhi"             # change to any city in the file (check missing % first)
START, END = "2015-01-01", "2019-12-31"   # 2020 excluded: COVID lockdown is a structural break
TEST_MONTHS = 12           # last 12 months are the test set
S = 12                     # seasonal period (monthly data)
USE_LOG = True             # log-transform to stabilise variance


# ------------------------------------------------------------------ 1. LOAD
def load_monthly(path, city):
    df = pd.read_csv(path, parse_dates=["Date"])
    d = df[df["City"] == city][["Date", "AQI"]].set_index("Date").sort_index()
    d = d.loc[START:END]
    print(f"[data] {city}: {len(d)} daily rows, {d['AQI'].isna().mean():.1%} missing AQI")
    m = d["AQI"].resample("MS").mean()
    print(f"[data] monthly points: {len(m)}, missing months: {m.isna().sum()}")
    m = m.interpolate(limit=2)  # fill only short gaps
    return m.dropna()


# ------------------------------------------------------------------ 2. EDA
def eda_plots(y):
    fig, ax = plt.subplots(figsize=(10, 4))
    y.plot(ax=ax, marker="o", ms=3)
    ax.set_title("Monthly mean AQI"); ax.set_ylabel("AQI")
    fig.tight_layout(); fig.savefig("01_series.png", dpi=150); plt.close(fig)

    dec = seasonal_decompose(y, model="additive", period=S)
    fig = dec.plot(); fig.set_size_inches(10, 7)
    fig.tight_layout(); fig.savefig("02_decomposition.png", dpi=150); plt.close(fig)


def stationarity_report(y, name):
    adf_p = adfuller(y.dropna(), autolag="AIC")[1]
    kpss_p = kpss(y.dropna(), regression="c", nlags="auto")[1]
    verdict = "stationary" if (adf_p < 0.05 and kpss_p > 0.05) else "NOT clearly stationary"
    print(f"[stationarity] {name:<22} ADF p={adf_p:.4f} | KPSS p={kpss_p:.4f} -> {verdict}")


def acf_pacf_plots(series, fname, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lags = min(24, len(series) // 2 - 1)
    plot_acf(series.dropna(), lags=lags, ax=ax[0]); ax[0].set_title(f"ACF - {title}")
    plot_pacf(series.dropna(), lags=min(lags, len(series) // 2 - 1), ax=ax[1], method="ywm")
    ax[1].set_title(f"PACF - {title}")
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


# ------------------------------------------------------------------ 3. MODELS
def fourier_terms(index, K=2):
    """Deterministic yearly seasonality as K sine/cosine pairs (4 columns for K=2)."""
    m = np.asarray(index.month)
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * m / 12)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * m / 12)
    return pd.DataFrame(cols, index=index)


def fit(y, order, seasonal=(0, 0, 0, 0), exog=None):
    trend = "c" if order[1] == 0 and seasonal[1] == 0 else "n"
    m = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal, trend=trend,
                enforce_stationarity=False, enforce_invertibility=False)
    return m.fit(disp=False, maxiter=200)


def grid_search(y, orders, seasonals=((0, 0, 0, 0),), exog=None):
    """Pick the order with the lowest AIC on the TRAINING data only."""
    best = (np.inf, None, None)
    for o, so in itertools.product(orders, seasonals):
        try:
            res = fit(y, o, so, exog)
            if np.isfinite(res.aic) and res.aic < best[0]:
                best = (res.aic, o, so)
        except Exception:
            continue
    return best


def select_models(train):
    """Select orders on the training set (AIC). d is fixed per model family
    because AIC is only comparable between models with the same differencing."""
    pq = list(itertools.product(range(0, 4), range(0, 4)))
    specs = {}
    specs["AR"] = grid_search(train, [(p, 0, 0) for p in range(1, 5)])
    specs["MA"] = grid_search(train, [(0, 0, q) for q in range(1, 5)])
    specs["ARMA"] = grid_search(train, [(p, 0, q) for p, q in pq if p + q > 0])
    specs["ARIMA"] = grid_search(train, [(p, 1, q) for p, q in pq])
    sar_orders = [(p, 0, q) for p in range(0, 3) for q in range(0, 3)]
    sar_seasonals = [(P, 1, Q, S) for P in range(0, 2) for Q in range(0, 2)]
    specs["SARIMA"] = grid_search(train, sar_orders, sar_seasonals)
    # ARIMA errors + Fourier seasonality (SARIMAX). Uses 4 seasonal parameters
    # instead of seasonal differencing, so it keeps all training observations.
    X_train = fourier_terms(train.index)
    specs["Harmonic-ARIMAX"] = grid_search(
        train, [(p, 0, q) for p in range(0, 3) for q in range(0, 3)], exog=X_train)
    print("\n[order selection by AIC on training data]")
    for k, (aic, o, so) in specs.items():
        print(f"  {k:<7} order={o} seasonal={so if so != (0,0,0,0) else '-'} AIC={aic:.1f}")
    return specs


# ------------------------------------------------------------------ 4. TRANSFORMS
def fwd(x):  return np.log1p(x) if USE_LOG else x
def inv(x):  return np.expm1(x) if USE_LOG else x


# ------------------------------------------------------------------ 5. WALK-FORWARD
def walk_forward(y_t, y_raw, n_test, specs):
    """Expanding-window, refit at every step, 1-step-ahead forecasts.
    y_t = transformed series, y_raw = original scale."""
    n = len(y_t)
    preds = {k: [] for k in list(specs) + ["Naive", "Seasonal naive"]}
    actual, dates = [], []
    for i in range(n - n_test, n):
        hist_t, hist_raw = y_t.iloc[:i], y_raw.iloc[:i]
        actual.append(y_raw.iloc[i]); dates.append(y_raw.index[i])
        preds["Naive"].append(hist_raw.iloc[-1])
        preds["Seasonal naive"].append(hist_raw.iloc[-S])
        for name, (_, o, so) in specs.items():
            try:
                if name == "Harmonic-ARIMAX":
                    res = fit(hist_t, o, so, fourier_terms(hist_t.index))
                    f = res.forecast(1, exog=fourier_terms(y_t.index[i:i + 1])).iloc[0]
                else:
                    f = fit(hist_t, o, so).forecast(1).iloc[0]
                preds[name].append(float(inv(f)))
            except Exception:
                preds[name].append(np.nan)
    idx = pd.DatetimeIndex(dates)
    return pd.Series(actual, idx), {k: pd.Series(v, idx) for k, v in preds.items()}


def metrics(actual, pred):
    e = actual - pred
    return {"RMSE": np.sqrt(np.nanmean(e ** 2)),
            "MAE": np.nanmean(np.abs(e)),
            "MAPE %": np.nanmean(np.abs(e / actual)) * 100}


def diebold_mariano(actual, p1, p2):
    """DM test (1-step, squared error). Negative stat => model 1 better than model 2."""
    d = (actual - p1) ** 2 - (actual - p2) ** 2
    d = d.dropna()
    n = len(d)
    stat = d.mean() / np.sqrt(d.var(ddof=1) / n)
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return stat, p


# ------------------------------------------------------------------ MAIN
def main():
    y = load_monthly(CSV_PATH, CITY)
    y_t = fwd(y)
    n_test = TEST_MONTHS
    train_t, train_raw = y_t.iloc[:-n_test], y.iloc[:-n_test]

    # --- EDA and stationarity (train only: no peeking at test data)
    eda_plots(y)
    print()
    stationarity_report(train_t, "level")
    stationarity_report(train_t.diff().dropna(), "first difference (d=1)")
    stationarity_report(train_t.diff(S).dropna(), f"seasonal diff (D=1, s={S})")
    stationarity_report(train_t.diff(S).diff().dropna(), "seasonal + first diff")
    acf_pacf_plots(train_t, "03_acf_pacf_level.png", "level")
    acf_pacf_plots(train_t.diff(S).dropna(), "04_acf_pacf_seasonal_diff.png", f"seasonal diff (s={S})")

    # --- order selection on training data only
    specs = select_models(train_t)

    # --- walk-forward evaluation on the test window
    actual, preds = walk_forward(y_t, y, n_test, specs)

    rows = {}
    for k, p in preds.items():
        rows[k] = metrics(actual, p)
        if k in specs:
            ex = fourier_terms(train_t.index) if k == "Harmonic-ARIMAX" else None
            res = fit(train_t, specs[k][1], specs[k][2], ex)
            rows[k]["AIC"] = res.aic
            rows[k]["BIC"] = res.bic
    table = pd.DataFrame(rows).T.round(3).sort_values("RMSE")
    print("\n[test results - walk-forward, 1-step ahead]\n", table.to_string())
    table.to_csv("results_table.csv")

    # --- Diebold-Mariano: SARIMA vs baselines
    print("\n[Diebold-Mariano, SARIMA vs baseline (negative stat = SARIMA better)]")
    for base in ["Naive", "Seasonal naive"]:
        s, p = diebold_mariano(actual, preds["SARIMA"], preds[base])
        print(f"  vs {base:<15} DM={s:.3f}  p={p:.3f}")
    print("[Diebold-Mariano, Harmonic-ARIMAX vs baseline (negative stat = Harmonic better)]")
    for base in ["Naive", "Seasonal naive", "SARIMA"]:
        s, p = diebold_mariano(actual, preds["Harmonic-ARIMAX"], preds[base])
        print(f"  vs {base:<15} DM={s:.3f}  p={p:.3f}")

    # --- Residual diagnostics (SARIMA and Harmonic-ARIMAX), fit on training data
    for name in ["SARIMA", "Harmonic-ARIMAX"]:
        ex = fourier_terms(train_t.index) if name == "Harmonic-ARIMAX" else None
        _, o, so = specs[name]
        best = fit(train_t, o, so, ex)
        burn = S + 1 if so[1] == 1 else 1       # drop start-up residuals
        resid = best.resid.iloc[burn:]
        df_model = o[0] + o[2] + so[0] + so[2]  # fitted ARMA parameters
        lags = [l for l in (6, 12) if l > df_model and l < len(resid)]
        lb = acorr_ljungbox(resid, lags=lags, model_df=df_model, return_df=True)
        print(f"\n[Ljung-Box on {name} residuals - p > 0.05 means white noise]\n",
              lb.round(4).to_string())
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        ax[0, 0].plot(resid.index, resid.values); ax[0, 0].axhline(0, color="k", lw=0.8)
        ax[0, 0].set_title("Residuals over time")
        ax[0, 1].hist(resid.values, bins=12, edgecolor="k"); ax[0, 1].set_title("Histogram")
        stats.probplot(resid.values, dist="norm", plot=ax[1, 0]); ax[1, 0].set_title("Normal Q-Q")
        plot_acf(resid, lags=min(12, len(resid) // 2 - 1), ax=ax[1, 1]); ax[1, 1].set_title("Residual ACF")
        fig.suptitle(f"{name} residual diagnostics")
        fig.tight_layout()
        fig.savefig(f"05_{name.lower().replace('-', '_')}_residuals.png", dpi=150); plt.close(fig)

    # --- Forecast plot
    fig, ax = plt.subplots(figsize=(11, 4.5))
    y.iloc[-(n_test + 24):].plot(ax=ax, color="black", label="Actual", marker="o", ms=3)
    for k, c in [("SARIMA", "tab:red"), ("Harmonic-ARIMAX", "tab:purple"), ("Seasonal naive", "tab:green")]:
        preds[k].plot(ax=ax, label=k, color=c, ls="--")
    ax.set_title(f"{CITY}: walk-forward 1-step forecasts (test window)")
    ax.set_ylabel("AQI"); ax.legend(); fig.tight_layout()
    fig.savefig("06_forecast_comparison.png", dpi=150); plt.close(fig)

    # --- Largest SARIMA errors (for error analysis slide)
    best_name = table.index[0] if table.index[0] in preds else "SARIMA"
    err = (actual - preds[best_name]).abs().sort_values(ascending=False)
    print(f"\n[largest absolute errors of the best model ({best_name}) - explain these in error analysis]")
    print(err.head(3).round(1).to_string())
    print("\nSaved: PNG plots and results_table.csv")


if __name__ == "__main__":
    main()