"""
Fit the final model (Harmonic-ARIMAX, monthly) on ALL available data and forecast ahead.
Run:  python aqi_forecast.py city_day.csv          (default: 6 months ahead)
      python aqi_forecast.py city_day.csv 12       (12 months ahead)
Outputs: forecast_table.csv and forecast_plot.png
"""
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "city_day.csv"
HORIZON = int(sys.argv[2]) if len(sys.argv) > 2 else 6
CITY = "Delhi"                       # same city as your other runs
START, END = "2015-01-01", "2019-12-31"
K = 2                                # Fourier pairs (same as monthly pipeline)


def load_monthly(path, city):
    df = pd.read_csv(path, parse_dates=["Date"])
    d = df[df["City"] == city][["Date", "AQI"]].set_index("Date").sort_index().loc[START:END]
    return d["AQI"].resample("MS").mean().interpolate(limit=2).dropna()


def fourier_terms(index, K=K):
    m = np.asarray(index.month)
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * m / 12)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * m / 12)
    return pd.DataFrame(cols, index=index)


def fit(y, order, exog):
    m = SARIMAX(y, exog=exog, order=order, trend="c",
                enforce_stationarity=False, enforce_invertibility=False)
    return m.fit(disp=False, maxiter=200)


def main():
    y = load_monthly(CSV_PATH, CITY)
    y_log = np.log1p(y)
    X = fourier_terms(y_log.index)

    # choose ARMA order by AIC on all available data
    best = (np.inf, None)
    for p, q in itertools.product(range(3), range(3)):
        try:
            r = fit(y_log, (p, 0, q), X)
            if r.aic < best[0]:
                best = (r.aic, (p, 0, q))
        except Exception:
            pass
    order = best[1]
    res = fit(y_log, order, X)
    print(f"[model] Harmonic-ARIMAX order={order}, AIC={res.aic:.1f}, trained on {len(y)} months "
          f"({y.index[0]:%b %Y} to {y.index[-1]:%b %Y})")

    fut_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
    fc = res.get_forecast(steps=HORIZON, exog=fourier_terms(fut_idx))
    mean = np.expm1(fc.predicted_mean)
    ci = np.expm1(fc.conf_int(alpha=0.05))
    out = pd.DataFrame({"forecast_AQI": mean.values,
                        "lower_95": ci.iloc[:, 0].values,
                        "upper_95": ci.iloc[:, 1].values}, index=fut_idx).round(1)
    out.index.name = "month"
    print("\n[forecast]\n", out.to_string())
    out.to_csv("forecast_table.csv")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    y.iloc[-36:].plot(ax=ax, color="black", marker="o", ms=3, label="Observed")
    out["forecast_AQI"].plot(ax=ax, color="tab:purple", marker="o", ms=4, label="Forecast")
    ax.fill_between(out.index, out["lower_95"], out["upper_95"], color="tab:purple",
                    alpha=0.2, label="95% interval")
    ax.set_title(f"{CITY}: AQI forecast, next {HORIZON} months (Harmonic-ARIMAX)")
    ax.set_ylabel("AQI"); ax.legend(); fig.tight_layout()
    fig.savefig("forecast_plot.png", dpi=150)
    print("\nSaved: forecast_table.csv and forecast_plot.png")


if __name__ == "__main__":
    main()
