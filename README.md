# Predictive-Analytics-Model-
The project evaluates and shows implementation of past AQI pattern which follows a seasonal trend . 
# AQI Forecasting Using SARIMA

## Overview
This project forecasts the Air Quality Index (AQI) using classical time
series models. It was built for the Predictive Analytics course, with the
main goal of testing and comparing model accuracy using interpretable
statistical methods. Instead of jumping straight to a complex model, the
project builds up step by step through a model ladder:

AR → MA → ARMA → ARIMA → SARIMA

Each step shows what the new component adds and how much it improves the
forecast.

## Why Time Series Models?
Air quality today depends strongly on air quality over the previous days,
so ordinary regression, which assumes independent observations, is a poor
fit. AR, MA and ARIMA-family models are designed to capture exactly this
dependence. AQI also follows a clear seasonal pattern across the year,
which is why the final model is SARIMA.

## Methodology
1. **Data exploration:** plot the series, inspect trend and seasonality,
   and handle missing values.
2. **Stationarity check:** use the Augmented Dickey-Fuller (ADF) test and
   apply differencing where required.
3. **Order selection:** use ACF and PACF plots to choose p, d, q (and
   seasonal P, D, Q).
4. **Model fitting:** fit AR, MA, ARMA, ARIMA and SARIMA models using
   statsmodels.
5. **Model comparison:** compare models using AIC/BIC for fit quality and
   RMSE/MAE on a held-out test set.
6. **Diagnostics:** use residual plots and the Ljung-Box test to confirm
   the residuals behave like white noise.
7. **Forecasting:** predict future AQI values with confidence intervals.

## Key Concepts Covered
Stationarity, autocorrelation and partial autocorrelation, differencing,
seasonal differencing, the Box-Jenkins methodology, and AIC/BIC model
selection.


