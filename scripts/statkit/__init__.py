"""statkit — statistical foundation toolkit for the statistical-arbitrage project.

Pure-Python research layer (the simulation/backtest core lives in C++, Parts 3-4).
Provides, with unit tests on synthetic data:

  stationarity   ADF, KPSS (opposite nulls) + joint interpretation
  cointegration  Engle-Granger two-step; Johansen rank test
  autocorr       Durbin-Watson; Ljung-Box
  heterosk       Breusch-Pagan; White
  hac            Newey-West HAC standard errors
  ou             Ornstein-Uhlenbeck fit + implied half-life
  forecast       Diebold-Mariano forecast comparison
  multtest       Bonferroni; White Reality Check; Hansen SPA
  misspec        Ramsey RESET test for omitted nonlinearity
  bootstrap      stationary (block) bootstrap

Run the test suite:  python3 scripts/run_statkit_tests.py
"""
from . import stationarity, cointegration, autocorr, heterosk, hac, ou, forecast, multtest, misspec, bootstrap  # noqa: F401

__version__ = "0.1.0"
