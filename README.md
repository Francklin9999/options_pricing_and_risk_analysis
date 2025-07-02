# Options Pricing & Risk Analysis Platform

## Overview
A professional, modular platform for pricing, risk analysis, and portfolio management of financial derivatives. Built in Python with Streamlit, it supports advanced option models, full Greeks, portfolio analytics, live market data, and interactive visualizations.

## Features
- **Multiple Option Pricing Models:**
  - Black-Scholes (European)
  - Binomial Tree (European & American)
  - Heston (Stochastic Volatility)
  - Hull-White (Interest Rate Options)
  - Barrier Options (up-and-in, up-and-out, down-and-in, down-and-out)
  - Lookback Options (fixed, floating)
  - American Options
- **Full Greeks Calculation:** Delta, Gamma, Vega, Theta, Rho, Vanna, Volga, Cross-Gamma, DV01 (where applicable)
- **Portfolio Risk Dashboard:**
  - Add/remove/manage positions
  - Portfolio valuation (real-time)
  - Aggregated risk metrics and Greeks
  - Parametric and Monte Carlo VaR/ES
  - Scenario analysis, stress testing, and risk-return analytics
- **Live Market Data Integration:**
  - Fetch spot, strikes, expiries, and implied volatilities from Yahoo Finance (yfinance)
  - Auto-populate model parameters from live data
- **Advanced Visualizations:**
  - Interactive price and Greeks heatmaps (Plotly)
  - Side-by-side call/put value cards with clear color coding
  - LaTeX model formulas and descriptions
- **Robust UI/UX:**
  - Streamlit-based, with dynamic forms, sidebar navigation, and error handling
  - All widgets have unique keys to avoid UI errors
- **Comprehensive Testing:**
  - Pytest-based unit tests for all models and Greeks
  - Edge case and error handling coverage

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd optionspricing
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run app/ui/main.py
   ```

## Directory Structure
- `app/models/` — All pricing models (Black-Scholes, Binomial Tree, Heston, Hull-White, Barrier, Lookback, American)
- `app/ui/` — Streamlit UI logic, forms, plots, and main navigation
- `app/data/` — Live data utilities (yfinance integration)
- `tests/` — Unit tests for all models

## Requirements
- Python 3.8+
- See `requirements.txt` for package list

---

For more details, see inline documentation and module docstrings. 