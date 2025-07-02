import streamlit as st
import numpy as np
from app.ui.forms import black_scholes_form, binomial_tree_form, heston_form, hull_white_form
from app.models.black_scholes import black_scholes_price, black_scholes_greeks_full
from app.models.black_scholes import black_scholes_vanna, black_scholes_volga, black_scholes_cross_gamma, black_scholes_cross_greeks
from app.models.binomial_tree import binomial_tree_price, binomial_tree_greeks_full, binomial_tree_vanna, binomial_tree_volga, binomial_tree_cross_gamma, binomial_tree_cross_greeks
from app.models.heston import heston_price, heston_greeks_full, heston_vanna, heston_volga, heston_cross_gamma, heston_cross_greeks
from app.models.hull_white import hull_white_bond_option, hull_white_dv01
from numpy import std, sqrt, log
from app.models.barrier import barrier_option_price, barrier_greeks_full, barrier_vanna, barrier_volga, barrier_cross_gamma, barrier_cross_greeks
from app.models.lookback import lookback_option_price, lookback_greeks_full, lookback_vanna, lookback_volga, lookback_cross_gamma, lookback_cross_greeks
from app.models.american import american_option_price, american_greeks_full, american_vanna, american_volga, american_cross_gamma, american_cross_greeks
from app.ui.model_ui import model_ui
import yfinance as yf
from datetime import datetime

def model_analysis_page():
    st.title("Options Pricing & Risk Analysis Platform")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Fetch Live Option Parameters")
    ticker = st.sidebar.text_input("Ticker (e.g. AAPL, MSFT, SPY)", value="AAPL", key="live_ticker")

    if ticker:
        ticker_obj = yf.Ticker(ticker)
        spot = ticker_obj.history(period="1d")["Close"].iloc[-1]
        expirations = ticker_obj.options

        expiry = st.sidebar.selectbox("Expiry Date", expirations, key="live_expiry")
        if expiry:
            opt_chain = ticker_obj.option_chain(expiry)
            calls = opt_chain.calls
            puts = opt_chain.puts
            strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
            strike = st.sidebar.selectbox("Strike", strikes, key="live_strike")
            call_row = calls[calls["strike"] == strike]
            put_row = puts[puts["strike"] == strike]
            implied_vol = None
            if not call_row.empty:
                implied_vol = call_row.iloc[0]["impliedVolatility"]
            elif not put_row.empty:
                implied_vol = put_row.iloc[0]["impliedVolatility"]
            expiration = datetime.strptime(expiry, "%Y-%m-%d")
            today = datetime.now()
            T = max((expiration - today).days / 365.0, 0.001)
            risk_free_rate = 0.045
            implied_vol_str = f"{implied_vol:.2%}" if implied_vol else "N/A"
            st.sidebar.success(f"Fetched: S={spot:.2f}, K={strike}, T={T:.3f}y, σ={implied_vol_str}, r={risk_free_rate:.3f}")
        else:
            st.sidebar.warning("No expirations found for this ticker.")

    model = st.sidebar.selectbox("Select Model", [
        "Black-Scholes",
        "Binomial Tree",
        "Heston",
        # "Hull-White",
        "Barrier Option",
        "Lookback Option",
        "American Option"
    ])

    param_config_black_scholes = [
        {"name": "S", "label": "Current Asset Price", "default": 100.0, "step": 1.0, "format": "%.2f"},
        {"name": "K", "label": "Strike Price", "default": 100.0, "step": 1.0, "format": "%.2f"},
        {"name": "T", "label": "Time to Maturity (Years)", "default": 1.0, "step": 0.01, "format": "%.2f"},
        {"name": "sigma", "label": "Volatility (σ)", "default": 0.2, "step": 0.01, "format": "%.2f"},
        {"name": "r", "label": "Risk-Free Interest Rate", "default": 0.05, "step": 0.001, "format": "%.3f"},
    ]
    param_config_binomial_tree = [
        {"name": "S", "label": "Current Asset Price", "default": 100.0, "step": 1.0, "format": "%.2f"},
        {"name": "K", "label": "Strike Price", "default": 100.0, "step": 1.0, "format": "%.2f"},
        {"name": "T", "label": "Time to Maturity (Years)", "default": 1.0, "step": 0.01, "format": "%.2f"},
        {"name": "sigma", "label": "Volatility (σ)", "default": 0.2, "step": 0.01, "format": "%.2f"},
        {"name": "r", "label": "Risk-Free Interest Rate", "default": 0.05, "step": 0.001, "format": "%.3f"},
        {"name": "steps", "label": "Number of Steps", "default": 100, "step": 1, "format": "%d"},
    ]
    param_config_heston = [
        {"name": "S", "label": "Current Asset Price", "default": 100.0, "step": 1.0, "format": "%.2f"},
        {"name": "K", "label": "Strike Price", "default": 100.0, "step": 1.0, "format": "%.2f"},
        {"name": "T", "label": "Time to Maturity (Years)", "default": 1.0, "step": 0.01, "format": "%.2f"},
        {"name": "r", "label": "Risk-Free Interest Rate", "default": 0.05, "step": 0.001, "format": "%.3f"},
        {"name": "v0", "label": "Initial Variance (v0)", "default": 0.04, "step": 0.01, "format": "%.3f"},
        {"name": "kappa", "label": "Mean Reversion (kappa)", "default": 2.0, "step": 0.1, "format": "%.2f"},
        {"name": "theta", "label": "Long Run Variance (theta)", "default": 0.04, "step": 0.01, "format": "%.3f"},
        {"name": "sigma", "label": "Vol of Vol (σ)", "default": 0.5, "step": 0.01, "format": "%.2f"},
        {"name": "rho", "label": "Correlation (rho)", "default": -0.7, "step": 0.01, "format": "%.2f"},
    ]
    param_config_hull_white = [
        {"name": "r0", "label": "Initial Short Rate (r0)", "default": 0.01, "step": 0.001, "format": "%.3f"},
        {"name": "t", "label": "Valuation Time (t)", "default": 0.0, "step": 0.01, "format": "%.2f"},
        {"name": "T", "label": "Option Expiry (T)", "default": 3.0, "step": 0.01, "format": "%.2f"},
        {"name": "S", "label": "Bond Maturity (S)", "default": 5.0, "step": 0.01, "format": "%.2f"},
        {"name": "K", "label": "Strike", "default": 0.8, "step": 0.01, "format": "%.2f"},
        {"name": "a", "label": "Mean Reversion (a)", "default": 0.03, "step": 0.01, "format": "%.2f"},
        {"name": "sigma", "label": "Volatility (σ)", "default": 0.03, "step": 0.001, "format": "%.3f"},
    ]
    param_config_barrier = [
        {"name": "S", "label": "Spot Price (S)", "default": 100.0, "step": 1.0, "format": "%.2f"},
        {"name": "K", "label": "Strike Price (K)", "default": 100.0, "step": 1.0, "format": "%.2f"},
        {"name": "H", "label": "Barrier Level (H)", "default": 120.0, "step": 1.0, "format": "%.2f"},
        {"name": "T", "label": "Time to Maturity (years)", "default": 1.0, "step": 0.01, "format": "%.2f"},
        {"name": "r", "label": "Risk-Free Rate (r)", "default": 0.01, "step": 0.001, "format": "%.3f"},
        {"name": "sigma", "label": "Volatility (σ)", "default": 0.2, "step": 0.01, "format": "%.2f"},
        {"name": "rebate", "label": "Rebate (if knocked out)", "default": 0.0, "step": 0.01, "format": "%.2f"},
    ]
    param_config_lookback = [
        {"name": "S", "label": "Spot Price (S)", "default": 100.0, "step": 1.0, "format": "%.2f"},
        {"name": "K", "label": "Strike Price (K)", "default": 100.0, "step": 1.0, "format": "%.2f"},
        {"name": "T", "label": "Time to Maturity (years)", "default": 1.0, "step": 0.01, "format": "%.2f"},
        {"name": "r", "label": "Risk-Free Rate (r)", "default": 0.01, "step": 0.001, "format": "%.3f"},
        {"name": "sigma", "label": "Volatility (σ)", "default": 0.2, "step": 0.01, "format": "%.2f"},
    ]
    param_config_american = [
        {"name": "S", "label": "Spot Price (S)", "default": 100.0, "step": 1.0, "format": "%.2f"},
        {"name": "K", "label": "Strike Price (K)", "default": 100.0, "step": 1.0, "format": "%.2f"},
        {"name": "T", "label": "Time to Maturity (years)", "default": 1.0, "step": 0.01, "format": "%.2f"},
        {"name": "r", "label": "Risk-Free Rate (r)", "default": 0.01, "step": 0.001, "format": "%.3f"},
        {"name": "sigma", "label": "Volatility (σ)", "default": 0.2, "step": 0.01, "format": "%.2f"},
        {"name": "steps", "label": "Binomial Steps", "default": 100, "step": 1, "format": "%d"},
    ]

    if model == "Black-Scholes":
        model_ui(
            model_name="Black-Scholes",
            param_config=param_config_black_scholes,
            price_func=lambda S, K, T, r, sigma, option_type: black_scholes_price(S, K, T, r, sigma, option_type=option_type),
            greeks_func=lambda S, K, T, r, sigma, option_type: black_scholes_greeks_full(S, K, T, r, sigma, option_type=option_type),
            price_types=("call", "put"),
            heatmap_params={"min_S": 80.0, "max_S": 120.0, "min_sigma": 0.10, "max_sigma": 0.30},
        )
    elif model == "Binomial Tree":
        exercise = st.sidebar.selectbox("Exercise Type", ["european", "american"], key=f"exercise_{model}")
        model_ui(
            model_name="Binomial Tree",
            param_config=param_config_binomial_tree,
            price_func=lambda S, K, T, r, sigma, steps, option_type: binomial_tree_price(S, K, T, r, sigma, steps, option_type=option_type, exercise=exercise),
            greeks_func=lambda S, K, T, r, sigma, steps, option_type: binomial_tree_greeks_full(S, K, T, r, sigma, steps, option_type=option_type, exercise=exercise),
            price_types=("call", "put"),
            heatmap_params={"min_S": 80.0, "max_S": 120.0, "min_sigma": 0.10, "max_sigma": 0.30},
        )
    elif model == "Heston":
        model_ui(
            model_name="Heston",
            param_config=param_config_heston,
            price_func=lambda S, K, T, r, v0, kappa, theta, sigma, rho, option_type: heston_price(S, K, T, r, v0, kappa, theta, sigma, rho, option_type=option_type),
            greeks_func=lambda S, K, T, r, v0, kappa, theta, sigma, rho, option_type: heston_greeks_full(S, K, T, r, v0, kappa, theta, sigma, rho, option_type=option_type),
            price_types=("call", "put"),
            heatmap_params={"min_S": 80.0, "max_S": 120.0, "min_sigma": 0.10, "max_sigma": 0.30},
        )
    elif model == "Hull-White":
        model_ui(
            model_name="Hull-White",
            param_config=param_config_hull_white,
            price_func=lambda r0, t, T, S, K, a, sigma, option_type: hull_white_bond_option(r0, t, T, S, K, a, sigma, P0T=lambda T_: np.exp(-r0 * T_), option_type=option_type),
            greeks_func=lambda r0, t, T, S, K, a, sigma, option_type: {"DV01": hull_white_dv01(r0, t, T, S, K, a, sigma, P0T=lambda T_: np.exp(-r0 * T_), option_type=option_type)},
            price_types=("call", "put"),
            heatmap_params={"min_S": 80.0, "max_S": 120.0, "min_sigma": 0.10, "max_sigma": 0.30},
        )
    elif model == "Barrier Option":
        barrier_type = st.sidebar.selectbox("Barrier Type", ["up-and-out", "down-and-out", "up-and-in", "down-and-in"], index=2, key=f"barrier_type_{model}")
        H = None
        for p in param_config_barrier:
            if p["name"] == "H":
                H = st.sidebar.number_input(p["label"], value=p["default"], step=p.get("step", 1.0), format=p.get("format", "%.2f"), key="barrier_H")
                break
        if H is not None:
            if barrier_type.startswith("up"):
                min_S = 0.8 * H
                max_S = 0.99 * H
            elif barrier_type.startswith("down"):
                min_S = 1.01 * H
                max_S = 1.2 * H
            else:
                min_S = 80.0
                max_S = 120.0
        else:
            min_S = 80.0
            max_S = 120.0
        heatmap_params = {
            "min_S": min_S,
            "max_S": max_S,
            "min_sigma": 0.10,
            "max_sigma": 0.30
        }
        model_ui(
            model_name="Barrier Option",
            param_config=param_config_barrier,
            price_func=lambda S, K, H, T, r, sigma, rebate, option_type: barrier_option_price(S, K, H, T, r, sigma, option_type=option_type, barrier_type=barrier_type, rebate=rebate),
            greeks_func=lambda S, K, H, T, r, sigma, rebate, option_type: barrier_greeks_full(S, K, H, T, r, sigma, option_type=option_type, barrier_type=barrier_type, rebate=rebate),
            price_types=("call", "put"),
            heatmap_params=heatmap_params,
        )
    elif model == "Lookback Option":
        lookback_type = st.sidebar.selectbox("Lookback Type", ["fixed", "floating"], key=f"lookback_type_{model}")
        model_ui(
            model_name="Lookback Option",
            param_config=param_config_lookback,
            price_func=lambda S, K, T, r, sigma, option_type: lookback_option_price(S, K, T, r, sigma, option_type=option_type, lookback_type=lookback_type),
            greeks_func=lambda S, K, T, r, sigma, option_type: lookback_greeks_full(S, K, T, r, sigma, option_type=option_type, lookback_type=lookback_type),
            price_types=("call", "put"),
            heatmap_params={"min_S": 80.0, "max_S": 120.0, "min_sigma": 0.10, "max_sigma": 0.30},
        )
    elif model == "American Option":
        model_ui(
            model_name="American Option",
            param_config=param_config_american,
            price_func=lambda S, K, T, r, sigma, steps, option_type: american_option_price(S, K, T, r, sigma, steps, option_type=option_type),
            greeks_func=lambda S, K, T, r, sigma, steps, option_type: american_greeks_full(S, K, T, r, sigma, steps, option_type=option_type),
            price_types=("call", "put"),
            heatmap_params={"min_S": 80.0, "max_S": 120.0, "min_sigma": 0.10, "max_sigma": 0.30},
        )