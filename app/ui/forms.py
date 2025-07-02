import streamlit as st

def black_scholes_form():
    with st.form("bs_form"):
        S = st.number_input("Spot Price (S)", min_value=0.01, value=100.0)
        K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0)
        T = st.number_input("Time to Maturity (years, T)", min_value=0.01, value=1.0)
        r = st.number_input("Risk-Free Rate (r, annualized)", value=0.05)
        sigma = st.number_input("Volatility (sigma, annualized)", min_value=0.0001, value=0.2)
        option_type = st.selectbox("Option Type", ["call", "put"])
        submitted = st.form_submit_button("Calculate")
    if submitted:
        return S, K, T, r, sigma, option_type
    return None

def binomial_tree_form():
    with st.form("bt_form"):
        S = st.number_input("Spot Price (S)", min_value=0.01, value=100.0, key="bt_S")
        K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0, key="bt_K")
        T = st.number_input("Time to Maturity (years, T)", min_value=0.01, value=1.0, key="bt_T")
        r = st.number_input("Risk-Free Rate (r, annualized)", value=0.05, key="bt_r")
        sigma = st.number_input("Volatility (sigma, annualized)", min_value=0.0001, value=0.2, key="bt_sigma")
        steps = st.number_input("Number of Steps", min_value=1, max_value=1000, value=100, step=1, key="bt_steps")
        option_type = st.selectbox("Option Type", ["call", "put"], key="bt_option_type")
        exercise = st.selectbox("Exercise Type", ["european", "american"], key="bt_exercise")
        submitted = st.form_submit_button("Calculate")
    if submitted:
        return S, K, T, r, sigma, int(steps), option_type, exercise
    return None

def heston_form():
    with st.form("heston_form"):
        S = st.number_input("Spot Price (S)", min_value=0.01, value=100.0, key="heston_S")
        K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0, key="heston_K")
        T = st.number_input("Time to Maturity (years, T)", min_value=0.01, value=1.0, key="heston_T")
        r = st.number_input("Risk-Free Rate (r, annualized)", value=0.05, key="heston_r")
        v0 = st.number_input("Initial Variance (v0)", min_value=0.0001, value=0.04, key="heston_v0")
        kappa = st.number_input("Mean Reversion Speed (kappa)", min_value=0.0001, value=2.0, key="heston_kappa")
        theta = st.number_input("Long-term Variance (theta)", min_value=0.0001, value=0.04, key="heston_theta")
        sigma = st.number_input("Vol of Vol (sigma)", min_value=0.0001, value=0.5, key="heston_sigma")
        rho = st.number_input("Correlation (rho)", min_value=-1.0, max_value=1.0, value=-0.7, key="heston_rho")
        option_type = st.selectbox("Option Type", ["call", "put"], key="heston_option_type")
        submitted = st.form_submit_button("Calculate")
    if submitted:
        return S, K, T, r, v0, kappa, theta, sigma, rho, option_type
    return None

def hull_white_form():
    with st.form("hw_form"):
        r0 = st.number_input("Initial Short Rate (r0)", value=0.03, key="hw_r0")
        t = st.number_input("Current Time (t)", min_value=0.0, value=0.0, key="hw_t")
        T = st.number_input("Option Expiry (T)", min_value=0.01, value=2.0, key="hw_T")
        S = st.number_input("Bond Maturity (S)", min_value=0.01, value=5.0, key="hw_S")
        K = st.number_input("Strike Price (K)", min_value=0.01, value=0.8, key="hw_K")
        a = st.number_input("Mean Reversion Speed (a)", min_value=0.0001, value=0.1, key="hw_a")
        sigma = st.number_input("Volatility (sigma)", min_value=0.0001, value=0.01, key="hw_sigma")
        submitted = st.form_submit_button("Calculate")
    if submitted:
        return r0, t, T, S, K, a, sigma
    return None 