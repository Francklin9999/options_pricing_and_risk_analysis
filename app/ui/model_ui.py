import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

model_descriptions = {
    "Black-Scholes": (
        "Closed-form model for European options assuming log-normal returns, "
        "constant volatility, friction-less markets, continuous trading and no arbitrage. "
        "Dividends can be handled by a continuous dividend yield q.\n\n"
        "**Parameters:**\n"
        "- S  : Current spot price of the underlying asset\n"
        "- K  : Strike price\n"
        "- T  : Time to maturity (in years)\n"
        "- r  : Continuously-compounded risk-free rate\n"
        "- q  : Continuous dividend yield (0 if none)\n"
        "- σ  : Constant volatility of the underlying asset return",
        [
            r"C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)",
            r"P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)",
            r"d_1 = \frac{\ln(S/K) + (r - q + 0.5\sigma^{2})T}{\sigma\sqrt{T}}",
            r"d_2 = d_1 - \sigma\sqrt{T}",
        ]
    ),

    "Binomial Tree": (
        "Discrete-time lattice that models the underlying price as moving up by factor *u* or "
        "down by factor *d* each step. Risk-neutral probability *p* is chosen so that the tree "
        "replicates the risk-free drift. Early exercise is handled by comparing intrinsic and "
        "continuation values at each node, making the method suitable for American and exotic "
        "features.\n\n"
        "**Parameters:**\n"
        "- S  : Current spot price\n"
        "- K  : Strike price\n"
        "- T  : Time to maturity (years)\n"
        "- r  : Risk-free rate\n"
        "- σ  : Volatility (used to set *u*, *d*)\n"
        "- N  : Number of time steps in the tree",
        [
            r"u = e^{\sigma\sqrt{\Delta t}},\; d = e^{-\sigma\sqrt{\Delta t}},\;"
            r"p = \frac{e^{r\Delta t} - d}{u - d},\; \Delta t = T/N",
            r"Option\ value = \text{discounted expected value under } p \text{ with early-exercise checks}"
        ]
    ),

    "Heston": (
        "Stochastic-volatility model in which variance itself follows a square-root (CIR) "
        "process, producing volatility smiles/skews observed in markets.\n\n"
        "**Parameters:**\n"
        "- S      : Current spot price\n"
        "- v0     : Current variance (σ₀²)\n"
        "- κ      : Mean-reversion speed of the variance process\n"
        "- θ      : Long-run (equilibrium) variance level\n"
        "- σ_v    : Volatility of variance (\"vol-of-vol\")\n"
        "- ρ      : Correlation between Brownian motions of price and variance\n"
        "- r      : Risk-free rate\n"
        "- q      : Dividend yield (if any)\n"
        "- T      : Time to maturity",
        [
            r"dS_t = (r - q) S_t dt + \sqrt{v_t}\, S_t dW_t^{(1)}",
            r"dv_t = \kappa(\theta - v_t)dt + \sigma_v \sqrt{v_t}\, dW_t^{(2)}",
            r"dW_t^{(1)} dW_t^{(2)} = \rho dt"
        ]
    ),

    "Hull-White (1-factor)": (
        "Short-rate model for fixed-income derivatives; the continuously-compounded short rate "
        "reverts to a time-dependent mean level, enabling analytic zero-bond prices and efficient "
        "tree/Monte-Carlo methods for bond options, swaptions, etc.\n\n"
        "**Parameters:**\n"
        "- r0 : Initial short rate *r(0)*\n"
        "- a  : Mean-reversion speed (higher ⇒ faster pull to θ)\n"
        "- θ(t): Time-dependent mean level calibrated to the initial yield curve\n"
        "- σ  : Volatility of the short rate\n"
        "- T  : Horizon of the derivative being priced",
        [
            r"dr_t = \bigl(\theta(t) - a\,r_t\bigr) dt + \sigma\, dW_t"
        ]
    ),

    "Barrier Option": (
        "Path-dependent option that is knocked *in* or *out* when the underlying breaches a "
        "preset barrier *H*.  Pricing under Black-Scholes uses closed-form solutions that adjust "
        "vanilla prices for the probability of hitting the barrier.\n\n"
        "**Parameters:**\n"
        "- S  : Current spot price\n"
        "- K  : Strike price\n"
        "- H  : Barrier level (up/down)\n"
        "- T  : Time to maturity\n"
        "- r  : Risk-free rate\n"
        "- q  : Dividend yield\n"
        "- σ  : Volatility\n"
        "- barrier_type : 'up-and-out', 'down-and-in', etc.",
        [
            r"C_{\text{up-out}} = C_{\text{BS}} - C_{\text{adj}} \quad (\text{one example})"
        ]
    ),

    "Lookback Option": (
        "Path-dependent contract whose payoff depends on the maximum (or minimum) underlying "
        "price observed over the life of the option.  Analytical formulas exist for European "
        "lookbacks under Black-Scholes; American versions require trees/MC.\n\n"
        "**Parameters:**\n"
        "- S         : Current spot price\n"
        "- K         : Strike price (used for **fixed-strike** variant; omit for floating)\n"
        "- S_max/S_min: Running maximum or minimum of the spot (captured during pricing)\n"
        "- T         : Time to maturity\n"
        "- r         : Risk-free rate\n"
        "- q         : Dividend yield\n"
        "- σ         : Volatility",
        [
            r"Payoff_{fixed\ strike\ call} = \max(S_{\max} - K,\; 0)",
            r"Payoff_{floating\ strike\ call} = \max(S_T - S_{\min},\; 0)"
        ]
    ),

    "American Option": (
        "Option exercisable at any time up to expiry. No closed-form price under "
        "Black-Scholes; lattice (binomial/trinomial), finite-difference, or simulation "
        "methods with early-exercise logic are used.\n\n"
        "**Parameters:**\n"
        "- S  : Current spot price\n"
        "- K  : Strike price\n"
        "- T  : Time to maturity\n"
        "- r  : Risk-free rate\n"
        "- q  : Dividend yield\n"
        "- σ  : Volatility\n"
        "- method : Numerical scheme (e.g., 'binomial', 'finite_difference')",
        [
            r"Price = \max\bigl(\text{intrinsic},\ \text{discounted expected continuation}\bigr)"
        ]
    ),
}


def model_ui(
    model_name,
    param_config,
    price_func,
    greeks_func=None,
    price_types=("call", "put"),
    heatmap_params=None,
):
    st.sidebar.markdown(f"## :bar_chart: {model_name} Model")
    params = {}
    for p in param_config:
        params[p["name"]] = st.sidebar.number_input(
            p["label"], value=p["default"], step=p.get("step", 0.01), format=p.get("format", "%.2f")
        )

    heatmap_param_names = [p["name"] for p in param_config if p["name"] not in ("S", "sigma")]
    heatmap_params_dict = {k: params[k] for k in heatmap_param_names}

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Heatmap Parameters")
    min_S = st.sidebar.number_input("Min Spot Price", value=heatmap_params.get("min_S", 80.0), step=1.0)
    max_S = st.sidebar.number_input("Max Spot Price", value=heatmap_params.get("max_S", 120.0), step=1.0)
    min_sigma = st.sidebar.number_input("Min Volatility for Heatmap", value=heatmap_params.get("min_sigma", 0.10), step=0.01)
    max_sigma = st.sidebar.number_input("Max Volatility for Heatmap", value=heatmap_params.get("max_sigma", 0.30), step=0.01)

    st.markdown(f"<h1 style='text-align: center;'>{model_name} Pricing Model</h1>", unsafe_allow_html=True)

    desc, formulas = model_descriptions.get(model_name, ("", []))
    st.info(f"**Model Description:** {desc}")

    if formulas:
        st.markdown("**Model Formula:**")
        for formula in formulas:
            st.latex(formula)

    cols = st.columns(len(price_types))
    for i, opt_type in enumerate(price_types):
        if len(price_types) == 1:
            price = price_func(**params)
        else:
            price = price_func(**params, option_type=opt_type)
        color = "#b6fcb6" if opt_type.lower() == "call" else "#fcb6b6"
        with cols[i]:
            st.markdown(
                f"<div style='background-color:{color};color:#000;padding:10px;border-radius:10px;text-align:center;'>"
                f"<h3>{opt_type.upper()} Value</h3><h2>${price:.2f}</h2></div>",
                unsafe_allow_html=True
            )

    st.markdown("## Options Price - Interactive Heatmap")

    S_grid = np.linspace(min_S, max_S, 10)
    sigma_grid = np.linspace(min_sigma, max_sigma, 10)
    for opt_type in price_types:
        matrix = np.zeros((len(sigma_grid), len(S_grid)))
        for i, sig in enumerate(sigma_grid):
            for j, s in enumerate(S_grid):
                call_params = heatmap_params_dict.copy()
                if "S" in params:
                    call_params["S"] = s
                if "sigma" in params:
                    call_params["sigma"] = sig
                if len(price_types) == 1:
                    matrix[i, j] = price_func(**call_params)
                else:
                    matrix[i, j] = price_func(**call_params, option_type=opt_type)
        matrix_df = pd.DataFrame(matrix, index=np.round(sigma_grid, 2), columns=np.round(S_grid, 2))
        text = np.round(matrix, 2).astype(str)
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=np.round(S_grid, 2),
                y=np.round(sigma_grid, 2),
                colorscale='Viridis',
                colorbar=dict(title=f"{opt_type.capitalize()} Price"),
                showscale=True,
                zmin=np.min(matrix),
                zmax=np.max(matrix),
                text=text,
                texttemplate="%{text}",
                hovertemplate="Spot: %{x}<br>Vol: %{y}<br>Price: %{z:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title="Spot Price (S)",
            yaxis_title="Volatility (σ)",
            xaxis=dict(showgrid=True, tickmode='array', tickvals=np.round(S_grid, 2)),
            yaxis=dict(showgrid=True, tickmode='array', tickvals=np.round(sigma_grid, 2)),
            margin=dict(l=40, r=40, t=40, b=40),
            font=dict(size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"#### {opt_type.capitalize()} Price Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if greeks_func:
                st.markdown(f"#### {opt_type.capitalize()} Greeks")
                if len(price_types) == 1:
                    greeks = greeks_func(**params)
                else:
                    greeks = greeks_func(**params, option_type=opt_type)
                df = pd.DataFrame([greeks]).T
                df.columns = ['']
                st.table(df) 