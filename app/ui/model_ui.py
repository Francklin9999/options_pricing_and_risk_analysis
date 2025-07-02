import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

model_descriptions = {
    "Black-Scholes": (
        "The Black-Scholes model is a closed-form solution for pricing European options. "
        "It assumes lognormal asset returns, constant volatility, and no dividends. "
        "The model provides analytical formulas for call and put prices, as well as all Greeks.\n\n"
        "**Parameters:**\n"
        "- S: Spot price of the underlying asset\n"
        "- K: Strike price\n"
        "- T: Time to maturity\n"
        "- r: Risk-free interest rate\n"
        "- σ: Volatility of the underlying asset",
        [
            r"C = S N(d_1) - K e^{-rT} N(d_2)",
            r"P = K e^{-rT} N(-d_2) - S N(-d_1)",
            r"d_1 = \frac{\ln\left(\frac{S}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}}",
            r"d_2 = d_1 - \sigma\sqrt{T}",
        ]
    ),
    "Binomial Tree": (
        "The Binomial Tree model prices options by simulating possible paths for the underlying asset using discrete time steps. "
        "It can handle American and European options, and allows for early exercise.\n\n"
        "**Parameters:**\n"
        "- S: Spot price\n"
        "- K: Strike price\n"
        "- T: Time to maturity\n"
        "- r: Risk-free rate\n"
        "- σ: Volatility\n"
        "- N: Number of steps",
        [
            r"\text{Option price} = \sum_{i=0}^N \binom{N}{i} p^i (1-p)^{N-i} V_i",
            r"p = \text{risk-neutral probability}, \quad V_i = \text{value at node } i"
        ]
    ),
    "Heston": (
        "The Heston model is a stochastic volatility model for option pricing. It assumes that the asset volatility follows its own mean-reverting process, "
        "allowing for volatility smiles and skews.\n\n"
        "**Parameters:**\n"
        "- S: Spot price\n"
        "- v₀: Initial variance\n"
        "- κ: Mean reversion speed\n"
        "- θ: Long-term variance\n"
        "- σ: Volatility of variance (vol of vol)\n"
        "- ρ: Correlation between asset and variance\n"
        "- r: Risk-free rate\n"
        "- T: Time to maturity",
        [
            r"dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^S",
            r"dv_t = \kappa(\theta - v_t)dt + \sigma\sqrt{v_t} dW_t^v"
        ]
    ),
    "Hull-White": (
        "The Hull-White model is used for pricing interest rate derivatives. It models the short rate as a mean-reverting process, allowing for closed-form solutions "
        "for zero-coupon bonds and bond options.\n\n"
        "**Parameters:**\n"
        "- r₀: Initial short rate\n"
        "- a: Mean reversion speed\n"
        "- θ(t): Time-dependent drift term\n"
        "- σ: Volatility\n"
        "- T: Maturity",
        [
            r"dr_t = (\theta(t) - a r_t)dt + \sigma dW_t"
        ]
    ),
    "Barrier Option": (
        "Barrier options are path-dependent options that are activated or extinguished if the underlying asset crosses a specified barrier level. Pricing formulas depend "
        "on the type of barrier (up-and-out, down-and-in, etc.).\n\n"
        "**Parameters:**\n"
        "- S: Spot price\n"
        "- K: Strike price\n"
        "- H: Barrier level\n"
        "- T: Time to maturity\n"
        "- r: Risk-free rate\n"
        "- σ: Volatility",
        [
            r"C_{\text{UO}} = C_{\text{BS}} - C_{\text{adj}}",
            r"\text{(Up-and-out call as example)}"
        ]
    ),
    "Lookback Option": (
        "Lookback options are path-dependent options whose payoff depends on the maximum or minimum asset price during the option's life. "
        "They are useful for hedging against missed opportunities.\n\n"
        "**Parameters:**\n"
        "- S: Spot price\n"
        "- K: Strike price\n"
        "- S_{max} or S_{min}: Observed extreme price\n"
        "- T: Time to maturity\n"
        "- r: Risk-free rate\n"
        "- σ: Volatility",
        [
            r"\text{Payoff (fixed strike call)} = \max(S_{\max} - K, 0)"
        ]
    ),
    "American Option": (
        "American options can be exercised at any time before expiry. They are typically priced using binomial trees or finite difference methods, as there is no closed-form solution.\n\n"
        "**Parameters:**\n"
        "- S: Spot price\n"
        "- K: Strike price\n"
        "- T: Time to maturity\n"
        "- r: Risk-free rate\n"
        "- σ: Volatility",
        [
            r"\text{Price} = \max(\text{Intrinsic Value}, \text{Discounted Expected Value})"
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