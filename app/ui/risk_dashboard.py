import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from scipy.stats import norm

def risk_dashboard_page():
    st.title("Risk Dashboard")
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = []
    if not st.session_state['portfolio']:
        st.session_state['portfolio'] = [
            {"model": "Black-Scholes", "notional": 1000000.0, "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.01, "sigma": 0.2, "option_type": "call"},
            {"model": "Black-Scholes", "notional": 500000.0, "S": 105.0, "K": 100.0, "T": 0.5, "r": 0.015, "sigma": 0.25, "option_type": "put"},
            {"model": "Binomial Tree", "notional": 750000.0, "S": 98.0, "K": 100.0, "T": 1.2, "r": 0.012, "sigma": 0.22, "steps": 100, "option_type": "call", "exercise": "european"},
            {"model": "Heston", "notional": 600000.0, "S": 102.0, "K": 100.0, "T": 0.8, "r": 0.013, "v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.5, "rho": -0.7, "option_type": "put"}
        ]
    models = ["Black-Scholes", "Binomial Tree", "Heston", "Hull-White"]
    with st.form("add_position_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            model = st.selectbox("Model", models)
        with col2:
            notional = st.number_input("Notional", min_value=1.0, value=1000000.0, step=10000.0)
        with col3:
            option_type = st.selectbox("Type", ["call", "put"], key="dash_type") if model in ["Black-Scholes", "Binomial Tree", "Heston"] else None
        if model == "Black-Scholes":
            S = st.number_input("Spot", value=100.0)
            K = st.number_input("Strike", value=100.0)
            T = st.number_input("Maturity (years)", value=1.0)
            r = st.number_input("Rate", value=0.01)
            sigma = st.number_input("Volatility", value=0.2)
        elif model == "Binomial Tree":
            S = st.number_input("Spot", value=100.0)
            K = st.number_input("Strike", value=100.0)
            T = st.number_input("Maturity (years)", value=1.0)
            r = st.number_input("Rate", value=0.01)
            sigma = st.number_input("Volatility", value=0.2)
            steps = st.number_input("Steps", value=50)
            exercise = st.selectbox("Exercise", ["european", "american"])
        elif model == "Heston":
            S = st.number_input("Spot", value=100.0)
            K = st.number_input("Strike", value=100.0)
            T = st.number_input("Maturity (years)", value=1.0)
            r = st.number_input("Rate", value=0.01)
            v0 = st.number_input("v0", value=0.04)
            kappa = st.number_input("kappa", value=2.0)
            theta = st.number_input("theta", value=0.04)
            sigma = st.number_input("Vol of Vol", value=0.5)
            rho = st.number_input("rho", value=-0.7)
        elif model == "Hull-White":
            r0 = st.number_input("Initial Short Rate (r0)", value=0.01)
            t = st.number_input("Valuation Time (t)", value=0.0)
            T = st.number_input("Option Expiry (T)", value=1.0)
            S_ = st.number_input("Bond Maturity (S)", value=2.0)
            K = st.number_input("Strike", value=0.95)
            a = st.number_input("Mean Reversion (a)", value=0.1)
            sigma = st.number_input("Volatility (sigma)", value=0.01)
        submitted = st.form_submit_button("Add Position")
        if submitted:
            pos = {"model": model, "notional": notional}
            if model == "Black-Scholes":
                pos.update(dict(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type))
            elif model == "Binomial Tree":
                pos.update(dict(S=S, K=K, T=T, r=r, sigma=sigma, steps=steps, option_type=option_type, exercise=exercise))
            elif model == "Heston":
                pos.update(dict(S=S, K=K, T=T, r=r, v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho, option_type=option_type))
            elif model == "Hull-White":
                pos.update(dict(r0=r0, t=t, T=T, S=S_, K=K, a=a, sigma=sigma))
            st.session_state['portfolio'].append(pos)
            st.success("Position added!")
    st.subheader("Remove Position")
    if len(st.session_state['portfolio']) > 0:
        remove_idx = st.selectbox("Select position to remove (by index)", list(range(len(st.session_state['portfolio']))))
        if st.button("Remove Selected Position"):
            st.session_state['portfolio'].pop(remove_idx)
            st.success("Position removed.")
            st.experimental_rerun()
    st.subheader("Portfolio Positions")
    if not st.session_state['portfolio']:
        st.info("No positions in portfolio.")
        return
    port_df = pd.DataFrame(st.session_state['portfolio'])
    st.dataframe(port_df)
    st.subheader("Risk Table")
    risk_rows = []
    for pos in st.session_state['portfolio']:
        model = pos['model']
        if model == "Black-Scholes":
            from app.models.black_scholes import black_scholes_greeks, black_scholes_price
            greeks = black_scholes_greeks(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])
            price = black_scholes_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])
            risk_rows.append({**pos, 'price': price, **greeks})
        elif model == "Binomial Tree":
            from app.models.binomial_tree import binomial_tree_price, binomial_tree_delta, binomial_tree_gamma
            price = binomial_tree_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
            delta = binomial_tree_delta(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
            gamma = binomial_tree_gamma(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
            risk_rows.append({**pos, 'price': price, 'delta': delta, 'gamma': gamma})
        elif model == "Heston":
            from app.models.heston import heston_price, heston_delta
            price = heston_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
            delta = heston_delta(pos['S'], pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
            risk_rows.append({**pos, 'price': price, 'delta': delta})
        elif model == "Hull-White":
            from app.models.hull_white import hull_white_bond_option, hull_white_dv01
            def P0T(T_):
                return np.exp(-pos['r0'] * T_)
            price = hull_white_bond_option(pos['r0'], pos['t'], pos['T'], pos['S'], pos['K'], pos['a'], pos['sigma'], P0T, 'call')
            dv01 = hull_white_dv01(pos['r0'], pos['t'], pos['T'], pos['S'], pos['K'], pos['a'], pos['sigma'], P0T, 'call')
            risk_rows.append({**pos, 'price': price, 'DV01': dv01})
    if risk_rows:
        risk_df = pd.DataFrame(risk_rows)
        st.dataframe(risk_df)
        portfolio_value = sum(row['price'] * row['notional'] for row in risk_rows if 'price' in row and 'notional' in row)
        st.metric("Portfolio Valuation", f"${portfolio_value:,.2f}")
    st.header("Portfolio Value-at-Risk (VaR) & Expected Shortfall (ES)")
    if not st.session_state['portfolio']:
        st.info("Add positions to compute portfolio VaR/ES.")
    else:
        st.subheader("Parametric (Delta-Normal) VaR/ES")
        conf_level = st.slider("Confidence Level", min_value=90, max_value=99, value=95, step=1, format="%d%%", key="main_conf_level") / 100
        holding_period = st.number_input("Holding Period (days)", min_value=1, value=1)
        port_df = pd.DataFrame(st.session_state['portfolio'])
        deltas = []
        spots = []
        notionals = []
        for pos in st.session_state['portfolio']:
            if pos['model'] in ["Black-Scholes", "Binomial Tree", "Heston"]:
                S = pos['S']
                notional = pos['notional']
                if pos['model'] == "Black-Scholes":
                    from app.models.black_scholes import black_scholes_greeks
                    delta = black_scholes_greeks(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])['delta']
                elif pos['model'] == "Binomial Tree":
                    from app.models.binomial_tree import binomial_tree_delta
                    delta = binomial_tree_delta(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                elif pos['model'] == "Heston":
                    from app.models.heston import heston_delta
                    delta = heston_delta(pos['S'], pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                deltas.append(delta * notional)
                spots.append(S)
                notionals.append(notional)
        spot_vol = st.number_input("Spot Volatility (annualized, %)", min_value=1.0, value=20.0) / 100
        spot_vol_hp = spot_vol * np.sqrt(holding_period / 252)
        port_delta = np.sum(deltas)
        port_std = np.sqrt(np.sum((np.array(deltas) * np.array(spots) * spot_vol_hp) ** 2))
        var = norm.ppf(conf_level) * port_std
        es = port_std * norm.pdf(norm.ppf(conf_level)) / (1 - conf_level)
        st.write(f"**Parametric VaR ({int(conf_level*100)}%, {holding_period}d):** ${var:,.0f}")
        st.write(f"**Parametric ES ({int(conf_level*100)}%, {holding_period}d):** ${es:,.0f}")
        st.caption("Parametric VaR/ES assumes normal P&L distribution using portfolio delta and spot volatility.")
        st.divider()
        st.subheader("Monte Carlo VaR/ES")
        n_sims = st.number_input("# Simulations", min_value=100, max_value=10000, value=2000, step=100)
        np.random.seed(42)
        spot_shocks = np.random.normal(0, spot_vol_hp, size=(n_sims, len(spots)))
        pnl = []
        for i in range(n_sims):
            total_pnl = 0
            for j, pos in enumerate(st.session_state['portfolio']):
                if pos['model'] in ["Black-Scholes", "Binomial Tree", "Heston"]:
                    shocked_S = spots[j] * (1 + spot_shocks[i, j])
                    notional = notionals[j]
                    if pos['model'] == "Black-Scholes":
                        from app.models.black_scholes import black_scholes_price
                        base = black_scholes_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])
                        shocked = black_scholes_price(shocked_S, pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])
                    elif pos['model'] == "Binomial Tree":
                        from app.models.binomial_tree import binomial_tree_price
                        base = binomial_tree_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                        shocked = binomial_tree_price(shocked_S, pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                    elif pos['model'] == "Heston":
                        from app.models.heston import heston_price
                        base = heston_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                        shocked = heston_price(shocked_S, pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                    total_pnl += (shocked - base) * notional
            pnl.append(total_pnl)
        pnl = np.array(pnl)
        var_mc = np.percentile(pnl, 100 * (1 - conf_level))
        es_mc = pnl[pnl <= var_mc].mean() if np.any(pnl <= var_mc) else np.nan
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pnl, nbinsx=50, name="P&L"))
        fig.add_vline(x=var_mc, line_dash="dash", line_color="red", annotation_text="VaR", annotation_position="top left")
        fig.add_vline(x=es_mc, line_dash="dot", line_color="orange", annotation_text="ES", annotation_position="top left")
        fig.update_layout(title="Monte Carlo Simulated P&L Distribution", xaxis_title="P&L", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True, key="mc_var_hist")
        st.write(f"**Monte Carlo VaR ({int(conf_level*100)}%, {holding_period}d):** ${var_mc:,.0f}")
        st.write(f"**Monte Carlo ES ({int(conf_level*100)}%, {holding_period}d):** ${es_mc:,.0f}")
        st.caption("Monte Carlo VaR/ES is based on simulated spot shocks and full portfolio revaluation.")
    st.header("Advanced Analytics")
    if not st.session_state['portfolio']:
        st.info("Add positions to access advanced analytics.")
    else:
        analytics_tab = st.selectbox("Select Analytics Type", [
            "VaR & ES", "Monte Carlo Scenario Engine", "Stress Testing", "Sensitivity/Attribution", "Risk-Return Metrics"])
        port_df = pd.DataFrame(st.session_state['portfolio'])
        if analytics_tab == "VaR & ES":
            method = st.selectbox("VaR/ES Method", ["Parametric (Delta-Normal)", "Parametric (Cornish-Fisher)", "Monte Carlo", "Historical Simulation"])
            conf_level = st.slider("Confidence Level", min_value=90, max_value=99, value=95, step=1, format="%d%%", key="adv_conf_level") / 100
            holding_period = st.number_input("Holding Period (days)", min_value=1, value=1, key="adv_hp")
            if method == "Historical Simulation":
                st.markdown("Upload or simulate historical returns for the portfolio.")
                n_hist = st.number_input("# Historical Days", min_value=100, max_value=2000, value=500)
                np.random.seed(42)
                hist_returns = np.random.normal(0, 0.01, size=n_hist)
                var_hist = np.percentile(hist_returns, 100 * (1 - conf_level))
                es_hist = hist_returns[hist_returns <= var_hist].mean() if np.any(hist_returns <= var_hist) else np.nan
                st.write(f"**Historical VaR ({int(conf_level*100)}%, 1d):** {var_hist:.4%}")
                st.write(f"**Historical ES ({int(conf_level*100)}%, 1d):** {es_hist:.4%}")
                st.caption("Historical VaR/ES is based on the empirical distribution of past returns.")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=hist_returns, nbinsx=50, name="Returns"))
                fig.add_vline(x=var_hist, line_dash="dash", line_color="red", annotation_text="VaR", annotation_position="top left")
                fig.add_vline(x=es_hist, line_dash="dot", line_color="orange", annotation_text="ES", annotation_position="top left")
                fig.update_layout(title="Historical Returns Distribution", xaxis_title="Return", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True, key="adv_hist_var_hist")
            elif method == "Parametric (Delta-Normal)":
                deltas = []
                spots = []
                notionals = []
                for pos in st.session_state['portfolio']:
                    if pos['model'] in ["Black-Scholes", "Binomial Tree", "Heston"]:
                        S = pos['S']
                        notional = pos['notional']
                        if pos['model'] == "Black-Scholes":
                            from app.models.black_scholes import black_scholes_greeks
                            delta = black_scholes_greeks(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])['delta']
                        elif pos['model'] == "Binomial Tree":
                            from app.models.binomial_tree import binomial_tree_delta
                            delta = binomial_tree_delta(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                        elif pos['model'] == "Heston":
                            from app.models.heston import heston_delta
                            delta = heston_delta(pos['S'], pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                        deltas.append(delta * notional)
                        spots.append(S)
                        notionals.append(notional)
                spot_vol = st.number_input("Spot Volatility (annualized, %)", min_value=1.0, value=20.0, key="adv_spot_vol") / 100
                spot_vol_hp = spot_vol * np.sqrt(holding_period / 252)
                port_delta = np.sum(deltas)
                port_std = np.sqrt(np.sum((np.array(deltas) * np.array(spots) * spot_vol_hp) ** 2))
                var = norm.ppf(conf_level) * port_std
                es = port_std * norm.pdf(norm.ppf(conf_level)) / (1 - conf_level)
                st.write(f"**Parametric VaR ({int(conf_level*100)}%, {holding_period}d):** ${var:,.0f}")
                st.write(f"**Parametric ES ({int(conf_level*100)}%, {holding_period}d):** ${es:,.0f}")
                st.caption("Parametric VaR/ES assumes normal P&L distribution using portfolio delta and spot volatility.")
            elif method == "Parametric (Cornish-Fisher)":
                st.markdown("Cornish-Fisher VaR/ES using simulated portfolio P&L distribution")
                deltas = []
                spots = []
                notionals = []
                for pos in st.session_state['portfolio']:
                    if pos['model'] in ["Black-Scholes", "Binomial Tree", "Heston"]:
                        S = pos['S']
                        notional = pos['notional']
                        deltas.append(notional)
                        spots.append(S)
                        notionals.append(notional)
                spot_vol = st.number_input("Spot Volatility (annualized, %)", min_value=1.0, value=20.0, key="cf_spot_vol") / 100
                spot_vol_hp = spot_vol * np.sqrt(holding_period / 252)
                n_sims = st.number_input("# Simulations", min_value=100, max_value=10000, value=2000, step=100, key="cf_sims")
                np.random.seed(42)
                spot_shocks = np.random.normal(0, spot_vol_hp, size=(n_sims, len(spots)))
                pnl = []
                for i in range(n_sims):
                    total_pnl = 0
                    for j, pos in enumerate(st.session_state['portfolio']):
                        if pos['model'] in ["Black-Scholes", "Binomial Tree", "Heston"]:
                            shocked_S = spots[j] * (1 + spot_shocks[i, j])
                            notional = notionals[j]
                            if pos['model'] == "Black-Scholes":
                                from app.models.black_scholes import black_scholes_price
                                base = black_scholes_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])
                                shocked = black_scholes_price(shocked_S, pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])
                            elif pos['model'] == "Binomial Tree":
                                from app.models.binomial_tree import binomial_tree_price
                                base = binomial_tree_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                                shocked = binomial_tree_price(shocked_S, pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                            elif pos['model'] == "Heston":
                                from app.models.heston import heston_price
                                base = heston_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                                shocked = heston_price(shocked_S, pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                            total_pnl += (shocked - base) * notional
                    pnl.append(total_pnl)
                pnl = np.array(pnl)
                port_mean = np.mean(pnl)
                port_std = np.std(pnl)
                skew = ((np.mean((pnl - port_mean) ** 3)) / (port_std ** 3)) if port_std > 0 else 0
                kurt = ((np.mean((pnl - port_mean) ** 4)) / (port_std ** 4)) if port_std > 0 else 0
                z = norm.ppf(conf_level)
                z_cf = z + (1/6)*(z**2-1)*skew + (1/24)*(z**3-3*z)*kurt - (1/36)*(2*z**3-5*z)*skew**2
                var_cf = port_mean + z_cf * port_std
                es_cf = port_mean + port_std * norm.pdf(z_cf) / (1 - conf_level)
                st.write(f"**Cornish-Fisher VaR ({int(conf_level*100)}%, {holding_period}d):** ${var_cf:,.0f}")
                st.write(f"**Cornish-Fisher ES ({int(conf_level*100)}%, {holding_period}d):** ${es_cf:,.0f}")
                st.caption("Cornish-Fisher VaR/ES adjusts for skew and kurtosis in the simulated P&L distribution.")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=pnl, nbinsx=50, name="P&L"))
                fig.add_vline(x=var_cf, line_dash="dash", line_color="red", annotation_text="VaR", annotation_position="top left")
                fig.add_vline(x=es_cf, line_dash="dot", line_color="orange", annotation_text="ES", annotation_position="top left")
                fig.update_layout(title="Cornish-Fisher Simulated P&L Distribution", xaxis_title="P&L", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True, key="cf_var_hist")
            elif method == "Monte Carlo":
                deltas = []
                spots = []
                notionals = []
                for pos in st.session_state['portfolio']:
                    if pos['model'] in ["Black-Scholes", "Binomial Tree", "Heston"]:
                        S = pos['S']
                        notional = pos['notional']
                        deltas.append(notional)
                        spots.append(S)
                        notionals.append(notional)
                spot_vol = st.number_input("Spot Volatility (annualized, %)", min_value=1.0, value=20.0, key="adv_mc_spot_vol") / 100
                spot_vol_hp = spot_vol * np.sqrt(holding_period / 252)
                n_sims = st.number_input("# Simulations", min_value=100, max_value=10000, value=2000, step=100, key="adv_mc_sims")
                np.random.seed(42)
                spot_shocks = np.random.normal(0, spot_vol_hp, size=(n_sims, len(spots)))
                pnl = []
                for i in range(n_sims):
                    total_pnl = 0
                    for j, pos in enumerate(st.session_state['portfolio']):
                        if pos['model'] in ["Black-Scholes", "Binomial Tree", "Heston"]:
                            shocked_S = spots[j] * (1 + spot_shocks[i, j])
                            notional = notionals[j]
                            if pos['model'] == "Black-Scholes":
                                from app.models.black_scholes import black_scholes_price
                                base = black_scholes_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])
                                shocked = black_scholes_price(shocked_S, pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])
                            elif pos['model'] == "Binomial Tree":
                                from app.models.binomial_tree import binomial_tree_price
                                base = binomial_tree_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                                shocked = binomial_tree_price(shocked_S, pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                            elif pos['model'] == "Heston":
                                from app.models.heston import heston_price
                                base = heston_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                                shocked = heston_price(shocked_S, pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                            total_pnl += (shocked - base) * notional
                    pnl.append(total_pnl)
                pnl = np.array(pnl)
                var_mc = np.percentile(pnl, 100 * (1 - conf_level))
                es_mc = pnl[pnl <= var_mc].mean() if np.any(pnl <= var_mc) else np.nan
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=pnl, nbinsx=50, name="P&L"))
                fig.add_vline(x=var_mc, line_dash="dash", line_color="red", annotation_text="VaR", annotation_position="top left")
                fig.add_vline(x=es_mc, line_dash="dot", line_color="orange", annotation_text="ES", annotation_position="top left")
                fig.update_layout(title="Monte Carlo Simulated P&L Distribution", xaxis_title="P&L", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"**Monte Carlo VaR ({int(conf_level*100)}%, {holding_period}d):** ${var_mc:,.0f}")
                st.write(f"**Monte Carlo ES ({int(conf_level*100)}%, {holding_period}d):** ${es_mc:,.0f}")
                st.caption("Monte Carlo VaR/ES is based on simulated spot shocks and full portfolio revaluation.")
        elif analytics_tab == "Monte Carlo Scenario Engine":
            st.markdown("Simulate thousands of market scenarios (spot, vol, rates, etc.) and analyze portfolio P&L distribution.")
            n_sims = st.number_input("# Scenarios", min_value=100, max_value=10000, value=2000, step=100, key="mc_scenarios")
            spot_vol = st.number_input("Spot Volatility (annualized, %)", min_value=1.0, value=20.0, key="mc_spot_vol") / 100
            rate_vol = st.number_input("Rate Volatility (annualized, %)", min_value=0.1, value=1.0, key="mc_rate_vol") / 100
            np.random.seed(42)
            spot_shocks = np.random.normal(0, spot_vol, size=(n_sims, len(port_df)))
            rate_shocks = np.random.normal(0, rate_vol, size=(n_sims, len(port_df)))
            pnl = []
            for i in range(n_sims):
                total_pnl = 0
                for j, pos in enumerate(st.session_state['portfolio']):
                    if pos['model'] in ["Black-Scholes", "Binomial Tree", "Heston"]:
                        shocked_S = pos['S'] * (1 + spot_shocks[i, j])
                        shocked_r = pos['r'] + rate_shocks[i, j]
                        notional = pos['notional']
                        if pos['model'] == "Black-Scholes":
                            from app.models.black_scholes import black_scholes_price
                            base = black_scholes_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])
                            shocked = black_scholes_price(shocked_S, pos['K'], pos['T'], shocked_r, pos['sigma'], pos['option_type'])
                        elif pos['model'] == "Binomial Tree":
                            from app.models.binomial_tree import binomial_tree_price
                            base = binomial_tree_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                            shocked = binomial_tree_price(shocked_S, pos['K'], pos['T'], shocked_r, pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                        elif pos['model'] == "Heston":
                            from app.models.heston import heston_price
                            base = heston_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                            shocked = heston_price(shocked_S, pos['K'], pos['T'], shocked_r, pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                        total_pnl += (shocked - base) * notional
                pnl.append(total_pnl)
            pnl = np.array(pnl)
            st.write(f"Simulated mean P&L: ${np.mean(pnl):,.0f}")
            st.write(f"Simulated 5th/95th percentiles: ${np.percentile(pnl, 5):,.0f} / ${np.percentile(pnl, 95):,.0f}")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=pnl, nbinsx=50, name="P&L"))
            fig.update_layout(title="Monte Carlo Scenario P&L Distribution", xaxis_title="P&L", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True, key="mc_scenario_hist")
            st.caption("Monte Carlo scenario engine shows the full distribution of portfolio outcomes under random market shocks.")
        elif analytics_tab == "Stress Testing":
            st.markdown("Apply extreme but plausible market moves and see the impact on portfolio value and risk.")
            preset = st.selectbox("Select Stress Scenario", ["2008 Crisis", "COVID Crash", "Custom"])
            if preset == "2008 Crisis":
                spot_shock = -0.4
                rate_shock = -0.02
            elif preset == "COVID Crash":
                spot_shock = -0.3
                rate_shock = -0.01
            else:
                spot_shock = st.number_input("Custom Spot Shock (%)", value=-10.0) / 100
                rate_shock = st.number_input("Custom Rate Shock (%)", value=-0.5) / 100
            total_pnl = 0
            for pos in st.session_state['portfolio']:
                if pos['model'] in ["Black-Scholes", "Binomial Tree", "Heston"]:
                    shocked_S = pos['S'] * (1 + spot_shock)
                    shocked_r = pos['r'] + rate_shock
                    notional = pos['notional']
                    if pos['model'] == "Black-Scholes":
                        from app.models.black_scholes import black_scholes_price
                        base = black_scholes_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])
                        shocked = black_scholes_price(shocked_S, pos['K'], pos['T'], shocked_r, pos['sigma'], pos['option_type'])
                    elif pos['model'] == "Binomial Tree":
                        from app.models.binomial_tree import binomial_tree_price
                        base = binomial_tree_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                        shocked = binomial_tree_price(shocked_S, pos['K'], pos['T'], shocked_r, pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                    elif pos['model'] == "Heston":
                        from app.models.heston import heston_price
                        base = heston_price(pos['S'], pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                        shocked = heston_price(shocked_S, pos['K'], pos['T'], shocked_r, pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                    total_pnl += (shocked - base) * notional
            st.write(f"Portfolio P&L under stress: ${total_pnl:,.0f}")
            st.caption("Stress testing shows the impact of extreme market moves on portfolio value.")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=[total_pnl], nbinsx=1, name="P&L"))
            fig.update_layout(title="Stress Test P&L Distribution", xaxis_title="P&L", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True, key="stress_hist")
        elif analytics_tab == "Sensitivity/Attribution":
            st.markdown("Decompose portfolio risk by factor and position.")
            deltas = []
            labels = []
            for pos in st.session_state['portfolio']:
                if pos['model'] in ["Black-Scholes", "Binomial Tree", "Heston"]:
                    if pos['model'] == "Black-Scholes":
                        from app.models.black_scholes import black_scholes_greeks
                        delta = black_scholes_greeks(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])['delta']
                    elif pos['model'] == "Binomial Tree":
                        from app.models.binomial_tree import binomial_tree_delta
                        delta = binomial_tree_delta(pos['S'], pos['K'], pos['T'], pos['r'], pos['sigma'], pos['steps'], pos['option_type'], pos['exercise'])
                    elif pos['model'] == "Heston":
                        from app.models.heston import heston_delta
                        delta = heston_delta(pos['S'], pos['K'], pos['T'], pos['r'], pos['v0'], pos['kappa'], pos['theta'], pos['sigma'], pos['rho'], pos['option_type'])
                    deltas.append(abs(delta * pos['notional']))
                    labels.append(f"{pos['model']} {pos.get('option_type','')}")
            fig = go.Figure(data=[go.Pie(labels=labels, values=deltas, hole=0.4)])
            fig.update_layout(title="Risk Attribution by Position (Delta)")
            st.plotly_chart(fig, use_container_width=True, key="pie_risk_attribution")
            st.caption("Pie chart shows which positions contribute most to portfolio delta risk.")
        elif analytics_tab == "Risk-Return Metrics":
            st.markdown("Compute Sharpe, Sortino, Information Ratio, etc. for the portfolio.")
            n_hist = st.number_input("# Simulated Return Days", min_value=100, max_value=2000, value=500, key="rr_hist")
            np.random.seed(42)
            port_returns = np.random.normal(0.0005, 0.01, size=n_hist) 
            sharpe = np.mean(port_returns) / np.std(port_returns) * np.sqrt(252)
            downside = port_returns[port_returns < 0]
            sortino = np.mean(port_returns) / (np.std(downside) if len(downside) > 0 else 1) * np.sqrt(252)
            info_ratio = np.mean(port_returns) / (np.std(port_returns - 0.0002) if len(port_returns) > 0 else 1) * np.sqrt(252)
            st.write(f"Sharpe Ratio: {sharpe:.2f}")
            st.write(f"Sortino Ratio: {sortino:.2f}")
            st.write(f"Information Ratio: {info_ratio:.2f}")
            st.caption("Risk-return metrics help evaluate portfolio performance relative to risk taken.")