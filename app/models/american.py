import numpy as np
from typing import Literal

OptionType = Literal['call', 'put']

def american_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int = 100,
    option_type: OptionType = 'call'
) -> float:
    """
    Price an American option using the Binomial Tree method.
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        steps: Number of time steps
        option_type: 'call' or 'put'
    Returns:
        Option price
    """
    if option_type not in ['call', 'put']:
        raise ValueError("option_type must be 'call' or 'put'")
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    ST = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])
    if option_type == 'call':
        values = np.maximum(ST - K, 0)
    else:
        values = np.maximum(K - ST, 0)
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:i+2] + (1 - p) * values[0:i+1])
        ST = np.array([S * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
        if option_type == 'call':
            values = np.maximum(values, ST - K)
        else:
            values = np.maximum(values, K - ST)
    return values[0]

def american_option_delta(S, K, T, r, sigma, steps, option_type, eps=1e-4):
    p_up = american_option_price(S + eps, K, T, r, sigma, steps, option_type)
    p_down = american_option_price(S - eps, K, T, r, sigma, steps, option_type)
    return (p_up - p_down) / (2 * eps)

def american_option_gamma(S, K, T, r, sigma, steps, option_type, eps=1e-4):
    p_up = american_option_price(S + eps, K, T, r, sigma, steps, option_type)
    p = american_option_price(S, K, T, r, sigma, steps, option_type)
    p_down = american_option_price(S - eps, K, T, r, sigma, steps, option_type)
    return (p_up - 2 * p + p_down) / (eps ** 2)

def american_option_vega(S, K, T, r, sigma, steps, option_type, eps=1e-4):
    return (american_option_price(S, K, T, r, sigma + eps, steps, option_type) - american_option_price(S, K, T, r, sigma - eps, steps, option_type)) / (2 * eps)

def american_option_theta(S, K, T, r, sigma, steps, option_type, eps=1e-5):
    return (american_option_price(S, K, T - eps, r, sigma, steps, option_type) - american_option_price(S, K, T + eps, r, sigma, steps, option_type)) / (2 * eps)

def american_option_rho(S, K, T, r, sigma, steps, option_type, eps=1e-4):
    return (american_option_price(S, K, T, r + eps, sigma, steps, option_type) - american_option_price(S, K, T, r - eps, sigma, steps, option_type)) / (2 * eps)

def american_greeks_full(S, K, T, r, sigma, steps, option_type):
    delta = american_option_delta(S, K, T, r, sigma, steps, option_type)
    gamma = american_option_gamma(S, K, T, r, sigma, steps, option_type)
    vega = american_option_vega(S, K, T, r, sigma, steps, option_type)
    theta = american_option_theta(S, K, T, r, sigma, steps, option_type)
    rho = american_option_rho(S, K, T, r, sigma, steps, option_type)
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

def american_vanna(S, K, T, r, sigma, steps, option_type, eps=1e-4):
    v_up = american_option_vega(S + eps, K, T, r, sigma, steps, option_type)
    v_down = american_option_vega(S - eps, K, T, r, sigma, steps, option_type)
    return (v_up - v_down) / (2 * eps)

def american_volga(S, K, T, r, sigma, steps, option_type, eps=1e-4):
    v_up = american_option_vega(S, K, T, r, sigma + eps, steps, option_type)
    v_down = american_option_vega(S, K, T, r, sigma - eps, steps, option_type)
    return (v_up - v_down) / (2 * eps)

def american_cross_gamma(S, K, T, r, sigma, steps, option_type, eps=1e-4):
    dS_up = american_option_delta(S + eps, K, T, r, sigma, steps, option_type)
    dS_down = american_option_delta(S - eps, K, T, r, sigma, steps, option_type)
    return (dS_up - dS_down) / (2 * eps)

def american_cross_greeks(S, K, T, r, sigma, steps, option_type):
    vanna = american_vanna(S, K, T, r, sigma, steps, option_type)
    volga = american_volga(S, K, T, r, sigma, steps, option_type)
    cross_gamma = american_cross_gamma(S, K, T, r, sigma, steps, option_type)
    return {'vanna': vanna, 'volga': volga, 'cross_gamma': cross_gamma} 