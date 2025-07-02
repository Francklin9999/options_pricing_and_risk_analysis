import numpy as np
from typing import Literal

OptionType = Literal['call', 'put']
ExerciseType = Literal['european', 'american']

def binomial_tree_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int = 100,
    option_type: OptionType = 'call',
    exercise: ExerciseType = 'european'
) -> float:
    """
    Price an option using the Cox-Ross-Rubinstein binomial tree model.
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        steps: Number of time steps in the tree
        option_type: 'call' or 'put'
        exercise: 'european' or 'american'
    Returns:
        Option price
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    ST = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])
    if option_type == 'call':
        values = np.maximum(ST - K, 0)
    elif option_type == 'put':
        values = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:i+2] + (1 - p) * values[0:i+1])
        if exercise == 'american':
            ST = np.array([S * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
            if option_type == 'call':
                values = np.maximum(values, ST - K)
            else:
                values = np.maximum(values, K - ST)
    return values[0]

def binomial_tree_delta(S, K, T, r, sigma, steps, option_type, exercise, eps=1e-2):
    p_up = binomial_tree_price(S + eps, K, T, r, sigma, steps, option_type, exercise)
    p_down = binomial_tree_price(S - eps, K, T, r, sigma, steps, option_type, exercise)
    return (p_up - p_down) / (2 * eps)

def binomial_tree_gamma(S, K, T, r, sigma, steps, option_type, exercise, eps=1e-2):
    p_up = binomial_tree_price(S + eps, K, T, r, sigma, steps, option_type, exercise)
    p = binomial_tree_price(S, K, T, r, sigma, steps, option_type, exercise)
    p_down = binomial_tree_price(S - eps, K, T, r, sigma, steps, option_type, exercise)
    return (p_up - 2 * p + p_down) / (eps ** 2)

def binomial_tree_vega(S, K, T, r, sigma, steps, option_type, exercise, eps=1e-4):
    return (binomial_tree_price(S, K, T, r, sigma + eps, steps, option_type, exercise) - binomial_tree_price(S, K, T, r, sigma - eps, steps, option_type, exercise)) / (2 * eps)

def binomial_tree_theta(S, K, T, r, sigma, steps, option_type, exercise, eps=1e-5):
    return (binomial_tree_price(S, K, T - eps, r, sigma, steps, option_type, exercise) - binomial_tree_price(S, K, T + eps, r, sigma, steps, option_type, exercise)) / (2 * eps)

def binomial_tree_rho(S, K, T, r, sigma, steps, option_type, exercise, eps=1e-4):
    return (binomial_tree_price(S, K, T, r + eps, sigma, steps, option_type, exercise) - binomial_tree_price(S, K, T, r - eps, sigma, steps, option_type, exercise)) / (2 * eps)

def binomial_tree_greeks_full(S, K, T, r, sigma, steps, option_type, exercise):
    delta = binomial_tree_delta(S, K, T, r, sigma, steps, option_type, exercise)
    gamma = binomial_tree_gamma(S, K, T, r, sigma, steps, option_type, exercise)
    vega = binomial_tree_vega(S, K, T, r, sigma, steps, option_type, exercise)
    theta = binomial_tree_theta(S, K, T, r, sigma, steps, option_type, exercise)
    rho = binomial_tree_rho(S, K, T, r, sigma, steps, option_type, exercise)
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

def binomial_tree_vanna(S, K, T, r, sigma, steps, option_type, exercise, eps=1e-4):
    v_up = binomial_tree_vega(S + eps, K, T, r, sigma, steps, option_type, exercise)
    v_down = binomial_tree_vega(S - eps, K, T, r, sigma, steps, option_type, exercise)
    return (v_up - v_down) / (2 * eps)

def binomial_tree_volga(S, K, T, r, sigma, steps, option_type, exercise, eps=1e-4):
    v_up = binomial_tree_vega(S, K, T, r, sigma + eps, steps, option_type, exercise)
    v_down = binomial_tree_vega(S, K, T, r, sigma - eps, steps, option_type, exercise)
    return (v_up - v_down) / (2 * eps)

def binomial_tree_cross_gamma(S, K, T, r, sigma, steps, option_type, exercise, eps=1e-4):
    dS_up = binomial_tree_delta(S + eps, K, T, r, sigma, steps, option_type, exercise)
    dS_down = binomial_tree_delta(S - eps, K, T, r, sigma, steps, option_type, exercise)
    return (dS_up - dS_down) / (2 * eps)

def binomial_tree_cross_greeks(S, K, T, r, sigma, steps, option_type, exercise):
    vanna = binomial_tree_vanna(S, K, T, r, sigma, steps, option_type, exercise)
    volga = binomial_tree_volga(S, K, T, r, sigma, steps, option_type, exercise)
    cross_gamma = binomial_tree_cross_gamma(S, K, T, r, sigma, steps, option_type, exercise)
    return {'vanna': vanna, 'volga': volga, 'cross_gamma': cross_gamma} 