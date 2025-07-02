import numpy as np
from scipy.stats import norm
from typing import Literal

OptionType = Literal['call', 'put']
LookbackType = Literal['fixed', 'floating']

def lookback_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = 'call',
    lookback_type: LookbackType = 'fixed',
    S_min: float = None,
    S_max: float = None
) -> float:
    """
    Price a European lookback option (fixed or floating strike) using analytical formulas.
    Args:
        S: Spot price
        K: Strike price (for fixed strike)
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
        lookback_type: 'fixed' or 'floating'
        S_min: Minimum spot over life (for floating strike put)
        S_max: Maximum spot over life (for floating strike call)
    Returns:
        Option price
    """
    if lookback_type == 'fixed':
        if option_type == 'call':
            a1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            a2 = a1 - sigma * np.sqrt(T)
            price = S * norm.cdf(a1) - K * np.exp(-r * T) * norm.cdf(a2)
            price += S * (sigma ** 2) / (2 * r) * (norm.cdf(-a1 + sigma * np.sqrt(T)) - np.exp(-r * T) * norm.cdf(-a2 + sigma * np.sqrt(T)))
            return price
        else:
            a1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            a2 = a1 - sigma * np.sqrt(T)
            price = K * np.exp(-r * T) * norm.cdf(-a2) - S * norm.cdf(-a1)
            price += S * (sigma ** 2) / (2 * r) * (np.exp(-r * T) * norm.cdf(a2 - sigma * np.sqrt(T)) - norm.cdf(a1 - sigma * np.sqrt(T)))
            return price
    elif lookback_type == 'floating':
        if option_type == 'call':
            if S_min is None:
                S_min = S
            price = S - S_min * np.exp(-r * T) + lookback_option_price(S, S_min, T, r, sigma, 'put', 'fixed')
            return price
        else:
            if S_max is None:
                S_max = S
            price = S_max * np.exp(-r * T) - S + lookback_option_price(S, S_max, T, r, sigma, 'call', 'fixed')
            return price
    else:
        raise ValueError("Invalid lookback_type")

def lookback_option_delta(S, K, T, r, sigma, option_type, lookback_type, S_min=None, S_max=None, eps=1e-4):
    p_up = lookback_option_price(S + eps, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    p_down = lookback_option_price(S - eps, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    return (p_up - p_down) / (2 * eps)

def lookback_option_gamma(S, K, T, r, sigma, option_type, lookback_type, S_min=None, S_max=None, eps=1e-4):
    p_up = lookback_option_price(S + eps, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    p = lookback_option_price(S, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    p_down = lookback_option_price(S - eps, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    return (p_up - 2 * p + p_down) / (eps ** 2)

def lookback_option_vega(S, K, T, r, sigma, option_type, lookback_type, S_min=None, S_max=None, eps=1e-4):
    return (lookback_option_price(S, K, T, r, sigma + eps, option_type, lookback_type, S_min, S_max) - lookback_option_price(S, K, T, r, sigma - eps, option_type, lookback_type, S_min, S_max)) / (2 * eps)

def lookback_option_theta(S, K, T, r, sigma, option_type, lookback_type, S_min=None, S_max=None, eps=1e-5):
    return (lookback_option_price(S, K, T - eps, r, sigma, option_type, lookback_type, S_min, S_max) - lookback_option_price(S, K, T + eps, r, sigma, option_type, lookback_type, S_min, S_max)) / (2 * eps)

def lookback_option_rho(S, K, T, r, sigma, option_type, lookback_type, S_min=None, S_max=None, eps=1e-4):
    return (lookback_option_price(S, K, T, r + eps, sigma, option_type, lookback_type, S_min, S_max) - lookback_option_price(S, K, T, r - eps, sigma, option_type, lookback_type, S_min, S_max)) / (2 * eps)

def lookback_greeks_full(S, K, T, r, sigma, option_type, lookback_type, S_min=None, S_max=None):
    delta = lookback_option_delta(S, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    gamma = lookback_option_gamma(S, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    vega = lookback_option_vega(S, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    theta = lookback_option_theta(S, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    rho = lookback_option_rho(S, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

def lookback_vanna(S, K, T, r, sigma, option_type, lookback_type, S_min=None, S_max=None, eps=1e-4):
    v_up = lookback_option_vega(S + eps, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    v_down = lookback_option_vega(S - eps, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    return (v_up - v_down) / (2 * eps)

def lookback_volga(S, K, T, r, sigma, option_type, lookback_type, S_min=None, S_max=None, eps=1e-4):
    v_up = lookback_option_vega(S, K, T, r, sigma + eps, option_type, lookback_type, S_min, S_max)
    v_down = lookback_option_vega(S, K, T, r, sigma - eps, option_type, lookback_type, S_min, S_max)
    return (v_up - v_down) / (2 * eps)

def lookback_cross_gamma(S, K, T, r, sigma, option_type, lookback_type, S_min=None, S_max=None, eps=1e-4):
    dS_up = lookback_option_delta(S + eps, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    dS_down = lookback_option_delta(S - eps, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    return (dS_up - dS_down) / (2 * eps)

def lookback_cross_greeks(S, K, T, r, sigma, option_type, lookback_type, S_min=None, S_max=None):
    vanna = lookback_vanna(S, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    volga = lookback_volga(S, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    cross_gamma = lookback_cross_gamma(S, K, T, r, sigma, option_type, lookback_type, S_min, S_max)
    return {'vanna': vanna, 'volga': volga, 'cross_gamma': cross_gamma} 