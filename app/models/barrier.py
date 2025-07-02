import numpy as np
from scipy.stats import norm
from typing import Literal

OptionType = Literal['call', 'put']
BarrierType = Literal['up-and-out', 'down-and-out', 'up-and-in', 'down-and-in']

def barrier_option_price(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = 'call',
    barrier_type: BarrierType = 'up-and-out',
    rebate: float = 0.0
) -> float:
    """
    Price a European barrier option using the Reiner-Rubinstein formula.
    Args:
        S: Spot price
        K: Strike price
        H: Barrier level
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
        barrier_type: 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
        rebate: Rebate paid if option is knocked out (default 0)
    Returns:
        Option price
    """
    if (option_type not in ['call', 'put']) or (barrier_type not in ['up-and-out', 'down-and-out', 'up-and-in', 'down-and-in']):
        raise ValueError("Invalid option_type or barrier_type")
    if (barrier_type.startswith('up') and H <= S) or (barrier_type.startswith('down') and H >= S):
        return rebate * np.exp(-r * T)  # Already knocked out
    mu = (r - 0.5 * sigma ** 2) / sigma ** 2
    lambd = np.sqrt(mu ** 2 + 2 * r / sigma ** 2)
    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y1 = np.log(H ** 2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    z = np.log(H / S) / (sigma * np.sqrt(T)) + lambd * sigma * np.sqrt(T)
    
    def phi(d):
        return norm.cdf(d)
    def psi(d):
        return norm.pdf(d)
    
    def vanilla_price(S_, K_, T_, r_, sigma_, option_type_):
        d1 = (np.log(S_ / K_) + (r_ + 0.5 * sigma_ ** 2) * T_) / (sigma_ * np.sqrt(T_))
        d2 = d1 - sigma_ * np.sqrt(T_)
        if option_type_ == 'call':
            return S_ * phi(d1) - K_ * np.exp(-r_ * T_) * phi(d2)
        else:
            return K_ * np.exp(-r_ * T_) * phi(-d2) - S_ * phi(-d1)
    
    if option_type == 'call' and barrier_type == 'up-and-out':
        if K >= H or S >= H:
            return max(rebate * np.exp(-r * T), 0)
        A = vanilla_price(S, K, T, r, sigma, 'call')
        B = (H / S) ** (2 * (mu + 1)) * vanilla_price(H ** 2 / S, K, T, r, sigma, 'call')
        C = (H / S) ** (2 * mu) * vanilla_price(H ** 2 / S, K, T, r, sigma, 'put')
        D = vanilla_price(S, K, T, r, sigma, 'put')
        price = A - B + C - D + rebate * np.exp(-r * T)
        return max(price, 0)
    elif option_type == 'call' and barrier_type == 'down-and-out':
        if S <= H or K <= H:
            return max(rebate * np.exp(-r * T), 0)
        A = vanilla_price(S, K, T, r, sigma, 'call')
        B = (H / S) ** (2 * (mu + 1)) * vanilla_price(H ** 2 / S, K, T, r, sigma, 'call')
        price = A - B + rebate * np.exp(-r * T)
        return max(price, 0)
    elif option_type == 'put' and barrier_type == 'up-and-out':
        if S >= H or K >= H:
            return max(rebate * np.exp(-r * T), 0)
        A = vanilla_price(S, K, T, r, sigma, 'put')
        B = (H / S) ** (2 * mu) * vanilla_price(H ** 2 / S, K, T, r, sigma, 'put')
        price = A - B + rebate * np.exp(-r * T)
        return max(price, 0)
    elif option_type == 'put' and barrier_type == 'down-and-out':
        if S <= H or K <= H:
            return max(rebate * np.exp(-r * T), 0)
        A = vanilla_price(S, K, T, r, sigma, 'put')
        B = (H / S) ** (2 * mu) * vanilla_price(H ** 2 / S, K, T, r, sigma, 'put')
        C = (H / S) ** (2 * (mu + 1)) * vanilla_price(H ** 2 / S, K, T, r, sigma, 'call')
        D = vanilla_price(S, K, T, r, sigma, 'call')
        price = A - B + C - D + rebate * np.exp(-r * T)
        return max(price, 0)
    elif barrier_type.endswith('in'):
        out_type = barrier_type.replace('in', 'out')
        out_price = barrier_option_price(S, K, H, T, r, sigma, option_type, out_type, rebate)
        vanilla = vanilla_price(S, K, T, r, sigma, option_type)
        price = vanilla - out_price
        return max(price, 0)
    else:
        raise NotImplementedError("Barrier type not implemented")

def barrier_option_delta(S, K, H, T, r, sigma, option_type, barrier_type, rebate=0.0, eps=1e-4):
    p_up = barrier_option_price(S + eps, K, H, T, r, sigma, option_type, barrier_type, rebate)
    p_down = barrier_option_price(S - eps, K, H, T, r, sigma, option_type, barrier_type, rebate)
    return (p_up - p_down) / (2 * eps)

def barrier_option_gamma(S, K, H, T, r, sigma, option_type, barrier_type, rebate=0.0, eps=1e-4):
    p_up = barrier_option_price(S + eps, K, H, T, r, sigma, option_type, barrier_type, rebate)
    p = barrier_option_price(S, K, H, T, r, sigma, option_type, barrier_type, rebate)
    p_down = barrier_option_price(S - eps, K, H, T, r, sigma, option_type, barrier_type, rebate)
    return (p_up - 2 * p + p_down) / (eps ** 2)

def barrier_option_vega(S, K, H, T, r, sigma, option_type, barrier_type, rebate=0.0, eps=1e-4):
    return (barrier_option_price(S, K, H, T, r, sigma + eps, option_type, barrier_type, rebate) - barrier_option_price(S, K, H, T, r, sigma - eps, option_type, barrier_type, rebate)) / (2 * eps)

def barrier_option_theta(S, K, H, T, r, sigma, option_type, barrier_type, rebate=0.0, eps=1e-5):
    return (barrier_option_price(S, K, H, T - eps, r, sigma, option_type, barrier_type, rebate) - barrier_option_price(S, K, H, T + eps, r, sigma, option_type, barrier_type, rebate)) / (2 * eps)

def barrier_option_rho(S, K, H, T, r, sigma, option_type, barrier_type, rebate=0.0, eps=1e-4):
    return (barrier_option_price(S, K, H, T, r + eps, sigma, option_type, barrier_type, rebate) - barrier_option_price(S, K, H, T, r - eps, sigma, option_type, barrier_type, rebate)) / (2 * eps)

def barrier_greeks_full(S, K, H, T, r, sigma, option_type, barrier_type, rebate=0.0):
    delta = barrier_option_delta(S, K, H, T, r, sigma, option_type, barrier_type, rebate)
    gamma = barrier_option_gamma(S, K, H, T, r, sigma, option_type, barrier_type, rebate)
    vega = barrier_option_vega(S, K, H, T, r, sigma, option_type, barrier_type, rebate)
    theta = barrier_option_theta(S, K, H, T, r, sigma, option_type, barrier_type, rebate)
    rho = barrier_option_rho(S, K, H, T, r, sigma, option_type, barrier_type, rebate)
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

def barrier_vanna(S, K, H, T, r, sigma, option_type, barrier_type, rebate=0.0, eps=1e-4):
    v_up = barrier_option_vega(S + eps, K, H, T, r, sigma, option_type, barrier_type, rebate)
    v_down = barrier_option_vega(S - eps, K, H, T, r, sigma, option_type, barrier_type, rebate)
    return (v_up - v_down) / (2 * eps)

def barrier_volga(S, K, H, T, r, sigma, option_type, barrier_type, rebate=0.0, eps=1e-4):
    v_up = barrier_option_vega(S, K, H, T, r, sigma + eps, option_type, barrier_type, rebate)
    v_down = barrier_option_vega(S, K, H, T, r, sigma - eps, option_type, barrier_type, rebate)
    return (v_up - v_down) / (2 * eps)

def barrier_cross_gamma(S, K, H, T, r, sigma, option_type, barrier_type, rebate=0.0, eps=1e-4):
    dS_up = barrier_option_delta(S + eps, K, H, T, r, sigma, option_type, barrier_type, rebate)
    dS_down = barrier_option_delta(S - eps, K, H, T, r, sigma, option_type, barrier_type, rebate)
    return (dS_up - dS_down) / (2 * eps)

def barrier_cross_greeks(S, K, H, T, r, sigma, option_type, barrier_type, rebate=0.0):
    vanna = barrier_vanna(S, K, H, T, r, sigma, option_type, barrier_type, rebate)
    volga = barrier_volga(S, K, H, T, r, sigma, option_type, barrier_type, rebate)
    cross_gamma = barrier_cross_gamma(S, K, H, T, r, sigma, option_type, barrier_type, rebate)
    return {'vanna': vanna, 'volga': volga, 'cross_gamma': cross_gamma} 