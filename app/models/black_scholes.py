import numpy as np
from scipy.stats import norm
from typing import Literal, Dict

OptionType = Literal['call', 'put']

def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 for Black-Scholes formula."""
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 for Black-Scholes formula."""
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = 'call'
) -> float:
    """
    Price a European option using the Black-Scholes formula.
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
    Returns:
        Option price
    """
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    if option_type == 'call':
        price = S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = 'call'
) -> Dict[str, float]:
    """
    Calculate Black-Scholes Greeks for a European option.
    Returns a dictionary with Delta, Gamma, Vega, Theta, Rho.
    All values are annualized (per year), except Delta and Gamma (unitless).
    """
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d_1)
    cdf_d1 = norm.cdf(d_1)
    cdf_d2 = norm.cdf(d_2)
    if option_type == 'call':
        delta = cdf_d1
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * cdf_d2)
        rho = K * T * np.exp(-r * T) * cdf_d2
    elif option_type == 'put':
        delta = cdf_d1 - 1
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d_2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d_2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T)
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }

def black_scholes_vega(S, K, T, r, sigma, option_type, eps=1e-4):
    return (black_scholes_price(S, K, T, r, sigma + eps, option_type) - black_scholes_price(S, K, T, r, sigma - eps, option_type)) / (2 * eps)

def black_scholes_theta(S, K, T, r, sigma, option_type, eps=1e-5):
    return (black_scholes_price(S, K, T - eps, r, sigma, option_type) - black_scholes_price(S, K, T + eps, r, sigma, option_type)) / (2 * eps)

def black_scholes_rho(S, K, T, r, sigma, option_type, eps=1e-4):
    return (black_scholes_price(S, K, T, r + eps, sigma, option_type) - black_scholes_price(S, K, T, r - eps, sigma, option_type)) / (2 * eps)

def black_scholes_greeks_full(S, K, T, r, sigma, option_type):
    greeks = black_scholes_greeks(S, K, T, r, sigma, option_type)
    vega = black_scholes_vega(S, K, T, r, sigma, option_type)
    theta = black_scholes_theta(S, K, T, r, sigma, option_type)
    rho = black_scholes_rho(S, K, T, r, sigma, option_type)
    greeks['vega'] = vega
    greeks['theta'] = theta
    greeks['rho'] = rho
    return greeks

def black_scholes_vanna(S, K, T, r, sigma, option_type, eps=1e-4):
    v_up = black_scholes_vega(S + eps, K, T, r, sigma, option_type)
    v_down = black_scholes_vega(S - eps, K, T, r, sigma, option_type)
    return (v_up - v_down) / (2 * eps)

def black_scholes_volga(S, K, T, r, sigma, option_type, eps=1e-4):
    v_up = black_scholes_vega(S, K, T, r, sigma + eps, option_type)
    v_down = black_scholes_vega(S, K, T, r, sigma - eps, option_type)
    return (v_up - v_down) / (2 * eps)

def black_scholes_cross_gamma(S, K, T, r, sigma, option_type, eps=1e-4):
    dS_up = black_scholes_delta(S + eps, K, T, r, sigma, option_type)
    dS_down = black_scholes_delta(S - eps, K, T, r, sigma, option_type)
    return (dS_up - dS_down) / (2 * eps)

def black_scholes_cross_greeks(S, K, T, r, sigma, option_type):
    vanna = black_scholes_vanna(S, K, T, r, sigma, option_type)
    volga = black_scholes_volga(S, K, T, r, sigma, option_type)
    cross_gamma = black_scholes_cross_gamma(S, K, T, r, sigma, option_type)
    return {'vanna': vanna, 'volga': volga, 'cross_gamma': cross_gamma}

def black_scholes_delta(S, K, T, r, sigma, option_type):
    d_1 = d1(S, K, T, r, sigma)
    if option_type == 'call':
        return norm.cdf(d_1)
    elif option_type == 'put':
        return norm.cdf(d_1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'") 