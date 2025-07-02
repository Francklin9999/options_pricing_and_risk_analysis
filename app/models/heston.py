import numpy as np
from scipy.integrate import quad
from typing import Literal
from app.models.black_scholes import black_scholes_price

OptionType = Literal['call', 'put']

def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    option_type: OptionType = 'call'
) -> float:
    """
    Price a European option using the Heston stochastic volatility model.
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma: Volatility of variance (vol of vol)
        rho: Correlation between asset and variance
        option_type: 'call' or 'put'
    Returns:
        Option price
    """
    if sigma == 0:
        return black_scholes_price(S, K, T, r, np.sqrt(v0), option_type)

    def integrand(phi, Pnum):
        return np.real(
            np.exp(-1j * phi * np.log(K)) * heston_cf(phi, Pnum)
            / (1j * phi)
        )

    def heston_cf(phi, Pnum):
        if Pnum == 1:
            u = 0.5
            b = kappa - rho * sigma
        else:
            u = -0.5
            b = kappa
        a = kappa * theta
        x = np.log(S)
        d = np.sqrt((rho * sigma * 1j * phi - b) ** 2 - sigma ** 2 * (2 * u * 1j * phi - phi ** 2))
        g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)
        C = r * 1j * phi * T + a / sigma ** 2 * (
            (b - rho * sigma * 1j * phi + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
        )
        D = (b - rho * sigma * 1j * phi + d) / sigma ** 2 * (
            (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
        )
        return np.exp(C + D * v0 + 1j * phi * x)

    P1 = 0.5 + 1 / np.pi * quad(lambda phi: integrand(phi, 1), 1e-8, 100, limit=100)[0]
    P2 = 0.5 + 1 / np.pi * quad(lambda phi: integrand(phi, 2), 1e-8, 100, limit=100)[0]
    call = S * P1 - K * np.exp(-r * T) * P2
    if option_type == 'call':
        return call
    elif option_type == 'put':
        return call - S + K * np.exp(-r * T)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def heston_delta(S, K, T, r, v0, kappa, theta, sigma, rho, option_type, eps=1e-2):
    p_up = heston_price(S + eps, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    p_down = heston_price(S - eps, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    return (p_up - p_down) / (2 * eps)

def heston_vega(S, K, T, r, v0, kappa, theta, sigma, rho, option_type, eps=1e-4):
    return (heston_price(S, K, T, r, v0, kappa, theta, sigma + eps, rho, option_type) - heston_price(S, K, T, r, v0, kappa, theta, sigma - eps, rho, option_type)) / (2 * eps)

def heston_theta(S, K, T, r, v0, kappa, theta_, sigma, rho, option_type, eps=1e-5):
    return (heston_price(S, K, T - eps, r, v0, kappa, theta_, sigma, rho, option_type) - heston_price(S, K, T + eps, r, v0, kappa, theta_, sigma, rho, option_type)) / (2 * eps)

def heston_rho(S, K, T, r, v0, kappa, theta, sigma, rho, option_type, eps=1e-4):
    return (heston_price(S, K, T, r + eps, v0, kappa, theta, sigma, rho, option_type) - heston_price(S, K, T, r - eps, v0, kappa, theta, sigma, rho, option_type)) / (2 * eps)

def heston_greeks_full(S, K, T, r, v0, kappa, theta, sigma, rho, option_type):
    delta = heston_delta(S, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    vega = heston_vega(S, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    theta_ = heston_theta(S, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    rho_ = heston_rho(S, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    return {'delta': delta, 'vega': vega, 'theta': theta_, 'rho': rho_}

def heston_vanna(S, K, T, r, v0, kappa, theta, sigma, rho, option_type, eps=1e-4):
    v_up = heston_vega(S + eps, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    v_down = heston_vega(S - eps, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    return (v_up - v_down) / (2 * eps)

def heston_volga(S, K, T, r, v0, kappa, theta, sigma, rho, option_type, eps=1e-4):
    v_up = heston_vega(S, K, T, r, v0, kappa, theta, sigma + eps, rho, option_type)
    v_down = heston_vega(S, K, T, r, v0, kappa, theta, sigma - eps, rho, option_type)
    return (v_up - v_down) / (2 * eps)

def heston_cross_gamma(S, K, T, r, v0, kappa, theta, sigma, rho, option_type, eps=1e-4):
    dS_up = heston_delta(S + eps, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    dS_down = heston_delta(S - eps, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    return (dS_up - dS_down) / (2 * eps)

def heston_cross_greeks(S, K, T, r, v0, kappa, theta, sigma, rho, option_type):
    vanna = heston_vanna(S, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    volga = heston_volga(S, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    cross_gamma = heston_cross_gamma(S, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    return {'vanna': vanna, 'volga': volga, 'cross_gamma': cross_gamma} 