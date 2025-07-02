import numpy as np
from scipy.stats import norm
from typing import Literal

OptionType = Literal['call', 'put']

def hull_white_bond_price(
    r0: float,
    t: float,
    T: float,
    a: float,
    sigma: float,
    P0T: callable
) -> float:
    """
    Price a zero-coupon bond under the Hull-White model.
    Args:
        r0: Initial short rate
        t: Current time
        T: Maturity time
        a: Mean reversion speed
        sigma: Volatility
        P0T: Function returning discount factor P(0, T)
    Returns:
        Bond price at time t
    """
    B = (1 - np.exp(-a * (T - t))) / a
    f0t = r0
    A = (P0T(T) / P0T(t)) * np.exp(
        B * f0t - (sigma ** 2) / (4 * a) * (1 - np.exp(-a * (T - t))) ** 2 * (1 - np.exp(-2 * a * t)) / a
    )
    return A * np.exp(-B * r0)

def hull_white_bond_option(
    r0: float,
    t: float,
    T: float,
    S: float,
    K: float,
    a: float,
    sigma: float,
    P0T: callable,
    option_type: OptionType = 'call'
) -> float:
    """
    Price a European option on a zero-coupon bond under the Hull-White model (Jamshidian's formula).
    Args:
        r0: Initial short rate
        t: Current time
        T: Option expiry
        S: Bond maturity
        K: Strike price
        a: Mean reversion speed
        sigma: Volatility
        P0T: Function returning discount factor P(0, T)
        option_type: 'call' or 'put'
    Returns:
        Option price
    """
    B = (1 - np.exp(-a * (S - T))) / a
    P_tT = hull_white_bond_price(r0, t, T, a, sigma, P0T)
    P_tS = hull_white_bond_price(r0, t, S, a, sigma, P0T)
    sigma_p = sigma * np.sqrt((1 - np.exp(-2 * a * (T - t))) / (2 * a)) * B
    h = (np.log(P_tS / (P_tT * K)) / sigma_p) + 0.5 * sigma_p
    if option_type == 'call':
        price = P_tS * norm.cdf(h) - K * P_tT * norm.cdf(h - sigma_p)
    elif option_type == 'put':
        price = K * P_tT * norm.cdf(-h + sigma_p) - P_tS * norm.cdf(-h)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price 

def hull_white_dv01(r0, t, T, S, K, a, sigma, P0T, option_type, eps=1e-4):
    price = hull_white_bond_option(r0, t, T, S, K, a, sigma, P0T, option_type)
    price_up = hull_white_bond_option(r0 + eps, t, T, S, K, a, sigma, P0T, option_type)
    return (price_up - price) / eps 