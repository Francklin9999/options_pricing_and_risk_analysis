import pytest
import numpy as np
from app.models.heston import heston_price
from app.models.black_scholes import black_scholes_price

def test_heston_matches_black_scholes_when_sigma_zero():
    S, K, T, r = 100, 100, 1, 0.05
    v0 = 0.04
    kappa, theta, sigma, rho = 1.0, 0.04, 0.0, 0.0
    price_heston = heston_price(S, K, T, r, v0, kappa, theta, sigma, rho, 'call')
    price_bs = black_scholes_price(S, K, T, r, np.sqrt(v0), 'call')
    assert np.isclose(price_heston, price_bs, atol=1e-2)

def test_heston_typical_parameters():
    S, K, T, r = 100, 100, 1, 0.05
    v0 = 0.04
    kappa, theta, sigma, rho = 2.0, 0.04, 0.5, -0.7
    price = heston_price(S, K, T, r, v0, kappa, theta, sigma, rho, 'call')
    assert price > 0

def test_heston_put_call_parity():
    S, K, T, r = 100, 100, 1, 0.05
    v0 = 0.04
    kappa, theta, sigma, rho = 2.0, 0.04, 0.5, -0.7
    call = heston_price(S, K, T, r, v0, kappa, theta, sigma, rho, 'call')
    put = heston_price(S, K, T, r, v0, kappa, theta, sigma, rho, 'put')
    assert np.isclose(call - put, S - K * np.exp(-r * T), atol=1e-2)

def test_heston_invalid_option_type():
    with pytest.raises(ValueError):
        heston_price(100, 100, 1, 0.05, 0.04, 2.0, 0.04, 0.5, -0.7, 'invalid') 