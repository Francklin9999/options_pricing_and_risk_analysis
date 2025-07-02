import pytest
import numpy as np
from app.models.american import (
    american_option_price, american_option_delta, american_option_gamma, american_option_vega, american_option_theta, american_option_rho,
    american_greeks_full, american_vanna, american_volga, american_cross_gamma, american_cross_greeks
)

def test_american_option_price_call():
    price = american_option_price(100, 100, 1, 0.05, 0.2, 100, 'call')
    assert price >= 0

def test_american_option_price_put():
    price = american_option_price(100, 100, 1, 0.05, 0.2, 100, 'put')
    assert price >= 0

def test_american_option_invalid_type():
    with pytest.raises(ValueError):
        american_option_price(100, 100, 1, 0.05, 0.2, 100, 'invalid')

def test_american_greeks_full():
    greeks = american_greeks_full(100, 100, 1, 0.05, 0.2, 100, 'call')
    for g in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        assert g in greeks

def test_american_cross_greeks():
    cross = american_cross_greeks(100, 100, 1, 0.05, 0.2, 100, 'call')
    for g in ['vanna', 'volga', 'cross_gamma']:
        assert g in cross 