import pytest
import numpy as np
from app.models.lookback import (
    lookback_option_price, lookback_option_delta, lookback_option_gamma, lookback_option_vega, lookback_option_theta, lookback_option_rho,
    lookback_greeks_full, lookback_vanna, lookback_volga, lookback_cross_gamma, lookback_cross_greeks
)

def test_lookback_option_price_fixed_call():
    price = lookback_option_price(100, 100, 1, 0.05, 0.2, 'call', 'fixed')
    assert price >= 0

def test_lookback_option_price_floating_put():
    price = lookback_option_price(100, 100, 1, 0.05, 0.2, 'put', 'floating', S_max=120)
    assert price >= 0

def test_lookback_option_invalid_type():
    with pytest.raises(ValueError):
        lookback_option_price(100, 100, 1, 0.05, 0.2, 'call', 'invalid')

def test_lookback_greeks_full():
    greeks = lookback_greeks_full(100, 100, 1, 0.05, 0.2, 'call', 'fixed')
    for g in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        assert g in greeks

def test_lookback_cross_greeks():
    cross = lookback_cross_greeks(100, 100, 1, 0.05, 0.2, 'call', 'fixed')
    for g in ['vanna', 'volga', 'cross_gamma']:
        assert g in cross 