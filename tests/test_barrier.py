import pytest
import numpy as np
from app.models.barrier import (
    barrier_option_price, barrier_option_delta, barrier_option_gamma, barrier_option_vega, barrier_option_theta, barrier_option_rho,
    barrier_greeks_full, barrier_vanna, barrier_volga, barrier_cross_gamma, barrier_cross_greeks
)

def test_barrier_option_price_up_and_out_call():
    price = barrier_option_price(100, 100, 120, 1, 0.05, 0.2, 'call', 'up-and-out')
    assert price >= 0

def test_barrier_option_price_down_and_out_put():
    price = barrier_option_price(100, 100, 80, 1, 0.05, 0.2, 'put', 'down-and-out')
    assert price >= 0

def test_barrier_option_price_knocked_out():
    # Already knocked out: S <= H
    price = barrier_option_price(80, 100, 80, 1, 0.05, 0.2, 'call', 'down-and-out')
    assert np.isclose(price, 0, atol=1e-8)

def test_barrier_option_invalid_type():
    with pytest.raises(ValueError):
        barrier_option_price(100, 100, 120, 1, 0.05, 0.2, 'invalid', 'up-and-out')
    with pytest.raises(ValueError):
        barrier_option_price(100, 100, 120, 1, 0.05, 0.2, 'call', 'invalid')

def test_barrier_greeks_full():
    greeks = barrier_greeks_full(100, 100, 120, 1, 0.05, 0.2, 'call', 'up-and-out')
    for g in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        assert g in greeks

def test_barrier_cross_greeks():
    cross = barrier_cross_greeks(100, 100, 120, 1, 0.05, 0.2, 'call', 'up-and-out')
    for g in ['vanna', 'volga', 'cross_gamma']:
        assert g in cross 