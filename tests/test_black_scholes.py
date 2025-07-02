import pytest
import numpy as np
from app.models.black_scholes import black_scholes_price, black_scholes_greeks

def test_black_scholes_call_price():
    price = black_scholes_price(100, 100, 1, 0.05, 0.2, 'call')
    assert np.isclose(price, 10.4506, atol=1e-4)

def test_black_scholes_put_price():
    price = black_scholes_price(100, 100, 1, 0.05, 0.2, 'put')
    assert np.isclose(price, 5.5735, atol=1e-4)

def test_black_scholes_greeks_call():
    greeks = black_scholes_greeks(100, 100, 1, 0.05, 0.2, 'call')
    assert np.isclose(greeks['delta'], 0.6368, atol=1e-4)
    assert np.isclose(greeks['gamma'], 0.0188, atol=1e-4)
    assert np.isclose(greeks['vega'], 37.5240, atol=1e-3)
    assert np.isclose(greeks['theta'], -6.4140, atol=1e-3)
    assert np.isclose(greeks['rho'], 53.2325, atol=1e-3)

def test_black_scholes_greeks_put():
    greeks = black_scholes_greeks(100, 100, 1, 0.05, 0.2, 'put')
    assert np.isclose(greeks['delta'], -0.3632, atol=1e-4)
    assert np.isclose(greeks['gamma'], 0.0188, atol=1e-4)
    assert np.isclose(greeks['vega'], 37.5240, atol=1e-3)
    assert np.isclose(greeks['theta'], -1.6579, atol=1e-3)
    assert np.isclose(greeks['rho'], -41.8905, atol=1e-3)

def test_invalid_option_type():
    with pytest.raises(ValueError):
        black_scholes_price(100, 100, 1, 0.05, 0.2, 'invalid')
    with pytest.raises(ValueError):
        black_scholes_greeks(100, 100, 1, 0.05, 0.2, 'invalid') 