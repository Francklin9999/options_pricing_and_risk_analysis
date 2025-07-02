import pytest
import numpy as np
from app.models.hull_white import hull_white_bond_price, hull_white_bond_option

class FlatCurve:
    def __init__(self, r):
        self.r = r
    def __call__(self, T):
        return np.exp(-self.r * T)
    def derivative(self, t):
        return -self.r * np.exp(-self.r * t)

def test_hull_white_bond_price_flat_curve():
    r0 = 0.03
    t = 0.0
    T = 5.0
    a = 0.1
    sigma = 0.01
    P0T = FlatCurve(r0)
    price = hull_white_bond_price(r0, t, T, a, sigma, P0T)
    assert np.isclose(price, P0T(T), atol=1e-4)

def test_hull_white_bond_option_positive():
    r0 = 0.03
    t = 0.0
    T = 2.0
    S = 5.0
    K = 0.8
    a = 0.1
    sigma = 0.01
    P0T = FlatCurve(r0)
    price = hull_white_bond_option(r0, t, T, S, K, a, sigma, P0T, 'call')
    assert price > 0

def test_hull_white_bond_option_invalid_type():
    r0 = 0.03
    t = 0.0
    T = 2.0
    S = 5.0
    K = 0.8
    a = 0.1
    sigma = 0.01
    P0T = FlatCurve(r0)
    with pytest.raises(ValueError):
        hull_white_bond_option(r0, t, T, S, K, a, sigma, P0T, 'invalid') 