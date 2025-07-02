import pytest
import numpy as np
from app.models.binomial_tree import binomial_tree_price
from app.models.black_scholes import black_scholes_price

def test_binomial_tree_european_call_converges_to_black_scholes():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    price_binom = binomial_tree_price(S, K, T, r, sigma, steps=500, option_type='call', exercise='european')
    price_bs = black_scholes_price(S, K, T, r, sigma, 'call')
    assert np.isclose(price_binom, price_bs, atol=1e-2)

def test_binomial_tree_european_put_converges_to_black_scholes():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    price_binom = binomial_tree_price(S, K, T, r, sigma, steps=500, option_type='put', exercise='european')
    price_bs = black_scholes_price(S, K, T, r, sigma, 'put')
    assert np.isclose(price_binom, price_bs, atol=1e-2)

def test_binomial_tree_american_put_greater_than_european():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    price_amer = binomial_tree_price(S, K, T, r, sigma, steps=500, option_type='put', exercise='american')
    price_euro = binomial_tree_price(S, K, T, r, sigma, steps=500, option_type='put', exercise='european')
    assert price_amer >= price_euro

def test_binomial_tree_invalid_option_type():
    with pytest.raises(ValueError):
        binomial_tree_price(100, 100, 1, 0.05, 0.2, steps=100, option_type='invalid', exercise='european') 