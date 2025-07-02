import numpy as np
import plotly.graph_objs as go
from app.models.black_scholes import black_scholes_price, black_scholes_greeks
from app.models.binomial_tree import binomial_tree_price
from app.models.heston import heston_price
from app.models.hull_white import hull_white_bond_option

def plot_bs_price_surface(K, T, r, sigma, option_type, S=None):
    if S is None:
        S = K
    S_grid = np.linspace(0.5 * S, 1.5 * S, 50)
    vol = np.linspace(0.5 * sigma, 1.5 * sigma, 50)
    S_mesh, vol_mesh = np.meshgrid(S_grid, vol)
    price_grid = np.zeros_like(S_mesh)
    for i in range(S_mesh.shape[0]):
        for j in range(S_mesh.shape[1]):
            price_grid[i, j] = black_scholes_price(S_mesh[i, j], K, T, r, vol_mesh[i, j], option_type)
    fig = go.Figure(data=[go.Surface(z=price_grid, x=S_grid, y=vol, colorscale='Viridis')])
    fig.update_layout(title="Option Price Surface", scene=dict(xaxis_title="Spot Price (S)", yaxis_title="Volatility (sigma)", zaxis_title="Price"))
    return fig

def plot_bs_greeks_surface(K, T, r, sigma, option_type, greek="delta", S=None):
    if S is None:
        S = K
    S_grid = np.linspace(0.5 * S, 1.5 * S, 50)
    vol = np.linspace(0.5 * sigma, 1.5 * sigma, 50)
    S_mesh, vol_mesh = np.meshgrid(S_grid, vol)
    greek_grid = np.zeros_like(S_mesh)
    for i in range(S_mesh.shape[0]):
        for j in range(S_mesh.shape[1]):
            greek_grid[i, j] = black_scholes_greeks(S_mesh[i, j], K, T, r, vol_mesh[i, j], option_type)[greek]
    color_map = {
        "delta": 'Cividis',
        "gamma": 'Plasma',
        "vega": 'YlGnBu',
        "theta": 'Inferno',
        "rho": 'RdBu'
    }
    fig = go.Figure(data=[go.Surface(z=greek_grid, x=S_grid, y=vol, colorscale=color_map.get(greek, 'Viridis'))])
    fig.update_layout(title=f"{greek.capitalize()} Surface", scene=dict(xaxis_title="Spot Price (S)", yaxis_title="Volatility (sigma)", zaxis_title=greek.capitalize()))
    return fig

def plot_binomial_tree_price_surface(K, T, r, sigma, steps, option_type, exercise, S=None):
    if S is None:
        S = K
    S_grid = np.linspace(0.5 * S, 1.5 * S, 30)
    vol = np.linspace(0.5 * sigma, 1.5 * sigma, 30)
    S_mesh, vol_mesh = np.meshgrid(S_grid, vol)
    price_grid = np.zeros_like(S_mesh)
    for i in range(S_mesh.shape[0]):
        for j in range(S_mesh.shape[1]):
            price_grid[i, j] = binomial_tree_price(S_mesh[i, j], K, T, r, vol_mesh[i, j], steps, option_type, exercise)
    fig = go.Figure(data=[go.Surface(z=price_grid, x=S_grid, y=vol, colorscale='Blues')])
    fig.update_layout(title="Binomial Tree Price Surface", scene=dict(xaxis_title="Spot Price (S)", yaxis_title="Volatility (sigma)", zaxis_title="Price"))
    return fig

def plot_heston_price_surface(K, T, r, v0, kappa, theta, sigma, rho, option_type, S=None):
    if S is None:
        S = K
    S_grid = np.linspace(0.5 * S, 1.5 * S, 20)
    vol = np.linspace(0.5 * np.sqrt(theta), 1.5 * np.sqrt(theta), 20)
    S_mesh, vol_mesh = np.meshgrid(S_grid, vol)
    price_grid = np.zeros_like(S_mesh)
    for i in range(S_mesh.shape[0]):
        for j in range(S_mesh.shape[1]):
            price_grid[i, j] = heston_price(S_mesh[i, j], K, T, r, v0, kappa, theta, sigma, rho, option_type)
    fig = go.Figure(data=[go.Surface(z=price_grid, x=S_grid, y=vol, colorscale='Magma')])
    fig.update_layout(title="Heston Price Surface", scene=dict(xaxis_title="Spot Price (S)", yaxis_title="Volatility (sqrt(theta))", zaxis_title="Price"))
    return fig

def plot_hull_white_price_surface(r0, t, T, a, sigma, K):
    S = np.linspace(0.5, 2.0, 20)
    strike = np.linspace(0.5 * K, 1.5 * K, 20)
    S_grid, K_grid = np.meshgrid(S, strike)
    price_grid = np.zeros_like(S_grid)
    def P0T(T_):
        return np.exp(-r0 * T_)
    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            price_grid[i, j] = hull_white_bond_option(r0, t, T, S_grid[i, j], K_grid[i, j], a, sigma, P0T, 'call')
    fig = go.Figure(data=[go.Surface(z=price_grid, x=S, y=strike, colorscale='Greens')])
    fig.update_layout(title="Hull-White Bond Option Price Surface", scene=dict(xaxis_title="Bond Maturity (S)", yaxis_title="Strike (K)", zaxis_title="Price"))
    return fig

def plot_heatmap_2d(x, y, z, x_label, y_label, z_label, title, colorscale='Viridis'):
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale=colorscale, colorbar=dict(title=z_label)))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    return fig

def plot_bs_greek_heatmap(K, T, r, sigma, option_type, greek="delta", S=None):
    if S is None:
        S = K
    S_grid = np.linspace(0.5 * S, 1.5 * S, 50)
    vol = np.linspace(0.5 * sigma, 1.5 * sigma, 50)
    S_mesh, vol_mesh = np.meshgrid(S_grid, vol)
    greek_grid = np.zeros_like(S_mesh)
    for i in range(S_mesh.shape[0]):
        for j in range(S_mesh.shape[1]):
            greek_grid[i, j] = black_scholes_greeks(S_mesh[i, j], K, T, r, vol_mesh[i, j], option_type)[greek]
    return plot_heatmap_2d(S_grid, vol, greek_grid, "Spot Price (S)", "Volatility (sigma)", greek.capitalize(), f"Black-Scholes {greek.capitalize()} Heatmap")

def plot_binomial_tree_greek_heatmap(K, T, r, sigma, steps, option_type, exercise, greek="delta", S=None):
    if S is None:
        S = K
    S_grid = np.linspace(0.5 * S, 1.5 * S, 30)
    vol = np.linspace(0.5 * sigma, 1.5 * sigma, 30)
    S_mesh, vol_mesh = np.meshgrid(S_grid, vol)
    greek_grid = np.zeros_like(S_mesh)
    for i in range(S_mesh.shape[0]):
        for j in range(S_mesh.shape[1]):
            if greek == "delta":
                greek_grid[i, j] = binomial_tree_delta(S_mesh[i, j], K, T, r, vol_mesh[i, j], steps, option_type, exercise)
            elif greek == "gamma":
                greek_grid[i, j] = binomial_tree_gamma(S_mesh[i, j], K, T, r, vol_mesh[i, j], steps, option_type, exercise)
    return plot_heatmap_2d(S_grid, vol, greek_grid, "Spot Price (S)", "Volatility (sigma)", greek.capitalize(), f"Binomial Tree {greek.capitalize()} Heatmap")

def plot_heston_greek_heatmap(K, T, r, v0, kappa, theta, sigma, rho, option_type, greek="delta", S=None):
    if S is None:
        S = K
    S_grid = np.linspace(0.5 * S, 1.5 * S, 20)
    vol = np.linspace(0.5 * np.sqrt(theta), 1.5 * np.sqrt(theta), 20)
    S_mesh, vol_mesh = np.meshgrid(S_grid, vol)
    greek_grid = np.zeros_like(S_mesh)
    for i in range(S_mesh.shape[0]):
        for j in range(S_mesh.shape[1]):
            if greek == "delta":
                greek_grid[i, j] = heston_delta(S_mesh[i, j], K, T, r, v0, kappa, theta, sigma, rho, option_type)
    return plot_heatmap_2d(S_grid, vol, greek_grid, "Spot Price (S)", "Volatility (sqrt(theta))", greek.capitalize(), f"Heston {greek.capitalize()} Heatmap")

def plot_hull_white_dv01_heatmap(r0, t, T, a, sigma, K):
    S = np.linspace(0.5, 2.0, 20)
    strike = np.linspace(0.5 * K, 1.5 * K, 20)
    S_grid, K_grid = np.meshgrid(S, strike)
    dv01_grid = np.zeros_like(S_grid)
    def P0T(T_):
        return np.exp(-r0 * T_)
    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            dv01_grid[i, j] = hull_white_dv01(r0, t, T, S_grid[i, j], K_grid[i, j], a, sigma, P0T, 'call')
    return plot_heatmap_2d(S, strike, dv01_grid, "Bond Maturity (S)", "Strike (K)", "DV01", "Hull-White DV01 Heatmap", colorscale='Greens') 