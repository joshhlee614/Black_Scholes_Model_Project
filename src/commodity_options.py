import numpy as np
from scipy.stats import norm
from .black_scholes import d1, d2

def commodity_call_price(S, K, T, r, sigma, storage_cost, convenience_yield):
    """
    Calculate the price of a European call option on a commodity.
    
    Parameters:
    -----------
    S : float
        Current commodity price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the commodity price
    storage_cost : float
        Annual storage cost as a percentage
    convenience_yield : float
        Annual convenience yield as a percentage
    
    Returns:
    --------
    float
        Price of the call option
    """
    # Adjust the risk-free rate for storage costs and convenience yield
    adjusted_r = r + storage_cost - convenience_yield
    
    # Calculate d1 and d2 using the adjusted rate
    d1_val = d1(S, K, T, adjusted_r, sigma)
    d2_val = d2(d1_val, sigma, T)
    
    # Calculate the call price
    call_price = S * np.exp(-convenience_yield * T) * norm.cdf(d1_val) - \
                 K * np.exp(-r * T) * norm.cdf(d2_val)
    
    return call_price

def commodity_put_price(S, K, T, r, sigma, storage_cost, convenience_yield):
    """
    Calculate the price of a European put option on a commodity.
    
    Parameters:
    -----------
    S : float
        Current commodity price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the commodity price
    storage_cost : float
        Annual storage cost as a percentage
    convenience_yield : float
        Annual convenience yield as a percentage
    
    Returns:
    --------
    float
        Price of the put option
    """
    # Adjust the risk-free rate for storage costs and convenience yield
    adjusted_r = r + storage_cost - convenience_yield
    
    # Calculate d1 and d2 using the adjusted rate
    d1_val = d1(S, K, T, adjusted_r, sigma)
    d2_val = d2(d1_val, sigma, T)
    
    # Calculate the put price
    put_price = K * np.exp(-r * T) * norm.cdf(-d2_val) - \
                S * np.exp(-convenience_yield * T) * norm.cdf(-d1_val)
    
    return put_price

def commodity_forward_price(S, T, r, storage_cost, convenience_yield):
    """
    Calculate the forward price of a commodity.
    
    Parameters:
    -----------
    S : float
        Current commodity price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    storage_cost : float
        Annual storage cost as a percentage
    convenience_yield : float
        Annual convenience yield as a percentage
    
    Returns:
    --------
    float
        Forward price of the commodity
    """
    # Calculate the forward price
    forward_price = S * np.exp((r + storage_cost - convenience_yield) * T)
    return forward_price

def commodity_futures_price(S, T, r, storage_cost, convenience_yield):
    """
    Calculate the futures price of a commodity.
    This is equivalent to the forward price under the assumptions of the model.
    
    Parameters:
    -----------
    S : float
        Current commodity price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    storage_cost : float
        Annual storage cost as a percentage
    convenience_yield : float
        Annual convenience yield as a percentage
    
    Returns:
    --------
    float
        Futures price of the commodity
    """
    return commodity_forward_price(S, T, r, storage_cost, convenience_yield) 