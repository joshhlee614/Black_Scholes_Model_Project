import numpy as np
from scipy.stats import norm

# Black-Scholes option pricing model: only for European options

def d1(S, K, T, r, sigma):
    """Calculate d1 parameter for Black-Scholes model.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free interest rate
        sigma: Volatility
        
    Returns:
        d1 parameter
    """
    if sigma <= 0 or T <= 0:
        raise ValueError("Volatility and time to maturity must be positive")
    if S <= 0 or K <= 0:
        raise ValueError("Stock price and strike price must be positive")
    
    # Handle special cases to prevent divide by zero
    if T <= 1e-10:  # At expiration
        if abs(S - K) < 1e-10:  # At the money
            return 0
        return np.inf if S > K else -np.inf
    
    if abs(sigma * np.sqrt(T)) < 1e-10:
        return np.inf if S > K else -np.inf
        
    return (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    """Calculate d2 parameter for Black-Scholes model.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free interest rate
        sigma: Volatility
        
    Returns:
        d2 parameter
    """
    if sigma <= 0 or T <= 0:
        raise ValueError("Volatility and time to maturity must be positive")
    if S <= 0 or K <= 0:
        raise ValueError("Stock price and strike price must be positive")
    
    # Handle special cases
    if T <= 1e-10:  # At expiration
        if abs(S - K) < 1e-10:  # At the money
            return 0
        return np.inf if S > K else -np.inf
        
    if abs(sigma * np.sqrt(T)) < 1e-10:
        return np.inf if S > K else -np.inf
    
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def call_price(S, K, T, r, sigma):
    """Calculate European call option price using Black-Scholes model.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free interest rate
        sigma: Volatility
        
    Returns:
        Call option price
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("All parameters must be positive")
    if T <= 1e-10:  # At expiration
        return max(0, S - K)
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    return S * norm.cdf(d1_val) - K * np.exp(-r*T) * norm.cdf(d2_val)

def put_price(S, K, T, r, sigma):
    """Calculate European put option price using Black-Scholes model.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free interest rate
        sigma: Volatility
        
    Returns:
        Put option price
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("All parameters must be positive")
    if T <= 1e-10:  # At expiration
        return max(0, K - S)
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    return K * np.exp(-r*T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)

def call_delta(S, K, T, r, sigma):
    """Calculate delta for a call option.
    
    Delta measures the rate of change of the option price with respect to the underlying asset's price.
    For calls: ranges from 0 to 1
    """
    if T <= 1e-10:  # At expiration
        return 1.0 if S > K else 0.0
    return norm.cdf(d1(S, K, T, r, sigma))

def put_delta(S, K, T, r, sigma):
    """Calculate delta for a put option.
    
    Delta measures the rate of change of the option price with respect to the underlying asset's price.
    For puts: ranges from -1 to 0
    """
    if T <= 1e-10:  # At expiration
        return -1.0 if S < K else 0.0
    return -norm.cdf(-d1(S, K, T, r, sigma))

def gamma(S, K, T, r, sigma):
    """Calculate gamma (same for calls and puts).
    
    Gamma measures the rate of change of delta with respect to the underlying asset's price.
    Always positive, indicating that delta increases as the stock price increases.
    """
    if T <= 1e-10:  # At expiration
        return 0.0
    d1_val = d1(S, K, T, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))

def call_theta(S, K, T, r, sigma):
    """Calculate theta for a call option.
    
    Theta measures the rate of change of the option price with respect to time.
    Usually negative, indicating that the option loses value as time passes.
    Returns daily theta (price change per day).
    """
    if T <= 1e-10:  # At expiration
        return 0.0
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r*T) * norm.cdf(d2_val)
    return (term1 + term2) / 365  # Convert to daily theta

def put_theta(S, K, T, r, sigma):
    """Calculate theta for a put option.
    
    Theta measures the rate of change of the option price with respect to time.
    Usually negative, indicating that the option loses value as time passes.
    Returns daily theta (price change per day).
    """
    if T <= 1e-10:  # At expiration
        return 0.0
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r*T) * norm.cdf(-d2_val)
    return (term1 + term2) / 365  # Convert to daily theta

def vega(S, K, T, r, sigma):
    """Calculate vega (same for calls and puts).
    
    Vega measures the rate of change of the option price with respect to volatility.
    Always positive, indicating that option prices increase with volatility.
    Returns price change per 1% change in volatility.
    """
    if T <= 1e-10:  # At expiration
        return 0.0
    d1_val = d1(S, K, T, r, sigma)
    return S * np.sqrt(T) * norm.pdf(d1_val) / 100  # Convert to 1% change in volatility

def call_rho(S, K, T, r, sigma):
    """Calculate rho for a call option.
    
    Rho measures the rate of change of the option price with respect to the risk-free interest rate.
    For calls: positive, indicating that call prices increase with interest rates.
    Returns price change per 1% change in interest rate.
    """
    if T <= 1e-10:  # At expiration
        return 0.0
    d2_val = d2(S, K, T, r, sigma)
    return K * T * np.exp(-r*T) * norm.cdf(d2_val) / 100  # Convert to 1% change in interest rate

def put_rho(S, K, T, r, sigma):
    """Calculate rho for a put option.
    
    Rho measures the rate of change of the option price with respect to the risk-free interest rate.
    For puts: negative, indicating that put prices decrease with interest rates.
    Returns price change per 1% change in interest rate.
    """
    if T <= 1e-10:  # At expiration
        return 0.0
    d2_val = d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r*T) * norm.cdf(-d2_val) / 100  # Convert to 1% change in interest rate

# Export all functions
__all__ = [
    'd1', 'd2',
    'call_price', 'put_price',
    'call_delta', 'put_delta', 'gamma',
    'call_theta', 'put_theta', 'vega',
    'call_rho', 'put_rho'
]

if __name__ == "__main__":
    # Test the functions with example values
    S = 100  # stock price
    K = 100  # strike price
    T = 1    # time to maturity (in years)
    r = 0.05 # risk-free interest rate
    sigma = 0.2 # volatility

    # Test prices
    call = call_price(S, K, T, r, sigma)
    put = put_price(S, K, T, r, sigma)
    print(f"Call Option Price: ${call:.2f}")
    print(f"Put Option Price: ${put:.2f}")

    # Test Greeks
    print("\nGreeks:")
    print(f"Call Delta: {call_delta(S, K, T, r, sigma):.4f}")
    print(f"Put Delta: {put_delta(S, K, T, r, sigma):.4f}")
    print(f"Gamma: {gamma(S, K, T, r, sigma):.4f}")
    print(f"Call Theta: ${call_theta(S, K, T, r, sigma):.4f} per day")
    print(f"Put Theta: ${put_theta(S, K, T, r, sigma):.4f} per day")
    print(f"Vega: ${vega(S, K, T, r, sigma):.4f} per 1% vol change")
    print(f"Call Rho: ${call_rho(S, K, T, r, sigma):.4f} per 1% rate change")
    print(f"Put Rho: ${put_rho(S, K, T, r, sigma):.4f} per 1% rate change")

