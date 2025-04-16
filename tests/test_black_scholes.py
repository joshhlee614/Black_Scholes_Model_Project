import unittest
import numpy as np
from src.black_scholes import d1, d2, call_price, put_price

class TestBlackScholes(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.S = 100  # Stock price
        self.K = 100  # Strike price
        self.T = 1    # Time to maturity
        self.r = 0.05 # Risk-free rate
        self.sigma = 0.2 # Volatility

    def test_d1(self):
        """Test d1 calculation"""
        d1_val = d1(self.S, self.K, self.T, self.r, self.sigma)
        
        # For these parameters, d1 should be approximately 0.35
        self.assertAlmostEqual(d1_val, 0.35, places=6)
        
        # Additional checks for d1 properties
        self.assertGreater(d1_val, 0)  # d1 should be positive for at-the-money options
        
        # Test d1 with different parameters
        d1_high_vol = d1(self.S, self.K, self.T, self.r, 0.4)  # Higher volatility
        self.assertLess(d1_high_vol, d1_val)  # Higher volatility should decrease d1

    def test_d2(self):
        """Test d2 calculation"""
        d2_val = d2(self.S, self.K, self.T, self.r, self.sigma)
        
        # For these parameters, d2 should be approximately 0.15
        self.assertAlmostEqual(d2_val, 0.15, places=6)
        
        # Verify relationship between d1 and d2
        d1_val = d1(self.S, self.K, self.T, self.r, self.sigma)
        expected_d2 = d1_val - self.sigma * np.sqrt(self.T)
        self.assertAlmostEqual(d2_val, expected_d2, places=10)

    def test_call_price(self):
        """Test call option pricing"""
        call = call_price(self.S, self.K, self.T, self.r, self.sigma)
        
        # Basic checks
        self.assertGreater(call, 0)  # Call price should be positive
        self.assertLess(call, self.S)  # Call price should be less than stock price
        
        # Test with different parameters
        call_itm = call_price(200, 100, 1, 0.05, 0.2)  # In-the-money
        self.assertGreater(call_itm, 100)  # Should be worth more than intrinsic value
        
        call_otm = call_price(50, 100, 1, 0.05, 0.2)  # Out-of-the-money
        self.assertLess(call_otm, 1)  # Should be worth very little

    def test_put_price(self):
        """Test put option pricing"""
        put = put_price(self.S, self.K, self.T, self.r, self.sigma)
        
        # Basic checks
        self.assertGreater(put, 0)  # Put price should be positive
        
        # Test put-call parity
        call = call_price(self.S, self.K, self.T, self.r, self.sigma)
        lhs = call - put
        rhs = self.S - self.K * np.exp(-self.r * self.T)
        self.assertAlmostEqual(lhs, rhs, places=10)

def test_black_scholes_known_values():
    # Test case 1: Standard parameters
    S = 100  # Current stock price
    K = 100  # Strike price
    T = 1.0  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    # Expected values from standard Black-Scholes formula
    expected_call = 10.4506  # Approximate value
    expected_put = 5.5735    # Approximate value
    
    # Calculate actual values
    actual_call = call_price(S, K, T, r, sigma)
    actual_put = put_price(S, K, T, r, sigma)
    
    # Compare with 0.01 precision
    assert abs(actual_call - expected_call) < 0.01
    assert abs(actual_put - expected_put) < 0.01

def test_black_scholes_edge_cases():
    # Test case 2: At-the-money options
    S = 100
    K = 100
    T = 0.5
    r = 0.05
    sigma = 0.2
    
    call_atm = call_price(S, K, T, r, sigma)
    put_atm = put_price(S, K, T, r, sigma)
    
    # At-the-money call should be more expensive than put
    assert call_atm > put_atm
    
    # Test case 3: Deep in-the-money call
    S = 150
    K = 100
    call_itm = call_price(S, K, T, r, sigma)
    assert call_itm >= (S - K)  # Call price should be at least intrinsic value
    
    # Test case 4: Deep out-of-the-money put
    S = 50
    K = 100
    put_otm = put_price(S, K, T, r, sigma)
    assert put_otm >= (K - S)  # Put price should be at least intrinsic value
    assert put_otm > 0  # Put price should be positive

def test_black_scholes_put_call_parity():
    # Test put-call parity: C - P = S - K*e^(-rT)
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    C = call_price(S, K, T, r, sigma)
    P = put_price(S, K, T, r, sigma)
    rhs = S - K * np.exp(-r * T)
    
    # Check put-call parity holds within 0.01
    assert abs((C - P) - rhs) < 0.01

if __name__ == '__main__':
    unittest.main()

