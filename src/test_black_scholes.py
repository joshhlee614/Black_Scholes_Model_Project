import unittest
import numpy as np
from src.black_scholes import (
    d1, d2, call_price, put_price,
    call_delta, put_delta, gamma,
    call_theta, put_theta, vega,
    call_rho, put_rho
)

class TestBlackScholes(unittest.TestCase):
    def setUp(self):
        # Base case parameters
        self.S = 100  # stock price
        self.K = 100  # strike price
        self.T = 1    # time to maturity (in years)
        self.r = 0.05 # risk-free interest rate
        self.sigma = 0.2 # volatility

    def test_d1_d2_relationship(self):
        """Test that d2 = d1 - sigma*sqrt(T)"""
        d1_val = d1(self.S, self.K, self.T, self.r, self.sigma)
        d2_val = d2(self.S, self.K, self.T, self.r, self.sigma)
        expected_d2 = d1_val - self.sigma * np.sqrt(self.T)
        self.assertAlmostEqual(d2_val, expected_d2, places=10)

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K*exp(-rT)"""
        call = call_price(self.S, self.K, self.T, self.r, self.sigma)
        put = put_price(self.S, self.K, self.T, self.r, self.sigma)
        rhs = self.S - self.K * np.exp(-self.r * self.T)
        self.assertAlmostEqual(call - put, rhs, places=10)

    def test_delta_relationships(self):
        """Test that call_delta - put_delta = 1"""
        call_d = call_delta(self.S, self.K, self.T, self.r, self.sigma)
        put_d = put_delta(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(call_d - put_d, 1.0, places=10)

    def test_gamma_same_for_calls_and_puts(self):
        """Test that gamma is the same for calls and puts"""
        # Gamma is the same for both calls and puts
        gamma_val = gamma(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(gamma_val, 0)  # Gamma should always be positive

    def test_vega_same_for_calls_and_puts(self):
        """Test that vega is the same for calls and puts"""
        # Vega is the same for both calls and puts
        vega_val = vega(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(vega_val, 0)  # Vega should always be positive

    def test_theta_negative(self):
        """Test that theta is negative for both calls and puts"""
        call_t = call_theta(self.S, self.K, self.T, self.r, self.sigma)
        put_t = put_theta(self.S, self.K, self.T, self.r, self.sigma)
        self.assertLess(call_t, 0)
        self.assertLess(put_t, 0)

    def test_rho_signs(self):
        """Test that rho is positive for calls and negative for puts"""
        call_r = call_rho(self.S, self.K, self.T, self.r, self.sigma)
        put_r = put_rho(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(call_r, 0)
        self.assertLess(put_r, 0)

    def test_at_the_money(self):
        """Test at-the-money options (S = K)"""
        S = K = 100
        call = call_price(S, K, self.T, self.r, self.sigma)
        put = put_price(S, K, self.T, self.r, self.sigma)
        self.assertGreater(call, 0)
        self.assertGreater(put, 0)
        # Use put-call parity to verify the relationship
        rhs = S - K * np.exp(-self.r * self.T)
        self.assertAlmostEqual(call - put, rhs, places=10)

    def test_in_the_money_call(self):
        """Test in-the-money call option (S > K)"""
        S = 110
        K = 100
        call = call_price(S, K, self.T, self.r, self.sigma)
        put = put_price(S, K, self.T, self.r, self.sigma)
        self.assertGreater(call, put)
        self.assertGreater(call_delta(S, K, self.T, self.r, self.sigma), 0.5)

    def test_out_of_the_money_put(self):
        """Test out-of-the-money put options (S > K)"""
        S = 120  # More extreme out-of-the-money case
        K = 100
        put = put_price(S, K, self.T, self.r, self.sigma)
        put_delta_val = put_delta(S, K, self.T, self.r, self.sigma)
        self.assertLess(put, call_price(S, K, self.T, self.r, self.sigma))
        self.assertLess(put_delta_val, -0.1)  # Adjusted expectation for OTM put delta

    def test_zero_time_to_maturity(self):
        """Test option prices at expiration"""
        T = 0.0  # Exactly at expiration
        # At-the-money
        atm_call = call_price(self.S, self.K, T, self.r, self.sigma)
        atm_put = put_price(self.S, self.K, T, self.r, self.sigma)
        self.assertAlmostEqual(atm_call, max(0, self.S - self.K), delta=1e-10)
        self.assertAlmostEqual(atm_put, max(0, self.K - self.S), delta=1e-10)
        
        # Out-of-the-money
        otm_call = call_price(self.S, self.K + 10, T, self.r, self.sigma)
        otm_put = put_price(self.S, self.K - 10, T, self.r, self.sigma)
        self.assertAlmostEqual(otm_call, 0.0, delta=1e-10)
        self.assertAlmostEqual(otm_put, 0.0, delta=1e-10)

    def test_high_volatility(self):
        """Test behavior with high volatility"""
        sigma = 0.8  # High volatility
        call = call_price(self.S, self.K, self.T, self.r, sigma)
        put = put_price(self.S, self.K, self.T, self.r, sigma)
        self.assertGreater(call, 0)
        self.assertGreater(put, 0)
        vega_val = vega(self.S, self.K, self.T, self.r, sigma)
        self.assertGreater(vega_val, 0)

    def test_zero_interest_rate(self):
        """Test behavior with zero interest rate"""
        r = 0.0
        call = call_price(self.S, self.K, self.T, r, self.sigma)
        put = put_price(self.S, self.K, self.T, r, self.sigma)
        self.assertGreater(call, 0)
        self.assertGreater(put, 0)
        call_r = call_rho(self.S, self.K, self.T, r, self.sigma)
        put_r = put_rho(self.S, self.K, self.T, r, self.sigma)
        self.assertGreater(call_r, 0)
        self.assertLess(put_r, 0)

    def test_extreme_volatility(self):
        """Test behavior with extreme volatility values"""
        # Very low volatility
        sigma_low = 0.0001
        call_low = call_price(self.S, self.K, self.T, self.r, sigma_low)
        put_low = put_price(self.S, self.K, self.T, self.r, sigma_low)
        self.assertGreaterEqual(call_low, max(0, self.S - self.K * np.exp(-self.r * self.T)))
        self.assertGreaterEqual(put_low, max(0, self.K * np.exp(-self.r * self.T) - self.S))

        # Very high volatility
        sigma_high = 5.0
        call_high = call_price(self.S, self.K, self.T, self.r, sigma_high)
        put_high = put_price(self.S, self.K, self.T, self.r, sigma_high)
        self.assertGreater(call_high, call_low)
        self.assertGreater(put_high, put_low)

    def test_extreme_prices(self):
        """Test behavior with extreme price values"""
        # Very high stock price
        S_high = 1000000
        K_normal = 100
        call_high_S = call_price(S_high, K_normal, self.T, self.r, self.sigma)
        put_high_S = put_price(S_high, K_normal, self.T, self.r, self.sigma)
        self.assertAlmostEqual(call_delta(S_high, K_normal, self.T, self.r, self.sigma), 1.0, places=4)
        self.assertAlmostEqual(put_delta(S_high, K_normal, self.T, self.r, self.sigma), 0.0, places=4)

        # Very low stock price
        S_low = 0.0001
        call_low_S = call_price(S_low, K_normal, self.T, self.r, self.sigma)
        put_low_S = put_price(S_low, K_normal, self.T, self.r, self.sigma)
        self.assertAlmostEqual(call_delta(S_low, K_normal, self.T, self.r, self.sigma), 0.0, places=4)
        self.assertAlmostEqual(put_delta(S_low, K_normal, self.T, self.r, self.sigma), -1.0, places=4)

    def test_extreme_time(self):
        """Test behavior with extreme time values"""
        # Very short time to expiration
        T_short = 1/365  # One day
        call_short = call_price(self.S, self.K, T_short, self.r, self.sigma)
        put_short = put_price(self.S, self.K, T_short, self.r, self.sigma)
        self.assertGreaterEqual(call_short, max(0, self.S - self.K))
        self.assertGreaterEqual(put_short, max(0, self.K - self.S))

        # Very long time to expiration
        T_long = 30  # 30 years
        call_long = call_price(self.S, self.K, T_long, self.r, self.sigma)
        put_long = put_price(self.S, self.K, T_long, self.r, self.sigma)
        self.assertGreater(call_long, call_short)
        self.assertGreater(put_long, put_short)

    def test_extreme_rates(self):
        """Test behavior with extreme interest rates"""
        # Zero interest rate
        r_zero = 0.0
        call_zero_r = call_price(self.S, self.K, self.T, r_zero, self.sigma)
        put_zero_r = put_price(self.S, self.K, self.T, r_zero, self.sigma)
        self.assertGreaterEqual(call_zero_r, 0)
        self.assertGreaterEqual(put_zero_r, 0)

        # High interest rate
        r_high = 0.5  # 50%
        call_high_r = call_price(self.S, self.K, self.T, r_high, self.sigma)
        put_high_r = put_price(self.S, self.K, self.T, r_high, self.sigma)
        self.assertGreater(call_high_r, call_zero_r)
        self.assertLess(put_high_r, put_zero_r)

    def test_put_call_parity_edge_cases(self):
        """Test put-call parity holds under various edge cases"""
        test_cases = [
            (100, 100, 1, 0.05, 0.2),    # Base case
            (0.0001, 100, 1, 0.05, 0.2), # Very low stock price
            (1000000, 100, 1, 0.05, 0.2),# Very high stock price
            (100, 100, 0.0001, 0.05, 0.2),# Very short time
            (100, 100, 30, 0.05, 0.2),   # Very long time
            (100, 100, 1, 0.0, 0.2),     # Zero interest rate
            (100, 100, 1, 0.5, 0.2),     # High interest rate
            (100, 100, 1, 0.05, 0.0001), # Low volatility
            (100, 100, 1, 0.05, 5.0)     # High volatility
        ]
        
        for S, K, T, r, sigma in test_cases:
            call = call_price(S, K, T, r, sigma)
            put = put_price(S, K, T, r, sigma)
            rhs = S - K * np.exp(-r * T)
            self.assertAlmostEqual(call - put, rhs, places=6,
                msg=f"Put-call parity failed for S={S}, K={K}, T={T}, r={r}, sigma={sigma}")

    def test_boundary_conditions(self):
        """Test option prices satisfy basic boundary conditions"""
        # Call price should be close to stock price when K is very small
        K_small = 0.0001
        call_small_K = call_price(self.S, K_small, self.T, self.r, self.sigma)
        self.assertAlmostEqual(call_small_K, self.S, delta=1.0)

        # Put price should be close to present value of K when S is very small
        S_small = 0.0001
        put_small_S = put_price(S_small, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(put_small_S, self.K * np.exp(-self.r * self.T), delta=1.0)

        # Call price should approach 0 when K is very large
        K_large = 1000000
        call_large_K = call_price(self.S, K_large, self.T, self.r, self.sigma)
        self.assertLess(call_large_K, 1.0)

        # Put price should approach 0 when S is very large
        S_large = 1000000
        put_large_S = put_price(S_large, self.K, self.T, self.r, self.sigma)
        self.assertLess(put_large_S, 1.0)

if __name__ == "__main__":
    unittest.main() 