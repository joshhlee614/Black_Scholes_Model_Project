import streamlit as st
import numpy as np
from scipy.stats import norm

st.set_page_config(
    page_title="Black-Scholes Model - Theory",
    layout="wide"
)

st.title("The Black-Scholes Model")

st.markdown("""
## Introduction
The Black-Scholes model, developed by Fischer Black, Myron Scholes, and Robert Merton in 1973, revolutionized quantitative finance 
by providing a closed-form solution for European option pricing. This model serves as the foundation for modern derivatives pricing 
and risk management, particularly in commodities markets where options play a crucial role in hedging and speculation.

## Mathematical Foundation

### Key Components
The model incorporates five essential parameters:
1. **Spot Price (S)**: Current market price of the underlying asset
2. **Strike Price (K)**: Exercise price of the option
3. **Time to Maturity (T)**: Time until option expiration (in years)
4. **Volatility (σ)**: Annualized standard deviation of returns
5. **Risk-free Rate (r)**: Continuously compounded risk-free interest rate

### The Black-Scholes PDE
The model is derived from the following partial differential equation:

$$\\frac{\\partial V}{\\partial t} + \\frac{1}{2}\\sigma^2S^2\\frac{\\partial^2 V}{\\partial S^2} + rS\\frac{\\partial V}{\\partial S} - rV = 0$$

Where:
- $V$ is the option value
- $t$ is time
- $S$ is the stock price
- $\\sigma$ is volatility
- $r$ is the risk-free rate

### Closed-Form Solutions

#### Call Option Price
$$C = S \\cdot N(d_1) - K \\cdot e^{-rT} \\cdot N(d_2)$$

#### Put Option Price
$$P = K \\cdot e^{-rT} \\cdot N(-d_2) - S \\cdot N(-d_1)$$

#### Where:
$$d_1 = \\frac{\\ln(S/K) + (r + \\sigma^2/2)T}{\\sigma\\sqrt{T}}$$

$$d_2 = d_1 - \\sigma\\sqrt{T}$$

### The Greeks

#### Delta (Δ)
$$\\Delta_{call} = N(d_1)$$
$$\\Delta_{put} = N(d_1) - 1$$

#### Gamma (Γ)
$$\\Gamma = \\frac{N'(d_1)}{S\\sigma\\sqrt{T}}$$

#### Theta (Θ)
$$\\Theta_{call} = -\\frac{S\\sigma N'(d_1)}{2\\sqrt{T}} - rKe^{-rT}N(d_2)$$
$$\\Theta_{put} = -\\frac{S\\sigma N'(d_1)}{2\\sqrt{T}} + rKe^{-rT}N(-d_2)$$

#### Vega (ν)
$$\\nu = S\\sqrt{T}N'(d_1)$$

#### Rho (ρ)
$$\\rho_{call} = KTe^{-rT}N(d_2)$$
$$\\rho_{put} = -KTe^{-rT}N(-d_2)$$

## Professional Applications

### Risk Management
- **Delta Hedging**: Creating delta-neutral portfolios
- **Gamma Scalping**: Profiting from volatility through gamma trading
- **Vega Hedging**: Managing volatility risk exposure
- **Portfolio Optimization**: Using Greeks for risk allocation

### Market Making
- **Bid-Ask Spread Determination**: Using Greeks to set option prices
- **Inventory Management**: Delta-neutral position maintenance
- **Volatility Trading**: Capitalizing on implied vs. realized volatility

### Commodities Trading
- **Energy Derivatives**: Pricing options on crude oil, natural gas
- **Agricultural Options**: Hedging crop price risks
- **Metals Trading**: Gold and silver options pricing
- **Storage Arbitrage**: Incorporating storage costs in pricing

## Model Limitations and Extensions

### Key Limitations
- Constant volatility assumption
- Continuous trading assumption
- No transaction costs
- Log-normal distribution of returns
- European exercise only

### Common Extensions
- Local volatility models
- Stochastic volatility (Heston model)
- Jump-diffusion processes
- American option pricing
- Commodity-specific adjustments

## Practical Implementation

### Numerical Considerations
- Efficient computation of normal CDF
- Handling of edge cases (e.g., T → 0)
- Numerical stability in Greeks calculation
- Vectorized implementation for performance

### Market Calibration
- Implied volatility calculation
- Volatility surface construction
- Risk-neutral probability estimation
- Forward curve incorporation

## Application Features
This implementation demonstrates:
1. Real-time option pricing with parameter sensitivity
2. Comprehensive Greeks calculation and visualization
3. Multi-dimensional sensitivity analysis
4. Historical calculation tracking
5. Professional-grade risk metrics
""")

st.markdown("""
<style>
.model-section {
    background-color: rgba(49, 51, 63, 0.7);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True) 