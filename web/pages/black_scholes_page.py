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
The Black-Scholes model is a mathematical model for pricing European-style options. It was developed by Fischer Black, Myron Scholes, 
and Robert Merton in the early 1970s and revolutionized the field of quantitative finance.

## Key Components
The model takes into account five key parameters:
1. **Spot Price (S)**: Current price of the underlying asset
2. **Strike Price (K)**: Price at which the option can be exercised
3. **Time to Maturity (T)**: Time until the option expires
4. **Volatility (σ)**: Measure of the asset's price fluctuations
5. **Risk-free Rate (r)**: Theoretical return on a risk-free investment

## The Formulas
### Call Option Price
$C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$

### Put Option Price
$P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)$

### Where:
$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$

$d_2 = d_1 - \sigma\sqrt{T}$

## Applications in This App
This application demonstrates several key aspects of the Black-Scholes model:

1. **Price Sensitivity**: The heatmaps show how option prices change with different spot prices and volatility levels
2. **Put-Call Parity**: The relationship between call and put prices is maintained through the model
3. **Time Value**: The impact of time to maturity on option prices
4. **Volatility Impact**: How changes in volatility affect option prices

## Limitations
While the Black-Scholes model is foundational, it has some limitations:
- Assumes constant volatility
- Assumes continuous trading
- Assumes no transaction costs
- Assumes log-normal distribution of returns
- Only applies to European options

## Practical Use
In this application, you can:
1. Calculate option prices for different market scenarios
2. Visualize how prices change with different parameters
3. Understand the relationship between calls and puts
4. Explore the impact of volatility on option prices
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