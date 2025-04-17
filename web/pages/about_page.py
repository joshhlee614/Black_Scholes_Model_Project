import streamlit as st

st.set_page_config(
    page_title="About - Black-Scholes Model",
    layout="wide"
)

st.title("About This Application")

st.markdown("""
## Overview
This comprehensive implementation of the Black-Scholes option pricing model demonstrates advanced quantitative finance concepts 
and their practical applications in derivatives trading. The application serves as both an educational tool and a professional 
demonstration of quantitative modeling capabilities, with particular relevance to commodities and energy derivatives markets.

## Key Features

### Quantitative Analysis
- **Advanced Option Pricing**: Implementation of the Black-Scholes-Merton model for European options
- **Greeks Analysis**: Comprehensive calculation of Delta, Gamma, Theta, Vega, and Rho
- **Sensitivity Analysis**: Multi-dimensional analysis of option price sensitivity to market parameters
- **Put-Call Parity**: Validation of fundamental option pricing relationships

### Technical Implementation
- **Numerical Methods**: Efficient implementation using vectorized NumPy operations
- **Database Integration**: SQLite backend for historical calculation tracking
- **Real-time Computation**: Optimized calculations for interactive parameter adjustment
- **Visualization**: Advanced plotting using Plotly for interactive analysis

### Professional Applications
- **Risk Management**: Greeks analysis for portfolio hedging
- **Trading Strategies**: Analysis of option price sensitivity for strategy development
- **Market Making**: Understanding bid-ask spread dynamics through Greeks
- **Commodities Trading**: Special focus on energy and agricultural derivatives

## Technical Stack
- **Core**: Python 3.8+
- **Numerical Computing**: NumPy, SciPy
- **Data Management**: SQLAlchemy, pandas
- **Visualization**: Plotly, Matplotlib
- **Web Framework**: Streamlit
- **Development Tools**: pytest, mypy, black, flake8

## Mathematical Rigor
- Implementation of the Black-Scholes partial differential equation
- Proper handling of edge cases and numerical stability
- Validation against known analytical solutions
- Monte Carlo simulation capabilities (planned)

## Creator
Created by Joshua Lee as a demonstration of quantitative finance expertise and full-stack development capabilities. 
This project showcases:
- Strong mathematical foundation in derivatives pricing
- Advanced programming skills in Python
- Understanding of financial markets and trading
- Ability to create production-ready quantitative tools

## Future Development
- Extension to American options
- Implementation of local volatility models
- Addition of commodity-specific features
- Integration with market data APIs
- Advanced risk metrics calculation
""")

st.markdown("""
<style>
.about-section {
    background-color: rgba(49, 51, 63, 0.7);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True) 