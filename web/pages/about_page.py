import streamlit as st

st.set_page_config(
    page_title="About - Black-Scholes Model",
    layout="wide"
)

st.title("About This Application")

st.markdown("""
## Overview
This interactive application demonstrates the Black-Scholes option pricing model, a fundamental tool in quantitative finance. 
The app allows users to explore how option prices change with different market parameters and visualize these relationships 
through interactive heatmaps.

## Features
- **Interactive Option Pricing**: Calculate European call and put option prices using the Black-Scholes model
- **Parameter Sensitivity Analysis**: Visualize how option prices change with different spot prices and volatility levels
- **Educational Tools**: Step-by-step calculation breakdowns and detailed explanations
- **Real-time Visualization**: Interactive heatmaps showing price sensitivity to market parameters

## Technical Implementation
- Built with Python using Streamlit for the frontend
- Utilizes NumPy and SciPy for numerical computations
- Implements the Black-Scholes model with proper mathematical notation
- Features responsive design and interactive visualizations

## Creator
Created by Joshua Lee as a demonstration of quantitative finance concepts and interactive web application development.
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