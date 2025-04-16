import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime
import uuid
import time
from scipy.stats import norm
import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import from the src package
try:
    from src.black_scholes import (
        call_price,
        put_price,
        call_delta,
        put_delta,
        gamma,
        call_theta,
        put_theta,
        vega,
        call_rho,
        put_rho
    )
except ImportError as e:
    logger.error(f"Failed to import Black-Scholes functions: {e}")
    st.error("Failed to import required modules. Please check the installation.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Black-Scholes Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .description {
        background-color: rgba(49, 51, 63, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .heatmap-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
    }
    .creator-info {
        margin-top: 20px;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .linkedin-icon {
        color: #0077b5;
        font-size: 1.2rem;
    }
    .calculation {
        background-color: rgba(49, 51, 63, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .pnl-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .pnl-negative {
        color: #F44336;
        font-weight: bold;
    }
    .greek-card {
        background-color: rgba(49, 51, 63, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
    }
    .greek-value {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .greek-name {
        font-size: 1.1rem;
        color: #4CAF50;
    }
    .greek-description {
        font-size: 0.9rem;
        color: #B0BEC5;
    }
    .positive-value {
        color: #4CAF50;
    }
    .negative-value {
        color: #F44336;
    }
    .error-message {
        color: #ff4b4b;
        background-color: #ffebee;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-message {
        color: #4caf50;
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

# Initialize session state for input values
if 'input_values' not in st.session_state:
    st.session_state.input_values = {
        'asset_price': 100.00,
        'strike_price': 100.00,
        'time_to_maturity': 1.00,
        'volatility': 0.20,
        'risk_free_rate': 0.05,
        'min_spot_price': 80.00,
        'max_spot_price': 120.00,
        'min_volatility': 0.10,
        'max_volatility': 0.30,
        'call_purchase_price': 0.0,
        'put_purchase_price': 0.0
    }

# Initialize database with error handling
def init_db():
    try:
        conn = sqlite3.connect('black_scholes.db')
        c = conn.cursor()
        
        # Drop existing tables if they exist
        c.execute('DROP TABLE IF EXISTS outputs')
        c.execute('DROP TABLE IF EXISTS inputs')
        
        # Create inputs table
        c.execute('''
        CREATE TABLE IF NOT EXISTS inputs (
            calculation_id TEXT PRIMARY KEY,
            timestamp TEXT,
            asset_price REAL,
            strike_price REAL,
            time_to_maturity REAL,
            volatility REAL,
            risk_free_rate REAL,
            min_spot_price REAL,
            max_spot_price REAL,
            min_volatility REAL,
            max_volatility REAL,
            call_purchase_price REAL,
            put_purchase_price REAL
        )
        ''')
        
        # Create outputs table
        c.execute('''
        CREATE TABLE IF NOT EXISTS outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            calculation_id TEXT,
            spot_price REAL,
            volatility REAL,
            call_price REAL,
            put_price REAL,
            call_pnl REAL,
            put_pnl REAL,
            FOREIGN KEY (calculation_id) REFERENCES inputs (calculation_id)
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        st.error("Failed to initialize database. Some features may not be available.")

# Save calculation to database with error handling
def save_to_db(inputs, outputs_df):
    try:
        conn = sqlite3.connect('black_scholes.db')
        c = conn.cursor()
        
        calculation_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save inputs
        c.execute('''
        INSERT INTO inputs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            calculation_id,
            timestamp,
            inputs['asset_price'],
            inputs['strike_price'],
            inputs['time_to_maturity'],
            inputs['volatility'],
            inputs['risk_free_rate'],
            inputs['min_spot_price'],
            inputs['max_spot_price'],
            inputs['min_volatility'],
            inputs['max_volatility'],
            inputs['call_purchase_price'],
            inputs['put_purchase_price']
        ))
        
        # Save outputs
        for _, row in outputs_df.iterrows():
            c.execute('''
            INSERT INTO outputs (calculation_id, spot_price, volatility, call_price, put_price, call_pnl, put_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                calculation_id,
                row['spot_price'],
                row['volatility'],
                row['call_price'],
                row['put_price'],
                row['call_pnl'] if 'call_pnl' in row else 0.0,
                row['put_pnl'] if 'put_pnl' in row else 0.0
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Calculation saved successfully with ID: {calculation_id}")
        return calculation_id
    except Exception as e:
        logger.error(f"Failed to save calculation to database: {e}")
        st.error("Failed to save calculation. Some features may not be available.")
        return None

# Plot heatmap
def plot_heatmap(prices, volatilities, spot_prices, title, cmap="viridis", is_pnl=False):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For PnL heatmaps, use a diverging colormap centered at 0
    if is_pnl:
        # Find the absolute maximum value for symmetric color scaling
        abs_max = max(abs(np.min(prices)), abs(np.max(prices)))
        vmin, vmax = -abs_max, abs_max
        cmap = "RdYlGn"  # Red for negative, green for positive
    else:
        vmin, vmax = None, None
    
    # Create heatmap with better formatting
    sns.heatmap(
        prices, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap,
        xticklabels=[f"{x:.2f}" for x in spot_prices],
        yticklabels=[f"{y:.2f}" for y in volatilities[::-1]],  # Reverse for better visualization
        ax=ax,
        cbar_kws={'label': 'Option Price ($)' if not is_pnl else 'P&L ($)'},
        vmin=vmin,
        vmax=vmax,
        center=0 if is_pnl else None
    )
    
    # Set labels and title with better formatting
    ax.set_xlabel("Spot Price ($)", fontsize=12, labelpad=10)
    ax.set_ylabel("Volatility (σ)", fontsize=12, labelpad=10)
    ax.set_title(f"{title} {'P&L' if is_pnl else 'Price'} Heatmap", fontsize=14, pad=20)
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig

# Main application
def main():
    # Initialize database
    init_db()
    
    st.markdown('<h1 class="main-header">Black-Scholes Model</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown(
        '<div class="description">'
        'A comprehensive implementation of the Black-Scholes option pricing model with real-time '
        'calculations, visualizations, and Greeks analysis. This tool helps in understanding '
        'option pricing dynamics and risk measures.'
        '</div>',
        unsafe_allow_html=True
    )

    # Create columns for the layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<h2 class="sub-header">Parameters</h2>', unsafe_allow_html=True)
        
        def update_asset_price():
            st.session_state.input_values['asset_price'] = st.session_state.asset_price_input
        asset_price = st.number_input(
            "Asset Price", 
            min_value=1.0, 
            value=st.session_state.input_values['asset_price'], 
            step=1.0,
            key="asset_price_input",
            on_change=update_asset_price
        )
        
        def update_strike_price():
            st.session_state.input_values['strike_price'] = st.session_state.strike_price_input
        strike_price = st.number_input(
            "Strike Price", 
            min_value=1.0, 
            value=st.session_state.input_values['strike_price'], 
            step=1.0,
            key="strike_price_input",
            on_change=update_strike_price
        )
        
        def update_time_to_maturity():
            st.session_state.input_values['time_to_maturity'] = st.session_state.time_to_maturity_input
        time_to_maturity = st.number_input(
            "Time to Maturity (Years)", 
            min_value=0.01, 
            value=st.session_state.input_values['time_to_maturity'], 
            step=0.25,
            key="time_to_maturity_input",
            on_change=update_time_to_maturity
        )
        
        def update_volatility():
            st.session_state.input_values['volatility'] = st.session_state.volatility_input
        volatility = st.number_input(
            "Volatility (σ)", 
            min_value=0.01, 
            value=st.session_state.input_values['volatility'], 
            step=0.01,
            key="volatility_input",
            on_change=update_volatility
        )
        
        def update_risk_free_rate():
            st.session_state.input_values['risk_free_rate'] = st.session_state.risk_free_rate_input
        risk_free_rate = st.number_input(
            "Risk-Free Interest Rate", 
            min_value=0.0, 
            value=st.session_state.input_values['risk_free_rate'], 
            step=0.01,
            key="risk_free_rate_input",
            on_change=update_risk_free_rate
        )
        
        st.markdown("---")
        
        st.markdown('<div class="sidebar-header">Heatmap Parameters</div>', unsafe_allow_html=True)
        
        def update_min_spot_price():
            st.session_state.input_values['min_spot_price'] = st.session_state.min_spot_price_input
        min_spot_price = st.number_input(
            "Min Spot Price", 
            min_value=1.0, 
            value=st.session_state.input_values['min_spot_price'], 
            step=5.0,
            key="min_spot_price_input",
            on_change=update_min_spot_price
        )
        
        def update_max_spot_price():
            st.session_state.input_values['max_spot_price'] = st.session_state.max_spot_price_input
        max_spot_price = st.number_input(
            "Max Spot Price", 
            min_value=min_spot_price + 1.0, 
            value=st.session_state.input_values['max_spot_price'], 
            step=5.0,
            key="max_spot_price_input",
            on_change=update_max_spot_price
        )
        
        def update_min_volatility():
            st.session_state.input_values['min_volatility'] = st.session_state.min_volatility_input
        min_volatility = st.number_input(
            "Min Volatility", 
            min_value=0.01, 
            value=st.session_state.input_values['min_volatility'], 
            step=0.05,
            key="min_volatility_input",
            on_change=update_min_volatility
        )
        
        def update_max_volatility():
            st.session_state.input_values['max_volatility'] = st.session_state.max_volatility_input
        max_volatility = st.number_input(
            "Max Volatility", 
            min_value=min_volatility + 0.01, 
            value=st.session_state.input_values['max_volatility'], 
            step=0.05,
            key="max_volatility_input",
            on_change=update_max_volatility
        )
        
        # P&L Analysis inputs
        st.markdown("---")
        st.markdown('<div class="sidebar-header">P&L Analysis</div>', unsafe_allow_html=True)
        
        def update_call_purchase_price():
            st.session_state.input_values['call_purchase_price'] = st.session_state.call_purchase_price_input
        call_purchase_price = st.number_input(
            "Call Purchase Price", 
            min_value=0.0, 
            value=st.session_state.input_values['call_purchase_price'], 
            step=1.0,
            key="call_purchase_price_input",
            on_change=update_call_purchase_price
        )
        
        def update_put_purchase_price():
            st.session_state.input_values['put_purchase_price'] = st.session_state.put_purchase_price_input
        put_purchase_price = st.number_input(
            "Put Purchase Price", 
            min_value=0.0, 
            value=st.session_state.input_values['put_purchase_price'], 
            step=1.0,
            key="put_purchase_price_input",
            on_change=update_put_purchase_price
        )
        
        # Calculate button
        calculate = st.button("Calculate", key="calculate")
        
        # Creator information
        st.markdown("---")
        st.markdown("""
        <div class="creator-info">
            Created by: Joshua Lee
            <a href="https://www.linkedin.com/in/joshhlee614/" target="_blank">
                <i class="fab fa-linkedin linkedin-icon"></i>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">Option Prices & Greeks</h2>', unsafe_allow_html=True)
        
        # Calculate option prices and Greeks
        if all(v > 0 for v in [
            st.session_state.input_values['asset_price'],
            st.session_state.input_values['strike_price'],
            st.session_state.input_values['time_to_maturity'],
            st.session_state.input_values['volatility']
        ]):
            with st.spinner('Calculating...'):
                S = st.session_state.input_values['asset_price']
                K = st.session_state.input_values['strike_price']
                T = st.session_state.input_values['time_to_maturity']
                sigma = st.session_state.input_values['volatility']
                r = st.session_state.input_values['risk_free_rate']
                
                # Calculate prices
                call = call_price(S, K, T, r, sigma)
                put = put_price(S, K, T, r, sigma)
                
                # Calculate Greeks
                call_delta_val = call_delta(S, K, T, r, sigma)
                put_delta_val = put_delta(S, K, T, r, sigma)
                gamma_val = gamma(S, K, T, r, sigma)
                call_theta_val = call_theta(S, K, T, r, sigma)
                put_theta_val = put_theta(S, K, T, r, sigma)
                vega_val = vega(S, K, T, r, sigma)
                call_rho_val = call_rho(S, K, T, r, sigma)
                put_rho_val = put_rho(S, K, T, r, sigma)
                
                # Display prices
                price_col1, price_col2 = st.columns(2)
                with price_col1:
                    st.markdown(
                        f'<div class="calculation">'
                        f'<h3>Call Option Price</h3>'
                        f'<p style="font-size: 1.5rem;">${call:.2f}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with price_col2:
                    st.markdown(
                        f'<div class="calculation">'
                        f'<h3>Put Option Price</h3>'
                        f'<p style="font-size: 1.5rem;">${put:.2f}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Display Greeks in a grid
                st.markdown('<h3 class="sub-header">Greeks Analysis</h3>', unsafe_allow_html=True)
                greek_col1, greek_col2, greek_col3, greek_col4 = st.columns(4)
                
                with greek_col1:
                    st.markdown(
                        f'<div class="greek-card">'
                        f'<div class="greek-name">Δ Delta</div>'
                        f'<div class="greek-value">'
                        f'Call: <span class="{"positive-value" if call_delta_val > 0 else "negative-value"}">{call_delta_val:.4f}</span><br>'
                        f'Put: <span class="{("positive-value" if put_delta_val > 0 else "negative-value")}">{put_delta_val:.4f}</span>'
                        f'</div>'
                        f'<div class="greek-description">Price sensitivity to underlying asset</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with greek_col2:
                    st.markdown(
                        f'<div class="greek-card">'
                        f'<div class="greek-name">Γ Gamma</div>'
                        f'<div class="greek-value">'
                        f'<span class="positive-value">{gamma_val:.4f}</span>'
                        f'</div>'
                        f'<div class="greek-description">Delta sensitivity to underlying asset</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with greek_col3:
                    st.markdown(
                        f'<div class="greek-card">'
                        f'<div class="greek-name">Θ Theta</div>'
                        f'<div class="greek-value">'
                        f'Call: <span class="{("positive-value" if call_theta_val > 0 else "negative-value")}">${call_theta_val:.4f}</span><br>'
                        f'Put: <span class="{("positive-value" if put_theta_val > 0 else "negative-value")}">${put_theta_val:.4f}</span>'
                        f'</div>'
                        f'<div class="greek-description">Price sensitivity to time decay (per day)</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with greek_col4:
                    st.markdown(
                        f'<div class="greek-card">'
                        f'<div class="greek-name">ν Vega</div>'
                        f'<div class="greek-value">'
                        f'<span class="positive-value">${vega_val:.4f}</span>'
                        f'</div>'
                        f'<div class="greek-description">Price sensitivity to volatility (per 1% change)</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Add Rho in a new row
                rho_col1, rho_col2, rho_col3, rho_col4 = st.columns(4)
                with rho_col1:
                    st.markdown(
                        f'<div class="greek-card">'
                        f'<div class="greek-name">ρ Rho</div>'
                        f'<div class="greek-value">'
                        f'Call: <span class="{("positive-value" if call_rho_val > 0 else "negative-value")}">${call_rho_val:.4f}</span><br>'
                        f'Put: <span class="{("positive-value" if put_rho_val > 0 else "negative-value")}">${put_rho_val:.4f}</span>'
                        f'</div>'
                        f'<div class="greek-description">Price sensitivity to interest rate (per 1% change)</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # Add Greeks Sensitivity Analysis Section
                st.markdown('<h3 class="sub-header">Greeks Sensitivity Analysis</h3>', unsafe_allow_html=True)
                
                # Create tabs for different sensitivity analyses
                tab1, tab2, tab3, tab4 = st.tabs(["Stock Price", "Time to Maturity", "Volatility", "Interest Rate"])
                
                with tab1:
                    st.markdown('<h4>Delta & Gamma vs. Stock Price</h4>', unsafe_allow_html=True)
                    # Generate stock price range
                    S_range = np.linspace(max(1, S * 0.5), S * 1.5, 100)
                    # Calculate deltas and gamma
                    call_deltas = [call_delta(s, K, T, r, sigma) for s in S_range]
                    put_deltas = [put_delta(s, K, T, r, sigma) for s in S_range]
                    gammas = [gamma(s, K, T, r, sigma) for s in S_range]
                    
                    # Create the plot
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    ax2 = ax1.twinx()
                    
                    ax1.plot(S_range, call_deltas, 'g-', label='Call Delta')
                    ax1.plot(S_range, put_deltas, 'r-', label='Put Delta')
                    ax2.plot(S_range, gammas, 'b--', label='Gamma')
                    
                    ax1.set_xlabel('Stock Price')
                    ax1.set_ylabel('Delta')
                    ax2.set_ylabel('Gamma')
                    ax1.legend(loc='upper left')
                    ax2.legend(loc='upper right')
                    ax1.axvline(x=S, color='k', linestyle='--', alpha=0.3)
                    ax1.grid(True)
                    
                    st.pyplot(fig)
                
                with tab2:
                    st.markdown('<h4>Theta vs. Time to Maturity</h4>', unsafe_allow_html=True)
                    # Generate time range
                    T_range = np.linspace(0.01, T * 2, 100)
                    # Calculate thetas
                    call_thetas = [call_theta(S, K, t, r, sigma) for t in T_range]
                    put_thetas = [put_theta(S, K, t, r, sigma) for t in T_range]
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(T_range, call_thetas, 'g-', label='Call Theta')
                    ax.plot(T_range, put_thetas, 'r-', label='Put Theta')
                    
                    ax.set_xlabel('Time to Maturity (Years)')
                    ax.set_ylabel('Theta ($ per day)')
                    ax.legend()
                    ax.axvline(x=T, color='k', linestyle='--', alpha=0.3)
                    ax.grid(True)
                    
                    st.pyplot(fig)
                
                with tab3:
                    st.markdown('<h4>Vega vs. Volatility</h4>', unsafe_allow_html=True)
                    # Generate volatility range
                    sigma_range = np.linspace(0.01, sigma * 2, 100)
                    # Calculate vegas
                    vegas = [vega(S, K, T, r, s) for s in sigma_range]
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(sigma_range, vegas, 'b-', label='Vega')
                    
                    ax.set_xlabel('Volatility')
                    ax.set_ylabel('Vega ($ per 1% vol change)')
                    ax.legend()
                    ax.axvline(x=sigma, color='k', linestyle='--', alpha=0.3)
                    ax.grid(True)
                    
                    st.pyplot(fig)
                
                with tab4:
                    st.markdown('<h4>Rho vs. Interest Rate</h4>', unsafe_allow_html=True)
                    # Generate interest rate range
                    r_range = np.linspace(0, r * 2, 100)
                    # Calculate rhos
                    call_rhos = [call_rho(S, K, T, rate, sigma) for rate in r_range]
                    put_rhos = [put_rho(S, K, T, rate, sigma) for rate in r_range]
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(r_range, call_rhos, 'g-', label='Call Rho')
                    ax.plot(r_range, put_rhos, 'r-', label='Put Rho')
                    
                    ax.set_xlabel('Interest Rate')
                    ax.set_ylabel('Rho ($ per 1% rate change)')
                    ax.legend()
                    ax.axvline(x=r, color='k', linestyle='--', alpha=0.3)
                    ax.grid(True)
                    
                    st.pyplot(fig)

    if calculate:
        # Show a spinner while calculating
        with st.spinner("Calculating option prices..."):
            # Calculate option prices using Black-Scholes
            call_price_value = call_price(asset_price, strike_price, time_to_maturity, risk_free_rate, volatility)
            put_price_value = put_price(asset_price, strike_price, time_to_maturity, risk_free_rate, volatility)
            
            # Calculate P&L if purchase prices are provided
            call_pnl = call_price_value - call_purchase_price if call_purchase_price > 0 else 0
            put_pnl = put_price_value - put_purchase_price if put_purchase_price > 0 else 0
            
            # Display basic results
            st.markdown('<div class="sub-header">Options Price - Interactive Heatmap</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Call Option Price", f"${call_price_value:.2f}")
                if call_purchase_price > 0:
                    st.markdown(
                        f"P&L: <span class=\"{'pnl-positive' if call_pnl >= 0 else 'pnl-negative'}\">${call_pnl:.2f}</span>", 
                        unsafe_allow_html=True
                    )
            with col2:
                st.metric("Put Option Price", f"${put_price_value:.2f}")
                if put_purchase_price > 0:
                    st.markdown(
                        f"P&L: <span class=\"{'pnl-positive' if put_pnl >= 0 else 'pnl-negative'}\">${put_pnl:.2f}</span>", 
                        unsafe_allow_html=True
                    )
            
            # Generate heatmap data using Black-Scholes
            spot_prices = np.linspace(min_spot_price, max_spot_price, 10)
            volatilities = np.linspace(min_volatility, max_volatility, 10)
            
            # Create arrays for heatmap data
            call_prices = np.zeros((len(volatilities), len(spot_prices)))
            put_prices = np.zeros((len(volatilities), len(spot_prices)))
            call_pnl_matrix = np.zeros((len(volatilities), len(spot_prices)))
            put_pnl_matrix = np.zeros((len(volatilities), len(spot_prices)))
            
            for i, vol in enumerate(volatilities):
                for j, spot in enumerate(spot_prices):
                    call_prices[i, j] = call_price(spot, strike_price, time_to_maturity, risk_free_rate, vol)
                    put_prices[i, j] = put_price(spot, strike_price, time_to_maturity, risk_free_rate, vol)
                    
                    # Calculate P&L if purchase prices are provided
                    if call_purchase_price > 0:
                        call_pnl_matrix[i, j] = call_prices[i, j] - call_purchase_price
                    if put_purchase_price > 0:
                        put_pnl_matrix[i, j] = put_prices[i, j] - put_purchase_price
            
            # Create DataFrame for database storage
            data = []
            for i, vol in enumerate(volatilities):
                for j, spot in enumerate(spot_prices):
                    data_row = {
                        'spot_price': spot_prices[j],
                        'volatility': volatilities[i],
                        'call_price': call_prices[i, j],
                        'put_price': put_prices[i, j]
                    }
                    
                    if call_purchase_price > 0:
                        data_row['call_pnl'] = call_pnl_matrix[i, j]
                    if put_purchase_price > 0:
                        data_row['put_pnl'] = put_pnl_matrix[i, j]
                        
                    data.append(data_row)
            
            outputs_df = pd.DataFrame(data)
            
            # Save to database
            inputs = {
                'asset_price': asset_price,
                'strike_price': strike_price,
                'time_to_maturity': time_to_maturity,
                'volatility': volatility,
                'risk_free_rate': risk_free_rate,
                'min_spot_price': min_spot_price,
                'max_spot_price': max_spot_price,
                'min_volatility': min_volatility,
                'max_volatility': max_volatility,
                'call_purchase_price': call_purchase_price,
                'put_purchase_price': put_purchase_price
            }
            
            calculation_id = save_to_db(inputs, outputs_df)
            
            # Display heatmaps in a grid layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="sub-header">Call Price Heatmap</div>', unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)
                    call_fig = plot_heatmap(call_prices, volatilities, spot_prices, "Call", cmap="viridis")
                    st.pyplot(call_fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display Call P&L heatmap if purchase price is provided
                if call_purchase_price > 0:
                    st.markdown('<div class="sub-header">Call P&L Heatmap</div>', unsafe_allow_html=True)
                    with st.container():
                        st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)
                        call_pnl_fig = plot_heatmap(call_pnl_matrix, volatilities, spot_prices, "Call", is_pnl=True)
                        st.pyplot(call_pnl_fig)
                        st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="sub-header">Put Price Heatmap</div>', unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)
                    put_fig = plot_heatmap(put_prices, volatilities, spot_prices, "Put", cmap="magma")
                    st.pyplot(put_fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display Put P&L heatmap if purchase price is provided
                if put_purchase_price > 0:
                    st.markdown('<div class="sub-header">Put P&L Heatmap</div>', unsafe_allow_html=True)
                    with st.container():
                        st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)
                        put_pnl_fig = plot_heatmap(put_pnl_matrix, volatilities, spot_prices, "Put", is_pnl=True)
                        st.pyplot(put_pnl_fig)
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Display calculation ID for reference
            st.info(f"Calculation ID: {calculation_id}")

if __name__ == "__main__":
    main()