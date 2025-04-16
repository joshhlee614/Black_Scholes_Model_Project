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
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import json

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

# Initialize SQLAlchemy
Base = declarative_base()

class Calculation(Base):
    __tablename__ = 'calculations'
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    asset_price = Column(Float)
    strike_price = Column(Float)
    time_to_maturity = Column(Float)
    volatility = Column(Float)
    risk_free_rate = Column(Float)
    min_spot_price = Column(Float)
    max_spot_price = Column(Float)
    min_volatility = Column(Float)
    max_volatility = Column(Float)
    call_purchase_price = Column(Float)
    put_purchase_price = Column(Float)
    greeks = Column(String)  # Store Greeks as JSON
    
    # Relationship with results
    results = relationship("CalculationResult", back_populates="calculation")

class CalculationResult(Base):
    __tablename__ = 'results'
    
    id = Column(Integer, primary_key=True)
    calculation_id = Column(String, ForeignKey('calculations.id'))
    spot_price = Column(Float)
    volatility = Column(Float)
    call_price = Column(Float)
    put_price = Column(Float)
    call_pnl = Column(Float)
    put_pnl = Column(Float)
    
    # Relationship with calculation
    calculation = relationship("Calculation", back_populates="results")

def init_db():
    """Initialize the database and return a session."""
    try:
        # Create the database directory if it doesn't exist
        db_path = os.path.join(os.getcwd(), 'black_scholes.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create engine and session
        engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        logger.info("Database initialized successfully")
        return session
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        st.error("Failed to initialize database. Some features may not be available.")
        return None

def save_to_db(session, inputs, outputs_df, greeks):
    """Save calculation results to the database."""
    try:
        if session is None:
            raise ValueError("Database session is not initialized")
            
        # Validate inputs
        if not all(isinstance(v, (int, float)) and v > 0 for v in [
            inputs['asset_price'],
            inputs['strike_price'],
            inputs['time_to_maturity'],
            inputs['volatility'],
            inputs['risk_free_rate']
        ]):
            raise ValueError("All input parameters must be positive numbers")
            
        calculation_id = str(uuid.uuid4())
        
        # Create calculation record
        calculation = Calculation(
            id=calculation_id,
            timestamp=datetime.now(),
            asset_price=inputs['asset_price'],
            strike_price=inputs['strike_price'],
            time_to_maturity=inputs['time_to_maturity'],
            volatility=inputs['volatility'],
            risk_free_rate=inputs['risk_free_rate'],
            min_spot_price=inputs['min_spot_price'],
            max_spot_price=inputs['max_spot_price'],
            min_volatility=inputs['min_volatility'],
            max_volatility=inputs['max_volatility'],
            call_purchase_price=inputs.get('call_purchase_price', 0.0),
            put_purchase_price=inputs.get('put_purchase_price', 0.0),
            greeks=json.dumps(greeks)
        )
        
        # Create results records
        results = []
        for _, row in outputs_df.iterrows():
            result = CalculationResult(
                calculation_id=calculation_id,
                spot_price=row['spot_price'],
                volatility=row['volatility'],
                call_price=row['call_price'],
                put_price=row['put_price'],
                call_pnl=row.get('call_pnl'),
                put_pnl=row.get('put_pnl')
            )
            results.append(result)
        
        # Save to database
        session.add(calculation)
        session.add_all(results)
        session.commit()
        
        logger.info(f"Calculation saved successfully with ID: {calculation_id}")
        return calculation_id
    except Exception as e:
        logger.error(f"Failed to save calculation to database: {e}")
        session.rollback()  # Rollback in case of error
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

def get_historical_calculations(session):
    """Retrieve historical calculations from the database."""
    try:
        calculations = session.query(Calculation).order_by(Calculation.timestamp.desc()).all()
        return calculations
    except Exception as e:
        logger.error(f"Failed to retrieve historical calculations: {e}")
        return []

def get_calculation_details(session, calculation_id):
    """Retrieve detailed results for a specific calculation."""
    try:
        calculation = session.query(Calculation).filter(Calculation.id == calculation_id).first()
        results = session.query(CalculationResult).filter(CalculationResult.calculation_id == calculation_id).all()
        return calculation, results
    except Exception as e:
        logger.error(f"Failed to retrieve calculation details: {e}")
        return None, []

# Main application
def main():
    # Initialize database
    session = init_db()
    
    if session is None:
        st.error("Failed to initialize database. Some features may not be available.")
        return
    
    try:
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
                
                # Add parameter range controls
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="sidebar-header">Analysis Range</div>', unsafe_allow_html=True)
                    price_range = st.slider(
                        "Stock Price Range (%)",
                        min_value=50,
                        max_value=200,
                        value=(80, 120),
                        step=5,
                        help="Percentage range around current stock price"
                    )
                    time_range = st.slider(
                        "Time Range (years)",
                        min_value=0.0,
                        max_value=2.0,
                        value=(0.0, 1.0),
                        step=0.1,
                        help="Range of time to maturity"
                    )
                
                with col2:
                    st.markdown('<div class="sidebar-header">Volatility & Rate Range</div>', unsafe_allow_html=True)
                    vol_range = st.slider(
                        "Volatility Range (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=(0.0, 50.0),
                        step=5.0,
                        help="Range of volatility values"
                    )
                    rate_range = st.slider(
                        "Interest Rate Range (%)",
                        min_value=0.0,
                        max_value=20.0,
                        value=(0.0, 10.0),
                        step=0.5,
                        help="Range of risk-free interest rates"
                    )
                
                # Create tabs for different sensitivity analyses
                tab1, tab2, tab3, tab4 = st.tabs(["Stock Price", "Time to Maturity", "Volatility", "Interest Rate"])
                
                with tab1:
                    st.markdown('<h4>Delta & Gamma vs. Stock Price</h4>', unsafe_allow_html=True)
                    # Generate stock price range based on current price and selected range
                    S_min = S * (price_range[0] / 100)
                    S_max = S * (price_range[1] / 100)
                    S_range = np.linspace(S_min, S_max, 100)
                    
                    # Calculate deltas and gamma
                    call_deltas = [call_delta(s, K, T, r, sigma) for s in S_range]
                    put_deltas = [put_delta(s, K, T, r, sigma) for s in S_range]
                    gammas = [gamma(s, K, T, r, sigma) for s in S_range]
                    
                    # Create the plot with enhanced styling
                    fig, ax1 = plt.subplots(figsize=(12, 8))
                    ax2 = ax1.twinx()
                    
                    # Plot with better styling
                    ax1.plot(S_range, call_deltas, 'g-', label='Call Delta', linewidth=2)
                    ax1.plot(S_range, put_deltas, 'r-', label='Put Delta', linewidth=2)
                    ax2.plot(S_range, gammas, 'b--', label='Gamma', linewidth=2)
                    
                    # Add vertical line at current price
                    ax1.axvline(x=S, color='k', linestyle='--', alpha=0.3, label='Current Price')
                    
                    # Enhanced styling
                    ax1.set_xlabel('Stock Price ($)', fontsize=12)
                    ax1.set_ylabel('Delta', fontsize=12, color='g')
                    ax2.set_ylabel('Gamma', fontsize=12, color='b')
                    
                    # Add grid and legend
                    ax1.grid(True, linestyle='--', alpha=0.3)
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                    
                    # Add title and adjust layout
                    plt.title('Delta and Gamma Sensitivity to Stock Price', fontsize=14, pad=20)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Add explanation
                    st.markdown("""
                    <div class="description">
                    <p><strong>Delta</strong> shows how the option price changes with the stock price. 
                    Call deltas range from 0 to 1, while put deltas range from -1 to 0.</p>
                    <p><strong>Gamma</strong> shows how delta changes with the stock price. 
                    It's highest for at-the-money options and decreases as options move in or out of the money.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with tab2:
                    st.markdown('<h4>Theta vs. Time to Maturity</h4>', unsafe_allow_html=True)
                    # Generate time range
                    T_min, T_max = time_range
                    T_range = np.linspace(T_min, T_max, 100)
                    
                    # Calculate thetas
                    call_thetas = [call_theta(S, K, t, r, sigma) for t in T_range]
                    put_thetas = [put_theta(S, K, t, r, sigma) for t in T_range]
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.plot(T_range, call_thetas, 'g-', label='Call Theta', linewidth=2)
                    ax.plot(T_range, put_thetas, 'r-', label='Put Theta', linewidth=2)
                    
                    # Add vertical line at current time
                    ax.axvline(x=T, color='k', linestyle='--', alpha=0.3, label='Current Time')
                    
                    # Enhanced styling
                    ax.set_xlabel('Time to Maturity (Years)', fontsize=12)
                    ax.set_ylabel('Theta ($ per day)', fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.legend(loc='upper right')
                    
                    plt.title('Theta Sensitivity to Time to Maturity', fontsize=14, pad=20)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Add explanation
                    st.markdown("""
                    <div class="description">
                    <p><strong>Theta</strong> measures the time decay of an option's value. 
                    It's typically negative, showing how much value an option loses each day.</p>
                    <p>At-the-money options have the highest theta (most negative), 
                    while deep in/out-of-the-money options have theta closer to zero.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with tab3:
                    st.markdown('<h4>Vega vs. Volatility</h4>', unsafe_allow_html=True)
                    # Generate volatility range
                    vol_min, vol_max = vol_range
                    sigma_range = np.linspace(vol_min/100, vol_max/100, 100)
                    
                    # Calculate vegas
                    vegas = [vega(S, K, T, r, s) for s in sigma_range]
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.plot(sigma_range, vegas, 'b-', label='Vega', linewidth=2)
                    
                    # Add vertical line at current volatility
                    ax.axvline(x=sigma, color='k', linestyle='--', alpha=0.3, label='Current Volatility')
                    
                    # Enhanced styling
                    ax.set_xlabel('Volatility', fontsize=12)
                    ax.set_ylabel('Vega ($ per 1% vol change)', fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.legend(loc='upper right')
                    
                    plt.title('Vega Sensitivity to Volatility', fontsize=14, pad=20)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Add explanation
                    st.markdown("""
                    <div class="description">
                    <p><strong>Vega</strong> measures sensitivity to changes in volatility. 
                    It's always positive, showing how much an option's value increases with a 1% increase in volatility.</p>
                    <p>At-the-money options have the highest vega, while deep in/out-of-the-money options have lower vega.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with tab4:
                    st.markdown('<h4>Rho vs. Interest Rate</h4>', unsafe_allow_html=True)
                    # Generate interest rate range
                    rate_min, rate_max = rate_range
                    r_range = np.linspace(rate_min/100, rate_max/100, 100)
                    
                    # Calculate rhos
                    call_rhos = [call_rho(S, K, T, rate, sigma) for rate in r_range]
                    put_rhos = [put_rho(S, K, T, rate, sigma) for rate in r_range]
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.plot(r_range, call_rhos, 'g-', label='Call Rho', linewidth=2)
                    ax.plot(r_range, put_rhos, 'r-', label='Put Rho', linewidth=2)
                    
                    # Add vertical line at current rate
                    ax.axvline(x=r, color='k', linestyle='--', alpha=0.3, label='Current Rate')
                    
                    # Enhanced styling
                    ax.set_xlabel('Interest Rate', fontsize=12)
                    ax.set_ylabel('Rho ($ per 1% rate change)', fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.legend(loc='upper right')
                    
                    plt.title('Rho Sensitivity to Interest Rate', fontsize=14, pad=20)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Add explanation
                    st.markdown("""
                    <div class="description">
                    <p><strong>Rho</strong> measures sensitivity to changes in interest rates. 
                    Call options have positive rho (value increases with rates), 
                    while put options have negative rho (value decreases with rates).</p>
                    <p>Rho is more significant for long-term options and less important for short-term options.</p>
                    </div>
                    """, unsafe_allow_html=True)

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
                
                greeks = {
                    'call_delta': call_delta_val,
                    'put_delta': put_delta_val,
                    'gamma': gamma_val,
                    'call_theta': call_theta_val,
                    'put_theta': put_theta_val,
                    'vega': vega_val,
                    'call_rho': call_rho_val,
                    'put_rho': put_rho_val
                }
                
                calculation_id = save_to_db(session, inputs, outputs_df, greeks)
                if calculation_id:
                    st.info(f"Calculation saved with ID: {calculation_id}")
                
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

        # Add historical calculations section
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Historical Calculations</h2>', unsafe_allow_html=True)
        
        # Get historical calculations
        calculations = get_historical_calculations(session)
        
        if calculations:
            # Create a DataFrame for display
            history_data = []
            for calc in calculations:
                greeks = json.loads(calc.greeks)
                history_data.append({
                    'Timestamp': calc.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'Asset Price': f"${calc.asset_price:.2f}",
                    'Strike Price': f"${calc.strike_price:.2f}",
                    'Time to Maturity': f"{calc.time_to_maturity:.2f} years",
                    'Volatility': f"{calc.volatility:.2%}",
                    'Risk-Free Rate': f"{calc.risk_free_rate:.2%}",
                    'Call Delta': f"{greeks['call_delta']:.4f}",
                    'Put Delta': f"{greeks['put_delta']:.4f}",
                    'Gamma': f"{greeks['gamma']:.4f}",
                    'ID': calc.id
                })
            
            history_df = pd.DataFrame(history_data)
            
            # Display the historical calculations
            st.dataframe(
                history_df,
                column_config={
                    "ID": st.column_config.TextColumn("ID", disabled=True),
                    "Timestamp": st.column_config.DatetimeColumn("Timestamp"),
                    "Asset Price": st.column_config.TextColumn("Asset Price"),
                    "Strike Price": st.column_config.TextColumn("Strike Price"),
                    "Time to Maturity": st.column_config.TextColumn("Time to Maturity"),
                    "Volatility": st.column_config.TextColumn("Volatility"),
                    "Risk-Free Rate": st.column_config.TextColumn("Risk-Free Rate"),
                    "Call Delta": st.column_config.TextColumn("Call Delta"),
                    "Put Delta": st.column_config.TextColumn("Put Delta"),
                    "Gamma": st.column_config.TextColumn("Gamma")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Add a selectbox to choose a calculation to view in detail
            selected_id = st.selectbox(
                "Select a calculation to view in detail",
                options=history_df['ID'].tolist(),
                format_func=lambda x: f"Calculation from {history_df[history_df['ID'] == x]['Timestamp'].iloc[0]}"
            )
            
            if selected_id:
                calculation, results = get_calculation_details(session, selected_id)
                if calculation and results:
                    # Display calculation details
                    st.markdown('<h3 class="sub-header">Calculation Details</h3>', unsafe_allow_html=True)
                    
                    # Create two columns for input parameters and Greeks
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<h4>Input Parameters</h4>', unsafe_allow_html=True)
                        st.markdown(f"""
                        - Asset Price: ${calculation.asset_price:.2f}
                        - Strike Price: ${calculation.strike_price:.2f}
                        - Time to Maturity: {calculation.time_to_maturity:.2f} years
                        - Volatility: {calculation.volatility:.2%}
                        - Risk-Free Rate: {calculation.risk_free_rate:.2%}
                        """)
                    
                    with col2:
                        st.markdown('<h4>Greeks</h4>', unsafe_allow_html=True)
                        greeks = json.loads(calculation.greeks)
                        st.markdown(f"""
                        - Call Delta: {greeks['call_delta']:.4f}
                        - Put Delta: {greeks['put_delta']:.4f}
                        - Gamma: {greeks['gamma']:.4f}
                        - Call Theta: ${greeks['call_theta']:.4f}/day
                        - Put Theta: ${greeks['put_theta']:.4f}/day
                        - Vega: ${greeks['vega']:.4f}/1% vol
                        - Call Rho: ${greeks['call_rho']:.4f}/1% rate
                        - Put Rho: ${greeks['put_rho']:.4f}/1% rate
                        """)
                    
                    # Display results in a table
                    st.markdown('<h4>Detailed Results</h4>', unsafe_allow_html=True)
                    results_data = []
                    for result in results:
                        results_data.append({
                            'Spot Price': f"${result.spot_price:.2f}",
                            'Volatility': f"{result.volatility:.2%}",
                            'Call Price': f"${result.call_price:.2f}",
                            'Put Price': f"${result.put_price:.2f}",
                            'Call P&L': f"${result.call_pnl:.2f}" if result.call_pnl is not None else "N/A",
                            'Put P&L': f"${result.put_pnl:.2f}" if result.put_pnl is not None else "N/A"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(
                        results_df,
                        hide_index=True,
                        use_container_width=True
                    )
        else:
            st.info("No historical calculations found. Perform a calculation to see it here.")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        st.error("An error occurred. Please try again.")
    finally:
        # Close the session
        if session:
            session.close()

if __name__ == "__main__":
    main()