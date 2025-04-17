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
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship
import json

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, 'app.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import Black-Scholes functions
try:
    # Import the module first
    from src import black_scholes
    
    # Then import all functions
    call_price = black_scholes.call_price
    put_price = black_scholes.put_price
    call_delta = black_scholes.call_delta
    put_delta = black_scholes.put_delta
    gamma = black_scholes.gamma
    call_theta = black_scholes.call_theta
    put_theta = black_scholes.put_theta
    vega = black_scholes.vega
    call_rho = black_scholes.call_rho
    put_rho = black_scholes.put_rho
    
    # Verify all required functions are imported
    required_functions = [
        call_price, put_price,
        call_delta, put_delta,
        gamma, call_theta, put_theta,
        vega, call_rho, put_rho
    ]
    
    logger.info("Successfully imported Black-Scholes functions")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Current directory: {os.getcwd()}")
except ImportError as e:
    logger.error(f"Failed to import Black-Scholes functions: {e}")
    logger.error(f"Python path: {sys.path}")
    logger.error(f"Current directory: {os.getcwd()}")
    st.error("Failed to import required modules. Please check the installation.")
    raise

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
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin: 1rem 0;
        padding: 0.5rem;
        background-color: rgba(49, 51, 63, 0.7);
        border-radius: 10px;
    }
    .creator-name {
        font-size: 1.1rem;
        font-weight: 500;
        color: #ffffff;
    }
    .linkedin-link {
        color: #0077b5;
        text-decoration: none;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .linkedin-link:hover {
        color: #005582;
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

# Add creator information at the top
st.markdown("""
<div class="creator-info">
    <span class="creator-name">Created by Joshua Lee</span>
    <a href="https://www.linkedin.com/in/joshhlee614" class="linkedin-link" target="_blank">
        <i class="fab fa-linkedin"></i>
    </a>
</div>
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

# Update Base class definition
class Base(DeclarativeBase):
    pass

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
        db_path = os.path.join(parent_dir, 'black_scholes.db')
        logger.info(f"Database path: {db_path}")
        
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create engine and session
        engine = create_engine(f'sqlite:///{db_path}')
        logger.info("Creating database tables...")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        logger.info("Database initialized successfully")
        return session
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error traceback: {e.__traceback__}")
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
        
        return calculation_id
    except Exception as e:
        logger.error(f"Failed to save calculation to database: {e}")
        session.rollback()  # Rollback in case of error
        st.error("Failed to save calculation. Some features may not be available.")
        return None

# Plot heatmap
def plot_heatmap(prices, volatilities, spot_prices, title, cmap="viridis", is_pnl=False):
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Ensure arrays are properly shaped
        prices = np.array(prices)
        volatilities = np.array(volatilities)
        spot_prices = np.array(spot_prices)
        
        # For PnL heatmaps, use a diverging colormap centered at 0
        if is_pnl:
            # Find the absolute maximum value for symmetric color scaling
            abs_max = max(abs(np.nanmin(prices)), abs(np.nanmax(prices)))
            vmin, vmax = -abs_max, abs_max
            cmap = "RdYlGn"  # Red for negative, green for positive
        else:
            vmin, vmax = np.nanmin(prices), np.nanmax(prices)
        
        # Create heatmap with better formatting
        sns.heatmap(
            prices,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            xticklabels=[f"${x:.0f}" for x in spot_prices],
            yticklabels=[f"{y:.2%}" for y in volatilities[::-1]],  # Reverse for better visualization
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
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return fig
    except Exception as e:
        logger.error(f"Error in plot_heatmap: {e}")
        raise

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
    try:
        # Initialize database
        logger.info("Initializing database...")
        session = init_db()
        
        if session is None:
            st.error("Failed to initialize database. Some features may not be available.")
            return
        
        # Initialize session state for input values if not already done
        logger.info("Initializing session state...")
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
        
        logger.info("Setting up UI...")
        st.markdown('<h1 class="main-header">Black-Scholes Model</h1>', unsafe_allow_html=True)
        
        # Description
        st.markdown(
            '<div class="description">'
            'A sophisticated implementation of the Black-Scholes-Merton option pricing model featuring real-time computation of option prices, Greeks, and sensitivity analysis. '
            'This professional-grade tool demonstrates quantitative finance concepts through interactive visualizations of option pricing dynamics, risk measures, and P&L scenarios. '
            'Features include real-time Greeks calculation, volatility surface analysis, and comprehensive risk metrics visualization.'
            '</div>',
            unsafe_allow_html=True
        )
        
        # Create columns for the layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<h2 class="sub-header">Parameters</h2>', unsafe_allow_html=True)
            
            # Use a form to handle input changes
            with st.form("input_form"):
                # Get input values
                S = st.number_input(
                    "Asset Price", 
                    min_value=1.0, 
                    value=st.session_state.input_values['asset_price'], 
                    step=1.0,
                    key="asset_price"
                )
                
                K = st.number_input(
                    "Strike Price", 
                    min_value=1.0, 
                    value=st.session_state.input_values['strike_price'], 
                    step=1.0,
                    key="strike_price"
                )
                
                T = st.number_input(
                    "Time to Maturity (Years)", 
                    min_value=0.01, 
                    value=st.session_state.input_values['time_to_maturity'], 
                    step=0.25,
                    key="time_to_maturity"
                )
                
                sigma = st.number_input(
                    "Volatility (σ)", 
                    min_value=0.01, 
                    value=st.session_state.input_values['volatility'], 
                    step=0.01,
                    key="volatility"
                )
                
                r = st.number_input(
                    "Risk-Free Interest Rate", 
                    min_value=0.0, 
                    value=st.session_state.input_values['risk_free_rate'], 
                    step=0.01,
                    key="risk_free_rate"
                )

                st.markdown('<div class="sidebar-header">Heatmap Parameters</div>', unsafe_allow_html=True)

                min_spot_price = st.number_input(
                    "Min Spot Price", 
                    min_value=1.0, 
                    value=st.session_state.input_values['min_spot_price'], 
                    step=5.0,
                    key="min_spot_price"
                )

                max_spot_price = st.number_input(
                    "Max Spot Price", 
                    min_value=min_spot_price + 1.0, 
                    value=st.session_state.input_values['max_spot_price'], 
                    step=5.0,
                    key="max_spot_price"
                )

                min_volatility = st.number_input(
                    "Min Volatility", 
                    min_value=0.01, 
                    value=st.session_state.input_values['min_volatility'], 
                    step=0.05,
                    key="min_volatility"
                )

                max_volatility = st.number_input(
                    "Max Volatility", 
                    min_value=min_volatility + 0.01, 
                    value=st.session_state.input_values['max_volatility'], 
                    step=0.05,
                    key="max_volatility"
                )

                # P&L Analysis inputs
                st.markdown("---")
                st.markdown('<div class="sidebar-header">P&L Analysis</div>', unsafe_allow_html=True)

                call_purchase_price = st.number_input(
                    "Call Purchase Price", 
                    min_value=0.0, 
                    value=st.session_state.input_values['call_purchase_price'], 
                    step=1.0,
                    key="call_purchase_price"
                )

                put_purchase_price = st.number_input(
                    "Put Purchase Price", 
                    min_value=0.0, 
                    value=st.session_state.input_values['put_purchase_price'], 
                    step=1.0,
                    key="put_purchase_price"
                )

                # Submit button for the form
                submitted = st.form_submit_button("Update Parameters")
                
                if submitted:
                    # Update all session state values at once
                    st.session_state.input_values.update({
                        'asset_price': S,
                        'strike_price': K,
                        'time_to_maturity': T,
                        'volatility': sigma,
                        'risk_free_rate': r,
                        'min_spot_price': min_spot_price,
                        'max_spot_price': max_spot_price,
                        'min_volatility': min_volatility,
                        'max_volatility': max_volatility,
                        'call_purchase_price': call_purchase_price,
                        'put_purchase_price': put_purchase_price
                    })

            # Calculate button outside the form
            calculate = st.button("Calculate", key="calculate")
        
        with col2:
            st.markdown('<h2 class="sub-header">Option Prices & Greeks</h2>', unsafe_allow_html=True)
            
            # Calculate option prices and Greeks
            if all(v > 0 for v in [S, K, T, sigma]):
                try:
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
                    
                    # Add parameter range controls for sensitivity analysis
                    st.markdown('<h3 class="sub-header">Sensitivity Analysis</h3>', unsafe_allow_html=True)
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
                            min_value=0,
                            max_value=100,
                            value=(10, 30),
                            step=5,
                            help="Range of volatility values"
                        )
                        rate_range = st.slider(
                            "Interest Rate Range (%)",
                            min_value=0,
                            max_value=20,
                            value=(0, 10),
                            step=1,
                            help="Range of risk-free interest rates"
                        )
                    
                    if calculate:
                        try:
                            with st.spinner('Calculating...'):
                                # Display sensitivity analysis graphs
                                tab1, tab2, tab3, tab4 = st.tabs(["Stock Price", "Time to Maturity", "Volatility", "Interest Rate"])
                                
                                with tab1:
                                    st.markdown('<h4>Option Prices vs. Stock Price</h4>', unsafe_allow_html=True)
                                    try:
                                        # Generate stock price range
                                        price_min, price_max = price_range
                                        S_range = np.linspace(S * price_min/100, S * price_max/100, 50)
                                        
                                        # Calculate option prices
                                        call_prices_s = np.zeros_like(S_range)
                                        put_prices_s = np.zeros_like(S_range)
                                        
                                        for i, s in enumerate(S_range):
                                            try:
                                                call_prices_s[i] = call_price(s, K, T, r, sigma)
                                                put_prices_s[i] = put_price(s, K, T, r, sigma)
                                            except Exception as e:
                                                logger.error(f"Error calculating prices at stock price {s}: {e}")
                                                st.error(f"Error calculating prices at stock price ${s:.2f}")
                                                raise
                                        
                                        # Create the plot
                                        fig, ax = plt.subplots(figsize=(12, 8))
                                        ax.plot(S_range, call_prices_s, 'g-', label='Call Price', linewidth=2)
                                        ax.plot(S_range, put_prices_s, 'r-', label='Put Price', linewidth=2)
                                        ax.axvline(x=S, color='k', linestyle='--', alpha=0.3, label='Current Price')
                                        ax.axhline(y=K, color='gray', linestyle=':', alpha=0.3, label='Strike Price')
                                        ax.set_xlabel('Stock Price ($)', fontsize=12)
                                        ax.set_ylabel('Option Price ($)', fontsize=12)
                                        ax.grid(True, linestyle='--', alpha=0.3)
                                        ax.legend(loc='upper right')
                                        plt.title('Option Prices vs. Stock Price', fontsize=14, pad=20)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                        
                                    except Exception as e:
                                        logger.error(f"Error in stock price sensitivity analysis: {e}")
                                        st.error("Error calculating stock price sensitivity. Please check your inputs.")
                                
                                with tab2:
                                    st.markdown('<h4>Option Prices vs. Time to Maturity</h4>', unsafe_allow_html=True)
                                    try:
                                        # Generate time range
                                        t_min, t_max = time_range
                                        t_min = max(0.01, t_min)  # Ensure minimum time is at least 0.01
                                        T_range = np.linspace(t_min, t_max, 50)
                                        
                                        # Calculate option prices
                                        call_prices_t = np.zeros_like(T_range)
                                        put_prices_t = np.zeros_like(T_range)
                                        
                                        for i, t in enumerate(T_range):
                                            try:
                                                call_prices_t[i] = call_price(S, K, t, r, sigma)
                                                put_prices_t[i] = put_price(S, K, t, r, sigma)
                                            except Exception as e:
                                                logger.error(f"Error calculating prices at time {t}: {e}")
                                                st.error(f"Error calculating prices at time {t:.2f} years")
                                                raise
                                        
                                        # Create the plot
                                        fig, ax = plt.subplots(figsize=(12, 8))
                                        ax.plot(T_range, call_prices_t, 'g-', label='Call Price', linewidth=2)
                                        ax.plot(T_range, put_prices_t, 'r-', label='Put Price', linewidth=2)
                                        ax.axvline(x=T, color='k', linestyle='--', alpha=0.3, label='Current Time')
                                        ax.set_xlabel('Time to Maturity (Years)', fontsize=12)
                                        ax.set_ylabel('Option Price ($)', fontsize=12)
                                        ax.grid(True, linestyle='--', alpha=0.3)
                                        ax.legend(loc='upper right')
                                        plt.title('Option Prices vs. Time to Maturity', fontsize=14, pad=20)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                        
                                    except Exception as e:
                                        logger.error(f"Error in time sensitivity analysis: {e}")
                                        st.error("Error calculating time sensitivity. Please check your inputs.")
                                
                                with tab3:
                                    st.markdown('<h4>Vega vs. Volatility</h4>', unsafe_allow_html=True)
                                    try:
                                        # Generate volatility range
                                        vol_min, vol_max = vol_range
                                        sigma_range = np.linspace(vol_min/100, vol_max/100, 50)
                                        
                                        # Calculate vegas
                                        vegas = np.zeros_like(sigma_range)
                                        for i, s in enumerate(sigma_range):
                                            try:
                                                vegas[i] = vega(S, K, T, r, s)
                                            except Exception as e:
                                                logger.error(f"Error calculating vega at volatility {s}: {e}")
                                                st.error(f"Error calculating vega at volatility {s:.2%}")
                                                raise
                                        
                                        # Create the plot
                                        fig, ax = plt.subplots(figsize=(12, 8))
                                        ax.plot(sigma_range * 100, vegas, 'b-', label='Vega', linewidth=2)
                                        ax.axvline(x=sigma * 100, color='k', linestyle='--', alpha=0.3, label='Current Volatility')
                                        ax.set_xlabel('Volatility (%)', fontsize=12)
                                        ax.set_ylabel('Vega ($ per 1% vol change)', fontsize=12)
                                        ax.grid(True, linestyle='--', alpha=0.3)
                                        ax.legend(loc='upper right')
                                        plt.title('Vega Sensitivity to Volatility', fontsize=14, pad=20)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                        
                                    except Exception as e:
                                        logger.error(f"Error in Vega sensitivity analysis: {e}")
                                        st.error("Error calculating Vega sensitivity. Please check your inputs.")
                                
                                with tab4:
                                    st.markdown('<h4>Rho vs. Interest Rate</h4>', unsafe_allow_html=True)
                                    try:
                                        # Generate interest rate range
                                        rate_min, rate_max = rate_range
                                        r_range = np.linspace(rate_min/100, rate_max/100, 50)
                                        
                                        # Calculate rhos
                                        call_rhos = np.zeros_like(r_range)
                                        put_rhos = np.zeros_like(r_range)
                                        
                                        for i, rate in enumerate(r_range):
                                            try:
                                                call_rhos[i] = call_rho(S, K, T, rate, sigma)
                                                put_rhos[i] = put_rho(S, K, T, rate, sigma)
                                            except Exception as e:
                                                logger.error(f"Error calculating rho at rate {rate}: {e}")
                                                st.error(f"Error calculating rho at rate {rate:.2%}")
                                                raise
                                        
                                        # Create the plot
                                        fig, ax = plt.subplots(figsize=(12, 8))
                                        ax.plot(r_range * 100, call_rhos, 'g-', label='Call Rho', linewidth=2)
                                        ax.plot(r_range * 100, put_rhos, 'r-', label='Put Rho', linewidth=2)
                                        ax.axvline(x=r * 100, color='k', linestyle='--', alpha=0.3, label='Current Rate')
                                        ax.set_xlabel('Interest Rate (%)', fontsize=12)
                                        ax.set_ylabel('Rho ($ per 1% rate change)', fontsize=12)
                                        ax.grid(True, linestyle='--', alpha=0.3)
                                        ax.legend(loc='upper right')
                                        plt.title('Rho Sensitivity to Interest Rate', fontsize=14, pad=20)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                        
                                    except Exception as e:
                                        logger.error(f"Error in Rho sensitivity analysis: {e}")
                                        st.error("Error calculating Rho sensitivity. Please check your inputs.")
                                
                                # Initialize spot prices and volatilities for heatmap
                                spot_prices = np.linspace(min_spot_price, max_spot_price, 10)
                                volatilities = np.linspace(min_volatility, max_volatility, 10)
                                
                                # Initialize arrays for prices and P&L
                                call_prices = np.zeros((len(volatilities), len(spot_prices)))
                                put_prices = np.zeros((len(volatilities), len(spot_prices)))
                                call_pnl_matrix = np.zeros((len(volatilities), len(spot_prices)))
                                put_pnl_matrix = np.zeros((len(volatilities), len(spot_prices)))
                                
                                # Calculate prices
                                for i in range(len(volatilities)):
                                    for j in range(len(spot_prices)):
                                        try:
                                            call_prices[i, j] = call_price(spot_prices[j], K, T, r, volatilities[i])
                                            put_prices[i, j] = put_price(spot_prices[j], K, T, r, volatilities[i])
                                            
                                            if call_purchase_price > 0:
                                                call_pnl_matrix[i, j] = call_prices[i, j] - call_purchase_price
                                            if put_purchase_price > 0:
                                                put_pnl_matrix[i, j] = put_prices[i, j] - put_purchase_price
                                        except Exception as e:
                                            logger.error(f"Error calculating prices at spot={spot_prices[j]}, vol={volatilities[i]}: {e}")
                                            st.error(f"Error calculating prices at spot=${spot_prices[j]:.2f}, volatility={volatilities[i]:.2%}")
                                            raise
                                
                                # Display heatmaps
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown('<div class="sub-header">Call Price Heatmap</div>', unsafe_allow_html=True)
                                    try:
                                        call_fig = plot_heatmap(call_prices, volatilities, spot_prices, "Call", cmap="viridis")
                                        st.pyplot(call_fig)
                                        
                                        if call_purchase_price > 0:
                                            st.markdown('<div class="sub-header">Call P&L Heatmap</div>', unsafe_allow_html=True)
                                            call_pnl_fig = plot_heatmap(call_pnl_matrix, volatilities, spot_prices, "Call", is_pnl=True)
                                            st.pyplot(call_pnl_fig)
                                    except Exception as e:
                                        logger.error(f"Error plotting call heatmaps: {e}")
                                        st.error("Error displaying call option heatmaps")
                                
                                with col2:
                                    st.markdown('<div class="sub-header">Put Price Heatmap</div>', unsafe_allow_html=True)
                                    try:
                                        put_fig = plot_heatmap(put_prices, volatilities, spot_prices, "Put", cmap="magma")
                                        st.pyplot(put_fig)
                                        
                                        if put_purchase_price > 0:
                                            st.markdown('<div class="sub-header">Put P&L Heatmap</div>', unsafe_allow_html=True)
                                            put_pnl_fig = plot_heatmap(put_pnl_matrix, volatilities, spot_prices, "Put", is_pnl=True)
                                            st.pyplot(put_pnl_fig)
                                    except Exception as e:
                                        logger.error(f"Error plotting put heatmaps: {e}")
                                        st.error("Error displaying put option heatmaps")
                                
                                # Create DataFrame for database storage
                                data = []
                                for i in range(len(volatilities)):
                                    for j in range(len(spot_prices)):
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
                                try:
                                    inputs = {
                                        'asset_price': S,
                                        'strike_price': K,
                                        'time_to_maturity': T,
                                        'volatility': sigma,
                                        'risk_free_rate': r,
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
                                except Exception as e:
                                    logger.error(f"Error saving to database: {e}")
                                    st.error("Error saving calculation to database. Your results are still displayed but won't be saved.")
                        
                        except Exception as e:
                            logger.error(f"Error in calculation: {e}")
                            st.error("An error occurred during calculation. Please check your inputs and try again.")
                
                except Exception as e:
                    logger.error(f"Error calculating option prices and Greeks: {e}")
                    st.error("Error calculating option prices and Greeks. Please check your inputs.")
            
            # Add historical calculations section
            st.markdown("---")
            st.markdown('<h2 class="sub-header">Historical Calculations</h2>', unsafe_allow_html=True)
            
            # Get historical calculations
            calculations = get_historical_calculations(session)
            
            if calculations:
                # Create a DataFrame for display
                history_data = []
                for calc in calculations:
                    try:
                        greeks = json.loads(calc.greeks) if calc.greeks else {}
                        history_data.append({
                            'Timestamp': calc.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'Asset Price': f"${calc.asset_price:.2f}",
                            'Strike Price': f"${calc.strike_price:.2f}",
                            'Time to Maturity': f"{calc.time_to_maturity:.2f} years",
                            'Volatility': f"{calc.volatility:.2%}",
                            'Risk-Free Rate': f"{calc.risk_free_rate:.2%}",
                            'Call Delta': f"{greeks.get('call_delta', 'N/A'):.4f}" if 'call_delta' in greeks else 'N/A',
                            'Put Delta': f"{greeks.get('put_delta', 'N/A'):.4f}" if 'put_delta' in greeks else 'N/A',
                            'Gamma': f"{greeks.get('gamma', 'N/A'):.4f}" if 'gamma' in greeks else 'N/A',
                            'ID': calc.id
                        })
                    except Exception as e:
                        logger.error(f"Error processing calculation {calc.id}: {e}")
                        continue
                
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
        logger.error(f"Error in main function: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error traceback: {e.__traceback__}")
        st.error(f"An error occurred: {str(e)}")
        raise
    finally:
        if 'session' in locals() and session is not None:
            session.close()

if __name__ == "__main__":
    main()
