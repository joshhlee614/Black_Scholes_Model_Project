# Black-Scholes Option Pricing Model

A comprehensive implementation of the Black-Scholes option pricing model with interactive visualizations and Greeks analysis.

## Features

- European option pricing (calls and puts)
- Greeks calculations (Delta, Gamma, Theta, Vega, Rho)
- Interactive sensitivity analysis
- Real-time visualizations
- Historical calculation storage
- Modern web interface using Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Black_Scholes_Model_Project.git
cd Black_Scholes_Model_Project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run web/app.py
```

The app will be available at:
- Local URL: http://localhost:8507
- Network URL: http://your-ip:8507

## Project Structure

```
Black_Scholes_Model_Project/
├── src/
│   └── black_scholes.py      # Core Black-Scholes implementation
├── web/
│   └── app.py               # Streamlit web application
├── tests/
│   └── test_black_scholes.py # Unit tests
├── requirements.txt         # Project dependencies
├── setup.py                # Package configuration
└── README.md               # Project documentation
```

## Features

### Option Pricing
- Calculate European call and put option prices
- Real-time price updates
- Interactive parameter adjustment

### Greeks Analysis
- Delta: Price sensitivity to underlying asset
- Gamma: Delta sensitivity to underlying asset
- Theta: Time decay sensitivity
- Vega: Volatility sensitivity
- Rho: Interest rate sensitivity

### Visualization
- Interactive sensitivity plots
- Greeks heatmaps
- P&L analysis
- Historical calculation tracking

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Black-Scholes-Merton model
- Streamlit for the web interface
- NumPy and SciPy for numerical computations 