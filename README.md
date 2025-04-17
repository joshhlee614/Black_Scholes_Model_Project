# Black-Scholes Option Pricing Model

A comprehensive implementation of the Black-Scholes option pricing model with interactive visualizations, Greeks analysis, and real-time sensitivity calculations.

## Features

- **Option Pricing**
  - European call and put option pricing
  - Real-time price updates
  - Interactive parameter adjustment
  - Put-Call parity validation

- **Greeks Analysis**
  - Delta: Price sensitivity to underlying asset
  - Gamma: Delta sensitivity to underlying asset
  - Theta: Time decay sensitivity
  - Vega: Volatility sensitivity
  - Rho: Interest rate sensitivity
  - Interactive Greeks visualization

- **Sensitivity Analysis**
  - Stock price sensitivity
  - Time to maturity analysis
  - Volatility impact
  - Interest rate effects
  - Interactive heatmaps

- **Data Management**
  - SQLite database integration
  - Historical calculation storage
  - Calculation history tracking
  - Export capabilities

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
│   ├── black_scholes.py      # Core Black-Scholes implementation
│   └── database.py          # Database operations
├── web/
│   ├── app.py               # Main Streamlit application
│   ├── pages/
│   │   └── black_scholes_page.py  # Black-Scholes theory page
│   └── static/              # Static assets
├── tests/
│   ├── test_black_scholes.py # Unit tests
│   └── test_database.py     # Database tests
├── requirements.txt         # Project dependencies
├── setup.py                # Package configuration
└── README.md               # Project documentation
```

## Development

### Testing
Run the test suite:
```bash
pytest tests/
```

Generate test coverage report:
```bash
pytest --cov=src tests/
```

### Code Quality
- Format code with Black:
```bash
black .
```
- Check for linting issues:
```bash
flake8 .
```
- Sort imports:
```bash
isort .
```
- Type checking:
```bash
mypy .
```

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
- Plotly for interactive visualizations
- SQLAlchemy for database operations 