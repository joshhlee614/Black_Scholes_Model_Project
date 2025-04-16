# Black-Scholes Option Pricing Model

A comprehensive implementation of the Black-Scholes option pricing model with interactive visualizations and Greeks analysis.

## Features

- European option pricing (calls and puts)
- Greeks calculations (Delta, Gamma, Theta, Vega, Rho)
- Interactive sensitivity analysis
- Real-time visualizations
- Historical calculation storage

## Local Development

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

4. Run the application:
```bash
streamlit run web/app.py
```

## Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository and main file (web/app.py)
6. Click "Deploy"

### Hugging Face Spaces

1. Create a Hugging Face account
2. Create a new Space
3. Choose Streamlit template
4. Push your code to the Space
5. Configure requirements.txt

### Render

1. Create a Render account
2. Create a new Web Service
3. Connect your GitHub repository
4. Configure build settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run web/app.py`
5. Deploy

## Project Structure

```
.
├── src/                    # Source code
│   ├── black_scholes.py    # Black-Scholes implementation
│   └── __init__.py         # Package initialization
├── web/                    # Web application
│   └── app.py              # Streamlit application
├── tests/                  # Test files
├── requirements.txt        # Dependencies
└── setup.py               # Package configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 