# Derivatives Pricing Application

A Streamlit web application for analyzing stock options and simulating price movements.

## Features

- Real-time option chain data from Yahoo Finance
- Option pricing analysis with derivatives
- In-the-money probability calculations
- Price simulation using:
  - T-Student distribution
  - Bootstrapping method
- Interactive visualizations:
  - Option derivatives vs strike price
  - Option derivatives vs option price
  - Fitted T-Student distribution
  - Random walk simulations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/robertcercos/derivates_prices_app.git
cd derivates_prices_app
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

1. Start the Streamlit application:
```bash
streamlit run src/app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)

4. Select an expiration date from the available options

5. Click "Get Options and Generate Plots" to analyze the options and view simulations

## Project Structure

```
derivates_prices_app/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── option_model.py
│   │   └── simulation_model.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py
│   ├── __init__.py
│   └── app.py
├── requirements.txt
└── README.md
```

## License

MIT License 