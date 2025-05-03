import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Literal, Tuple

class SimulationModel:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.historical_data = None
        self.t_params = None
        
    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical stock data and calculate daily returns."""
        import yfinance as yf
        stock = yf.Ticker(self.ticker)
        df = stock.history(period="max")
        df['Daily Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)
        self.historical_data = df
        return df
    
    def fit_t_distribution(self) -> Tuple[float, float, float]:
        """Fit a T-distribution to the daily returns."""
        returns = self.historical_data['Daily Return'].values
        self.t_params = stats.t.fit(returns)
        return self.t_params
    
    def simulate_random_walks(self, technique: Literal['t-student', 'bootstrap'], 
                            days: int = 252, sims: int = 10000) -> np.ndarray:
        """Simulate random walks using either T-Student or Bootstrap method."""
        if technique == "t-student":
            if self.t_params is None:
                self.fit_t_distribution()
            random_returns = stats.t.rvs(*self.t_params, size=(sims, days))
        elif technique == "bootstrap":
            random_returns = np.random.choice(
                self.historical_data['Daily Return'].values,
                size=(sims, days),
                replace=True
            )
        else:
            raise ValueError("Invalid technique. Choose 't-student' or 'bootstrap'.")
        
        # Clip extreme returns
        random_returns = np.clip(random_returns, -0.3, 0.3)
        
        # Convert returns to cumulative product for random walks
        random_walks = np.cumprod(1 + random_returns, axis=1)
        return random_walks
    
    def calculate_strike_probabilities(self, option_type: Literal['call', 'put'],
                                    strike_prices: np.ndarray,
                                    random_walks: np.ndarray,
                                    initial_price: float) -> np.ndarray:
        """Calculate probability of options being in-the-money at expiration."""
        probabilities = []
        for strike in strike_prices:
            if option_type == "call":
                hits = np.mean(random_walks * initial_price >= strike, axis=1)
            else:  # put
                hits = np.mean(random_walks * initial_price <= strike, axis=1)
            
            prob = np.mean(hits)
            probabilities.append(prob)
        
        # Convert to percentages with 1 decimal point
        return np.round(np.array(probabilities) * 100, 1) 