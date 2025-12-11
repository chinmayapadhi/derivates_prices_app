import numpy as np
import pandas as pd
from typing import Tuple, Literal

class OptionModel:
    def __init__(self, ticker: str, expiration_date: str):
        self.ticker = ticker
        self.expiration_date = expiration_date
        self.stock_price = None
        self.calls_df = None
        self.puts_df = None
        
    def fetch_option_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch option chain data from yfinance."""
        import yfinance as yf
        stock = yf.Ticker(self.ticker)
        options_chain = stock.option_chain(self.expiration_date)
        
        # Get current stock price
        self.stock_price = stock.history(period="1d")['Close'].iloc[0]
        
        # Process calls
        calls = options_chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume']]
        calls.columns = ['Strike Price (K)', 'Option Price (C)', 'Bid', 'Ask', 'Volume']
        
        # Process puts
        puts = options_chain.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume']]
        puts.columns = ['Strike Price (K)', 'Option Price (P)', 'Bid', 'Ask', 'Volume']
        
        return calls, puts
    
    def filter_options(self, df: pd.DataFrame, option_type: Literal['call', 'put']) -> pd.DataFrame:
        """Filter options based on price and volume criteria."""
        lower_bound = self.stock_price * 0.5
        upper_bound = self.stock_price * 1.5
        option_price_limit = self.stock_price * 0.3
        
        price_col = 'Option Price (C)' if option_type == 'call' else 'Option Price (P)'
        
        df = df[
            (df['Strike Price (K)'] >= lower_bound) & 
            (df['Strike Price (K)'] <= upper_bound) & 
            (df[price_col] <= option_price_limit)
        ]
        
        # Remove low volume options
        df = df.dropna(subset=['Volume'])
        if self.ticker == "^XSP":
            df = df[df['Volume'] >= 50]
        elif self.ticker == "^SPX":
            df = df[df['Volume'] >= 1000]
        else:
            df = df[df['Volume'] >= 500]
        return df
    
    def calculate_derivatives(self, df: pd.DataFrame, option_type: Literal['call', 'put']) -> pd.DataFrame:
        """Calculate derivatives with respect to strike price."""
        price_col = 'Option Price (C)' if option_type == 'call' else 'Option Price (P)'
        derivative_col = 'dC/dK' if option_type == 'call' else 'dP/dK'
        
        derivatives = []
        for i in range(len(df) - 1):
            price_current = df[price_col].iloc[i]
            price_next = df[price_col].iloc[i + 1]
            strike_current = df['Strike Price (K)'].iloc[i]
            strike_next = df['Strike Price (K)'].iloc[i + 1]
            
            derivative = (price_next - price_current) / (strike_next - strike_current)
            derivatives.append(derivative)
        
        df[derivative_col] = [np.nan] + derivatives
        return df
    
    def calculate_itm_probability(self, df: pd.DataFrame, option_type: Literal['call', 'put']) -> pd.DataFrame:
        """Calculate in-the-money probability from derivatives."""
        derivative_col = 'dC/dK' if option_type == 'call' else 'dP/dK'
        
        if option_type == 'call':
            df['in-the-money prob'] = (-df[derivative_col] * 100).clip(0, 100).round(1).astype(str) + '%'
        else:
            df['in-the-money prob'] = (df[derivative_col] * 100).clip(0, 100).round(1).astype(str) + '%'
        
        df['in-the-money prob'] = df['in-the-money prob'].replace('nan%', '-')
        return df
    
    def format_dataframe(self, df: pd.DataFrame, option_type: Literal['call', 'put']) -> pd.DataFrame:
        """Format the dataframe with appropriate decimal places."""
        price_col = 'Option Price (C)' if option_type == 'call' else 'Option Price (P)'
        
        # Round decimals for option prices, bid, and ask
        df[price_col] = df[price_col].round(2)
        df['Bid'] = df['Bid'].round(2)
        df['Ask'] = df['Ask'].round(2)
        
        # Convert other columns to integers
        df['Strike Price (K)'] = df['Strike Price (K)'].astype(int)
        df['Volume'] = df['Volume'].astype(int)
        
        return df 
