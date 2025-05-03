import streamlit as st
from datetime import timedelta
import pandas as pd
from typing import Literal

from models.option_model import OptionModel
from models.simulation_model import SimulationModel
from visualization.plots import OptionVisualizer, SimulationVisualizer

def run_app():
    st.title("Options Analysis")
    
    # User input
    ticker = st.text_input("Enter the stock ticker (e.g., AAPL):")
    
    if ticker:
        try:
            # Initialize models
            option_model = OptionModel(ticker, None)  # expiration_date will be set later
            sim_model = SimulationModel(ticker)
            
            # Get available expiration dates
            import yfinance as yf
            stock = yf.Ticker(ticker)
            expiration_dates = stock.options
            expiration_date = st.selectbox("Select the expiration date:", expiration_dates)
            
            if st.button("Get Options and Generate Plots"):
                # Set expiration date
                option_model.expiration_date = expiration_date
                
                # Get historical data and fit distribution
                df = sim_model.fetch_historical_data()
                t_params = sim_model.fit_t_distribution()
                
                # Get current stock price and calculate days to expiration
                stock_price = stock.history(period="1d")['Close'].iloc[0]
                days_to_expiration = (pd.to_datetime(expiration_date).normalize() - 
                                    pd.to_datetime("today").normalize()).days
                
                # Get option data
                calls_df, puts_df = option_model.fetch_option_data()
                
                # Filter and process options
                calls_df = option_model.filter_options(calls_df, 'call')
                puts_df = option_model.filter_options(puts_df, 'put')
                
                # Calculate derivatives and probabilities
                calls_df = option_model.calculate_derivatives(calls_df, 'call')
                puts_df = option_model.calculate_derivatives(puts_df, 'put')
                
                calls_df = option_model.calculate_itm_probability(calls_df, 'call')
                puts_df = option_model.calculate_itm_probability(puts_df, 'put')
                
                # Format dataframes
                calls_df = option_model.format_dataframe(calls_df, 'call')
                puts_df = option_model.format_dataframe(puts_df, 'put')
                
                # Generate random walks
                random_walks_t_student = sim_model.simulate_random_walks(
                    technique="t-student", 
                    days=days_to_expiration
                )
                random_walks_bootstrap = sim_model.simulate_random_walks(
                    technique="bootstrap", 
                    days=days_to_expiration
                )
                
                # Calculate ITM probabilities
                calls_df['ITM T-Student'] = sim_model.calculate_strike_probabilities(
                    'call', 
                    calls_df['Strike Price (K)'].values,
                    random_walks_t_student,
                    stock_price
                )
                calls_df['ITM Bootstrapping'] = sim_model.calculate_strike_probabilities(
                    'call', 
                    calls_df['Strike Price (K)'].values,
                    random_walks_bootstrap,
                    stock_price
                )
                puts_df['ITM T-Student'] = sim_model.calculate_strike_probabilities(
                    'put', 
                    puts_df['Strike Price (K)'].values,
                    random_walks_t_student,
                    stock_price
                )
                puts_df['ITM Bootstrapping'] = sim_model.calculate_strike_probabilities(
                    'put', 
                    puts_df['Strike Price (K)'].values,
                    random_walks_bootstrap,
                    stock_price
                )
                
                # Create tabs for call and put options
                tab1, tab2 = st.tabs(["Call Options", "Put Options"])
                
                # Display call options
                with tab1:
                    st.subheader("Call Option Prices")
                    st.dataframe(calls_df)
                
                # Display put options
                with tab2:
                    st.subheader("Put Option Prices")
                    st.dataframe(puts_df)
                
                # Plot derivatives
                OptionVisualizer.plot_derivatives_vs_strike(calls_df, puts_df, stock_price)
                OptionVisualizer.plot_derivatives_vs_option_price(calls_df, puts_df)
                
                # Plot fitted distribution
                st.subheader(f"Fitted T-Student Distribution for Daily Returns of {ticker}")
                SimulationVisualizer.plot_fitted_distribution(df, t_params, ticker)
                
                # Plot random walks
                sim_start_date = pd.to_datetime("today").strftime('%Y-%m-%d')
                sim_end_date = (pd.to_datetime("today") + timedelta(days=days_to_expiration)).strftime('%Y-%m-%d')
                
                st.subheader("Random Walks with T-Student Distribution")
                SimulationVisualizer.plot_random_walks(
                    random_walks_t_student,
                    stock_price,
                    ticker,
                    sim_start_date,
                    sim_end_date,
                    "t-student"
                )
                
                st.subheader("Random Walks with Bootstrapping")
                SimulationVisualizer.plot_random_walks(
                    random_walks_bootstrap,
                    stock_price,
                    ticker,
                    sim_start_date,
                    sim_end_date,
                    "bootstrap"
                )
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    run_app() 