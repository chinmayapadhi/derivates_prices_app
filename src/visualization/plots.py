import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import scipy.stats as stats
from typing import Literal

class OptionVisualizer:
    @staticmethod
    def plot_derivatives_vs_strike(calls_df: pd.DataFrame, puts_df: pd.DataFrame, stock_price: float):
        """Plot derivatives with respect to strike price."""
        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')
        
        # Drop NaN values
        calls_df = calls_df.dropna(subset=['dC/dK'])
        puts_df = puts_df.dropna(subset=['dP/dK'])
        
        # Plot derivatives
        plt.scatter(calls_df['Strike Price (K)'], calls_df['dC/dK'], 
                   color='#FF9999', marker='o', label='Calls (dC/dK)', alpha=0.7)
        plt.scatter(puts_df['Strike Price (K)'], puts_df['dP/dK'], 
                   color='#99CCFF', marker='o', label='Puts (dP/dK)', alpha=0.7)
        
        # Add stock price line
        plt.axvline(x=stock_price, color='#99CC99', linestyle='--', 
                   label=f'Current Price: {stock_price:.2f}', lw=2)
        
        # Style the plot
        plt.ylim(-2, 2)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('white')
        
        plt.title('Option Derivatives vs Strike Price', fontsize=14, fontweight='light')
        plt.xlabel('Strike Price', fontsize=12)
        plt.ylabel('Derivative (dC/dK and dP/dK)', fontsize=12)
        plt.grid(True, color='gray', alpha=0.3)
        plt.legend(loc='best', frameon=False)
        
        st.pyplot(plt)
        plt.close()
    
    @staticmethod
    def plot_derivatives_vs_option_price(calls_df: pd.DataFrame, puts_df: pd.DataFrame):
        """Plot derivatives with respect to option price."""
        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')
        
        # Drop NaN values
        calls_df = calls_df.dropna(subset=['dC/dK'])
        puts_df = puts_df.dropna(subset=['dP/dK'])
        
        # Plot derivatives
        plt.scatter(calls_df['Option Price (C)'], calls_df['dC/dK'], 
                   color='#FF9999', marker='o', label='Calls (dC/dK)', alpha=0.7)
        plt.scatter(puts_df['Option Price (P)'], puts_df['dP/dK'], 
                   color='#99CCFF', marker='o', label='Puts (dP/dK)', alpha=0.7)
        
        # Style the plot
        plt.ylim(-2, 2)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('white')
        
        plt.title('Option Derivatives vs Option Price', fontsize=14, fontweight='light')
        plt.xlabel('Option Price', fontsize=12)
        plt.ylabel('Derivative (dC/dK and dP/dK)', fontsize=12)
        plt.grid(True, color='gray', alpha=0.3)
        plt.legend(loc='best', frameon=False)
        
        st.pyplot(plt)
        plt.close()

class SimulationVisualizer:
    @staticmethod
    def plot_fitted_distribution(df: pd.DataFrame, t_params: tuple, ticker: str):
        """Plot the fitted T-distribution with historical returns."""
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter data
        filtered_df = df[(df['Daily Return'] >= -0.2) & (df['Daily Return'] <= 0.2)]
        
        # Plot histogram
        ax.hist(filtered_df['Daily Return'], bins=100, density=True, 
                alpha=0.5, color='#1abc9c', edgecolor='none')
        
        # Plot T-distribution
        x = np.linspace(-0.2, 0.2, 1000)
        pdf_t = stats.t.pdf(x, *t_params)
        ax.plot(x, pdf_t, color='#e74c3c', lw=2, alpha=0.8)
        
        # Add percentiles
        t_percentiles = [stats.t.ppf(p, *t_params) for p in [0.05, 0.5, 0.95]]
        for p, value, v_offset, h_align in zip([5, 50, 95], t_percentiles, [0.1, 0.35, 0.1], ['right', 'center', 'left']):
            if -0.2 <= value <= 0.2:
                ax.axvline(value, color='#34495e', linestyle='--', lw=1, ymin=0, ymax=0.85, alpha=0.5)
                ax.text(value, max(pdf_t) * v_offset, 
                       f"p{p}\n({format_value_as_percentage(value)})", 
                       color='black', ha=h_align, va='top' if p == 50 else 'bottom')
        
        # Add stock info
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        ax.text(0.98, 1.05, f"{ticker.upper()} | {start_date} - {end_date}",
                transform=ax.transAxes, fontsize=10, ha='right', va='top', 
                color='black', fontweight='light', alpha=0.7)
        
        # Style the plot
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.set_xticks(np.linspace(-0.2, 0.2, 9))
        ax.set_xticklabels([f"{x * 100:.0f}%" for x in ax.get_xticks()])
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("Daily Return", fontsize=12)
        ax.tick_params(axis='x', which='both', bottom=True, labelsize=10)
        
        st.pyplot(plt)
        plt.close()
    
    @staticmethod
    def plot_random_walks(random_walks: np.ndarray, initial_price: float, ticker: str,
                         sim_start_date: str, sim_end_date: str, technique: Literal['t-student', 'bootstrap']):
        """Plot random walk simulations with distribution."""
        # Set colors based on technique
        colors = {
            't-student': {'main': '#00B3B3', '95': '#3498db', '5': '#e74c3c'},
            'bootstrap': {'main': '#FF6347', '95': '#FF8C00', '5': '#FF4500'}
        }
        
        # Calculate statistics
        path_median = np.quantile(random_walks, 0.5, axis=0)
        path_95 = np.quantile(random_walks, 0.95, axis=0)
        path_5 = np.quantile(random_walks, 0.05, axis=0)
        
        # Filter paths
        ending_prices = random_walks[:, -1] * initial_price
        p99 = np.percentile(ending_prices, 99)
        p1 = np.percentile(ending_prices, 1)
        valid_paths = np.all((random_walks * initial_price >= p1) & 
                            (random_walks * initial_price <= p99), axis=1)
        filtered_random_walks = random_walks[valid_paths]
        
        # Create figure
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=(3, 1))
        ax = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
        
        # Plot spaghetti paths
        selected_paths = np.random.choice(filtered_random_walks.shape[0], 1000, replace=False)
        for i in selected_paths:
            ax.plot(filtered_random_walks[i] * initial_price, 
                   color='lightgray', linewidth=0.5, alpha=0.3)
        
        # Plot main paths
        ax.plot(path_median * initial_price, 
                label=f'Median {technique} (Final: {path_median[-1] * initial_price:.2f})', 
                color=colors[technique]['main'])
        ax.plot(path_95 * initial_price, 
                label=f'$95^{{th}}$ Percentile {technique} (Final: {path_95[-1] * initial_price:.2f})', 
                color=colors[technique]['95'])
        ax.plot(path_5 * initial_price, 
                label=f'$5^{{th}}$ Percentile {technique} (Final: {path_5[-1] * initial_price:.2f})', 
                color=colors[technique]['5'])
        
        # Style the plot
        ax.fill_between(np.arange(random_walks.shape[1]), 
                       y1=path_5 * initial_price, 
                       y2=path_95 * initial_price, 
                       color=colors[technique]['main'], alpha=0.2)
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(visible=True, linestyle='--', alpha=0.5)
        
        # Add info text
        ax.text(0.02, -0.1, f"{ticker.upper()} | {sim_start_date} - {sim_end_date}", 
                transform=ax.transAxes, fontsize=10, color='#7f8c8d', 
                ha='left', va='center', fontstyle='italic')
        ax.text(0.98, 1.05, f"Initial Price: {initial_price:.2f}", 
                transform=ax.transAxes, fontsize=12, ha='right', va='top')
        
        # Plot histogram
        ending_returns = (random_walks[:, -1] - 1) * 100
        p99 = np.percentile(ending_returns, 99)
        p1 = np.percentile(ending_returns, 1)
        p95 = np.percentile(ending_returns, 95)
        p5 = np.percentile(ending_returns, 5)
        
        ax_hist.hist(ending_returns, orientation='horizontal', bins=40, 
                    color=colors[technique]['main'], alpha=0.3, range=(p1, p99))
        
        ax_hist.axhline(np.median(ending_returns), 
                       label=f'Median {technique} ({np.median(ending_returns):.2f}%)', 
                       color=colors[technique]['main'])
        ax_hist.axhline(p95, 
                       label=f'95th Percentile {technique} ({p95:.2f}%)', 
                       color=colors[technique]['95'])
        ax_hist.axhline(p5, 
                       label=f'5th Percentile {technique} ({p5:.2f}%)', 
                       color=colors[technique]['5'])
        
        ax_hist.set_ylabel('Compound Growth Rate (%)')
        ax_hist.set_xlabel('Frequency')
        ax_hist.legend()
        
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

def format_value_as_percentage(value: float) -> str:
    """Format a value as a percentage string."""
    return f"+{value * 100:.2f}%" if value >= 0 else f"{value * 100:.2f}%" 