import numpy as np
import pandas as pd
from scipy.stats import norm

class VarEngine:
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level

    def calculate_historical_var(self, returns):
        """Finds the actual percentile of past returns."""
        # Returns the value below which (1 - confidence_level)% of returns fall
        return np.percentile(returns, (1 - self.confidence_level) * 100)

    def calculate_parametric_var(self, returns):
        """Calculates VaR based on Normal Distribution assumptions."""
        mu = np.mean(returns)
        sigma = np.std(returns)
        # Z-score for the confidence level
        z_score = norm.ppf(1 - self.confidence_level)
        return mu + z_score * sigma
    
    def calculate_monte_carlo_var(self, returns, simulations=10000):
        """
        Simulates 10,000 potential future returns to find the 5th percentile risk.
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Generate random scenarios based on mean and standard deviation
        simulated_returns = np.random.normal(mu, sigma, simulations)
        
        # Find the 5th percentile (for 95% confidence)
        mc_var = np.percentile(simulated_returns, (1 - self.confidence_level) * 100)
        
        return float(mc_var)    