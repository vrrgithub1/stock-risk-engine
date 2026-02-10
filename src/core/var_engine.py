import numpy as np
import pandas as pd
from scipy.stats import norm

class VarEngine:
    def __init__(self, confidence_level=0.95):
        self.conf_level = confidence_level

    def calculate_historical_var(self, returns):
        """
        Non-parametric: Simply finds the percentile of past returns.
        Captures 'Fat Tails' better than Parametric.
        """
        return np.percentile(returns, (1 - self.conf_level) * 100)

    def calculate_parametric_var(self, returns):
        """
        Variance-Covariance Method: Assumes Normal Distribution.
        Great for clean, mathematical modeling.
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        return norm.ppf(1 - self.conf_level, mu, sigma)
    