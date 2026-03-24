import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

class StatsEngine:
    @staticmethod
    def calculate_psi(expected, actual, buckets=10):
        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= (np.max(input) / (max - min))
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        breakpoints = np.percentile(expected, breakpoints)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

        expected_percents = np.clip(expected_percents, 1e-6, 1)
        actual_percents = np.clip(actual_percents, 1e-6, 1)

        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi_value

    @staticmethod
    def calculate_ks(expected, actual):
        d_stat, p_value = ks_2samp(expected, actual)
        return float(d_stat), float(p_value)
