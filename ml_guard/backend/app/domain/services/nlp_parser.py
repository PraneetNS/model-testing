
from typing import List, Dict, Any
import re

class NLPParser:
    """
    Parses natural language commands to determine specific test intents.
    Maps queries to granular test categories for a customized testing experience.
    """
    def __init__(self):
        self.keywords = {
            'accuracy': ['accuracy', 'precision', 'recall', 'f1', 'score', 'correctness', 'metrics'],
            'performance': ['performance', 'auc', 'roc', 'latency', 'speed', 'inference'],
            'data_quality': ['missing', 'duplicate', 'null', 'quality', 'clean', 'imbalance', 'balance', 'outliers', 'distribution'],
            'bias': ['bias', 'fairness', 'discrimination', 'gender', 'race', 'fair', 'parity', 'ethical'],
            'drift': ['drift', 'psi', 'change', 'distribution', 'population', 'ks', 'shift'],
            'stability': ['stability', 'stable', 'correlation', 'robust'],
            'robustness': ['robust', 'edge', 'boundary', 'adversarial', 'reliable', 'safety'],
            'stress_test': ['stress', 'noise', 'perturbation', 'extreme', 'load', 'break'],
            'regression': ['regression', 'baseline', 'compare', 'previous', 'improvement', 'degradation'],
            'all': ['all', 'comprehensive', 'complete', 'everything', 'full', 'suite', 'standard']
        }

    def parse_query(self, query: str) -> List[str]:
        """
        Parses the query and returns a list of granular test categories.
        """
        query_lower = query.lower()
        selected_tests = []

        # Check for 'all' first
        for keyword in self.keywords['all']:
            if keyword in query_lower:
                return [
                    'accuracy', 'performance', 'data_quality', 
                    'bias', 'drift', 'stability', 
                    'robustness', 'stress_test'
                ]

        # Check for specific intents
        for test_type, keywords in self.keywords.items():
            if test_type == 'all':
                continue
            for keyword in keywords:
                if keyword in query_lower:
                    if test_type not in selected_tests:
                        selected_tests.append(test_type)
                    break

        # Default to a sane set if nothing is found
        if not selected_tests:
            selected_tests = ['accuracy', 'data_quality', 'drift']

        return selected_tests
