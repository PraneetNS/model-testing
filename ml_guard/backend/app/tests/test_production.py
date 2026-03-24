import pytest
import pandas as pd
import numpy as np
import os
from app.domain.services.trainer import Trainer
from app.domain.services.nlp_parser import NLPParser
from app.domain.services.ml_testing.engine import MLTestEngine

class TestProductionReadiness:
    
    def setup_method(self):
        self.trainer = Trainer(storage_path="storage/test_models")
        self.parser = NLPParser()
        self.engine = MLTestEngine()
        
        # Create a sample dataset
        self.df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.choice([0, 1], 100)
        })

    def test_trainer_pipeline_real_logic(self):
        """Confirm the trainer actually fits and evaluates a model."""
        result = self.trainer.train_model(
            df=self.df,
            target_column='target',
            model_type='random_forest',
            do_cv=False
        )
        
        assert result['status'] == 'success'
        assert 'metrics' in result
        assert result['metrics']['accuracy'] >= 0
        assert os.path.exists(result['model_path'])

    def test_nlp_parser_intent_mapping(self):
        """Ensure NLP parser maps keywords to real test categories."""
        categories = self.parser.parse_query("Check for model drift and accuracy")
        assert 'drift' in categories
        assert 'accuracy' in categories

    def test_failure_simulation_bad_data(self):
        """Ensure system handles corrupt/invalid data gracefully."""
        bad_df = pd.DataFrame({'feature1': [1, 2, 3]}) # Missing target
        result = self.trainer.train_model(bad_df, 'target')
        assert result['status'] == 'error'
        assert 'target' in result['message'].lower()

    def test_concurrency_test_simulation(self):
        """Briefly simulate high-load checks."""
        # This is more of a unit test for the logic, but it ensures no global state collisions
        t1 = Trainer(storage_path="storage/test_1")
        t2 = Trainer(storage_path="storage/test_2")
        
        res1 = t1.train_model(self.df, 'target')
        res2 = t2.train_model(self.df, 'target')
        
        assert res1['model_id'] != res2['model_id']

    def teardown_method(self):
        import shutil
        if os.path.exists("storage/test_models"):
            shutil.rmtree("storage/test_models")
        if os.path.exists("storage/test_1"):
            shutil.rmtree("storage/test_1")
        if os.path.exists("storage/test_2"):
            shutil.rmtree("storage/test_2")
