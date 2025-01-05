import torch
import pytest
from src.detoxai.methods.posthoc.reject_option_classification import ROCModelWrapper

class TestROCPredictionModification:
    @pytest.fixture
    def setup_wrapper(self):
        # Mock base model that returns pre-defined logits
        class MockModel(torch.nn.Module):
            def forward(self, x):
                return x
        
        base_model = MockModel()
        theta = 0.7
        L_values = {0: 0, 1: 1}  # Protected group 0 -> class 0, group 1 -> class 1
        
        return ROCModelWrapper(base_model, theta, L_values)

    def test_prediction_modification(self, setup_wrapper):
        wrapper = setup_wrapper
        
        # Create test inputs
        # High confidence predictions (>0.7)
        logits_high = torch.tensor([
            [0.9, 0.1],  # High confidence class 0
            [0.1, 0.9],  # High confidence class 1
        ])
        
        # Low confidence predictions (<0.7)
        logits_low = torch.tensor([
            [0.6, 0.4],  # Low confidence
            [0.55, 0.45],  # Low confidence
        ])
        
        input_logits = torch.cat([logits_high, logits_low])
        
        # Protected attributes: [0,1,0,1]
        sensitive_features = torch.tensor([0, 1, 0, 1])
        
        # Make predictions
        predictions = wrapper(input_logits, sensitive_features)
        
        # Expected outcomes:
        # - High confidence predictions (first 2) should remain unchanged
        # - Low confidence predictions should be modified based on protected group
        expected = torch.tensor([0, 1, 0, 1])
        
        assert torch.equal(predictions, expected), \
            f"Expected {expected}, but got {predictions}"

    def test_edge_cases(self, setup_wrapper):
        wrapper = setup_wrapper
        
        # Test exactly at threshold
        logits_threshold = torch.tensor([
            [0.7, 0.3],  # Exactly at threshold
        ])
        sensitive_features = torch.tensor([0])
        
        predictions = wrapper(logits_threshold, sensitive_features)
        expected = torch.tensor([0])  # Should be modified since θ ≤ 0.7
        
        assert torch.equal(predictions, expected), \
            "Failed to handle threshold case correctly"

    def test_different_L_values(self):
        # Test with opposite L_value assignments
        class MockModel(torch.nn.Module):
            def forward(self, x):
                return x
                
        wrapper = ROCModelWrapper(
            MockModel(),
            theta=0.7,
            L_values={0: 1, 1: 0}  # Reversed L_values
        )
        
        logits = torch.tensor([
            [0.6, 0.4],  # Low confidence
            [0.6, 0.4],  # Low confidence
        ])
        sensitive_features = torch.tensor([0, 1])
        
        predictions = wrapper(logits, sensitive_features)
        expected = torch.tensor([1, 0])  # Reversed predictions
        
        assert torch.equal(predictions, expected), \
            "Failed to apply reversed L_values correctly"