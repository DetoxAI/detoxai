import numpy as np
import pytest
import torch

from src.detoxai.methods.posthoc.reject_option_classification import (
    RejectOptionClassification,
)


class TestRejectOptionClassification:
    @pytest.fixture
    def mock_data(self):
        # Create mock predictions, targets, and sensitive features
        preds = torch.tensor(
            [
                [0.9, 0.1],  # High confidence class 0
                [0.1, 0.9],  # High confidence class 1
                [0.6, 0.4],  # Low confidence
                [0.55, 0.45],  # Low confidence
            ]
        )
        targets = torch.tensor([0, 1, 0, 1])
        sensitive_features = torch.tensor([0, 1, 0, 1])
        return preds, targets, sensitive_features

    @pytest.fixture
    def mock_roc(self):
        class MockDataLoader:
            def __iter__(self):
                return iter([])

        roc = RejectOptionClassification(
            model=torch.nn.Linear(1, 2),
            experiment_name="test",
            device="cpu",
            dataloader=MockDataLoader(),
            theta_range=(0.6, 0.8),
            theta_steps=3,
        )
        return roc

    def test_evaluate_parameters(self, mock_roc, mock_data):
        preds, targets, sensitive_features = mock_data
        theta = 0.7
        L_values = {0: 0, 1: 1}

        score = mock_roc._evaluate_parameters(
            preds=preds,
            targets=targets,
            sensitive_features=sensitive_features,
            theta=theta,
            L_values=L_values,
        )

        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert not np.isnan(score)

    def test_optimize_parameters_with_mock_data(self, mock_roc, mock_data):
        preds, targets, sensitive_features = mock_data

        # Mock _get_model_predictions to return our test data
        mock_roc._get_model_predictions = lambda x: (preds, targets, sensitive_features)

        theta, L_values = mock_roc._optimize_parameters()

        assert 0.6 <= theta <= 0.8
        assert isinstance(L_values, dict)
        assert set(L_values.keys()) == {0, 1}
        assert all(v in {0, 1} for v in L_values.values())

    def test_modified_prediction(self, mock_roc, mock_data):
        preds, _, sensitive_features = mock_data
        theta = 0.7
        L_values = {0: 0, 1: 1}

        modified = mock_roc._modified_prediction(
            theta=theta,
            probs=preds,
            sensitive_features=sensitive_features,
            L_values=L_values,
        )

        assert torch.is_tensor(modified)
        assert modified.shape == (4,)
        assert set(modified.tolist()).issubset({0, 1})

    def test_parameter_validation(self, mock_roc, mock_data):
        preds, targets, sensitive_features = mock_data

        # Invalid theta (too low)
        with pytest.raises(AssertionError):
            mock_roc._evaluate_parameters(
                preds=preds,
                targets=targets,
                sensitive_features=sensitive_features,
                theta=0.3,
                L_values={0: 0, 1: 1},
            )

        # Invalid theta (too high)
        with pytest.raises(AssertionError):
            mock_roc._evaluate_parameters(
                preds=preds,
                targets=targets,
                sensitive_features=sensitive_features,
                theta=1.1,
                L_values={0: 0, 1: 1},
            )

        # Invalid L_values
        with pytest.raises(KeyError):
            mock_roc._evaluate_parameters(
                preds=preds,
                targets=targets,
                sensitive_features=sensitive_features,
                theta=0.7,
                L_values={0: 0},  # Missing value for group 1
            )

    def test_edge_cases(self, mock_roc):
        # Test with all high confidence predictions
        preds = torch.tensor([[0.9, 0.1], [0.95, 0.05]])
        targets = torch.tensor([0, 0])
        sensitive_features = torch.tensor([0, 1])

        score = mock_roc._evaluate_parameters(
            preds=preds,
            targets=targets,
            sensitive_features=sensitive_features,
            theta=0.7,
            L_values={0: 0, 1: 1},
        )

        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert not np.isnan(score)
