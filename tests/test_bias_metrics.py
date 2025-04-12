import numpy as np
import pytest
import torch

from src.detoxai.metrics.bias_metrics import (
    BiasMetrics,
    calculate_bias_metric_torch,
    stabilize,
)


class TestBiasMetrics:
    @pytest.fixture
    def sample_data_torch(self):
        y_pred = torch.tensor([1, 0, 1, 0, 1, 1])
        y_true = torch.tensor([1, 0, 1, 1, 0, 1])
        protected_attribute = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        return y_pred, y_true, protected_attribute

    @pytest.fixture
    def sample_data_np(self):
        y_pred = np.array([1, 0, 1, 0, 1, 1])
        y_true = np.array([1, 0, 1, 1, 0, 1])
        protected_attribute = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        return y_pred, y_true, protected_attribute

    @pytest.fixture
    def edge_case_data_torch(self):
        # All zeros to test stabilize function
        y_pred = torch.zeros(6, dtype=torch.int64)
        y_true = torch.zeros(6, dtype=torch.int64)
        protected_attribute = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        return y_pred, y_true, protected_attribute

    @pytest.fixture
    def edge_case_data_np(self):
        # All zeros to test stabilize function
        y_pred = np.zeros(6, dtype=int)
        y_true = np.zeros(6, dtype=int)
        protected_attribute = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        return y_pred, y_true, protected_attribute

    def test_tpr_gap_torch(self, sample_data_torch):
        y_pred, y_true, protected_attribute = sample_data_torch
        bias = calculate_bias_metric_torch(
            BiasMetrics.TPR_GAP, y_pred, y_true, protected_attribute
        )
        # Group A (protected=1): TPR = 2/2 = 1.0
        # Group B (protected=0): TPR = 1/2 = 0.5
        # |1.0 - 0.5| = 0.5
        assert abs(bias.item() - 0.5) < 1e-6

    def test_fpr_gap_torch(self, sample_data_torch):
        y_pred, y_true, protected_attribute = sample_data_torch
        bias = calculate_bias_metric_torch(
            BiasMetrics.FPR_GAP, y_pred, y_true, protected_attribute
        )
        # Group A (protected=1): FPR = 0/1 = 0.0
        # Group B (protected=0): FPR = 1/1 = 1.0
        # |0.0 - 1.0| = 1.0
        assert abs(bias.item() - 1.0) < 1e-6

    def test_tnr_gap_torch(self, sample_data_torch):
        y_pred, y_true, protected_attribute = sample_data_torch
        bias = calculate_bias_metric_torch(
            BiasMetrics.TNR_GAP, y_pred, y_true, protected_attribute
        )
        assert abs(bias.item() - 1.0) < 1e-6

    def test_fnr_gap_torch(self, sample_data_torch):
        y_pred, y_true, protected_attribute = sample_data_torch
        bias = calculate_bias_metric_torch(
            BiasMetrics.FNR_GAP, y_pred, y_true, protected_attribute
        )
        assert abs(bias.item() - 0.5) < 1e-6

    def test_eo_gap_torch(self, sample_data_torch):
        y_pred, y_true, protected_attribute = sample_data_torch
        bias = calculate_bias_metric_torch(
            BiasMetrics.EO_GAP, y_pred, y_true, protected_attribute
        )
        # Group A (protected=1): TPR = 2/2 = 1.0
        # Group B (protected=0): TPR = 1/2 = 0.5
        # Group A (protected=1): FPR = 0/1 = 0.0
        # Group B (protected=0): FPR = 1/1 = 1.0
        # max(|1.0 - 0.5|, |0.0 - 1.0|) = 1.0
        assert abs(bias.item() - 1.0) < 1e-6

    def test_dp_gap_torch(self, sample_data_torch):
        y_pred, y_true, protected_attribute = sample_data_torch
        bias = calculate_bias_metric_torch(
            BiasMetrics.DP_GAP, y_pred, y_true, protected_attribute
        )
        # Group A (protected=1): PPR = 2/3 = 0.666
        # Group B (protected=0): PPR = 2/3 = 0.666
        # |0.666 - 0.666| = 0.0
        assert abs(bias.item()) < 1e-6

    def test_invalid_metric(self, sample_data_torch):
        y_pred, y_true, protected_attribute = sample_data_torch
        with pytest.raises(ValueError):
            calculate_bias_metric_torch(
                "INVALID_METRIC", y_pred, y_true, protected_attribute
            )

    def test_stabilize(self):
        zero = torch.tensor(0.0)
        one = torch.tensor(1.0)
        assert abs(stabilize(zero) - 1e-4).item() < 1e-10
        assert abs(stabilize(one) - 1.0001).item() <= 2e-4

    def test_edge_case_torch(self, edge_case_data_torch):
        y_pred, y_true, protected_attribute = edge_case_data_torch
        bias = calculate_bias_metric_torch(
            BiasMetrics.TPR_GAP, y_pred, y_true, protected_attribute
        )
        assert not torch.isnan(bias)
