import numpy as np
import pytest
import torch

from src.detoxai.metrics.metrics import comprehensive_metrics_torch


class TestBiasMetrics:
    @pytest.fixture
    def sample_data_torch(self):
        y_pred = torch.tensor([1, 0, 1, 0, 1, 1])
        y_true = torch.tensor([1, 0, 1, 1, 0, 1])
        protected_attribute = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        return y_pred, y_true, protected_attribute

    @pytest.fixture
    def edge_case_data_torch(self):
        # All zeros to test stabilize function
        y_pred = torch.zeros(6, dtype=torch.int64)
        y_true = torch.zeros(6, dtype=torch.int64)
        protected_attribute = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        return y_pred, y_true, protected_attribute

    def test_performance_metrics_torch(self, sample_data_torch):
        y_pred, y_true, protected_attribute = sample_data_torch
        res = comprehensive_metrics_torch(y_true, y_pred, protected_attribute)

        expected_accuracy = 4 / 6
        expected_recall = 3 / 4
        expected_precision = 3 / 4

        assert np.isclose(res["Accuracy"], expected_accuracy)
        assert np.isclose(res["Recall"], expected_recall)
        assert np.isclose(res["Precision"], expected_precision)

    def test_performance_metrics_torch_edgecase(self, edge_case_data_torch):
        y_pred, y_true, protected_attribute = edge_case_data_torch
        res = comprehensive_metrics_torch(y_true, y_pred, protected_attribute)

        expected_accuracy = 1
        expected_recall = 0
        expected_precision = 0

        assert np.isclose(res["Accuracy"], expected_accuracy)
        assert np.isclose(res["Recall"], expected_recall)
        assert np.isclose(res["Precision"], expected_precision)

    def test_fairness_metrics_torch(self, sample_data_torch):
        y_pred, y_true, protected_attribute = sample_data_torch
        res = comprehensive_metrics_torch(
            y_true, y_pred, protected_attribute, return_torch=False
        )

        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        protected_attribute = protected_attribute.numpy()

        expected_EO = 1.0
        expected_EOO = 0.5
        expected_DP = 0.0
        expected_ACC_PAR = 2 / 3

        assert np.isclose(res["Equal_opportunity"], expected_EOO)
        assert np.isclose(res["Equalized_odds"], expected_EO)
        assert np.isclose(res["Demographic_parity"], expected_DP)
        assert np.isclose(res["Accuracy_parity"], expected_ACC_PAR)

    def test_fairness_metrics_torch_edgecase(self, edge_case_data_torch):
        y_pred, y_true, protected_attribute = edge_case_data_torch
        res = comprehensive_metrics_torch(
            y_true, y_pred, protected_attribute, return_torch=False
        )
        expected_EO = 0.0
        expected_EOO = 0.0
        expected_DP = 0.0
        expected_ACC_PAR = 0.0

        assert np.isclose(res["Equal_opportunity"], expected_EOO)
        assert np.isclose(res["Equalized_odds"], expected_EO)
        assert np.isclose(res["Demographic_parity"], expected_DP)
        assert np.isclose(res["Accuracy_parity"], expected_ACC_PAR)

    def test_return_torch_type(self, sample_data_torch):
        y_pred, y_true, protected_attribute = sample_data_torch
        t = comprehensive_metrics_torch(
            y_true, y_pred, protected_attribute, return_torch=True
        )
        nt = comprehensive_metrics_torch(
            y_true, y_pred, protected_attribute, return_torch=False
        )
        assert isinstance(t["Accuracy"], torch.Tensor)
        assert isinstance(nt["Accuracy"], float)
