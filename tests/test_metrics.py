import unittest

import numpy as np

from src.training.metrics import regression_metrics


class RegressionMetricsTests(unittest.TestCase):
    def test_regression_metrics_applies_mask(self):
        # Спостережені — лише позиції 0 і 2; позиції 1 і 3 мають
        # навмисно «дикі» прогнози, які мали б зіпсувати похибку,
        # якби маска не застосовувалася.
        y_true = np.array([5.0, 0.0, 7.0, 0.0])
        y_pred = np.array([6.0, 999.0, 9.0, 999.0])
        mask = np.array([1.0, 0.0, 1.0, 0.0])

        result = regression_metrics(y_true, y_pred, mask)

        # MAE по спостережених: (|5-6| + |7-9|) / 2 = 1.5
        self.assertAlmostEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt((1.0 + 4.0) / 2.0))

    def test_regression_metrics_returns_zero_when_nothing_observed(self):
        y_true = np.array([5.0, 7.0])
        y_pred = np.array([6.0, 9.0])
        mask = np.zeros(2)

        result = regression_metrics(y_true, y_pred, mask)

        self.assertEqual(result, {"mae": 0.0, "rmse": 0.0, "r2": 0.0})


if __name__ == "__main__":
    unittest.main()
