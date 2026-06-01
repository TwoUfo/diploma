import unittest

import torch

from src.models.losses import MultiTaskLoss


class MultiTaskLossTests(unittest.TestCase):
    def test_masked_mse_ignores_unobserved_targets(self):
        criterion = MultiTaskLoss(n_classes=2)
        pred = torch.tensor([10.0, 99.0, 20.0])
        target = torch.tensor([12.0, -1000.0, 18.0])
        mask = torch.tensor([1.0, 0.0, 1.0])

        loss = criterion._masked_mse(pred, target, mask)

        self.assertAlmostEqual(loss.item(), 4.0)

    def test_masked_mse_returns_zero_when_every_target_is_missing(self):
        criterion = MultiTaskLoss(n_classes=2)
        pred = torch.tensor([10.0, 99.0])
        target = torch.tensor([12.0, -1000.0])
        mask = torch.tensor([0.0, 0.0])

        loss = criterion._masked_mse(pred, target, mask)

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(loss.device, pred.device)

    def test_forward_returns_total_loss_and_all_reported_components(self):
        criterion = MultiTaskLoss(n_classes=2)

        pred_class = torch.tensor([[3.0, 0.1], [0.2, 2.0]])
        pred_diam = torch.tensor([1.0, 2.0])
        pred_albedo = torch.tensor([-2.0, -1.5])
        rot_log_pi = torch.log_softmax(torch.tensor([[2.0, 0.5], [0.1, 1.0]]), dim=-1)
        rot_mu = torch.tensor([[0.9, 1.4], [2.1, 2.4]])
        rot_log_sigma = torch.zeros(2, 2)

        loss, metrics = criterion(
            pred_class,
            pred_diam,
            pred_albedo,
            rot_log_pi,
            rot_mu,
            rot_log_sigma,
            true_class=torch.tensor([0, 1]),
            true_diam=torch.tensor([1.1, 0.0]),
            true_albedo=torch.tensor([-2.2, 0.0]),
            true_rot=torch.tensor([1.0, 0.0]),
            mask_diam=torch.tensor([1.0, 0.0]),
            mask_albedo=torch.tensor([1.0, 0.0]),
            mask_rot=torch.tensor([1.0, 0.0]),
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(metrics),
            {
                "loss_total",
                "loss_class",
                "loss_diam",
                "loss_albedo",
                "loss_rot",
                "sigma_class",
                "sigma_diam",
                "sigma_albedo",
                "sigma_rot",
            },
        )


if __name__ == "__main__":
    unittest.main()
