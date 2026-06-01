import unittest

import torch

from src.models.mtl_model import AsteroidMTLModel, mdn_map, mdn_nll


class AsteroidMTLModelTests(unittest.TestCase):
    def test_forward_returns_expected_shapes_and_valid_mixture_weights(self):
        model = AsteroidMTLModel(
            n_features=24,
            n_classes=10,
            backbone_layers=[16, 8],
            backbone_dropouts=[0.0, 0.0],
            head_hidden=4,
            head_dropout=0.0,
            rot_mdn_components=3,
        )
        model.eval()
        x = torch.randn(5, 24)

        with torch.no_grad():
            out = model(x)

        self.assertEqual(out["class_logits"].shape, (5, 10))
        self.assertEqual(out["diameter_pred"].shape, (5,))
        self.assertEqual(out["albedo_pred"].shape, (5,))
        self.assertEqual(out["rot_pred"].shape, (5,))
        self.assertEqual(out["rot_log_pi"].shape, (5, 3))
        self.assertEqual(out["rot_mu"].shape, (5, 3))
        self.assertEqual(out["rot_log_sigma"].shape, (5, 3))
        self.assertEqual(out["shared_repr"].shape, (5, 8))
        self.assertTrue(torch.allclose(out["rot_log_pi"].exp().sum(dim=-1), torch.ones(5)))

    def test_albedo_prior_uses_true_class_labels_during_training_path(self):
        prior = torch.tensor([-3.0, -1.0])
        model = AsteroidMTLModel(
            n_features=3,
            n_classes=2,
            backbone_layers=[4],
            backbone_dropouts=[0.0],
            head_hidden=4,
            head_dropout=0.0,
            class_albedo_prior=prior,
        )
        model.eval()
        for parameter in model.albedo_residual_head.parameters():
            parameter.data.zero_()

        x = torch.randn(2, 3)
        labels = torch.tensor([0, 1])

        with torch.no_grad():
            out = model(x, class_label=labels)

        self.assertTrue(torch.allclose(out["albedo_pred"], prior[labels]))

    def test_mdn_map_selects_mu_of_most_likely_component(self):
        log_pi = torch.log(torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]))
        mu = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        pred = mdn_map(log_pi, mu)

        self.assertTrue(torch.equal(pred, torch.tensor([2.0, 4.0])))

    def test_mdn_nll_respects_zero_mask(self):
        log_pi = torch.log_softmax(torch.randn(2, 3), dim=-1)
        mu = torch.randn(2, 3)
        log_sigma = torch.zeros(2, 3)
        target = torch.randn(2)
        mask = torch.zeros(2)

        loss = mdn_nll(log_pi, mu, log_sigma, target, mask)

        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
