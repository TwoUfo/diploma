import math
import unittest

import numpy as np
import pandas as pd

from src.data.preprocessing import JUPITER_A, compute_tisserand, preprocess


class PreprocessingTests(unittest.TestCase):
    def test_leak_and_identifier_columns_are_excluded_from_features(self):
        raw = pd.DataFrame(
            {
                "id": ["a000001"],
                "full_name": ["1 Ceres"],
                "class": ["MBA"],
                "H": [3.4],
                "diameter": [939.0],
                "albedo": [0.09],
                "rot_per": [9.07],
                "neo": ["N"],
                "pha": ["N"],
                "a": [2.77],
                "e": [0.08],
                "i": [10.6],
                "ma": [95.0],
                "moid": [1.59],
                "rms": [0.4],
                "sigma_a": [1e-9],
                "q": [2.55],   # leak: перигелійна відстань (функція a, e)
                "ad": [2.98],  # leak: афелійна відстань
                "n": [0.21],   # leak: середній рух
            }
        )

        features, *_ = preprocess(raw)

        # Цілі та leak-колонки не повинні стати ознаками.
        for leaked in ("albedo", "diameter", "rot_per", "q", "ad", "n"):
            self.assertNotIn(leaked, features.columns)
        # Колонки-ідентифікатори також відсутні.
        for ident in ("id", "full_name", "class"):
            self.assertNotIn(ident, features.columns)
        # H лишається ознакою.
        self.assertIn("H", features.columns)

    def test_compute_tisserand_matches_formula(self):
        df = pd.DataFrame({"a": [2.5], "e": [0.1], "i": [10.0]})
        actual = compute_tisserand(df).iloc[0]

        expected = (
            JUPITER_A / 2.5
            + 2
            * math.cos(math.radians(10.0))
            * math.sqrt(2.5 / JUPITER_A * (1 - 0.1**2))
        )

        self.assertAlmostEqual(actual, expected)

    def test_preprocess_keeps_h_as_feature_and_extracts_current_targets(self):
        raw = pd.DataFrame(
            {
                "class": ["MBA", "HYA", "APO"],
                "H": [15.0, 12.0, np.nan],
                "diameter": [3.0, np.nan, 1.0],
                "albedo": [0.09, 0.04, 0.2],
                "rot_per": [8.0, np.nan, 5.0],
                "neo": ["N", "Y", "Y"],
                "pha": ["N", None, "N"],
                "a": [2.4, 1.8, 1.1],
                "e": [0.1, 0.2, 0.3],
                "i": [5.0, 10.0, 2.0],
                "ma": [100.0, 50.0, 12.0],
                "moid": [0.4, np.nan, 0.01],
                "rms": [0.2, 0.1, 0.3],
                "sigma_a": [0.001, np.nan, 0.002],
                "q": [2.16, 1.44, 0.77],
            }
        )

        features, target_class, target_diameter, target_albedo, target_rot = preprocess(raw)

        self.assertEqual(len(features), 2)
        self.assertIn("H", features.columns)
        self.assertIn("tisserand_j", features.columns)
        self.assertNotIn("albedo", features.columns)
        self.assertNotIn("diameter", features.columns)
        self.assertNotIn("rot_per", features.columns)
        self.assertEqual(target_class.tolist(), ["MBA", "Rare"])
        self.assertAlmostEqual(target_diameter.iloc[0], np.log1p(3.0))
        self.assertTrue(np.isnan(target_diameter.iloc[1]))
        self.assertAlmostEqual(target_albedo.iloc[0], np.log(0.09))
        self.assertAlmostEqual(target_rot.iloc[0], np.log(8.0))
        self.assertFalse(features.isna().any().any())


if __name__ == "__main__":
    unittest.main()
