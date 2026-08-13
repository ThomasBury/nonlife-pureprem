import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution

from tweedie_regr.pure_premium import (
    exposure_balanced_double_lift_table,
    exposure_balanced_lift_table,
    lift_diagnostics,
    lightgbm_params,
    lorenz_curve,
    lorenz_dominance_table,
    make_splits,
    portfolio_calibration_table,
    prepare_mtpl_data,
    target_and_weight,
)


class PurePremiumTutorialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frequency = pd.DataFrame(
            {
                "IDpol": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "ClaimNb": [2, 1, 1, 0, 0, 0, 1, 0, 1, 0],
                "Exposure": [1.2, 0.5, 1.0, 0.8, 0.7, 0.6, 0.9, 0.4, 0.3, 0.2],
                "Density": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )
        severity = pd.DataFrame(
            {
                "IDpol": [1, 1, 2, 3, 7, 9],
                "ClaimAmount": [120_000, 50_000, 0, 250_000, 10_000, 20_000],
            }
        )
        cls.data, cls.report = prepare_mtpl_data(frequency, severity)

    def test_per_claim_cap_and_reconciliation(self) -> None:
        self.assertEqual(self.data.loc[1, "ClaimAmountCapped"], 150_000)
        self.assertEqual(self.data.loc[3, "ClaimAmountCapped"], 100_000)
        self.assertEqual(self.data.loc[2, "ClaimNb"], 0)
        self.assertEqual(self.report["exposure_capped_rows"], 1)
        self.assertEqual(self.report["claim_count_mismatch_rows"], 1)
        self.assertAlmostEqual(
            self.report["capped_loss_share"],
            (450_000 - 280_000) / 450_000,
        )

    def test_targets_reconstruct_capped_totals(self) -> None:
        np.testing.assert_allclose(
            self.data["Frequency"] * self.data["Exposure"],
            self.data["ClaimNb"],
        )
        claims = self.data["ClaimNb"] > 0
        np.testing.assert_allclose(
            self.data.loc[claims, "Severity"] * self.data.loc[claims, "ClaimNb"],
            self.data.loc[claims, "ClaimAmountCapped"],
        )
        np.testing.assert_allclose(
            self.data["PurePremium"] * self.data["Exposure"],
            self.data["ClaimAmountCapped"],
        )

    def test_target_and_weight_definitions(self) -> None:
        frequency_rows, frequency, frequency_weight = target_and_weight(
            self.data, "frequency"
        )
        severity_rows, severity, severity_weight = target_and_weight(
            self.data, "severity"
        )
        premium_rows, premium, premium_weight = target_and_weight(
            self.data, "pure_premium"
        )
        self.assertEqual(len(frequency_rows), len(self.data))
        self.assertEqual(len(premium_rows), len(self.data))
        self.assertTrue((severity_rows["ClaimNb"] > 0).all())
        np.testing.assert_array_equal(frequency_weight, frequency_rows["Exposure"])
        np.testing.assert_array_equal(severity_weight, severity_rows["ClaimNb"])
        np.testing.assert_array_equal(premium_weight, premium_rows["Exposure"])
        np.testing.assert_allclose(
            frequency * frequency_weight, frequency_rows["ClaimNb"]
        )
        np.testing.assert_allclose(
            severity * severity_weight, severity_rows["ClaimAmountCapped"]
        )
        np.testing.assert_allclose(
            premium * premium_weight, premium_rows["ClaimAmountCapped"]
        )

    def test_split_isolation_and_reproducibility(self) -> None:
        first = make_splits(self.data)
        second = make_splits(self.data)
        self.assertEqual(set(first.test.index), set(second.test.index))
        self.assertTrue(set(first.train.index).isdisjoint(first.test.index))
        self.assertTrue(
            set(first.development_train.index).isdisjoint(
                first.development_validation.index
            )
        )
        self.assertTrue(set(first.development_train.index).issubset(first.train.index))
        self.assertTrue(
            set(first.development_validation.index).issubset(first.train.index)
        )

    def test_exposure_balanced_lift_and_lorenz_endpoints(self) -> None:
        observed = np.array([0.0, 1.0, 4.0, 9.0])
        predicted = np.array([0.5, 1.5, 3.0, 8.0])
        exposure = np.array([0.5, 1.5, 2.0, 4.0])
        lift = exposure_balanced_lift_table(observed, predicted, exposure, n_bins=4)
        np.testing.assert_allclose(lift["exposure"], exposure.sum() / 4)
        self.assertAlmostEqual(
            float((lift["observed_rate"] * lift["exposure"]).sum()),
            float((observed * exposure).sum()),
        )
        x, y = lorenz_curve(observed, predicted, exposure)
        np.testing.assert_array_equal([x[0], y[0]], [0.0, 0.0])
        np.testing.assert_array_equal([x[-1], y[-1]], [1.0, 1.0])

    def test_calibration_lorenz_lift_and_double_lift_diagnostics(self) -> None:
        prediction = {"model": np.full(len(self.data), 2.0)}
        calibration = portfolio_calibration_table(self.data, "frequency", prediction)
        observed = float(self.data["ClaimNb"].sum())
        self.assertAlmostEqual(
            float(calibration.loc["model", "observed_total"]), observed
        )
        self.assertAlmostEqual(
            float(calibration.loc["model", "predicted_total"]),
            2 * self.data["Exposure"].sum(),
        )

        y_true = np.array([1.0, 2.0, 10.0, 20.0])
        weight = np.ones(4)
        dominance = lorenz_dominance_table(
            y_true,
            {
                "good": np.array([1.0, 2.0, 10.0, 20.0]),
                "bad": np.array([20.0, 10.0, 2.0, 1.0]),
            },
            weight,
        )
        self.assertEqual(dominance.loc[0, "result"], "good dominates bad")
        crossing = lorenz_dominance_table(
            y_true,
            {"a": np.array([1.0, 4.0, 2.0, 8.0]), "b": np.array([2.0, 1.0, 8.0, 4.0])},
            weight,
        )
        self.assertEqual(crossing.loc[0, "result"], "crossing; no dominance")

        diagnostics = lift_diagnostics(y_true, {"model": y_true}, weight, n_bins=4)
        self.assertGreater(float(diagnostics.loc["model", "observed_range"]), 0)
        double_lift = exposure_balanced_double_lift_table(
            y_true, y_true, y_true[::-1], weight, n_bins=4
        )
        self.assertTrue(np.all(np.diff(double_lift["prediction_ratio"]) >= 0))
        self.assertAlmostEqual(
            float(double_lift["exposure"].sum()), float(weight.sum())
        )

    def test_tiny_glum_and_lightgbm_family_fits(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(60, 2))
        weight = rng.uniform(0.2, 1.0, size=60)
        mean = np.exp(0.2 + 0.3 * x[:, 0] - 0.2 * x[:, 1])
        targets = {
            "frequency": rng.poisson(mean * weight) / weight,
            "severity": rng.gamma(shape=2.0, scale=mean / 2.0),
            "pure_premium": np.where(
                rng.random(60) < 0.6,
                0.0,
                rng.gamma(shape=2.0, scale=mean / 2.0),
            ),
        }
        families = {
            "frequency": "poisson",
            "severity": "gamma",
            "pure_premium": TweedieDistribution(1.5),
        }
        for component, target in targets.items():
            glum_model = GeneralizedLinearRegressor(
                family=families[component],
                link="log",
                alpha=0,
            ).fit(x, target, sample_weight=weight)
            glum_prediction = glum_model.predict(x)
            self.assertTrue(np.all(np.isfinite(glum_prediction)))
            self.assertTrue(np.all(glum_prediction > 0))

            booster = lgb.train(
                lightgbm_params(component, 1.5, 7, 5),
                lgb.Dataset(x, label=target, weight=weight),
                num_boost_round=5,
            )
            lightgbm_prediction = np.asarray(booster.predict(x), dtype=float)
            self.assertTrue(np.all(np.isfinite(lightgbm_prediction)))
            self.assertTrue(np.all(lightgbm_prediction > 0))


if __name__ == "__main__":
    unittest.main()
