import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glum import (
    GeneralizedLinearRegressor,
    GeneralizedLinearRegressorCV,
    TweedieDistribution,
)

from tweedie_regr.pure_premium import (
    cli,
    evaluate_predictions,
    exposure_balanced_double_lift_table,
    exposure_balanced_lift_table,
    fit_glum_predictive,
    lift_diagnostics,
    lightgbm_params,
    lorenz_curve,
    lorenz_dominance_table,
    make_splits,
    portfolio_calibration_table,
    prepare_mtpl_data,
    save_core_figures,
    select_tweedie_power,
    target_and_weight,
)


def synthetic_pricing_frame(n_rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = np.arange(n_rows)
    exposure = 0.5 + (index % 5) / 10
    claim_nb = np.where(index % 4 == 0, 0, 1 + (index % 3 == 0))
    claim_nb[-1] = 0
    claim_amount = claim_nb * (1_000 + 5 * index)
    severity = np.full(n_rows, np.nan)
    np.divide(claim_amount, claim_nb, out=severity, where=claim_nb > 0)
    frame = pd.DataFrame(
        {
            "ClaimNb": claim_nb,
            "Exposure": exposure,
            "Frequency": claim_nb / exposure,
            "Severity": severity,
            "PurePremium": claim_amount / exposure,
            "ClaimAmountCapped": claim_amount,
            "VehAge": rng.uniform(1, 20, n_rows),
            "DrivAge": rng.uniform(20, 80, n_rows),
            "BonusMalus": rng.uniform(50, 100, n_rows),
            "LogDensity": rng.normal(4, 0.5, n_rows),
            "VehBrand": rng.choice(("B1", "B2", "B3"), n_rows),
            "VehPower": rng.choice(("P1", "P2", "P3"), n_rows),
            "VehGas": rng.choice(("Diesel", "Regular"), n_rows),
            "Region": rng.choice(("R1", "R2", "R3"), n_rows),
            "Area": rng.choice(("A", "B"), n_rows),
        }
    )
    frame.loc[n_rows - 1, "VehBrand"] = "rare-no-claim"
    for column in ("VehBrand", "VehPower", "VehGas", "Region", "Area"):
        frame[column] = frame[column].astype("category")
    return frame


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

    @patch("tweedie_regr.pure_premium.main")
    def test_cli_parses_and_forwards_options(self, mocked_main) -> None:
        cli(
            [
                "--n-samples",
                "100",
                "--n-alphas",
                "5",
                "--max-rounds",
                "20",
                "--early-stopping-rounds",
                "3",
                "--output-dir",
                "custom-output",
            ]
        )

        mocked_main.assert_called_once_with(
            n_samples=100,
            n_alphas=5,
            max_rounds=20,
            early_stopping_rounds=3,
            output_dir=Path("custom-output"),
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

    def test_double_lift_rejects_invalid_inputs(self) -> None:
        observed = np.array([1.0, 2.0])
        prediction = np.array([1.0, 2.0])
        exposure = np.ones(2)

        with self.assertRaisesRegex(ValueError, "n_bins"):
            exposure_balanced_double_lift_table(
                observed, prediction, prediction, exposure, n_bins=0
            )
        for invalid_exposure in (
            np.array([1.0, 0.0]),
            np.array([1.0, -1.0]),
            np.array([1.0, np.nan]),
            np.array([1.0, np.inf]),
        ):
            with (
                self.subTest(exposure=invalid_exposure),
                self.assertRaisesRegex(ValueError, "exposure"),
            ):
                exposure_balanced_double_lift_table(
                    observed, prediction, prediction, invalid_exposure
                )
        for invalid_prediction in (
            np.array([1.0, 0.0]),
            np.array([1.0, -1.0]),
            np.array([1.0, np.nan]),
            np.array([1.0, np.inf]),
        ):
            for side in ("a", "b"):
                with (
                    self.subTest(prediction=invalid_prediction, side=side),
                    self.assertRaisesRegex(ValueError, f"model_{side}_prediction"),
                ):
                    exposure_balanced_double_lift_table(
                        observed,
                        invalid_prediction if side == "a" else prediction,
                        invalid_prediction if side == "b" else prediction,
                        exposure,
                    )

    def test_evaluate_predictions_contract(self) -> None:
        prediction = np.ones(len(self.data))
        result = evaluate_predictions(
            self.data,
            "frequency",
            {"constant": prediction},
            tweedie_power=1.5,
        )
        self.assertEqual(list(result.index), ["constant"])
        self.assertTrue(np.isfinite(result.to_numpy()).all())

        with self.assertRaisesRegex(ValueError, "length"):
            evaluate_predictions(
                self.data,
                "frequency",
                {"short": prediction[:-1]},
                tweedie_power=1.5,
            )
        for invalid in (0.0, np.nan, np.inf):
            bad = prediction.copy()
            bad[0] = invalid
            with (
                self.subTest(prediction=invalid),
                self.assertRaisesRegex(ValueError, "non-finite or non-positive"),
            ):
                evaluate_predictions(
                    self.data,
                    "frequency",
                    {"bad": bad},
                    tweedie_power=1.5,
                )

    def test_severity_fit_adds_zero_weight_anchor_for_no_claim_category(self) -> None:
        data = synthetic_pricing_frame()
        captured_weight = None
        original_fit = GeneralizedLinearRegressorCV.fit

        def capture_fit(model, rows, target, sample_weight=None, **kwargs):
            nonlocal captured_weight
            captured_weight = np.asarray(sample_weight, dtype=float).copy()
            return original_fit(
                model,
                rows,
                target,
                sample_weight=sample_weight,
                **kwargs,
            )

        with patch.object(GeneralizedLinearRegressorCV, "fit", new=capture_fit):
            model = fit_glum_predictive(data, "severity", 1.5, n_alphas=2)

        self.assertIsNotNone(captured_weight)
        self.assertEqual(np.count_nonzero(captured_weight == 0), 1)
        prediction = model.predict(data)
        self.assertTrue(np.all(np.isfinite(prediction) & (prediction > 0)))

    def test_select_tweedie_power_profiles_supplied_candidates(self) -> None:
        data = synthetic_pricing_frame()
        data.loc[data.index[-1], "VehBrand"] = "B1"
        data["VehBrand"] = data["VehBrand"].cat.remove_unused_categories()
        powers = (1.3, 1.7)
        with patch(
            "tweedie_regr.pure_premium.RATING_FORMULA",
            "VehAge + DrivAge + LogDensity",
        ):
            selected, profile = select_tweedie_power(
                data.iloc[:50],
                data.iloc[50:],
                powers=powers,
            )
        self.assertEqual(list(profile["power"]), list(powers))
        self.assertEqual(len(profile), 2)
        self.assertTrue(np.isfinite(profile.to_numpy()).all())
        self.assertIn(selected, powers)

    def test_save_core_figures_contract_and_style_scope(self) -> None:
        data = synthetic_pricing_frame(24)
        pure_premium_predictions = {
            "GLUM frequency x severity": np.full(len(data), 1_000.0),
            "LightGBM frequency x severity": np.full(len(data), 1_100.0),
            "GLUM Tweedie": np.full(len(data), 1_200.0),
            "LightGBM Tweedie": np.full(len(data), 1_300.0),
        }
        missing = pure_premium_predictions.copy()
        missing.pop("GLUM Tweedie")
        extra = {**pure_premium_predictions, "extra": np.ones(len(data))}
        with TemporaryDirectory() as temp_dir:
            for invalid in (missing, extra):
                with (
                    self.subTest(keys=set(invalid)),
                    self.assertRaisesRegex(ValueError, "contain exactly"),
                ):
                    save_core_figures(data, {}, {}, invalid, Path(temp_dir))

            frequency_predictions = {
                "GLUM": np.ones(len(data)),
                "LightGBM": np.full(len(data), 1.1),
            }
            severity_rows, _, _ = target_and_weight(data, "severity")
            severity_predictions = {
                "GLUM": np.full(len(severity_rows), 1_000.0),
                "LightGBM": np.full(len(severity_rows), 1_100.0),
            }
            style_before = {
                key: plt.rcParams[key]
                for key in ("axes.facecolor", "axes.prop_cycle", "grid.color")
            }
            save_core_figures(
                data,
                frequency_predictions,
                severity_predictions,
                pure_premium_predictions,
                Path(temp_dir),
            )
            self.assertEqual(
                style_before,
                {key: plt.rcParams[key] for key in style_before},
            )
            self.assertEqual(len(list(Path(temp_dir).glob("*.png"))), 4)

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
