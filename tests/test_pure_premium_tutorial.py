import re
import unittest
from contextlib import ExitStack
from decimal import Decimal, localcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glum import (
    GeneralizedLinearRegressor,
    GeneralizedLinearRegressorCV,
    TweedieDistribution,
)
from scipy.optimize import OptimizeResult
from scipy.special import logsumexp
from scipy.stats import gamma, nbinom, ncx2, poisson

from nonlife_pureprem.pure_premium import (
    CONSOLE,
    _set_log_y_observed_floor,
    cli,
    conditional_dispersion_diagnostic,
    conditional_dispersion_table,
    evaluate_predictions,
    exposure_balanced_double_lift_table,
    exposure_balanced_lift_table,
    fit_gamma_dispersion,
    fit_glum_predictive,
    fit_negative_binomial_shape,
    fit_tweedie_dispersion,
    gamma_ccdf_diagnostic,
    gamma_cdf_diagnostic,
    gini,
    grouped_calibration,
    hexbin_grid,
    lift_diagnostics,
    lightgbm_params,
    lorenz_curve,
    lorenz_dominance_table,
    make_splits,
    poisson_ccdf_diagnostic,
    poisson_cdf_diagnostic,
    portfolio_calibration_table,
    prepare_mtpl_data,
    print_frame,
    save_core_figures,
    select_tweedie_power,
    target_and_weight,
    tweedie_ccdf_diagnostic,
    tweedie_cdf_diagnostic,
    tweedie_cdf_series,
    tweedie_log_likelihood,
    tweedie_sf_series,
    weighted_empirical_ccdf,
    weighted_empirical_cdf,
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
            "ClaimAmountOriginal": claim_amount,
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
        cls.data, cls.claims, cls.report = prepare_mtpl_data(frequency, severity)

    def test_per_claim_cap_and_reconciliation(self) -> None:
        self.assertEqual(self.data.loc[1, "ClaimAmountCapped"], 150_000)
        self.assertEqual(self.data.loc[3, "ClaimAmountCapped"], 100_000)
        self.assertEqual(self.data.loc[2, "ClaimNb"], 0)
        self.assertEqual(self.report["claim_cap"], 100_000.0)
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

    @patch("nonlife_pureprem.pure_premium.main")
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

    def test_print_frame_preserves_row_labels(self) -> None:
        frame = pd.DataFrame({"policies": [20, 10]}, index=["frequency", "severity"])
        for index in (frame.index, frame.index.rename("component")):
            with CONSOLE.capture() as capture:
                print_frame("Scope", frame.set_axis(index))
            output = capture.get()
            self.assertIn("frequency", output)
            self.assertIn("severity", output)
        with CONSOLE.capture() as capture:
            print_frame("Default rows", frame.reset_index(drop=True))
        self.assertNotIn("index", capture.get())
        with CONSOLE.capture() as capture:
            print_frame(
                "Named rows", frame.reset_index(drop=True).rename_axis("row_label")
            )
        self.assertIn("row_label", capture.get())

    def test_gini_area_normalization_and_tied_permutations(self) -> None:
        observed = np.array([0.0, 2.0])
        prediction = np.array([1.0, 2.0])
        weight = np.ones(2)
        x, loss = lorenz_curve(observed, prediction, weight)
        signed_area = np.trapezoid(x - loss, x)
        np.testing.assert_allclose(signed_area, 0.25)
        np.testing.assert_allclose(gini(observed, prediction, weight), 0.5)
        np.testing.assert_allclose(
            gini(observed, prediction, weight) / gini(observed, observed, weight),
            1.0,
        )
        frame = pd.DataFrame({"PurePremium": observed, "Exposure": weight})
        scores = evaluate_predictions(
            frame, "pure_premium", {"ordered": prediction, "constant": weight}, 1.5
        )
        np.testing.assert_allclose(scores["raw_gini"], [0.5, 0.0], atol=1e-12)
        np.testing.assert_allclose(scores["normalized_gini"], [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(gini(observed, prediction[::-1], weight), -0.5)

        # Fractional weights and tied scores exercise the polygon convention.
        observed = np.array([0.0, 2.0, 9.0, 1.0, 4.0])
        prediction = np.array([1.0, 1.0, 3.0, 3.0, 5.0])
        weight = np.array([0.1, 0.7, 0.3, 1.0, 0.2])
        for scores in (prediction, np.ones(5)):
            x, loss = lorenz_curve(observed, scores, weight)
            raw = gini(observed, scores, weight)
            np.testing.assert_allclose(raw, 2 * np.trapezoid(x - loss, x), atol=1e-12)
            for order in (np.arange(5)[::-1], np.array([1, 0, 3, 2, 4])):
                px, ploss = lorenz_curve(observed[order], scores[order], weight[order])
                np.testing.assert_allclose(px, x, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(ploss, loss, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(
                    gini(observed[order], scores[order], weight[order]),
                    raw,
                    rtol=1e-12,
                    atol=1e-12,
                )
            np.testing.assert_allclose(
                gini(observed, np.log(scores), weight), raw, rtol=1e-12, atol=1e-12
            )

    def test_double_lift_interpretation_counterexamples(self) -> None:
        observed = np.array([1.0, 2.0, 4.0])
        table = exposure_balanced_double_lift_table(
            observed, np.array([0.5, 2.0, 8.0]), observed, np.ones(3), n_bins=3
        )
        np.testing.assert_allclose(table["prediction_ratio"], [0.5, 1.0, 2.0])
        np.testing.assert_allclose(table["observed_rate"], observed)
        np.testing.assert_allclose(table["model_b_rate"], observed)
        np.testing.assert_allclose(table["model_a_rate"], [0.5, 2.0, 8.0])
        self.assertTrue(
            np.all(table["model_a_rate"].iloc[[0, -1]] != observed[[0, -1]])
        )

        table = exposure_balanced_double_lift_table(
            np.array([1.0, 2.0]),
            np.array([1.0, 4.0]),
            np.array([1.0, 2.0]),
            np.ones(2),
            n_bins=1,
        )
        np.testing.assert_allclose(table["prediction_ratio"], 1.5)
        np.testing.assert_allclose(table["model_a_rate"] / table["model_b_rate"], 5 / 3)

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
            "nonlife_pureprem.pure_premium.RATING_FORMULA",
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
            self.assertSetEqual(
                {path.name for path in Path(temp_dir).glob("*.png")},
                {
                    "component_calibration.png",
                    "pure_premium_lorenz.png",
                    "pure_premium_lift.png",
                    "pure_premium_double_lift.png",
                },
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

    def test_weighted_empirical_cdf_endpoints_and_monotonicity(self) -> None:
        values = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
        weights = np.array([0.5, 1.0, 0.5, 1.5, 1.0])
        sorted_x, cdf = weighted_empirical_cdf(values, weights)
        np.testing.assert_array_equal(sorted_x, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(cdf, np.array([2 / 9, 2 / 3, 1.0]), atol=1e-12)
        self.assertEqual(cdf[-1], 1.0)
        self.assertTrue(np.all(np.diff(cdf) >= 0))
        with self.assertRaises(ValueError):
            weighted_empirical_cdf(values, np.zeros_like(weights))

    def test_weighted_empirical_ccdf_inverts_cdf(self) -> None:
        values = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
        weights = np.array([0.5, 1.0, 0.5, 1.5, 1.0])
        sorted_x, ccdf = weighted_empirical_ccdf(values, weights)
        _, cdf = weighted_empirical_cdf(values, weights)
        np.testing.assert_array_equal(sorted_x, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(ccdf + cdf, np.ones_like(ccdf))
        self.assertEqual(ccdf[-1], 0.0)
        self.assertEqual(
            ccdf[0], 1.0 - float(weights[values == 1].sum() / weights.sum())
        )
        self.assertTrue(np.all(np.diff(ccdf) <= 0))

    def test_conditional_curves_match_heterogeneous_policy_references(self) -> None:
        data = pd.DataFrame(
            {
                "ClaimNb": [0, 1, 4, 2],
                "Exposure": [0.1, 0.4, 1.0, 0.7],
                "ClaimAmountCapped": [0.0, 2.0, 12.0, 8.0],
            },
            index=[91, 12, 7, 44],
        )
        data["Frequency"] = data.ClaimNb / data.Exposure
        data["Severity"] = data.ClaimAmountCapped / data.ClaimNb
        data["PurePremium"] = data.ClaimAmountCapped / data.Exposure
        cases = (
            ("frequency", poisson_cdf_diagnostic, poisson_ccdf_diagnostic, (), (0.4,)),
            ("severity", gamma_cdf_diagnostic, gamma_ccdf_diagnostic, (0.7,), (0.7,)),
            (
                "pure_premium",
                tweedie_cdf_diagnostic,
                tweedie_ccdf_diagnostic,
                (1.5, 2.0),
                (1.5, 2.0),
            ),
        )
        with ExitStack() as stack:
            for name in (
                "fit_gamma_dispersion",
                "fit_negative_binomial_shape",
                "fit_tweedie_dispersion",
                "minimize_scalar",
            ):
                stack.enter_context(
                    patch(
                        f"nonlife_pureprem.pure_premium.{name}",
                        side_effect=AssertionError("plotters must not fit"),
                    )
                )
            stack.enter_context(
                patch.object(
                    pd.DataFrame,
                    "sample",
                    side_effect=AssertionError("plotters must not sample"),
                )
            )
            for component, cdf_plot, sf_plot, cdf_args, sf_args in cases:
                rows, target, weight = target_and_weight(data, component)
                mean = pd.Series(np.linspace(1.0, 4.0, len(rows)), index=rows.index)
                values = (
                    rows.ClaimNb
                    if component == "frequency"
                    else rows.ClaimAmountCapped
                    if component == "pure_premium"
                    else target
                )
                for survival, plot, args in (
                    (False, cdf_plot, cdf_args),
                    (True, sf_plot, sf_args),
                ):
                    with self.subTest(component=component, survival=survival):
                        fig, axes = plot(rows, mean, *args)
                        for axis in np.asarray(axes).flat:
                            empirical, reference = axis.lines[:2]
                            x, actual = reference.get_data()
                            self.assertEqual(len(x), 128)
                            probabilities = []
                            for mu, w in zip(mean, weight):
                                if component == "frequency":
                                    probability = (
                                        poisson.sf if survival else poisson.cdf
                                    )(x, w * mu)
                                elif component == "severity":
                                    probability = (gamma.sf if survival else gamma.cdf)(
                                        x, w / 0.7, scale=0.7 * mu / w
                                    )
                                else:
                                    # Independent p=1.5 noncentral-chi-square identity.
                                    lam, beta = w * np.sqrt(mu), np.sqrt(mu)
                                    probability = (ncx2.cdf if survival else ncx2.sf)(
                                        2 * lam, 2, 2 * x / beta
                                    )
                                probabilities.append(probability)
                            np.testing.assert_allclose(
                                actual,
                                np.average(probabilities, axis=0, weights=weight),
                                rtol=2e-9,
                                atol=0,
                            )
                            empirical_x, empirical_y = empirical.get_data()
                            expected = [
                                np.average(
                                    np.asarray(values) > t
                                    if survival
                                    else np.asarray(values) <= t,
                                    weights=weight,
                                )
                                for t in empirical_x
                            ]
                            np.testing.assert_allclose(
                                empirical_y, expected, atol=1e-15
                            )
                            self.assertIn(f"n={len(rows):,}", axis.get_title())
                            if component == "frequency" and survival:
                                expected_nb = np.average(
                                    [
                                        nbinom.sf(x, 0.4, 0.4 / (0.4 + w * mu))
                                        for mu, w in zip(mean, weight)
                                    ],
                                    axis=0,
                                    weights=weight,
                                )
                                np.testing.assert_allclose(
                                    axis.lines[2].get_ydata(), expected_nb
                                )
                        plt.close(fig)
        fig, axis = poisson_ccdf_diagnostic(
            data, pd.Series(1.0, index=data.index), float("inf")
        )
        self.assertIn("Poisson limit", axis.lines[2].get_label())
        np.testing.assert_array_equal(
            axis.lines[1].get_ydata(), axis.lines[2].get_ydata()
        )
        plt.close(fig)

    def test_conditional_diagnostics_require_alignment_and_valid_parameters(
        self,
    ) -> None:
        rows, _, _ = target_and_weight(self.data, "severity")
        mean = pd.Series(10_000.0, index=rows.index)
        for invalid in (
            mean.iloc[::-1],
            mean.iloc[:-1],
            mean.to_numpy(),
            mean * np.nan,
            mean * 0,
        ):
            with self.subTest(mean=invalid), self.assertRaises(ValueError):
                gamma_cdf_diagnostic(rows, invalid, 1.0)
        with self.assertRaisesRegex(ValueError, "eligible"):
            gamma_cdf_diagnostic(self.data, pd.Series(1.0, index=self.data.index), 1.0)
        for invalid in (0, -1, np.nan, np.inf):
            with self.subTest(dispersion=invalid), self.assertRaises(ValueError):
                gamma_cdf_diagnostic(rows, mean, invalid)
        for invalid in (None, 0, -1, np.nan):
            with self.subTest(shape=invalid), self.assertRaises(ValueError):
                poisson_ccdf_diagnostic(
                    self.data, pd.Series(1.0, index=self.data.index), invalid
                )
        for invalid in (1, 2, np.nan):
            with self.subTest(power=invalid), self.assertRaises(ValueError):
                tweedie_cdf_diagnostic(
                    self.data, pd.Series(1.0, index=self.data.index), invalid, 1.0
                )

    def test_conditional_probability_batches_preserve_all_rows(self) -> None:
        data = synthetic_pricing_frame(513)
        mean = pd.Series(np.linspace(0.1, 2, len(data)), index=data.index)
        with patch(
            "nonlife_pureprem.pure_premium.poisson.cdf", wraps=poisson.cdf
        ) as cdf:
            fig, axes = poisson_cdf_diagnostic(data, mean)
        self.assertEqual(cdf.call_count, 32)
        for call in cdf.call_args_list:
            self.assertLessEqual(call.args[0].shape[0], 8)
            self.assertLessEqual(len(call.args[1]), 512)
        x, actual = axes[0].lines[1].get_data()
        expected = np.average(
            poisson.cdf(x[:, None], data.Exposure.to_numpy() * mean.to_numpy()),
            axis=1,
            weights=data.Exposure,
        )
        np.testing.assert_allclose(actual, expected)
        plt.close(fig)

    def test_gamma_and_nb_fixed_mean_fits(self) -> None:
        rng = np.random.default_rng(92)
        n = rng.integers(1, 8, 20_000)
        mean = rng.uniform(20, 100, len(n))
        y = rng.gamma(n / 0.8, 0.8 * mean / n)
        fitted = fit_gamma_dispersion(y, mean, n)
        self.assertAlmostEqual(fitted, 0.8, delta=0.03)

        def likelihood(phi):
            return gamma.logpdf(y, n / phi, scale=phi * mean / n).sum()

        self.assertGreater(likelihood(fitted), likelihood(fitted * 0.99))
        self.assertGreater(likelihood(fitted), likelihood(fitted * 1.01))
        mixed_y, mixed_n = (
            np.array([1.0, 2.0, 10.0, 20.0]),
            np.array([1.0, 2.0, 5.0, 10.0]),
        )
        mixed_mean = np.full(4, 5.0)
        fitted = fit_gamma_dispersion(mixed_y, mixed_mean, mixed_n)
        step = fitted * 1e-4

        def logpdf(phi):
            return gamma.logpdf(
                mixed_y, mixed_n / phi, scale=phi * mixed_mean / mixed_n
            )

        gradient = (logpdf(fitted + step) - logpdf(fitted - step)) / (2 * step)
        self.assertAlmostEqual(float(gradient.sum()), 0, delta=1e-5)
        self.assertGreater(abs(float(gradient @ mixed_n)), 0.01)
        count_mean = rng.uniform(0.2, 3, len(n))
        counts = rng.negative_binomial(0.35, 0.35 / (0.35 + count_mean))
        shape = fit_negative_binomial_shape(counts, count_mean)
        self.assertAlmostEqual(shape, 0.35, delta=0.03)
        self.assertEqual(
            fit_negative_binomial_shape(np.ones(100), np.ones(100)), np.inf
        )

    def test_distribution_fitters_reject_invalid_inputs_and_optimizer_failures(
        self,
    ) -> None:
        for args in (
            ([], [], []),
            ([1], [1, 2], [1]),
            ([0], [1], [1]),
            ([1], [np.nan], [1]),
            ([1], [1], [0.5]),
            ([1], [1], [1]),
        ):
            with self.subTest(gamma=args), self.assertRaises(ValueError):
                fit_gamma_dispersion(*args)
        for args in (([], []), ([1], [1, 2]), ([-1], [1]), ([0.5], [1]), ([1], [0])):
            with self.subTest(nb=args), self.assertRaises(ValueError):
                fit_negative_binomial_shape(*args)
        for result in (
            OptimizeResult(success=False, x=0, fun=1, message="failed"),
            OptimizeResult(success=True, x=np.nan, fun=np.nan, message="invalid"),
        ):
            with patch(
                "nonlife_pureprem.pure_premium.minimize_scalar", return_value=result
            ):
                with self.assertRaises(RuntimeError):
                    fit_gamma_dispersion([1, 3], [2, 2], [1, 2])
                with self.assertRaises(RuntimeError):
                    fit_negative_binomial_shape([0, 2], [1, 1])

    def test_conditional_dispersion_uses_fixed_power_and_row_denominator(self) -> None:
        data = synthetic_pricing_frame(8)
        data["ClaimNb"] = [1, 1, 1, 1, 4, 4, 4, 4]
        data["Exposure"] = [0.25] * 4 + [1.0] * 4
        data["Area"] = ["A"] * 4 + ["B"] * 4
        mean = pd.Series(100.0, index=data.index)
        for component, target, power, phi in (
            ("frequency", "Frequency", 1, 1),
            ("severity", "Severity", 2, 0.2),
            ("pure_premium", "PurePremium", 1.5, 0.8),
        ):
            weight = data.ClaimNb if component == "severity" else data.Exposure
            data[target] = mean + np.tile([-1, 1], 4) * np.sqrt(
                phi * mean**power / weight
            )
            table = conditional_dispersion_table(data, mean, "Area", component, 1.5)
            np.testing.assert_allclose(table.dispersion, phi)
            np.testing.assert_array_equal(table.n, [4, 4])
            np.testing.assert_allclose(
                table.risk_volume, [weight.iloc[:4].sum(), weight.iloc[4:].sum()]
            )
            np.testing.assert_allclose(table.observed_rate, 100)
            np.testing.assert_allclose(table.predicted_rate, 100)
        fig, axes = conditional_dispersion_diagnostic(
            data,
            {c: mean for c in ("frequency", "severity", "pure_premium")},
            "Area",
            1.5,
            {"severity": 0.2, "pure_premium": 0.8},
        )
        for axis, reference in zip(axes, [1, 0.2, 0.8]):
            np.testing.assert_allclose(axis.lines[0].get_ydata(), reference)
        plt.close(fig)

    def test_lift_ties_are_permutation_invariant_and_conserve_totals(self) -> None:
        observed = np.array([0.0, 10.0, 3.0, 20.0, 2.0])
        exposure = np.array([1.0, 4.0, 2.0, 3.0, 1.0])
        score = np.array([1.0, 1.0, 1.0, 3.0, 3.0])
        b = np.array([1.0, 2.0, 4.0, 2.0, 3.0])
        a = score * b
        order = np.array([4, 2, 1, 3, 0])
        for prediction in (score, np.ones(5)):
            table = exposure_balanced_lift_table(
                observed, prediction, exposure, n_bins=4
            )
            permuted = exposure_balanced_lift_table(
                observed[order], prediction[order], exposure[order], n_bins=4
            )
            np.testing.assert_allclose(table, permuted)
            np.testing.assert_allclose(table.exposure, exposure.sum() / 4)
            for column, values in (
                ("observed_rate", observed),
                ("predicted_rate", prediction),
            ):
                self.assertAlmostEqual(
                    table[column] @ table.exposure, values @ exposure
                )
            if np.all(prediction == 1):
                np.testing.assert_allclose(
                    table.observed_rate, np.average(observed, weights=exposure)
                )
        table = exposure_balanced_double_lift_table(observed, a, b, exposure, n_bins=4)
        permuted = exposure_balanced_double_lift_table(
            observed[order], a[order], b[order], exposure[order], n_bins=4
        )
        np.testing.assert_allclose(table, permuted)
        np.testing.assert_allclose(table.exposure, exposure.sum() / 4)
        for column, values in (
            ("observed_rate", observed),
            ("model_a_rate", a),
            ("model_b_rate", b),
            ("prediction_ratio", score),
        ):
            self.assertAlmostEqual(table[column] @ table.exposure, values @ exposure)
        # First tied block crosses two complete bins and part of a third.
        np.testing.assert_allclose(
            table.observed_rate.iloc[:2], np.average(observed[:3], weights=exposure[:3])
        )

    def test_lift_tables_reject_empty_misaligned_and_nonfinite_inputs(self) -> None:
        for lift, args in (
            (exposure_balanced_lift_table, ([1.0, 2.0], [1.0, 2.0], [1.0, 1.0])),
            (
                exposure_balanced_double_lift_table,
                ([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 1.0]),
            ),
        ):
            for bad in ([], [1.0], [[1.0, 2.0]], [np.nan, 1.0], [-1.0, 1.0]):
                for position in range(len(args)):
                    invalid = list(args)
                    invalid[position] = bad
                    with (
                        self.subTest(lift=lift.__name__, position=position, bad=bad),
                        self.assertRaises(ValueError),
                    ):
                        lift(*invalid)
            for bins in (0, 1.5):
                with self.assertRaises(ValueError):
                    lift(*args, n_bins=bins)

    def test_age_bands_remain_chronological(self) -> None:
        data = synthetic_pricing_frame()
        data["DrivAge"] = np.linspace(18, 100, len(data))
        table = grouped_calibration(
            data, "frequency", {"model": np.ones(len(data))}, n_bins=10
        )
        self.assertIsInstance(table.index, pd.CategoricalIndex)
        self.assertTrue(table.index.ordered)
        self.assertTrue(np.all(np.diff([band.left for band in table.index]) > 0))
        permuted = data.sample(frac=1, random_state=3)
        pd.testing.assert_frame_equal(
            table,
            grouped_calibration(permuted, "frequency", {"model": np.ones(len(data))}),
        )

    @patch("nonlife_pureprem.pure_premium.plt.show")
    def test_hexbin_zero_constant_and_empty_panels(self, _show) -> None:
        fig, axes = hexbin_grid(
            {
                "zero outcomes": (np.ones(5), np.zeros(5)),
                "outside extent": (np.ones(5), np.full(5, 100)),
                "empty": ([], []),
                "all zero": (np.zeros(5), np.zeros(5)),
            },
            "predicted",
            "observed",
            extent_mode="diag",
        )
        fig.canvas.draw()
        for axis in (axes.flat[0], axes.flat[3]):
            self.assertLessEqual(axis.get_ylim()[0], 0)
            self.assertGreater(axis.get_xlim()[1], axis.get_xlim()[0])
            self.assertEqual(axis.collections[0].get_array().sum(), 5)
        self.assertEqual(axes.flat[1].collections[0].get_array().size, 0)
        self.assertTrue(axes.flat[2].texts)
        plt.close(fig)
        fig, axes = hexbin_grid(
            {"no positives": (np.ones(5), np.zeros(5))},
            "predicted",
            "observed",
            logx=True,
            logy=True,
        )
        fig.canvas.draw()
        self.assertTrue(axes.flat[0].texts)
        self.assertFalse(axes.flat[0].collections)
        plt.close(fig)

    def test_chapter_diagnostics_fit_all_training_rows_then_reuse_samples(self) -> None:
        chapter = (
            Path(__file__).parents[1] / "book/pure_premium_tutorial.qmd"
        ).read_text()
        cells = re.findall(r"```\{python\}\n(.*?)\n```", chapter, re.DOTALL)
        labelled = {
            match[1]: cell
            for cell in cells
            if (match := re.search(r"^#\| label: (.+)$", cell, re.MULTILINE))
        }
        self.assertLess(
            chapter.index("splits = make_splits(data)"),
            chapter.index("avg_frequency ="),
        )
        self.assertLess(
            chapter.index("selected_power, power_profile ="),
            chapter.index("glum_models = {}"),
        )
        self.assertLess(
            chapter.index("glum_models = {}"),
            chapter.index("#| label: fit-diagnostic-parameters"),
        )
        data = synthetic_pricing_frame(30_010)
        claims = pd.DataFrame(
            {
                "IDpol": np.repeat(data.index, data.ClaimNb),
                "ClaimAmount": np.repeat(data.Severity, data.ClaimNb),
            }
        )
        fitters = {
            name: Mock(return_value=value)
            for name, value in (
                ("fit_gamma_dispersion", 0.7),
                ("fit_negative_binomial_shape", 0.4),
                ("fit_tweedie_dispersion", 2.0),
            )
        }
        scope = dict(
            globals(),
            **fitters,
            n_samples=None,
            selected_power=1.5,
            load_mtpl2=Mock(return_value=(data, claims, self.report)),
            print_frame=Mock(),
            target_weight_table=Mock(return_value=pd.DataFrame()),
            TARGET_SPECS=("frequency", "severity", "pure_premium"),
            MODEL_COLORS={"observed": "black"},
        )
        exec(labelled["prepare-and-split"], scope)  # noqa: S102 - execute trusted repository chapter cells
        scope["glum_models"] = {
            component: Mock(predict=lambda rows: 1 + rows.index.to_numpy() / 100)
            for component in scope["TARGET_SPECS"]
        }
        original_sample = pd.DataFrame.sample

        def sample_after_fitting(rows, *args, **kwargs):
            for fitter in fitters.values():
                self.assertEqual(fitter.call_count, 1)
            return original_sample(rows, *args, **kwargs)

        with patch.object(
            pd.DataFrame, "sample", autospec=True, side_effect=sample_after_fitting
        ):
            for label in ("fit-diagnostic-parameters", "sample-diagnostic-policies"):
                exec(labelled[label], scope)  # noqa: S102 - execute trusted repository chapter cells
        train = scope["splits"].train
        test_ids = set(scope["splits"].test.index)
        for component, rows in scope["training_rows"].items():
            eligible, _, _ = target_and_weight(train, component)
            pd.testing.assert_frame_equal(rows, eligible)
            sample = scope["diagnostic_rows"][component]
            pd.testing.assert_frame_equal(
                sample, original_sample(rows, n=min(20_000, len(rows)), random_state=42)
            )
            self.assertTrue(test_ids.isdisjoint(sample.index))
            np.testing.assert_array_equal(
                scope["diagnostic_predictions"][component],
                1 + sample.index.to_numpy() / 100,
            )
        self.assertEqual(len(scope["diagnostic_rows"]["frequency"]), 20_000)
        self.assertLess(len(scope["diagnostic_rows"]["severity"]), 20_000)
        severity_rows = scope["training_rows"]["severity"]
        gamma_args = fitters["fit_gamma_dispersion"].call_args.args
        pd.testing.assert_series_equal(gamma_args[0], severity_rows.Severity)
        pd.testing.assert_series_equal(
            gamma_args[1], scope["training_predictions"]["severity"]
        )
        pd.testing.assert_series_equal(gamma_args[2], severity_rows.ClaimNb)
        count_args = fitters["fit_negative_binomial_shape"].call_args.args
        pd.testing.assert_series_equal(count_args[0], train.ClaimNb)
        pd.testing.assert_series_equal(
            count_args[1], train.Exposure * scope["training_predictions"]["frequency"]
        )
        tweedie_args = fitters["fit_tweedie_dispersion"].call_args.args
        pd.testing.assert_series_equal(tweedie_args[0], train.PurePremium)
        pd.testing.assert_series_equal(
            tweedie_args[1], scope["training_predictions"]["pure_premium"]
        )
        pd.testing.assert_series_equal(tweedie_args[2], train.Exposure)
        self.assertEqual(tweedie_args[3], 1.5)
        exec(labelled["diagnostic-zero-mass"], scope)  # noqa: S102 - execute trusted repository chapter cells
        frequency_sample = scope["diagnostic_rows"]["frequency"]
        frequency_weight = frequency_sample.Exposure.to_numpy()
        frequency_mean = (
            frequency_weight * scope["diagnostic_predictions"]["frequency"].to_numpy()
        )
        pure_premium_sample = scope["diagnostic_rows"]["pure_premium"]
        pure_premium_weight = pure_premium_sample.Exposure.to_numpy()
        pure_premium_mean = (
            pure_premium_weight
            * scope["diagnostic_predictions"]["pure_premium"].to_numpy()
        )
        expected_zero_mass = pd.DataFrame(
            {
                "observed": [
                    np.average(
                        frequency_sample.ClaimNb.eq(0), weights=frequency_weight
                    ),
                    np.average(
                        pure_premium_sample.ClaimAmountCapped.eq(0),
                        weights=pure_premium_weight,
                    ),
                ],
                "fitted": [
                    np.average(
                        poisson.pmf(0, frequency_mean), weights=frequency_weight
                    ),
                    np.average(
                        np.exp(
                            -(pure_premium_mean**0.5)
                            / (0.5 * 2.0 * pure_premium_weight**-0.5)
                        ),
                        weights=pure_premium_weight,
                    ),
                ],
            },
            index=pd.Index(
                ["Policy claim count", "Capped policy total"], name="observable"
            ),
        )
        pd.testing.assert_frame_equal(
            scope["print_frame"].call_args.args[1], expected_zero_mass
        )
        for obsolete in ("frequency-cdf", "severity-cdf", "pure-premium-cdf"):
            self.assertNotIn(obsolete, labelled)
        for component, family in (
            ("frequency", "poisson"),
            ("severity", "gamma"),
            ("pure_premium", "tweedie"),
        ):
            plot = Mock()
            scope[f"{family}_ccdf_diagnostic"] = plot
            exec(labelled[f"{component.replace('_', '-')}-sf"], scope)  # noqa: S102 - execute trusted repository chapter cells
            self.assertIs(plot.call_args.args[0], scope["diagnostic_rows"][component])
            self.assertIs(
                plot.call_args.args[1], scope["diagnostic_predictions"][component]
            )
        with patch.object(plt, "show"):
            exec(labelled["uncapped-training-claims"], scope)  # noqa: S102 - execute trusted repository chapter cells
        self.assertTrue(test_ids.isdisjoint(scope["training_claims"].IDpol))
        pd.testing.assert_frame_equal(
            scope["training_claims"], claims.loc[claims.IDpol.isin(train.index)]
        )
        self.assertEqual(len(scope["training_claims"]), int(train.ClaimNb.sum()))
        plt.close(scope["fig"])

    def test_chapter_uncapped_exploration(self) -> None:
        chapter = (
            Path(__file__).parents[1] / "book/pure_premium_tutorial.qmd"
        ).read_text()
        cells = re.findall(r"```\{python\}\n(.*?)\n```", chapter, re.DOTALL)
        labelled = {
            match[1]: cell
            for cell in cells
            if (match := re.search(r"^#\| label: (.+)$", cell, re.MULTILINE))
        }
        labels = [
            "exploratory-frequency",
            "exploratory-severity-reference",
            "exploratory-severity-density",
            "exploratory-severity-survival",
            "uncapped-training-claims",
            "exploratory-severity-groups",
            "exploratory-tail-probabilities",
        ]
        for label in labels:
            self.assertLess(chapter.index("avg_frequency ="), chapter.index(label))
            for fitting in (
                "selected_power, power_profile =",
                "glum_models = {}",
                "lightgbm_models = {}",
            ):
                self.assertLess(chapter.index(label), chapter.index(fitting))
        # Unequal weights; three valid groups, one zero-variance and two sparse.
        counts = np.array([0, 1, 3, 2, 2, 1, 4, 2, 2, 4, 1, 1])
        y = np.array([0, 100, 900, 200, 800, 300, 1200, 600, 600, 1000, 200, 300])
        groups = ["zero", "A", "A", "B", "B", "C", "C", "D", "D", "E", "F", "F"]
        train = pd.DataFrame(
            {
                "ClaimNb": counts,
                "Exposure": np.linspace(0.1, 1, len(counts)),
                "ClaimAmountOriginal": counts * y,
                "Severity": np.minimum(y, 500),
                "VehPower": groups,
                "BonusMalus": groups,
            }
        )
        before = train.copy(deep=True)
        claims = pd.DataFrame(
            {
                "IDpol": [*np.repeat(train.index, counts), 999],
                "ClaimAmount": [*np.repeat(y, counts), 1e9],
            }
        )
        fitter = Mock(wraps=fit_gamma_dispersion)
        scope = dict(
            globals(),
            splits=SimpleNamespace(train=train),
            claims=claims,
            fit_gamma_dispersion=fitter,
            print_frame=Mock(),
            MODEL_COLORS={"observed": "black", "GLUM": "orange", "LightGBM": "blue"},
        )
        with (
            patch.object(plt, "show"),
            patch.object(
                pd.DataFrame, "sample", side_effect=AssertionError("must use all rows")
            ),
        ):
            for label in labels:
                exec(labelled[label], scope)  # noqa: S102 - trusted chapter cells
                if label == "exploratory-frequency":
                    exposure = train.Exposure.to_numpy()
                    rate = counts.sum() / exposure.sum()
                    expected = [
                        np.average(poisson.pmf(k, exposure * rate), weights=exposure)
                        for k in range(5)
                    ]
                    expected.append(
                        np.average(poisson.sf(4, exposure * rate), weights=exposure)
                    )
                    np.testing.assert_allclose(scope["count_reference"], expected)
                    self.assertFalse(np.isclose(expected[0], poisson.pmf(0, rate)))
                    np.testing.assert_allclose(
                        scope["count_observed"],
                        np.bincount(counts, weights=exposure, minlength=6)
                        / exposure.sum(),
                    )
                    self.assertAlmostEqual(scope["count_reference"].sum(), 1)
                    self.assertEqual(scope["count_observed"][-1], 0)
                    self.assertEqual(scope["axes"][1].get_yscale(), "log")
                elif label == "exploratory-severity-reference":
                    n = counts[counts > 0]
                    positive_y = y[counts > 0]
                    args = fitter.call_args.args
                    np.testing.assert_array_equal(args[0], positive_y)
                    np.testing.assert_array_equal(args[2], n)
                    mean = np.average(positive_y, weights=n)
                    np.testing.assert_allclose(args[1], mean)
                    phi, grid = scope["uncapped_phi"], scope["severity_grid"]
                    for name, probability in (
                        ("severity_density", gamma.pdf),
                        ("severity_survival", gamma.sf),
                    ):
                        expected = np.average(
                            probability(grid[:, None], a=n / phi, scale=phi * mean / n),
                            weights=n,
                            axis=1,
                        )
                        np.testing.assert_allclose(
                            scope[name], expected, rtol=1e-13, atol=0
                        )
                    self.assertEqual(grid[-1], positive_y.max())
                    expected_sf = [
                        n[positive_y > x].sum() / n.sum() for x in scope["severity_x"]
                    ]
                    np.testing.assert_allclose(
                        scope["severity_empirical_sf"], expected_sf, atol=1e-15
                    )
                    self.assertEqual(scope["severity_empirical_sf"][-1], 0)
                elif label == "exploratory-severity-density":
                    bars = scope["axis"].patches
                    self.assertAlmostEqual(
                        sum(bar.get_height() * bar.get_width() for bar in bars), 1
                    )
                    expected, _ = np.histogram(
                        y[counts > 0],
                        bins=scope["severity_bins"],
                        weights=counts[counts > 0],
                        density=True,
                    )
                    np.testing.assert_allclose(
                        [bar.get_height() for bar in bars], expected
                    )
                elif label == "exploratory-severity-survival":
                    for axis, scale in zip(scope["axes"], ("linear", "log")):
                        self.assertEqual(axis.get_xscale(), scale)
                        self.assertEqual(axis.get_yscale(), "log")
                        self.assertEqual(axis.get_xlim()[1], y.max())
                        steps = axis.patches[0].get_data()
                        np.testing.assert_array_equal(steps.edges, scope["severity_x"])
                        np.testing.assert_array_equal(
                            steps.values, scope["severity_empirical_sf"][:-1]
                        )
                        self.assertEqual(steps.edges[-1], y.max())
                        self.assertTrue(
                            all(np.all(line.get_ydata() > 0) for line in axis.lines)
                        )
                elif label == "uncapped-training-claims":
                    self.assertNotIn(999, scope["training_claims"].IDpol)
                    self.assertEqual(len(scope["claim_amounts"]), counts.sum())
                    np.testing.assert_allclose(
                        scope["claim_sf"],
                        [(np.repeat(y, counts) > x).mean() for x in scope["claim_x"]],
                        atol=1e-15,
                    )
                plt.close("all")
        pd.testing.assert_frame_equal(train, before)
        for table in scope["severity_group_tables"].values():
            self.assertEqual(table.loc["A", "policies"], 2)
            self.assertEqual(table.loc["A", "claims"], 4)
            self.assertEqual(table.loc["A", "mean"], 700)
            self.assertEqual(table.loc["A", "V"], 480000)
            self.assertEqual(
                table.loc["D", "status"], "zero variance: excluded from logs"
            )
            self.assertEqual(table.loc["E", "status"], "sparse: excluded")
            self.assertEqual(table.loc["F", "status"], "sparse: excluded")
        self.assertNotIn("np.polyfit", labelled["exploratory-severity-groups"])
        for axis in scope["axes"]:
            self.assertEqual(len(axis.collections), 1)
            self.assertEqual(len(axis.lines), 1)
            self.assertEqual(axis.lines[0].get_label(), r"Gamma: $\hat\phi\,\bar y^2$")
        rows = scope["uncapped_severity_rows"]
        rows = rows.loc[rows.VehPower.isin(["D", "E", "F"])]
        edge_scope = dict(scope, uncapped_severity_rows=rows)
        with patch.object(plt, "show"):
            exec(labelled["exploratory-severity-groups"], edge_scope)  # noqa: S102 - trusted chapter cell
        self.assertTrue(all(not axis.axison for axis in edge_scope["axes"]))
        plt.close("all")

    def test_prepare_mtpl_data_returns_per_claim_dataframe(self) -> None:
        frequency = pd.DataFrame(
            {
                "IDpol": [1, 2, 3, 4],
                "ClaimNb": [2, 1, 0, 1],
                "Exposure": [1.0, 0.5, 1.0, 0.8],
                "Density": [10, 20, 30, 40],
            }
        )
        severity = pd.DataFrame(
            {"IDpol": [1, 1, 2, 4], "ClaimAmount": [120_000, 50_000, 5_000, 250_000]}
        )
        _policies, claims, _report = prepare_mtpl_data(frequency, severity)
        list(claims.columns)
        self.assertEqual(
            list(claims.columns), ["IDpol", "ClaimAmount", "ClaimAmountCapped"]
        )
        self.assertEqual(len(claims), 4)
        self.assertEqual(
            int(claims["ClaimAmountCapped"].sum()), 100_000 + 50_000 + 5_000 + 100_000
        )
        self.assertGreater(
            float(claims["ClaimAmount"].sum()), float(claims["ClaimAmountCapped"].sum())
        )

    def test_set_log_y_observed_floor_pins_below_empirical_min(self) -> None:
        fig, axis = plt.subplots()
        empirical_min = 3e-6
        axis.set_ylim(bottom=1e-12)
        _set_log_y_observed_floor(axis, empirical_min)
        self.assertAlmostEqual(axis.get_ylim()[0], empirical_min / 100.0)
        plt.close(fig)

    def test_tweedie_sf_matches_one_minus_cdf(self) -> None:
        mu, phi, p = 250.0, 5.0, 1.5
        x_grid = np.linspace(0.0, 1500.0, 25)
        cdf = tweedie_cdf_series(x_grid, mu, phi, p)
        sf = tweedie_sf_series(x_grid, mu, phi, p)
        np.testing.assert_allclose(sf + cdf, 1.0, atol=1e-9)

    def test_tweedie_cdf_series_matches_simulated_distribution(self) -> None:
        rng = np.random.default_rng(7)
        mu, phi, p = 10.0, 5.0, 1.5
        alpha = (2 - p) / (p - 1)
        beta = phi * (p - 1) * mu ** (p - 1)
        lam = mu ** (2 - p) / (phi * (2 - p))
        counts = rng.poisson(lam, size=100_000)
        sample = rng.gamma(counts * alpha, beta)
        self.assertGreater(np.mean(sample == 0), 0.2)
        x_grid = np.r_[0, np.quantile(sample, np.linspace(0.3, 0.99, 20))]
        theoretical = tweedie_cdf_series(x_grid, mu, phi, p)
        empirical = np.searchsorted(np.sort(sample), x_grid, side="right") / len(sample)
        np.testing.assert_allclose(theoretical, empirical, atol=0.005, rtol=0)

    def test_tweedie_probability_endpoints_and_zero_mass(self) -> None:
        mu, phi, p = 1.0, 1.0, 1.5
        x = [-np.inf, -1, 0, np.inf]
        np.testing.assert_allclose(
            tweedie_cdf_series(x, mu, phi, p), [0, 0, np.exp(-2), 1], rtol=1e-14, atol=0
        )
        np.testing.assert_allclose(
            tweedie_sf_series(x, mu, phi, p),
            [1, 1, -np.expm1(-2), 0],
            rtol=1e-14,
            atol=0,
        )
        # exp(-lambda) rounds to one; survival must still retain this mass.
        np.testing.assert_allclose(
            tweedie_sf_series([0], 1.0, 2e20, p), [1e-20], rtol=1e-14, atol=0
        )

    def test_tweedie_probabilities_match_independent_noncentral_chi_square(
        self,
    ) -> None:
        # At p=1.5 the Gamma summands are exponential. If M~Poisson(x/beta)
        # independently of N~Poisson(lambda), survival is P(N > M), equal to
        # the noncentral chi-square CDF below (df=2, noncentrality=2*x/beta).
        mu = np.array([1.0, 100.0, 1.0])
        phi = np.array([1.0, 1.0, 0.002])  # last lambda=1000: exp(-lambda)=0
        x = np.array([0.1, 0.5, 1.0, 2.0])[:, None] * mu
        lam = 2 * np.sqrt(mu) / phi
        beta = phi * np.sqrt(mu) / 2
        cdf = tweedie_cdf_series(x, mu, phi, 1.5)
        sf = tweedie_sf_series(x, mu, phi, 1.5)
        self.assertEqual(cdf.shape, (4, 3))
        np.testing.assert_allclose(
            cdf, ncx2.sf(2 * lam, 2, 2 * x / beta), rtol=2e-9, atol=0
        )
        np.testing.assert_allclose(
            sf, ncx2.cdf(2 * lam, 2, 2 * x / beta), rtol=2e-9, atol=0
        )
        np.testing.assert_allclose(cdf + sf, 1, rtol=2e-10, atol=0)
        self.assertTrue(np.all(np.diff(cdf, axis=0) >= 0))
        self.assertTrue(np.all(np.diff(sf, axis=0) <= 0))

    def test_tweedie_tiny_survival_has_no_floor(self) -> None:
        x = np.array([10.0, 15.0, 20.0])
        # Independent 80-digit sum: integer-shape Gamma survival is an
        # exponential times a finite polynomial. SciPy's ncx2 CDF underflows
        # at x=20, although the survival probability is representable.
        expected = []
        with localcontext() as context:
            context.prec = 80
            lam = Decimal(20)
            for threshold in x:
                t = Decimal(20 * threshold)
                poisson_mass = (-lam).exp()
                gamma_term = (-t).exp()
                gamma_survival = gamma_term
                total = Decimal(0)
                for k in range(1, 601):
                    poisson_mass *= lam / k
                    total += poisson_mass * gamma_survival
                    gamma_term *= t / k
                    gamma_survival += gamma_term
                expected.append(float(total))
        # The omitted Poisson mass after 600 terms is below 1e-600.
        np.testing.assert_allclose(expected[0], 3.9829972e-43, rtol=1e-8, atol=0)
        np.testing.assert_allclose(
            tweedie_sf_series(x, 1, 0.1, 1.5), expected, rtol=1e-9, atol=0
        )

    def test_tweedie_probability_raises_on_unachieved_convergence(self) -> None:
        for probability in (tweedie_cdf_series, tweedie_sf_series):
            with self.subTest(probability=probability.__name__):
                with self.assertRaisesRegex(RuntimeError, "did not converge"):
                    probability([1.0], 1.0, 0.002, 1.5, max_terms=10)
                # Small early contributions do not justify stopping a tiny SF.
                with self.assertRaisesRegex(RuntimeError, "did not converge"):
                    probability([10.0], 1.0, 0.1, 1.5, max_terms=1)

    def test_tweedie_probability_rejects_invalid_inputs(self) -> None:
        for probability in (tweedie_cdf_series, tweedie_sf_series):
            for kwargs in (
                {"x": [np.nan]},
                {"mu": 0},
                {"mu": np.inf},
                {"phi": -1},
                {"phi": np.nan},
                {"tweedie_power": 2},
                {"tweedie_power": np.nan},
                {"max_terms": 0},
                {"max_terms": 1.5},
                {"tol": 0},
                {"tol": np.nan},
                {"mu": 1e-300, "phi": 1e-300},
            ):
                args = {"x": [1], "mu": 1, "phi": 1, "tweedie_power": 1.5, **kwargs}
                with (
                    self.subTest(probability=probability.__name__, kwargs=kwargs),
                    self.assertRaises(ValueError),
                ):
                    probability(**args)

    def test_exposure_likelihood_matches_independent_poisson_gamma_density(
        self,
    ) -> None:
        y = np.array([0.0, 0.2, 0.0, 4.0, 10.0])
        mean = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
        exposure = np.array([0.05, 0.25, 0.8, 1.5, 2.0])
        phi = 1.7
        for p in (1.2, 1.5, 1.8):
            with self.subTest(power=p):
                lam = exposure * mean ** (2 - p) / (phi * (2 - p))
                beta = phi / exposure * (p - 1) * mean ** (p - 1)
                alpha = (2 - p) / (p - 1)
                k = np.arange(1, 601)[:, None]
                positive = y > 0
                expected = (
                    -lam[~positive].sum()
                    + logsumexp(
                        poisson.logpmf(k, lam[positive])
                        + gamma.logpdf(y[positive], k * alpha, scale=beta[positive]),
                        axis=0,
                    ).sum()
                )
                actual = tweedie_log_likelihood(y, mean, exposure, p, phi)
                # Retain GLUM's density approximation: on these inputs its
                # log likelihood differs by up to 3.2e-6 from the direct sum.
                self.assertAlmostEqual(
                    actual, expected, delta=5e-6 if p != 1.5 else 1e-10
                )
                # A second reference calls GLUM separately at each phi/e_i.
                rowwise = sum(
                    TweedieDistribution(p).log_likelihood(
                        y[i : i + 1], mean[i : i + 1], dispersion=phi / exposure[i]
                    )
                    for i in range(len(y))
                )
                self.assertAlmostEqual(
                    actual, rowwise, delta=1e-6 if p != 1.5 else 1e-10
                )
                zeros = tweedie_log_likelihood(np.zeros_like(y), mean, exposure, p, phi)
                self.assertAlmostEqual(
                    zeros, -lam.sum(), delta=5e-6 if p != 1.5 else 1e-10
                )

    def test_tweedie_likelihood_rejects_invalid_inputs_and_transformations(
        self,
    ) -> None:
        for kwargs in (
            {"y": []},
            {"y": [1]},
            {"y": [-1, 2]},
            {"y": [np.nan, 1]},
            {"mean": [0, 1]},
            {"exposure": [0, 1]},
            {"exposure": [np.inf, 1]},
            {"dispersion": 0},
            {"dispersion": np.nan},
            {"tweedie_power": 2},
            {"y": [[1, 2]], "mean": [[1, 2]], "exposure": [[1, 1]]},
            {"exposure": [1e100, 1], "tweedie_power": 1.999},
            {"exposure": [1e-100, 1], "tweedie_power": 1.999},
            {"y": [1e-300, 2], "exposure": [1e-100, 1]},
            {"mean": [1e300, 2], "exposure": [1e100, 1]},
        ):
            args = {
                "y": [0, 2],
                "mean": [1, 2],
                "exposure": [0.5, 1],
                "tweedie_power": 1.5,
                "dispersion": 1,
                **kwargs,
            }
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                tweedie_log_likelihood(**args)
        with (
            patch.object(TweedieDistribution, "log_likelihood", return_value=np.nan),
            self.assertRaisesRegex(RuntimeError, "non-finite"),
        ):
            tweedie_log_likelihood([0, 2], [1, 2], [0.5, 1], 1.5, 1)

    def test_tweedie_dispersion_recovers_simulation_and_maximizes_likelihood(
        self,
    ) -> None:
        rng = np.random.default_rng(42)
        mean = rng.uniform(10, 80, 8_000)
        exposure = rng.uniform(0.05, 1.5, len(mean))
        phi = 4.0
        for p in (1.3, 1.5, 1.7):
            with self.subTest(power=p):
                counts = rng.poisson(exposure * mean ** (2 - p) / (phi * (2 - p)))
                y = rng.gamma(
                    counts * (2 - p) / (p - 1),
                    phi / exposure * (p - 1) * mean ** (p - 1),
                )
                self.assertGreater(np.mean(y == 0), 0.05)
                fitted = fit_tweedie_dispersion(y, mean, exposure, p)
                self.assertAlmostEqual(fitted, phi, delta=0.04 * phi)
                optimum = tweedie_log_likelihood(y, mean, exposure, p, fitted)
                moment = np.mean(exposure * (y - mean) ** 2 / mean**p)
                self.assertGreater(
                    optimum, tweedie_log_likelihood(y, mean, exposure, p, moment)
                )
                for alternative in (fitted * 0.99, fitted * 1.01):
                    self.assertGreater(
                        optimum,
                        tweedie_log_likelihood(y, mean, exposure, p, alternative),
                    )
                h = 1e-4
                gradient = (
                    tweedie_log_likelihood(y, mean, exposure, p, fitted * np.exp(h))
                    - tweedie_log_likelihood(y, mean, exposure, p, fitted * np.exp(-h))
                ) / (2 * h * len(y))
                self.assertAlmostEqual(gradient, 0.0, delta=1e-6)

    def test_tweedie_dispersion_does_not_fall_back_on_failure(self) -> None:
        for result in (
            OptimizeResult(success=False, x=0, fun=1, message="failed"),
            OptimizeResult(success=True, x=0, fun=np.nan, message="nonfinite"),
            OptimizeResult(success=True, x=np.inf, fun=1, message="nonfinite"),
        ):
            with (
                patch(
                    "nonlife_pureprem.pure_premium.minimize_scalar", return_value=result
                ),
                self.assertRaisesRegex(RuntimeError, "optimization failed"),
            ):
                fit_tweedie_dispersion([0, 2], [1, 1], [0.5, 1], 1.5)
        for y in ([0, 0], [1, 1]):
            with self.assertRaisesRegex(
                ValueError, "positive loss.*positive finite start"
            ):
                fit_tweedie_dispersion(y, [1, 1], [0.5, 1], 1.5)

    def test_power_selection_recovers_exposure_varying_simulation(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=10_000)
        mean = np.exp(3 + 0.5 * x)
        exposure = rng.uniform(0.05, 1.5, len(x))
        phi, p = 4.0, 1.5
        counts = rng.poisson(2 * exposure * np.sqrt(mean) / phi)
        y = rng.gamma(counts, phi * np.sqrt(mean) / (2 * exposure))
        data = pd.DataFrame({"PurePremium": y, "Exposure": exposure, "LogDensity": x})
        train, validation = data.iloc[:7_000], data.iloc[7_000:]
        self.assertGreater(np.mean(train.PurePremium == 0), 0.1)
        self.assertGreater(np.mean(validation.PurePremium == 0), 0.1)
        powers = (1.8, p, 1.2)
        models = []
        original_fit = GeneralizedLinearRegressor.fit

        def capture_fit(model, rows, target, sample_weight, **kwargs):
            np.testing.assert_array_equal(sample_weight, train.Exposure)
            models.append(model)
            return original_fit(
                model, rows, target, sample_weight=sample_weight, **kwargs
            )

        with (
            patch("nonlife_pureprem.pure_premium.RATING_FORMULA", "LogDensity"),
            patch.object(GeneralizedLinearRegressor, "fit", new=capture_fit),
            patch(
                "nonlife_pureprem.pure_premium.fit_tweedie_dispersion",
                wraps=fit_tweedie_dispersion,
            ) as dispersion_fit,
        ):
            selected, profile = select_tweedie_power(train, validation, powers)
        self.assertEqual(dispersion_fit.call_count, len(powers))
        for model, call in zip(models, dispersion_fit.call_args_list):
            np.testing.assert_array_equal(call.args[0], train.PurePremium)
            np.testing.assert_allclose(call.args[1], model.predict(train))
            np.testing.assert_array_equal(call.args[2], train.Exposure)
        self.assertEqual(selected, p)
        self.assertEqual(list(profile.power), list(powers))
        self.assertEqual(
            list(profile.columns),
            ["power", "training_dispersion", "validation_log_likelihood"],
        )
        for model, row in zip(models, profile.itertuples()):
            fitted = fit_tweedie_dispersion(
                train.PurePremium, model.predict(train), train.Exposure, row.power
            )
            np.testing.assert_allclose(
                row.training_dispersion, fitted, rtol=1e-6, atol=0
            )
            expected = tweedie_log_likelihood(
                validation.PurePremium,
                model.predict(validation),
                validation.Exposure,
                row.power,
                row.training_dispersion,
            )
            self.assertAlmostEqual(row.validation_log_likelihood, expected, places=8)
        self.assertAlmostEqual(
            profile.loc[profile.power == p, "training_dispersion"].item(),
            phi,
            delta=0.05 * phi,
        )


if __name__ == "__main__":
    unittest.main()
