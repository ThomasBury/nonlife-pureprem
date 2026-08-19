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
from scipy.stats import gamma

from nonlife_pureprem.pure_premium import (
    _fit_tweedie_mle,
    _set_log_y_observed_floor,
    cli,
    evaluate_predictions,
    exposure_balanced_double_lift_table,
    exposure_balanced_lift_table,
    fit_glum_predictive,
    gamma_ccdf_diagnostic,
    gamma_cdf_diagnostic,
    lift_diagnostics,
    lightgbm_params,
    lorenz_curve,
    lorenz_dominance_table,
    make_splits,
    mean_variance_diagnostic,
    mean_variance_table,
    poisson_ccdf_diagnostic,
    poisson_cdf_diagnostic,
    portfolio_calibration_table,
    prepare_mtpl_data,
    save_core_figures,
    select_tweedie_power,
    target_and_weight,
    tweedie_ccdf_diagnostic,
    tweedie_cdf_diagnostic,
    tweedie_cdf_series,
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
        np.testing.assert_array_equal(sorted_x, np.array([1.0, 1.0, 2.0, 2.0, 3.0]))
        np.testing.assert_allclose(
            cdf, np.array([0.111, 0.222, 0.444, 0.667, 1.0]), atol=1e-3
        )
        self.assertEqual(cdf[-1], 1.0)
        self.assertTrue(np.all(np.diff(cdf) >= 0))
        with self.assertRaises(ValueError):
            weighted_empirical_cdf(values, np.zeros_like(weights))

    def test_weighted_empirical_ccdf_inverts_cdf(self) -> None:
        values = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
        weights = np.array([0.5, 1.0, 0.5, 1.5, 1.0])
        sorted_x, ccdf = weighted_empirical_ccdf(values, weights)
        _, cdf = weighted_empirical_cdf(values, weights)
        np.testing.assert_array_equal(sorted_x, np.array([1.0, 1.0, 2.0, 2.0, 3.0]))
        np.testing.assert_allclose(ccdf + cdf, np.ones_like(ccdf))
        self.assertEqual(ccdf[-1], 0.0)
        self.assertEqual(
            ccdf[0], 1.0 - float(weights[weights > 0].min() / weights.sum())
        )
        self.assertTrue(np.all(np.diff(ccdf) <= 0))

    def test_gamma_fit_recovers_known_parameters(self) -> None:
        rng = np.random.default_rng(0)
        true_alpha, true_scale = 1.8, 450.0
        sample = gamma.rvs(true_alpha, scale=true_scale, size=20_000, random_state=rng)
        alpha_fit, _, beta_fit = gamma.fit(sample, floc=0)
        self.assertAlmostEqual(alpha_fit, true_alpha, delta=0.1)
        self.assertAlmostEqual(beta_fit, true_scale, delta=20.0)

    def test_mean_variance_fits_known_power(self) -> None:
        rng = np.random.default_rng(1)
        areas = ("A", "B", "C", "D", "E", "F")
        rows = []
        for area in areas:
            exposure = rng.uniform(0.4, 1.0, size=40)
            mean_rate = {
                "A": 0.05,
                "B": 0.08,
                "C": 0.10,
                "D": 0.12,
                "E": 0.15,
                "F": 0.20,
            }[area]
            claim_nb = rng.poisson(mean_rate * exposure)
            rows.append(
                pd.DataFrame(
                    {
                        "Area": area,
                        "Exposure": exposure,
                        "ClaimNb": claim_nb,
                        "Frequency": claim_nb / np.maximum(exposure, 1e-9),
                    }
                )
            )
        frame = pd.concat(rows, ignore_index=True)
        _, p_fit, _ = mean_variance_table(frame, by="Area", response="frequency")
        self.assertAlmostEqual(p_fit, 1.0, delta=0.25)

    def test_dist_diagnostic_helpers_smoke(self) -> None:
        data = synthetic_pricing_frame(80)
        for diagnostic in (
            lambda: poisson_cdf_diagnostic(data),
            lambda: gamma_cdf_diagnostic(data),
            lambda: tweedie_cdf_diagnostic(data, tweedie_power=1.5),
            lambda: mean_variance_diagnostic(data, by="VehPower", tweedie_power=1.5),
        ):
            fig, _axes = diagnostic()
            self.assertIsNotNone(fig)
            plt.close(fig)

    def test_eccdf_diagnostic_helpers_smoke(self) -> None:
        data = synthetic_pricing_frame(80)
        for diagnostic in (
            lambda: poisson_ccdf_diagnostic(data),
            lambda: gamma_ccdf_diagnostic(data),
            lambda: tweedie_ccdf_diagnostic(data, tweedie_power=1.5),
        ):
            fig, _axis = diagnostic()
            self.assertIsNotNone(fig)
            plt.close(fig)

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

    def test_gamma_ccdf_diagnostic_accepts_claims(self) -> None:
        data = synthetic_pricing_frame(80)
        claims = pd.DataFrame(
            {
                "IDpol": np.repeat(
                    data.index.to_numpy(), np.maximum(data["ClaimNb"].to_numpy(), 0)
                ),
                "ClaimAmount": np.clip(
                    np.random.default_rng(0).gamma(
                        2.0, 1500.0, size=int(data["ClaimNb"].sum())
                    ),
                    1.0,
                    None,
                ),
            }
        )
        claims["ClaimAmountCapped"] = claims["ClaimAmount"].clip(upper=100_000)
        fig_with, axis_with = gamma_ccdf_diagnostic(data, claims=claims)
        fig_without, _axis_without = gamma_ccdf_diagnostic(data)
        self.assertIsNotNone(fig_with)
        self.assertIsNotNone(fig_without)
        expected_cap = float(
            np.quantile(
                claims["ClaimAmount"].to_numpy(dtype=float),
                0.995,
                method="inverted_cdf",
            )
        )
        self.assertAlmostEqual(axis_with.get_xlim()[1], expected_cap, delta=1.0)
        plt.close(fig_with)
        plt.close(fig_without)

    def test_tweedie_ccdf_diagnostic_caps_x_at_995th_percentile(self) -> None:
        data = synthetic_pricing_frame(200).copy()
        data.loc[data.index[0], "ClaimAmountOriginal"] = 5_000_000.0
        data.loc[data.index[0], "ClaimAmountCapped"] = 5_000_000.0
        fig, axis = tweedie_ccdf_diagnostic(data, tweedie_power=1.5)
        positive = data["ClaimAmountOriginal"].to_numpy(dtype=float) > 0
        expected_cap = float(
            np.quantile(
                data.loc[positive, "ClaimAmountOriginal"].to_numpy(dtype=float),
                0.995,
                weights=data.loc[positive, "Exposure"].to_numpy(dtype=float),
                method="inverted_cdf",
            )
        )
        self.assertAlmostEqual(axis.get_xlim()[1], expected_cap, delta=1.0)
        plt.close(fig)

    def test_gamma_ccdf_diagnostic_drops_extreme_outlier(self) -> None:
        data = synthetic_pricing_frame(80)
        amounts = np.concatenate(
            [
                np.random.default_rng(0).gamma(2.0, 1500.0, size=200),
                np.array([5_000_000.0]),
            ]
        )
        claims = pd.DataFrame(
            {
                "IDpol": np.arange(len(amounts)),
                "ClaimAmount": amounts,
                "ClaimAmountCapped": np.minimum(amounts, 100_000),
            }
        )
        fig, axis = gamma_ccdf_diagnostic(data, claims=claims)
        # The 5,000,000 EUR outlier must be dropped from the visible x-range.
        self.assertLess(axis.get_xlim()[1], 5_000_000.0)
        plt.close(fig)

    def test_tweedie_sf_matches_one_minus_cdf(self) -> None:
        mu, phi, p = 250.0, 5.0, 1.5
        x_grid = np.linspace(0.0, 1500.0, 25)
        cdf = tweedie_cdf_series(x_grid, mu, phi, p)
        sf = tweedie_sf_series(x_grid, mu, phi, p)
        np.testing.assert_allclose(sf + cdf, 1.0, atol=1e-9)

    def test_tweedie_cdf_series_matches_simulated_distribution(self) -> None:
        rng = np.random.default_rng(7)
        mu, phi, p = 300.0, 4.0, 1.5
        alpha = (2 - p) / (p - 1)
        beta = phi * (p - 1) * mu ** (p - 1)
        lam = mu ** (2 - p) / (phi * (2 - p))
        draws = []
        for _ in range(8):
            n = rng.poisson(lam, size=20_000)
            severities = rng.gamma(n * alpha, beta)
            draws.append(severities[n > 0])
        simulated = np.concatenate(draws)
        sample = rng.choice(simulated, size=2_000, replace=False)
        x_grid = np.quantile(sample, np.linspace(0.05, 0.95, 19))
        theoretical = tweedie_cdf_series(x_grid, mu, phi, p)
        empirical = np.searchsorted(np.sort(sample), x_grid, side="right") / len(sample)
        np.testing.assert_allclose(theoretical, empirical, atol=0.03)

    def test_fit_tweedie_mle_recovers_known_parameters(self) -> None:
        rng = np.random.default_rng(7)
        true_mu, true_phi, p = 100.0, 5.0, 1.5
        alpha = (2 - p) / (p - 1)
        beta = true_phi * (p - 1) * true_mu ** (p - 1)
        lam = true_mu ** (2 - p) / (true_phi * (2 - p))
        counts = rng.poisson(lam, size=200_000)
        severities = rng.gamma(np.maximum(counts * alpha, 1e-300), beta)
        sample = rng.choice(severities[counts > 0], size=10_000, replace=False)
        weights = rng.uniform(0.5, 1.0, size=10_000)
        mu_hat, phi_hat = _fit_tweedie_mle(sample, p, weights)
        self.assertAlmostEqual(mu_hat, true_mu, delta=0.10 * true_mu)
        self.assertAlmostEqual(phi_hat, true_phi, delta=0.50 * true_phi)

    def test_fit_tweedie_mle_falls_back_on_all_zeros(self) -> None:
        zeros = np.zeros(50)
        weights = np.ones(50)
        mu_hat, phi_hat = _fit_tweedie_mle(zeros, 1.5, weights)
        self.assertEqual(mu_hat, 0.0)
        self.assertFalse(np.isfinite(phi_hat))

    def test_tweedie_cdf_diagnostic_uses_mle_fit(self) -> None:
        data = synthetic_pricing_frame(200)
        amounts = data["ClaimAmountCapped"].to_numpy(dtype=float)
        weights = data["Exposure"].to_numpy(dtype=float)
        fig, axes = tweedie_cdf_diagnostic(data, tweedie_power=1.5)
        expected_mu, expected_phi = _fit_tweedie_mle(amounts, 1.5, weights)
        legend_text = " ".join(t.get_text() for t in axes[0].get_legend().get_texts())
        self.assertIn(f"μ={expected_mu:.2f}", legend_text)
        self.assertIn(f"φ={expected_phi:.2f}", legend_text)
        plt.close(fig)

    def test_tweedie_ccdf_diagnostic_aligns_empirical_x_and_y(self) -> None:
        # The empirical step line and the cropped s[within] slice must align;
        # the old `s[:len(within)]` copy-paste from gamma_ccdf_diagnostic
        # plotted tail x-values against the near-1.0 zero-mass head of s.
        data = synthetic_pricing_frame(200)
        amounts = data["ClaimAmountOriginal"].to_numpy(dtype=float)
        weights = data["Exposure"].to_numpy(dtype=float)
        positive = amounts > 0
        positive_x = amounts[positive]
        positive_w = weights[positive]
        lower = float(
            np.quantile(positive_x, 0.01, weights=positive_w, method="inverted_cdf")
        )
        upper = float(
            np.quantile(positive_x, 0.995, weights=positive_w, method="inverted_cdf")
        )
        empirical_x, empirical_s = weighted_empirical_ccdf(amounts, weights)
        within = (empirical_x >= lower) & (empirical_x <= upper)
        expected_s = empirical_s[within]
        expected_x = empirical_x[within]
        fig, axis = tweedie_ccdf_diagnostic(data, tweedie_power=1.5)
        empirical_line = next(
            line
            for line in axis.get_lines()
            if line.get_label().startswith("Empirical")
        )
        plotted_x = np.asarray(empirical_line.get_xdata(), dtype=float)
        plotted_y = np.asarray(empirical_line.get_ydata(), dtype=float)
        # matplotlib step(where="post") repeats each x as the right edge of
        # its segment; the plotted series is therefore a 2*N prefix of the
        # N visible (x, s) pairs.
        n = len(expected_x)
        np.testing.assert_array_equal(plotted_x[: n - 1], expected_x[:-1])
        np.testing.assert_allclose(plotted_y[: n - 1], expected_s[:-1])
        # Floor: with the corrected crop the smallest positive survival
        # value drives matplotlib's log-scale auto-bottom. Use the
        # smallest *positive* s so the 0 at the last x (P(X>x_max)=0) is
        # excluded — that 0 is geometric noise, not a floor signal.
        positive_floor = float(expected_s[expected_s > 0].min())
        self.assertLess(axis.get_ylim()[0], positive_floor)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
