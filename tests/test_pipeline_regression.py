"""Exercise regression labels/metrics and early stopping with real XGBoost."""
import importlib.util
import io
from contextlib import redirect_stdout
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

if "titan" not in sys.modules and importlib.util.find_spec("titan") is None:
    for name in ("titan", "titan.common", "titan.common.data_download"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["titan.common.data_download"].hdfs_to_local = lambda **kw: None

from risk_model_pipeline import PipelineConfig, RiskModelPipeline
from risk_model_pipeline.metrics import calc_regression_metrics
import risk_model_pipeline.pipeline as pipeline_module


class RegressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        signal = rng.normal(size=240)
        self.df = pd.DataFrame({
            "po_id": np.arange(240), "mf_signal": signal,
            "mf_noise": rng.normal(size=240),
            "target": 500.125 + 20 * signal + rng.normal(size=240),
            "weight": rng.uniform(0.5, 2, 240),
            "seg_v": np.where(np.arange(240) < 160, "dev", "val"),
        })
        self.params = {"objective": "reg:squarederror", "max_depth": 2,
                       "learning_rate": 0.1, "min_child_weight": 1,
                       "tree_method": "hist", "seed": 5, "nthread": 1,
                       "disable_default_eval_metric": 1}

    def pipeline(self, **kwargs):
        options = dict(sample_table="sample", feature_table_list=[], target_col="target",
                       split_by_person=False, save_selected_data=False, top_n_feature=1,
                       num_boost_round=10, verbose_eval=False, xgb_params=self.params)
        options.update(kwargs)
        return RiskModelPipeline(PipelineConfig(**options))

    def test_run_preserves_decimal_large_targets_and_reports_regression_metrics(self):
        original = self.df.copy(deep=True)
        pipe = self.pipeline()
        with patch.object(pipeline_module, "calc_ks", side_effect=AssertionError("KS must not run")):
            with redirect_stdout(io.StringIO()) as capture:
                result = pipe.run(df=self.df, weight_col="weight", return_data=True)
        self.assertEqual(result.task_type, "regression")
        np.testing.assert_array_equal(result.data.target, self.df.target)
        pd.testing.assert_frame_equal(self.df, original)
        self.assertTrue(np.isnan(result.ks_dev) and np.isnan(result.ks_val))
        self.assertIn("RMSE", capture.getvalue())
        for seg, metrics in (("dev", result.metrics_dev), ("val", result.metrics_val)):
            subset = result.data.query("seg_v == @seg")
            pred = result.model.inplace_predict(subset[result.selected_vars])
            expected = calc_regression_metrics(subset.target, pred)
            self.assertEqual(set(metrics), {"rmse", "mae", "r2"})
            for key in metrics:
                self.assertAlmostEqual(metrics[key], expected[key], places=5)
        self.assertNotIn("eval_metric", self.params)
        self.assertEqual(self.params["disable_default_eval_metric"], 1)

    def test_sql_route_preserves_regression_target(self):
        pipe = self.pipeline()
        pipe.sample_cols = ["po_id", "target", "weight", "seg_v"]
        mapping = ({"mf_signal": "features"}, {"features": ["main_iou_id", "mf_signal"]})
        with patch.object(pipe, "load_table_metadata", return_value=mapping):
            with patch.object(pipeline_module, "hdfs_to_local", return_value=self.df.copy()):
                with redirect_stdout(io.StringIO()):
                    result = pipe.run(var_lst=["mf_signal"], weight_col="weight", return_data=True)
        np.testing.assert_array_equal(result.data.target, self.df.target)
        self.assertTrue(result.sql)
        self.assertTrue(np.isfinite(result.metrics_val["rmse"]))

    def test_train_model_uses_rmse_minimum_and_best_iteration_predictions(self):
        frame = self.df.copy()
        frame.loc[frame.seg_v == "val", "target"] = 500.125 - 20 * frame.loc[frame.seg_v == "val", "mf_signal"]
        # Explicit training params override a classification configuration.
        pipe = self.pipeline(xgb_params={"objective": "binary:logistic"})
        with patch.object(pipeline_module, "calc_ks", side_effect=AssertionError("KS must not run")):
            with redirect_stdout(io.StringIO()):
                result = pipe.train_model(
                    frame, ["mf_signal", "mf_noise"], params=self.params,
                    weight_col="weight", num_boost_round=100,
                    early_stopping_rounds=5, verbose_eval=False,
                )
        self.assertEqual(result.task_type, "regression")
        history = result.evals_result["val"]["rmse"]
        self.assertEqual(result.model.best_iteration, int(np.argmin(history)))
        self.assertNotIn("ks", result.evals_result["val"])
        val = frame.query('seg_v == "val"')
        pred = result.model.inplace_predict(
            val[result.feature_names], iteration_range=(0, result.model.best_iteration + 1)
        )
        expected = calc_regression_metrics(val.target, pred)
        self.assertAlmostEqual(result.metrics_val["rmse"], expected["rmse"], places=5)

    def test_classification_rejects_nonbinary_instead_of_int8_wrapping(self):
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "二分类目标"):
                self.pipeline(xgb_params={"objective": "binary:logistic"}).run(df=self.df)

    def test_missing_regression_label_fails_explicitly(self):
        frame = self.df.copy()
        frame.loc[0, "target"] = np.nan
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "空值"):
                self.pipeline().run(df=frame)

    def test_metrics_match_hand_calculation(self):
        metrics = calc_regression_metrics([1, 2, 3], [2, 2, 2])
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2 / 3))
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertEqual(metrics["r2"], 0)
        self.assertTrue(np.isnan(calc_regression_metrics([1, 1], [1, 1])["r2"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
