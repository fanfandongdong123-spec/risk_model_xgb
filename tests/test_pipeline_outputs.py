"""Small real XGBoost/Parquet tests; no Presto connection or business data."""
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


# The internal download client is only required when reading real Presto data.
if "titan" not in sys.modules and importlib.util.find_spec("titan") is None:
    for name in ("titan", "titan.common", "titan.common.data_download"):
        sys.modules[name] = types.ModuleType(name)

    def unavailable_download(**kwargs):
        raise RuntimeError("The test must mock the internal downloader")

    sys.modules["titan.common.data_download"].hdfs_to_local = unavailable_download

from risk_model_pipeline import PipelineConfig, RiskModelPipeline
import risk_model_pipeline.pipeline as pipeline_module


class PipelineOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        signal = rng.normal(size=160)
        self.df = pd.DataFrame({
            "po_id": np.arange(160, dtype=np.int64) + 2 ** 60,
            "person_uuid": np.repeat(np.arange(80), 2),
            "fpd15": (signal + rng.normal(scale=0.2, size=160) > 0).astype(int),
            "debt": np.arange(160) + 1000,
            "remark": ["case", None] * 80,
            "mf_signal": signal,
            "mf_noise": rng.normal(size=160),
            "mf_other": rng.uniform(size=160),
            "mf_missing": np.nan,
            "mf_constant": 1,
        })

    def pipeline(self, **kwargs):
        options = dict(
            sample_table="sample", feature_table_list=["features"],
            split_by_person=False, output_dir=str(self.root / "data"),
            top_n_feature=2, num_boost_round=3, feature_log_interval=2,
            verbose_eval=2,
            xgb_params={"objective": "binary:logistic", "tree_method": "hist",
                        "max_depth": 2, "nthread": 2, "seed": 42,
                        "disable_default_eval_metric": 1},
        )
        options.update(kwargs)
        return RiskModelPipeline(PipelineConfig(**options))

    def run_quietly(self, pipe, **kwargs):
        capture = io.StringIO()
        with redirect_stdout(capture):
            result = pipe.run(**kwargs)
        return result, capture.getvalue()

    def test_manual_input_saves_selected_columns_and_training_progress(self):
        original = self.df.copy(deep=True)
        result, logs = self.run_quietly(self.pipeline(), df=self.df, return_data=True)
        path = Path(result.selected_data_path)
        self.assertTrue(path.is_file())
        self.assertEqual(path.parent, self.root / "data")
        self.assertTrue(path.name.startswith("selected_data_top2_"))
        saved = pd.read_parquet(path)
        pd.testing.assert_frame_equal(saved, result.data.reset_index(drop=True))
        pd.testing.assert_frame_equal(original, self.df)
        self.assertEqual(saved.po_id.tolist(), self.df.po_id.tolist())
        self.assertEqual(set(saved.seg_v), {"dev", "val"})
        self.assertEqual(set(saved.columns) & set(self.df.filter(like="mf_").columns), set(result.selected_vars))
        self.assertNotIn("mf_missing", saved)
        self.assertNotIn("mf_constant", saved)
        for stage in ("类型转换: 2/5", "缺失率统计: 2/5", "单一值统计:",
                      "初筛模型训练: 1/3", "初筛模型训练: 3/3", "正式模型训练: 3/3",
                      "开始保存筛选后数据", "完整建模流程完成"):
            self.assertIn(stage, logs)
        self.assertLess(logs.index("保存筛选后数据为 Parquet完成"), logs.index("开始正式模型训练"))

    def test_file_is_saved_without_returning_dataframe(self):
        result, _ = self.run_quietly(self.pipeline(), df=self.df)
        self.assertIsNone(result.data)
        self.assertEqual(len(pd.read_parquet(result.selected_data_path)), len(self.df))

    def test_sql_input_keeps_all_sample_columns_and_generated_seg(self):
        pipe = self.pipeline()
        features = list(self.df.filter(like="mf_").columns)
        pipe.sample_cols = [col for col in self.df if col not in features]
        mapping = {feature: "features" for feature in features}
        with patch.object(pipe, "load_table_metadata", return_value=(mapping, {"features": features})):
            with patch.object(pipeline_module, "hdfs_to_local", return_value=self.df.copy()) as downloader:
                result, _ = self.run_quietly(pipe, var_lst=features)
        self.assertIn("s.*", downloader.call_args.kwargs["sql"])
        saved = pd.read_parquet(result.selected_data_path)
        for column in [*pipe.sample_cols, "seg_v", *result.selected_vars]:
            self.assertIn(column, saved)

    def test_disable_saving_and_verbose_preserves_existing_groups(self):
        df = self.df.copy()
        df["seg_v"] = np.where(np.arange(len(df)) < 100, "dev", "val")
        result, logs = self.run_quietly(
            self.pipeline(save_selected_data=False, verbose_eval=False), df=df, return_data=True
        )
        self.assertIsNone(result.selected_data_path)
        self.assertFalse((self.root / "data").exists())
        self.assertEqual(result.data.seg_v.tolist(), df.seg_v.tolist())
        self.assertIn("开始初筛模型训练", logs)
        self.assertNotIn("初筛模型训练: 1/3", logs)

    def test_file_survives_final_training_failure(self):
        original_train = pipeline_module.xgb.train
        calls = 0

        def fail_second_training(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.assertEqual(len(list((self.root / "data").glob("*.parquet"))), 1)
                raise RuntimeError("simulated final training failure")
            return original_train(*args, **kwargs)

        with patch.object(pipeline_module.xgb, "train", side_effect=fail_second_training):
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "simulated final"):
                    self.pipeline().run(df=self.df)
        files = list((self.root / "data").glob("*.parquet"))
        self.assertEqual(len(files), 1)
        self.assertEqual(len(pd.read_parquet(files[0])), len(self.df))

    def test_standalone_training_has_progress_but_does_not_save(self):
        pipe = self.pipeline()
        with redirect_stdout(io.StringIO()) as capture:
            result = pipe.train_model(
                self.df, ["mf_signal", "mf_noise"],
                num_boost_round=3, verbose_eval=2, early_stopping_rounds=None,
            )
        self.assertIn("单独模型训练: 3/3", capture.getvalue())
        self.assertIn("seg_v", result.data)
        self.assertFalse((self.root / "data").exists())

    def test_all_filtered_features_fail_before_training(self):
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "没有可用特征"):
                self.pipeline().run(df=self.df, var_lst=["mf_constant", "mf_missing"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
