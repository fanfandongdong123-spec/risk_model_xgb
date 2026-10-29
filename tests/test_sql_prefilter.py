"""Check prefilter SQL results and the pipeline download route on local fixtures."""
import importlib.util
import io
from contextlib import redirect_stdout
import sqlite3
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd


if "titan" not in sys.modules and importlib.util.find_spec("titan") is None:
    for name in ("titan", "titan.common", "titan.common.data_download"):
        sys.modules[name] = types.ModuleType(name)

    def unavailable_download(**kwargs):
        raise RuntimeError("The test must mock the internal downloader")

    sys.modules["titan.common.data_download"].hdfs_to_local = unavailable_download

from risk_model_pipeline import PipelineConfig, RiskModelPipeline
import risk_model_pipeline.pipeline as pipeline_module
from risk_model_pipeline.sql import build_join_sql


class PrefilterSQLTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.executescript("""
            CREATE TABLE sample (
                po_id INTEGER, fpd15 INTEGER, person_uuid TEXT, seg_v TEXT, debt INTEGER
            );
            INSERT INTO sample VALUES
                (1,0,'p1','dev',100),(2,1,'p2','val',200),(2,0,'p3','dev',300),
                (3,1,'p4','val',400),(4,0,'p5','dev',500),(NULL,0,'pn','dev',50);
            CREATE TABLE fa (main_iou_id INTEGER, mf_a REAL, extra_a REAL);
            INSERT INTO fa VALUES (1,0.1,9),(2,0.2,8),(4,0.4,7),(9999,0.9,6),(NULL,0.5,5);
            CREATE TABLE fb (main_iou_id INTEGER, mf_b REAL, extra_b REAL);
            INSERT INTO fb VALUES (1,1.0,9),(2,2.0,8),(3,3.0,7),(9999,9.0,6),(NULL,5.0,5);
            CREATE TABLE unused (main_iou_id INTEGER, mf_unused REAL);
        """)
        self.config = PipelineConfig(
            sample_table="sample", feature_table_list=["fa", "fb", "unused"],
            split_by_person=False, save_selected_data=False,
        )
        self.table_vars = {"fa": ["mf_a"], "fb": ["mf_b"]}
        self.sample_cols = ["po_id", "fpd15", "person_uuid", "seg_v", "debt"]

    def sql(self, prefilter=True, **kwargs):
        self.config.prefilter_features = prefilter
        return build_join_sql(self.config, self.table_vars, sample_cols=self.sample_cols, **kwargs)

    def query(self, sql):
        return pd.read_sql_query(sql, self.db)

    def assert_same(self, expected, actual):
        columns = expected.columns.tolist()
        pd.testing.assert_frame_equal(
            expected.sort_values(columns).reset_index(drop=True),
            actual.sort_values(columns).reset_index(drop=True),
        )

    def test_default_prefilter_matches_direct_inner_join(self):
        self.assertTrue(self.config.prefilter_features)
        direct = self.query(self.sql(prefilter=False))
        sql = self.sql()
        actual = self.query(sql)
        self.assert_same(direct, actual)
        self.assertEqual(len(actual), 3)
        self.assertEqual(actual.po_id.tolist(), [1, 2, 2])
        self.assertNotIn("unused", sql)
        self.assertNotIn("extra_a", sql)
        self.assertNotIn("extra_b", sql)

    def test_intermediate_features_exclude_unrelated_rows(self):
        sql = self.sql()
        ctes = sql.rsplit("\nSELECT\n", 1)[0]
        features = self.query(ctes + "\nSELECT * FROM __dlf_feature_0")
        self.assertEqual(features.columns.tolist(), ["main_iou_id", "mf_a"])
        self.assertEqual(set(features.main_iou_id), {1, 2, 4})
        self.assertEqual(len(features), 3)

    def test_feature_duplicates_are_preserved(self):
        self.db.execute("INSERT INTO fa VALUES (1,0.9,0)")
        expected = self.query(self.sql(prefilter=False))
        actual = self.query(self.sql())
        self.assert_same(expected, actual)
        self.assertEqual(len(actual), 4)

    def test_include_all_sample_columns_preserves_output(self):
        expected = self.query(self.sql(prefilter=False, include_all_sample_cols=True))
        actual = self.query(self.sql(include_all_sample_cols=True))
        self.assert_same(expected, actual)
        self.assertEqual(actual.columns.tolist(), [*self.sample_cols, "mf_a", "mf_b"])

    def test_missing_optional_sample_columns_are_not_selected(self):
        sql = build_join_sql(
            self.config, self.table_vars, sample_cols=["po_id", "fpd15"]
        )
        result = self.query(sql)
        self.assertEqual(result.columns.tolist(), ["po_id", "fpd15", "mf_a", "mf_b"])

    def test_special_feature_and_reserved_key_are_quoted(self):
        self.db.executescript("""
            CREATE TABLE special_sample ("order" INTEGER, fpd15 INTEGER);
            INSERT INTO special_sample VALUES (1,0),(2,1);
            CREATE TABLE special_feature ("key" INTEGER, "mf_a@x" REAL);
            INSERT INTO special_feature VALUES (1,0.1),(2,0.2),(9999,0.9);
        """)
        self.config.sample_table = "special_sample"
        self.config.join_key_sample = "order"
        self.config.join_key_feature = "key"
        sql = build_join_sql(
            self.config, {"special_feature": ["mf_a@x"]},
            sample_cols=["order", "fpd15"],
        )
        result = self.query(sql)
        self.assertEqual(result.columns.tolist(), ["order", "fpd15", "mf_a@x"])
        self.assertEqual(len(result), 2)

    def test_empty_feature_list_does_not_emit_unused_ctes(self):
        sql = build_join_sql(self.config, {}, sample_cols=["po_id", "fpd15"])
        self.assertNotIn("WITH", sql)
        self.assertEqual(len(self.query(sql)), 6)

    def test_load_dataset_executes_generated_prefilter_sql(self):
        pipeline = RiskModelPipeline(self.config)
        pipeline.sample_cols = self.sample_cols
        metadata = ({"mf_a": "fa", "mf_b": "fb"}, {
            "fa": ["main_iou_id", "mf_a"], "fb": ["main_iou_id", "mf_b"]
        })
        with patch.object(pipeline, "load_table_metadata", return_value=metadata):
            with patch.object(pipeline_module, "hdfs_to_local", side_effect=self.query) as download:
                with redirect_stdout(io.StringIO()) as capture:
                    result, sql, lost, features = pipeline.load_dataset(["mf_a", "mf_b"])
        self.assertTrue(sql.startswith("WITH"))
        self.assertEqual(download.call_args.kwargs["sql"], sql)
        self.assertEqual(len(result), 3)
        self.assertEqual((lost, features), ([], ["mf_a", "mf_b"]))
        self.assertIn("预过滤", capture.getvalue())
        self.assertEqual(result.mf_a.dtype, "float32")


if __name__ == "__main__":
    unittest.main(verbosity=2)
