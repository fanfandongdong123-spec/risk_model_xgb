import ast
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import os
from pathlib import Path
import types
import unittest

import numpy as np
import pandas as pd

from psi_iv_xgb import calc_psi_iv_by_group, get_xgb_feature_bins


class SmallModel:
    feature_names = ["a", "unused"]
    feature_types = ["float", "float"]

    def trees_to_dataframe(self):
        return pd.DataFrame({
            "Feature": ["a", "a", "a", "Leaf"],
            "Split": [0.1, 0.2, 0.1, np.nan],
        })


class PsiIvTests(unittest.TestCase):
    def setUp(self):
        self.model = SmallModel()
        self.df = pd.DataFrame({
            "a": [-1, 0.1, 0.15, 0.2, 1, None,
                  -2, 0.1, 0.2, 0.3, None, None],
            "unused": [0] * 12,
            "month": ["2025-01"] * 6 + ["2025-02"] * 6,
            "y": [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
        })

    def call(self, **kwargs):
        params = dict(df=self.df, xgb_mod=self.model, target="y", group_var="month",
                      date_base="2025-01", verbose=False)
        params.update(kwargs)
        return calc_psi_iv_by_group(**params)

    def test_counts_missing_and_detail_reconcile(self):
        summary, detail = self.call(return_detail=True)
        self.assertEqual(len(summary), 4)
        self.assertEqual(len(detail), 12)
        base = summary.query("Variable == 'a' and month == '2025-01'").iloc[0]
        self.assertEqual(base.PSI, 0)
        self.assertEqual(base.N, 6)
        self.assertEqual(base.split_count, 2)
        self.assertAlmostEqual(base.missing_rate, 1 / 6)
        for row in summary.itertuples():
            bins = detail[(detail.Variable == row.Variable) & (detail.month == row.month)]
            self.assertEqual(bins.N.sum(), row.N)
            self.assertAlmostEqual(bins.N_pct.sum(), 1)
            self.assertAlmostEqual(bins.PSI.sum(), row.PSI)
            self.assertAlmostEqual(bins.IV.sum(), row.IV)

    def test_float32_and_left_closed_split_boundary(self):
        near = np.nextafter(float(np.float32(0.1)), -np.inf)
        frame = pd.DataFrame({"a": [0.1, near, 0.2], "month": ["m"] * 3, "y": [0, 1, 0]})
        _, bins = self.call(df=frame, date_base="m", var_lst=["a"], return_detail=True)
        self.assertEqual(bins.N.tolist(), [0, 2, 1, 0])

    def test_pure_class_bins_use_pysam_woe_cap(self):
        frame = pd.DataFrame({"a": [-1, -2, 1, 2], "month": ["m"] * 4, "y": [0, 0, 1, 1]})
        summary, detail = self.call(df=frame, date_base="m", var_lst=["a"], return_detail=True)
        self.assertAlmostEqual(summary.IV.iloc[0], 6)
        self.assertEqual(detail.loc[detail.N > 0, "WOE"].tolist(), [-3, 3])

    def test_new_missing_bin_produces_finite_positive_psi(self):
        frame = pd.DataFrame({"a": [-1, -1, -1, None], "month": ["b", "b", "n", "n"],
                              "y": [0, 1, 0, 1]})
        summary = self.call(df=frame, date_base="b", var_lst=["a"])
        value = summary.loc[summary.month == "n", "PSI"].iloc[0]
        expected = (0.5 - 1) * np.log((0.5 + 1e-30) / (1 + 1e-30))
        expected += 0.5 * np.log((0.5 + 1e-30) / 1e-30)
        self.assertAlmostEqual(value, expected)

    def test_single_class_and_missing_labels_are_not_treated_as_zero(self):
        frame = self.df.copy()
        frame.loc[frame.month == "2025-02", "y"] = 1
        summary = self.call(df=frame)
        self.assertTrue(summary.loc[summary.month == "2025-02", "IV"].isna().all())
        self.assertEqual(set(summary.loc[summary.month == "2025-02", "iv_status"]), {"single_class"})
        frame.loc[frame.month == "2025-02", "y"] = np.nan
        summary = self.call(df=frame)
        self.assertEqual(set(summary.loc[summary.month == "2025-02", "iv_status"]), {"no_labeled_rows"})
        self.assertTrue((summary.loc[summary.month == "2025-02", "N_labeled"] == 0).all())
        self.assertTrue(np.isfinite(summary.PSI).all())

    def test_missing_target_only_affects_iv(self):
        frame = self.df.copy()
        frame.loc[0, "y"] = np.nan
        summary = self.call(df=frame)
        original = self.call()
        np.testing.assert_allclose(summary.PSI, original.PSI)
        self.assertTrue((summary.loc[summary.month == "2025-01", "N_labeled"] == 5).all())

    def test_unused_feature_all_missing_and_input_unchanged(self):
        original = self.df.copy(deep=True)
        self.call()
        pd.testing.assert_frame_equal(self.df, original)
        edges = get_xgb_feature_bins(self.model)
        np.testing.assert_array_equal(edges["unused"], [-np.inf, np.inf])
        frame = self.df.copy()
        frame["a"] = np.nan
        summary = self.call(df=frame, var_lst=["a"])
        self.assertTrue((summary.PSI == 0).all())
        self.assertTrue((summary.IV == 0).all())
        self.assertTrue((summary.missing_rate == 1).all())

    def test_invalid_target_base_numeric_and_categorical_fail(self):
        for column, value in [("y", 12), ("a", "invalid"), ("a", np.inf), ("month", None)]:
            with self.subTest(column=column, value=value):
                frame = self.df.astype({"a": object}).copy()
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    self.call(df=frame)
        with self.assertRaises(ValueError):
            self.call(date_base="2025-01-01")
        with self.assertRaises(ValueError):
            self.call(var_lst=["does_not_exist"])
        categorical = copy.copy(self.model)
        categorical.feature_types = ["c", "float"]
        with self.assertRaises(ValueError):
            get_xgb_feature_bins(categorical)

    def test_real_xgboost_model_and_sklearn_style_wrapper(self):
        import xgboost as xgb
        x = pd.DataFrame({"real_feature": np.arange(40, dtype=float), "constant": 0.0})
        y = (x.real_feature > 18).astype(int)
        model = xgb.train({"max_depth": 2, "objective": "binary:logistic", "nthread": 1},
                          xgb.DMatrix(x, label=y), num_boost_round=3)
        self.assertGreater(len(get_xgb_feature_bins(model)["real_feature"]), 2)
        x["month"] = ["a", "b"] * 20
        x["y"] = y
        wrapped = types.SimpleNamespace(get_booster=lambda: model)
        result = self.call(df=x, xgb_mod=wrapped, date_base="a")
        self.assertEqual(set(result.Variable), {"real_feature", "constant"})
        self.assertTrue(np.isfinite(result.IV).all())

    @unittest.skipUnless(os.environ.get("PYSAM_SOURCE_ROOT"), "Local Pysam source not supplied")
    def test_matches_actual_pysam_binning_and_get_metrics(self):
        root = Path(os.environ["PYSAM_SOURCE_ROOT"]) / "src" / "pysam"
        source = ast.parse((root / "feature_engineering" / "__init__.py").read_text(encoding="utf-8"))
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Xgboost")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_bin")
        method.body = [n for n in method.body if not isinstance(n, ast.ImportFrom)]
        adapter = types.SimpleNamespace(
            get_features=lambda: (self.model.feature_names, [], ["a"]),
            trees_to_dataframe=self.model.trees_to_dataframe,
        )
        namespace = {"pd": pd, "np": np,
                     "models": types.SimpleNamespace(Xgboost=lambda pm: adapter)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "pysam_actual_get_bin", "exec"), namespace)
        bins = namespace["_get_bin"](types.SimpleNamespace(pysam=None))

        transform_source = ast.parse((root / "feature_engineering" / "woe" / "woe_multiprocess.py").read_text(encoding="utf-8"))
        transform = next(n for n in transform_source.body if isinstance(n, ast.FunctionDef) and n.name == "woe_trans_runner")
        # Pandas 3 no longer supports bool-list & Series. Convert only the
        # local reference's mask container; interval rules stay unchanged.
        for node in ast.walk(transform):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "idx_normal" for t in node.targets)
                    and isinstance(node.value, ast.BinOp)
                    and isinstance(node.value.left, ast.List)):
                node.value = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="asarray", ctx=ast.Load()),
                    args=[node.value], keywords=[],
                )
        ast.fix_missing_locations(transform)
        namespace["tqdm"] = lambda iterable, **kw: iterable
        exec(compile(ast.Module(body=[transform], type_ignores=[]), "pysam_actual_transform", "exec"), namespace)
        params = {"woe_param": {"keyword": "Bin"},
                  "logger": types.SimpleNamespace(debug=lambda *a, **k: None)}
        with redirect_stdout(io.StringIO()):
            transformed = namespace["woe_trans_runner"]((self.df, ["a"], bins, "", params))
        spec = importlib.util.spec_from_file_location("pysam_actual_metrics", root / "utils" / "calc_PSI.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = pd.DataFrame({"wa": transformed.wa, "y": self.df.y, "month": self.df.month}).set_index("month")
        reference = module.get_metrics(data, "2025-01", bins=bins)
        summary, detail = self.call(var_lst=["a"], return_detail=True)
        for period in summary.month:
            actual = detail.loc[detail.month == period].reset_index(drop=True)
            expected = reference.loc[reference.Date == period].reset_index(drop=True)
            self.assertEqual(actual.Bin.tolist(), expected.Bin.tolist())
            for column in ("N", "Y1", "Y0", "N_pct", "Y1_pct", "Y0_pct", "PSI", "WOE", "IV"):
                np.testing.assert_allclose(actual[column], expected[column], atol=1e-12, err_msg=column)


if __name__ == "__main__":
    unittest.main(verbosity=2)
