import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
from titan.common.data_download import hdfs_to_local

from .config import PipelineConfig
from .metrics import calc_ks
from .sql import build_feature_table_mapping, build_join_sql, group_vars_by_table


@dataclass
class PipelineResult:
    model: xgb.Booster
    selected_vars: list[str]
    importance_df: pd.DataFrame
    ks_dev: float
    ks_val: float
    sql: str
    lost_vars: list[str]


class RiskModelPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feat2table: dict[str, str] | None = None
        self.table_cols: dict[str, list[str]] | None = None

    def _create_engine(self):
        return create_engine(self.config.presto_url(), connect_args={"protocol": "https"})

    def load_table_metadata(self, refresh: bool = False):
        if self.feat2table is not None and self.table_cols is not None and not refresh:
            return self.feat2table, self.table_cols

        engine = self._create_engine()
        try:
            self.feat2table, self.table_cols = build_feature_table_mapping(
                self.config.feature_table_list,
                engine,
            )
        finally:
            engine.dispose()

        return self.feat2table, self.table_cols

    def get_all_features(self, prefix: str | None = None) -> list[str]:
        _, table_cols = self.load_table_metadata()
        prefix = self.config.feature_prefix if prefix is None else prefix
        return sorted({c for cols in table_cols.values() for c in cols if c.startswith(prefix)})

    def build_sql(self, var_lst: list[str]) -> tuple[str, list[str], dict[str, list[str]]]:
        feat2table, _ = self.load_table_metadata()
        table_vars, lost_vars = group_vars_by_table(var_lst, feat2table)
        sql = build_join_sql(self.config, table_vars)
        return sql, lost_vars, table_vars

    def load_dataset(self, var_lst: list[str]) -> tuple[pd.DataFrame, str, list[str], list[str]]:
        sql, lost_vars, table_vars = self.build_sql(var_lst)

        print("开始拉取样本+特征宽表...")
        df = hdfs_to_local(sql=sql)
        print(f"数据拉取完成: shape={df.shape}")

        var_candidate = [v for vars_ in table_vars.values() for v in vars_ if v in df.columns]
        df = self._auto_split_seg(df)
        df = self._reduce_memory(df, var_candidate)
        gc.collect()
        return df, sql, lost_vars, var_candidate

    def run(self, var_lst: list[str] | None = None) -> PipelineResult:
        if var_lst is None:
            var_lst = self.get_all_features()
            print(f"汇总全部 {self.config.feature_prefix} 特征数: {len(var_lst)}")

        df, sql, lost_vars, var_candidate = self.load_dataset(var_lst)

        var_after_miss = self.filter_by_missing(df, var_candidate)
        var_after_single = self.filter_by_single_value(df, var_after_miss)

        dev_mask = df[self.config.seg_col].eq(self.config.dev_value).to_numpy()
        val_mask = df[self.config.seg_col].eq(self.config.val_value).to_numpy()

        y_dev = df.loc[dev_mask, self.config.target_col].to_numpy()
        y_val = df.loc[val_mask, self.config.target_col].to_numpy()

        print("开始初筛模型训练...")
        dtrain_tmp = xgb.DMatrix(
            df.loc[dev_mask, var_after_single],
            label=y_dev,
            feature_names=var_after_single,
        )
        model_tmp = xgb.train(
            self.config.xgb_params,
            dtrain_tmp,
            num_boost_round=self.config.num_boost_round,
        )
        importance_df = self.get_importance_df(model_tmp, var_after_single)
        selected_vars = importance_df.head(self.config.top_n_feature)["feature"].tolist()

        del dtrain_tmp, model_tmp
        gc.collect()

        print(f"Top{self.config.top_n_feature} 特征筛选完成，开始正式训练...")
        dtrain = xgb.DMatrix(
            df.loc[dev_mask, selected_vars],
            label=y_dev,
            feature_names=selected_vars,
        )
        dval = xgb.DMatrix(
            df.loc[val_mask, selected_vars],
            label=y_val,
            feature_names=selected_vars,
        )

        model = xgb.train(
            self.config.xgb_params,
            dtrain,
            num_boost_round=self.config.num_boost_round,
            evals=[(dtrain, "dev"), (dval, "val")],
            verbose_eval=self.config.verbose_eval,
        )

        pred_dev = model.predict(dtrain)
        pred_val = model.predict(dval)
        ks_dev = calc_ks(y_dev, pred_dev)
        ks_val = calc_ks(y_val, pred_val)

        print("\n==== 最终模型结果 ====")
        print(f"Dev KS: {ks_dev:.4f}")
        print(f"Val KS: {ks_val:.4f}")

        return PipelineResult(
            model=model,
            selected_vars=selected_vars,
            importance_df=importance_df,
            ks_dev=ks_dev,
            ks_val=ks_val,
            sql=sql,
            lost_vars=lost_vars,
        )

    def _auto_split_seg(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.seg_col in df.columns:
            print(f"检测到 {self.config.seg_col} 字段，使用原始分组")
            return df

        rng = np.random.RandomState(self.config.random_seed)
        df[self.config.seg_col] = np.where(rng.rand(len(df)) < 0.7, self.config.dev_value, self.config.val_value)
        return df

    def _reduce_memory(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        for col in feature_cols:
            if col in df.columns and df[col].dtype != "float32":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        if self.config.target_col in df.columns:
            df[self.config.target_col] = df[self.config.target_col].astype("int8")

        return df

    def filter_by_missing(self, df: pd.DataFrame, vars_: list[str]) -> list[str]:
        miss_rate = df[vars_].isna().mean()
        keep = miss_rate[miss_rate <= self.config.miss_threshold].index.tolist()
        print(f"缺失率过滤: {len(vars_)} -> {len(keep)}")
        return keep

    def filter_by_single_value(self, df: pd.DataFrame, vars_: list[str]) -> list[str]:
        n = len(df)
        keep = []

        for col in vars_:
            top_ratio = df[col].value_counts(dropna=False).iloc[0] / n
            if top_ratio <= self.config.single_value_threshold:
                keep.append(col)

        print(f"单一值过滤: {len(vars_)} -> {len(keep)}")
        return keep

    @staticmethod
    def get_importance_df(model: xgb.Booster, feature_names: list[str]) -> pd.DataFrame:
        imp = model.get_score(importance_type="gain")
        return (
            pd.DataFrame({"feature": feature_names})
            .assign(imp=lambda d: d["feature"].map(imp).fillna(0.0))
            .sort_values("imp", ascending=False)
            .reset_index(drop=True)
        )
