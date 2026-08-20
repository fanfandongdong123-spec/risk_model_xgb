import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
from titan.common.data_download import hdfs_to_local

from .config import PipelineConfig
from .metrics import calc_ks
from .sql import build_feature_table_mapping, build_join_sql, get_table_columns, group_vars_by_table


@dataclass
class PipelineResult:
    model: xgb.Booster
    selected_vars: list[str]
    importance_df: pd.DataFrame
    ks_dev: float
    ks_val: float
    sql: str
    lost_vars: list[str]
    data: pd.DataFrame | None = None


class RiskModelPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feat2table: dict[str, str] | None = None
        self.table_cols: dict[str, list[str]] | None = None
        self.sample_cols: list[str] | None = None

    def _create_engine(self):
        return create_engine(self.config.presto_url(), connect_args={"protocol": "https"})

    def load_table_metadata(self, refresh: bool = False):
        if (
            self.feat2table is not None
            and self.table_cols is not None
            and self.sample_cols is not None
            and not refresh
        ):
            return self.feat2table, self.table_cols

        engine = self._create_engine()
        try:
            self.feat2table, self.table_cols = build_feature_table_mapping(
                self.config.feature_table_list,
                engine,
            )
            self.sample_cols = get_table_columns(self.config.sample_table, engine)
        finally:
            engine.dispose()

        return self.feat2table, self.table_cols

    def get_all_features(self, prefix: str | None = None) -> list[str]:
        _, table_cols = self.load_table_metadata()
        prefix = self.config.feature_prefix if prefix is None else prefix
        return sorted({c for cols in table_cols.values() for c in cols if c.startswith(prefix)})

    def build_sql(
        self,
        var_lst: list[str],
        include_all_sample_cols: bool = False,
    ) -> tuple[str, list[str], dict[str, list[str]]]:
        feat2table, _ = self.load_table_metadata()
        table_vars, lost_vars = group_vars_by_table(var_lst, feat2table)
        sql = build_join_sql(self.config, table_vars, include_all_sample_cols)
        return sql, lost_vars, table_vars

    def load_dataset(
        self,
        var_lst: list[str],
        include_all_sample_cols: bool = False,
    ) -> tuple[pd.DataFrame, str, list[str], list[str]]:
        sql, lost_vars, table_vars = self.build_sql(var_lst, include_all_sample_cols)

        print("开始拉取样本+特征宽表...")
        df = hdfs_to_local(sql=sql)
        print(f"数据拉取完成: shape={df.shape}")

        var_candidate = [v for vars_ in table_vars.values() for v in vars_ if v in df.columns]
        df = self._auto_split_seg(df)
        df = self._reduce_memory(df, var_candidate)
        gc.collect()
        return df, sql, lost_vars, var_candidate

    def run(
        self,
        var_lst: list[str] | None = None,
        df: pd.DataFrame | None = None,
        return_data: bool = False,
    ) -> PipelineResult:
        if df is None:
            if var_lst is None:
                var_lst = self.get_all_features()
                print(f"汇总全部 {self.config.feature_prefix} 特征数: {len(var_lst)}")

            df, sql, lost_vars, var_candidate = self.load_dataset(
                var_lst,
                include_all_sample_cols=return_data,
            )
        else:
            df, lost_vars, var_candidate = self._prepare_input_dataframe(df, var_lst)
            sql = ""

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

        result_data = None
        if return_data:
            sample_cols = (
                self.sample_cols
                if sql and self.sample_cols is not None
                else [col for col in df.columns if col not in var_candidate]
            )
            model_features = model.feature_names or selected_vars
            output_cols = list(
                dict.fromkeys(
                    [col for col in sample_cols if col in df.columns]
                    + [col for col in model_features if col in df.columns]
                )
            )
            result_data = df.loc[:, output_cols].copy()

        return PipelineResult(
            model=model,
            selected_vars=selected_vars,
            importance_df=importance_df,
            ks_dev=ks_dev,
            ks_val=ks_val,
            sql=sql,
            lost_vars=lost_vars,
            data=result_data,
        )

    def _prepare_input_dataframe(
        self,
        df: pd.DataFrame,
        var_lst: list[str] | None,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是 pandas DataFrame")
        if df.empty:
            raise ValueError("输入的 df 不能为空")
        if self.config.target_col not in df.columns:
            raise KeyError(f"输入 df 缺少目标字段: {self.config.target_col}")

        df = df.copy()
        if var_lst is None:
            var_candidate = sorted(
                col for col in df.columns if col.startswith(self.config.feature_prefix)
            )
            lost_vars = []
            print(f"从输入 df 汇总全部 {self.config.feature_prefix} 特征数: {len(var_candidate)}")
        else:
            var_candidate = list(dict.fromkeys(var for var in var_lst if var in df.columns))
            lost_vars = list(dict.fromkeys(var for var in var_lst if var not in df.columns))
            if lost_vars:
                print(f"警告: {len(lost_vars)} 个变量不在输入 df 中，已丢弃，示例: {lost_vars[:20]}")

        if not var_candidate:
            raise ValueError("输入 df 中没有可用于训练的特征")

        df = self._auto_split_seg(df)
        df = self._reduce_memory(df, var_candidate)
        gc.collect()
        print(f"使用手动输入 DataFrame 训练: shape={df.shape}")
        return df, lost_vars, var_candidate

    def _auto_split_seg(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.seg_col in df.columns:
            print(f"检测到 {self.config.seg_col} 字段，使用原始分组")
            return df

        person_col = self.config.person_uuid_col
        if person_col not in df.columns:
            raise KeyError(f"缺少按人划分所需字段: {person_col}")

        dev_persons = (
            df[person_col]
            .drop_duplicates()
            .sample(frac=0.7, random_state=self.config.random_seed)
        )
        df[self.config.seg_col] = np.where(
            df[person_col].isin(dev_persons),
            self.config.dev_value,
            self.config.val_value,
        )
        print(f"按 {person_col} 划分 dev/val，同一人的所有订单保持在同一分组")
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
