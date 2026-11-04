import gc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from importlib.util import find_spec
from inspect import signature
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
from titan.common.data_download import hdfs_to_local

from .config import PipelineConfig
from .metrics import calc_ks, calc_regression_metrics
from .sql import build_feature_table_mapping, build_join_sql, get_table_columns, group_vars_by_table


def _log(message: str):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


class _TrainingProgress(xgb.callback.TrainingCallback):
    def __init__(self, stage: str, rounds: int, period: int):
        self.stage = stage
        self.rounds = rounds
        self.period = max(1, period)
        self.started = 0.0

    def before_training(self, model):
        self.started = perf_counter()
        return model

    def after_iteration(self, model, epoch, evals_log):
        completed = epoch + 1
        if completed == 1 or completed % self.period == 0 or completed == self.rounds:
            _log(
                f"{self.stage}: {completed}/{self.rounds} 轮，"
                f"耗时 {perf_counter() - self.started:.1f} 秒"
            )
        return False


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
    selected_data_path: str | None = None
    task_type: str = "classification"
    metrics_dev: dict[str, float] = field(default_factory=dict)
    metrics_val: dict[str, float] = field(default_factory=dict)


@dataclass
class DatasetSplit:
    data: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame


@dataclass
class ModelTrainingResult:
    model: xgb.Booster
    ks_dev: float
    ks_val: float
    evals_result: dict
    feature_names: list[str]
    data: pd.DataFrame
    task_type: str = "classification"
    metrics_dev: dict[str, float] = field(default_factory=dict)
    metrics_val: dict[str, float] = field(default_factory=dict)


class RiskModelPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feat2table: dict[str, str] | None = None
        self.table_cols: dict[str, list[str]] | None = None
        self.sample_cols: list[str] | None = None

    @contextmanager
    def _stage(self, name: str):
        started = perf_counter()
        _log(f"开始{name}...")
        try:
            yield
        except Exception:
            _log(f"{name}失败，耗时 {perf_counter() - started:.1f} 秒")
            raise
        else:
            _log(f"{name}完成，耗时 {perf_counter() - started:.1f} 秒")

    def _feature_progress(self, stage: str, completed: int, total: int, started: float):
        interval = max(1, self.config.feature_log_interval)
        if completed == 1 or completed % interval == 0 or completed == total:
            _log(
                f"{stage}: {completed}/{total} 个特征，"
                f"耗时 {perf_counter() - started:.1f} 秒"
            )

    def _training_callbacks(self, stage: str, rounds: int, verbose: int | bool):
        if not verbose:
            return []
        return [_TrainingProgress(stage, rounds, int(verbose))]

    @staticmethod
    def _is_regression(params: dict) -> bool:
        objective = params.get("objective", "binary:logistic")
        if not isinstance(objective, str):
            raise ValueError("流水线需要明确的字符串 objective 来判断二分类或回归")
        if objective.startswith("reg:") or objective == "count:poisson":
            return True
        if objective.startswith("binary:"):
            return False
        raise ValueError(f"暂不支持该 objective: {objective}，请使用二分类或回归目标")

    def _training_params(self, params: dict | None = None) -> dict:
        resolved = dict(params) if params else dict(self.config.xgb_params)
        if self._is_regression(resolved):
            resolved.setdefault("eval_metric", "rmse")
            resolved["disable_default_eval_metric"] = 0
        return resolved

    def _target_values(self, values: pd.Series, params: dict) -> np.ndarray:
        try:
            numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"目标字段 {values.name} 必须是数值") from exc
        if not np.isfinite(numeric).all():
            raise ValueError(f"目标字段 {values.name} 包含空值或无穷值，请先处理")
        if self._is_regression(params):
            return numeric
        if not np.isin(numeric, [0, 1]).all():
            raise ValueError(f"二分类目标 {values.name} 必须是0/1；数值预测请设置回归 objective")
        return numeric.astype("int8")

    def _evaluation_kwargs(self, params: dict) -> dict:
        if self._is_regression(params):
            return {"maximize": False}
        metric_arg = "custom_metric" if "custom_metric" in signature(xgb.train).parameters else "feval"
        return {metric_arg: self._xgb_eval_ks, "maximize": True}

    @staticmethod
    def _predict_best(model: xgb.Booster, data: xgb.DMatrix) -> np.ndarray:
        if hasattr(model, "best_iteration"):
            return model.predict(data, iteration_range=(0, model.best_iteration + 1))
        return model.predict(data)

    def _evaluate_model(self, model, dtrain, dval, y_dev, y_val, params):
        regression = self._is_regression(params)
        metric_name = "RMSE / MAE / R²" if regression else "KS"
        with self._stage(f"预测并计算 Dev/Val {metric_name}"):
            pred_dev = self._predict_best(model, dtrain)
            pred_val = self._predict_best(model, dval)
            if regression:
                metrics_dev = calc_regression_metrics(y_dev, pred_dev)
                metrics_val = calc_regression_metrics(y_val, pred_val)
                ks_dev = ks_val = np.nan
            else:
                ks_dev = calc_ks(y_dev, pred_dev)
                ks_val = calc_ks(y_val, pred_val)
                metrics_dev, metrics_val = {"ks": ks_dev}, {"ks": ks_val}
        print("\n==== 模型结果 ====")
        for split, metrics in (("Dev", metrics_dev), ("Val", metrics_val)):
            print(split + ": " + ", ".join(f"{key.upper()}={value:.4f}" for key, value in metrics.items()))
        return ks_dev, ks_val, metrics_dev, metrics_val

    def _selected_output_columns(
        self,
        df: pd.DataFrame,
        var_candidate: list[str],
        selected_vars: list[str],
        from_sql: bool,
    ) -> list[str]:
        sample_cols = (
            self.sample_cols
            if from_sql and self.sample_cols is not None
            else [col for col in df.columns if col not in var_candidate]
        )
        return list(dict.fromkeys(
            col for col in [*sample_cols, self.config.seg_col, *selected_vars]
            if col in df.columns
        ))

    def _save_selected_dataset(self, df: pd.DataFrame, output_cols: list[str], n_features: int) -> str:
        output_dir = Path(self.config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"selected_data_top{n_features}_{datetime.now():%Y%m%d_%H%M%S_%f}.parquet"
        output_path = output_dir / filename
        with self._stage("保存筛选后数据为 Parquet"):
            _log(f"保存路径: {output_path}；shape=({len(df)}, {len(output_cols)})")
            try:
                df.loc[:, output_cols].to_parquet(output_path, index=False)
            except ImportError as exc:
                raise ImportError("保存 Parquet 需要 pyarrow 或 fastparquet，请先运行 pip install pyarrow") from exc
            _log(f"文件大小: {output_path.stat().st_size / 1024 ** 2:.1f} MB")
        return str(output_path)

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
        sql = build_join_sql(
            self.config, table_vars, include_all_sample_cols, sample_cols=self.sample_cols
        )
        mode = "去重样本键预过滤各特征表后关联" if self.config.prefilter_features else "直接关联特征表"
        _log(f"取数方式: {mode}；实际关联 {len(table_vars)} 张特征表，最终使用 INNER JOIN")
        return sql, lost_vars, table_vars

    def load_dataset(
        self,
        var_lst: list[str],
        include_all_sample_cols: bool = False,
    ) -> tuple[pd.DataFrame, str, list[str], list[str]]:
        with self._stage("生成样本与特征关联 SQL"):
            sql, lost_vars, table_vars = self.build_sql(var_lst, include_all_sample_cols)

        with self._stage("拉取样本+特征宽表"):
            df = hdfs_to_local(sql=sql)
            _log(f"数据拉取完成: shape={df.shape}")

        var_candidate = [v for vars_ in table_vars.values() for v in vars_ if v in df.columns]
        with self._stage("检查并划分 Dev/Val"):
            df = self._auto_split_seg(df)
        df = self._reduce_memory(df, var_candidate)
        with self._stage("预处理后内存回收"):
            gc.collect()
        return df, sql, lost_vars, var_candidate

    def run(
        self,
        var_lst: list[str] | None = None,
        df: pd.DataFrame | None = None,
        return_data: bool = False,
        weight_col: str | None = None,
    ) -> PipelineResult:
        run_started = perf_counter()
        _log("开始完整建模流程")
        train_params = self._training_params()
        if self.config.save_selected_data:
            with self._stage("检查 Parquet 保存依赖"):
                if find_spec("pyarrow") is None and find_spec("fastparquet") is None:
                    raise ImportError("保存 Parquet 需要 pyarrow 或 fastparquet，请先运行 pip install pyarrow")
        if df is None:
            if var_lst is None:
                with self._stage("读取表结构并汇总候选特征"):
                    var_lst = self.get_all_features()
                    _log(f"汇总全部 {self.config.feature_prefix} 特征数: {len(var_lst)}")

            df, sql, lost_vars, var_candidate = self.load_dataset(
                var_lst,
                include_all_sample_cols=(
                    return_data or weight_col is not None or self.config.save_selected_data
                ),
            )
        else:
            with self._stage("准备输入 DataFrame"):
                df, lost_vars, var_candidate = self._prepare_input_dataframe(df, var_lst)
            sql = ""

        var_after_miss = self.filter_by_missing(df, var_candidate)
        var_after_single = self.filter_by_single_value(df, var_after_miss)
        if not var_after_single:
            raise ValueError("缺失率和单一值过滤后没有可用特征，请检查数据或调整阈值")

        dev_mask = df[self.config.seg_col].eq(self.config.dev_value).to_numpy()
        val_mask = df[self.config.seg_col].eq(self.config.val_value).to_numpy()
        if not dev_mask.any() or not val_mask.any():
            raise ValueError(f"{self.config.seg_col} 必须同时包含 Dev 和 Val 分组")

        y_dev = self._target_values(df.loc[dev_mask, self.config.target_col], train_params)
        y_val = self._target_values(df.loc[val_mask, self.config.target_col], train_params)
        train_weight = self._get_train_weight(df, dev_mask, weight_col)

        with self._stage(f"构建初筛 DMatrix（{len(y_dev)} 行、{len(var_after_single)} 个特征）"):
            dtrain_tmp = xgb.DMatrix(
                df.loc[dev_mask, var_after_single],
                label=y_dev,
                weight=train_weight,
                feature_names=var_after_single,
            )
        with self._stage("初筛模型训练"):
            model_tmp = xgb.train(
                train_params,
                dtrain_tmp,
                num_boost_round=self.config.num_boost_round,
                callbacks=self._training_callbacks(
                    "初筛模型训练", self.config.num_boost_round, self.config.verbose_eval
                ),
            )
        with self._stage(f"按 gain 选择 Top{self.config.top_n_feature} 特征"):
            importance_df = self.get_importance_df(model_tmp, var_after_single)
            selected_vars = importance_df.head(self.config.top_n_feature)["feature"].tolist()
            if not selected_vars:
                raise ValueError("未选出任何特征，请确认 top_n_feature 大于 0")
            _log(f"实际选中特征数: {len(selected_vars)}")

        del dtrain_tmp, model_tmp
        with self._stage("初筛后内存回收"):
            gc.collect()

        output_cols = self._selected_output_columns(
            df, var_candidate, selected_vars, from_sql=bool(sql)
        )
        selected_data_path = None
        if self.config.save_selected_data:
            selected_data_path = self._save_selected_dataset(df, output_cols, len(selected_vars))

        with self._stage(f"构建正式训练 Dev DMatrix（{len(selected_vars)} 个特征）"):
            dtrain = xgb.DMatrix(
                df.loc[dev_mask, selected_vars],
                label=y_dev,
                weight=train_weight,
                feature_names=selected_vars,
            )
        with self._stage("构建正式训练 Val DMatrix"):
            dval = xgb.DMatrix(
                df.loc[val_mask, selected_vars],
                label=y_val,
                feature_names=selected_vars,
            )
        with self._stage("正式模型训练"):
            model = xgb.train(
                train_params,
                dtrain,
                num_boost_round=self.config.num_boost_round,
                evals=[(dtrain, "dev"), (dval, "val")],
                verbose_eval=self.config.verbose_eval,
                callbacks=self._training_callbacks(
                    "正式模型训练", self.config.num_boost_round, self.config.verbose_eval
                ),
            )
        ks_dev, ks_val, metrics_dev, metrics_val = self._evaluate_model(
            model, dtrain, dval, y_dev, y_val, train_params
        )

        result_data = None
        if return_data:
            with self._stage("准备返回筛选后 DataFrame"):
                result_data = df.loc[:, output_cols].copy()

        _log(f"完整建模流程完成，总耗时 {perf_counter() - run_started:.1f} 秒")

        return PipelineResult(
            model=model,
            selected_vars=selected_vars,
            importance_df=importance_df,
            ks_dev=ks_dev,
            ks_val=ks_val,
            sql=sql,
            lost_vars=lost_vars,
            data=result_data,
            selected_data_path=selected_data_path,
            task_type="regression" if self._is_regression(train_params) else "classification",
            metrics_dev=metrics_dev,
            metrics_val=metrics_val,
        )

    def split_dataset(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.7,
        random_state: int | None = None,
    ) -> DatasetSplit:
        """按配置划分 Dev/Val；默认按用户，split_by_person=False 时按行随机划分。"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是 pandas DataFrame")
        if df.empty:
            raise ValueError("输入的 df 不能为空")
        if not 0 < train_frac < 1:
            raise ValueError("train_frac 必须在 0 和 1 之间")

        started = perf_counter()
        _log(f"开始划分 Dev/Val，样本行数: {len(df)}")
        random_state = self.config.random_seed if random_state is None else random_state
        data = df.copy()
        if self.config.split_by_person:
            person_col = self.config.person_uuid_col
            if person_col not in data.columns:
                raise KeyError(f"缺少按人划分所需字段: {person_col}")
            dev_persons = (
                data[person_col]
                .drop_duplicates()
                .sample(frac=train_frac, random_state=random_state)
            )
            dev_mask = data[person_col].isin(dev_persons).to_numpy()
            split_description = f"按 {person_col}"
        else:
            # 使用行位置，避免原始索引重复时把不同样本错误地绑定到同组。
            dev_positions = (
                pd.Series(np.arange(len(data)))
                .sample(frac=train_frac, random_state=random_state)
                .to_numpy()
            )
            dev_mask = np.zeros(len(data), dtype=bool)
            dev_mask[dev_positions] = True
            split_description = "按样本行随机"

        data[self.config.seg_col] = np.where(
            dev_mask,
            self.config.dev_value,
            self.config.val_value,
        )
        train = data.loc[data[self.config.seg_col].eq(self.config.dev_value)].copy()
        val = data.loc[data[self.config.seg_col].eq(self.config.val_value)].copy()
        if train.empty or val.empty:
            raise ValueError("划分后 Dev 或 Val 为空，请增加样本/用户数或调整 train_frac")

        _log(
            f"{split_description}划分完成: "
            f"Dev={len(train)}, Val={len(val)}, train_frac={train_frac}，"
            f"耗时 {perf_counter() - started:.1f} 秒"
        )
        return DatasetSplit(data=data, train=train, val=val)

    def train_model(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        target_col: str | None = None,
        weight_col: str | None = None,
        params: dict | None = None,
        num_boost_round: int | None = None,
        early_stopping_rounds: int | None = 70,
        verbose_eval: int | bool | None = 50,
        auto_split: bool = True,
        train_frac: float = 0.7,
        random_state: int | None = None,
    ) -> ModelTrainingResult:
        """单独训练 XGBoost；缺少 seg 字段时按配置自动划分。"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是 pandas DataFrame")
        if df.empty:
            raise ValueError("输入的 df 不能为空")

        if self.config.seg_col not in df.columns:
            if not auto_split:
                raise KeyError(
                    f"输入 df 缺少分组字段: {self.config.seg_col}; "
                    "请先调用 split_dataset，或设置 auto_split=True"
                )
            print(f"未检测到 {self.config.seg_col}，按配置自动划分 Dev/Val")
            df = self.split_dataset(
                df,
                train_frac=train_frac,
                random_state=random_state,
            ).data
        else:
            df = df.copy()

        target_col = self.config.target_col if target_col is None else target_col
        if target_col not in df.columns:
            raise KeyError(f"输入 df 缺少目标字段: {target_col}")

        feature_names = list(dict.fromkeys(feature_names))
        missing_features = [col for col in feature_names if col not in df.columns]
        if missing_features:
            raise KeyError(f"输入 df 缺少特征，示例: {missing_features[:20]}")
        if not feature_names:
            raise ValueError("feature_names 不能为空")

        dev_mask = df[self.config.seg_col].eq(self.config.dev_value).to_numpy()
        val_mask = df[self.config.seg_col].eq(self.config.val_value).to_numpy()
        if not dev_mask.any() or not val_mask.any():
            raise ValueError(
                f"{self.config.seg_col} 必须同时包含 "
                f"{self.config.dev_value!r} 和 {self.config.val_value!r}"
            )

        train_params = self._training_params(params)
        y_dev = self._target_values(df.loc[dev_mask, target_col], train_params)
        y_val = self._target_values(df.loc[val_mask, target_col], train_params)
        train_weight = self._get_train_weight(df, dev_mask, weight_col)

        with self._stage("构建单独训练 Dev DMatrix"):
            dtrain = xgb.DMatrix(
                df.loc[dev_mask, feature_names],
                label=y_dev,
                weight=train_weight,
                feature_names=feature_names,
            )
        with self._stage("构建单独训练 Val DMatrix"):
            dval = xgb.DMatrix(
                df.loc[val_mask, feature_names],
                label=y_val,
                feature_names=feature_names,
            )

        rounds = self.config.num_boost_round if num_boost_round is None else num_boost_round
        verbose = self.config.verbose_eval if verbose_eval is None else verbose_eval
        evals_result = {}
        with self._stage("单独模型训练"):
            model = xgb.train(
                train_params,
                dtrain,
                num_boost_round=rounds,
                evals=[(dtrain, "dev"), (dval, "val")],
                **self._evaluation_kwargs(train_params),
                verbose_eval=verbose,
                early_stopping_rounds=early_stopping_rounds,
                evals_result=evals_result,
                callbacks=self._training_callbacks("单独模型训练", rounds, verbose),
            )

        ks_dev, ks_val, metrics_dev, metrics_val = self._evaluate_model(
            model, dtrain, dval, y_dev, y_val, train_params
        )
        if hasattr(model, "best_iteration"):
            print(f"Best iteration: {model.best_iteration + 1}")

        gc.collect()
        return ModelTrainingResult(
            model=model,
            ks_dev=ks_dev,
            ks_val=ks_val,
            evals_result=evals_result,
            feature_names=feature_names,
            data=df,
            task_type="regression" if self._is_regression(train_params) else "classification",
            metrics_dev=metrics_dev,
            metrics_val=metrics_val,
        )

    def _get_train_weight(
        self,
        df: pd.DataFrame,
        dev_mask: np.ndarray,
        weight_col: str | None,
    ) -> np.ndarray | None:
        if weight_col is None:
            return None
        if weight_col not in df.columns:
            raise KeyError(f"输入 df 缺少样本权重字段: {weight_col}")

        weights = pd.to_numeric(
            df.loc[dev_mask, weight_col],
            errors="coerce",
        ).to_numpy(dtype=float)
        if not np.isfinite(weights).all():
            raise ValueError(f"样本权重字段 {weight_col} 包含空值或非数值")
        if (weights < 0).any():
            raise ValueError(f"样本权重字段 {weight_col} 不能包含负数")
        if weights.sum() <= 0:
            raise ValueError(f"样本权重字段 {weight_col} 的训练集权重和必须大于 0")
        return weights

    @staticmethod
    def _xgb_eval_ks(preds: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[str, float]:
        return "ks", calc_ks(dmatrix.get_label(), preds)

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

        with self._stage("复制输入 DataFrame"):
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

        with self._stage("检查并划分 Dev/Val"):
            df = self._auto_split_seg(df)
        df = self._reduce_memory(df, var_candidate)
        with self._stage("预处理后内存回收"):
            gc.collect()
        _log(f"使用手动输入 DataFrame 训练: shape={df.shape}")
        return df, lost_vars, var_candidate

    def _auto_split_seg(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.seg_col in df.columns:
            _log(f"检测到 {self.config.seg_col} 字段，使用原始分组")
            return df

        return self.split_dataset(df).data

    def _reduce_memory(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        with self._stage("特征数值转换与 float32 压缩"):
            started = perf_counter()
            for i, col in enumerate(feature_cols, 1):
                if col in df.columns and df[col].dtype != "float32":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
                self._feature_progress("类型转换", i, len(feature_cols), started)

            if self.config.target_col in df.columns:
                df[self.config.target_col] = self._target_values(
                    df[self.config.target_col], self.config.xgb_params
                )

        return df

    def filter_by_missing(self, df: pd.DataFrame, vars_: list[str]) -> list[str]:
        keep = []
        with self._stage("缺失率过滤"):
            started = perf_counter()
            # 逐列统计，避免为整张宽表一次性创建缺失标记矩阵。
            for i, col in enumerate(vars_, 1):
                if df[col].isna().mean() <= self.config.miss_threshold:
                    keep.append(col)
                self._feature_progress("缺失率统计", i, len(vars_), started)
            _log(f"缺失率过滤: {len(vars_)} -> {len(keep)}")
        return keep

    def filter_by_single_value(self, df: pd.DataFrame, vars_: list[str]) -> list[str]:
        n = len(df)
        keep = []

        with self._stage("单一值过滤"):
            started = perf_counter()
            for i, col in enumerate(vars_, 1):
                top_ratio = df[col].value_counts(dropna=False).iloc[0] / n
                if top_ratio <= self.config.single_value_threshold:
                    keep.append(col)
                self._feature_progress("单一值统计", i, len(vars_), started)
            _log(f"单一值过滤: {len(vars_)} -> {len(keep)}")
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
