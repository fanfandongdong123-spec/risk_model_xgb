# risk_model_xgb

风控二分类 XGBoost 建模工具：关联样本与特征 → 划分 Dev/Val → 过滤变量 → 按 gain 选 Top N → 保存数据 → 训练并输出 KS。全程显示阶段进度与耗时。

## 环境

需要 pandas、NumPy、SQLAlchemy、XGBoost 和 Parquet 引擎（推荐 `pip install pyarrow`）。从 Presto 取数还需要对应的 SQLAlchemy 驱动及内部 `titan.common.data_download.hdfs_to_local`。

连接配置见 [config.py](risk_model_pipeline/config.py)。默认运行时隐藏输入密码，不要在代码中写入密码。

## 快速使用

```python
from risk_model_pipeline import PipelineConfig, RiskModelPipeline

config = PipelineConfig(
    sample_table="hive_store.risk_model.your_sample_table",
    feature_table_list=["hive_store.risk_model.your_feature_table"],
    target_col="fpd15",
    join_key_sample="po_id",
    join_key_feature="main_iou_id",
    split_by_person=True,  # False：按样本行随机划分
    top_n_feature=300,
    output_dir="./data",
)

pipeline = RiskModelPipeline(config)
result = pipeline.run()

model = result.model
selected_vars = result.selected_vars
print("Dev / Val KS:", result.ks_dev, result.ks_val)
print("Parquet:", result.selected_data_path)
```

默认先生成去重样本键集合，每张特征表只选关联键和所需特征，并用样本键做 `INNER JOIN` 预过滤，再拼回样本。最终仍使用 `INNER JOIN`，仅保留全部匹配成功的样本；关联键不唯一时可能放大行数。

设置 `config.prefilter_features=False` 可切回直接关联，便于对比。CTE 不保证物化或提速；可用 `sql, _, _ = pipeline.build_sql(var_lst)` 和 `print(sql)` 预览SQL，再在集群测试。

## 常用设置

| 设置 | 默认行为 |
| --- | --- |
| 候选变量 | 未指定列表时，读取 `mf_` 前缀特征 |
| 取数 | `prefilter_features=True`，先按样本键预过滤各特征表 |
| 变量过滤 | 删除缺失率或单一值占比大于 95% 的变量 |
| Top N | 初筛模型按 gain 排序，最多取 `top_n_feature=300` 个 |
| 训练 | `num_boost_round=800`；参数通过 `config.xgb_params` 调整 |
| 进度 | `feature_log_interval=100` 个特征；`verbose_eval=20` 轮 |

已有 `seg_v` 时沿用其中的 `dev` / `val` 分组；否则默认随机抽取约 70% 的用户作为 Dev，其余作为 Val，同一用户保持同组。设置 `split_by_person=False` 则按行随机划分，不需要 `person_uuid`，同一用户可能出现在两组。

随机种子默认 `random_seed=42`；复现划分还要求数据和行顺序一致。如需自定义比例或覆盖已有分组：

```python
split = pipeline.split_dataset(df_input, train_frac=0.8, random_state=42)
result = pipeline.run(df=split.data)
```

## 数据保存与其他调用

`run()` 在选出 Top N 后、正式训练前自动保存样本字段、标签、`seg_v` 和入选特征。目录相对于 Python 当前工作目录，自动创建，文件名带时间戳：

```text
./data/selected_data_top300_时间戳.parquet
```

设置 `save_selected_data=False` 可关闭保存。文件保存不要求 `return_data=True`；后者用于同时通过 `result.data` 返回筛选后的 DataFrame。

```python
# 直接使用已有 DataFrame，跳过 Presto
result = pipeline.run(df=df_input, return_data=True)

# 指定候选变量；weight 列只用于 Dev 训练，不用于 Val 评估
result = pipeline.run(var_lst=var_lst, df=df_input, weight_col="weight")

# 单独训练，不执行 Top N 筛选和自动保存
train_result = pipeline.train_model(df=df_input, feature_names=var_in)
```

完整调用见 [示例脚本](example_risk_model_pipeline_usage.py)。本地小样本测试：`python -m unittest discover -s tests -v`。
