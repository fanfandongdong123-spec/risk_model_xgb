# risk_model_xgb

用于风控二分类场景的 XGBoost 自动建模 pipeline。

## 功能

- 读取多张特征表的字段，并自动定位变量所在表
- 以样本表为主表，使用 `JOIN` 动态拼接所需特征表
- 自动过滤高缺失率和高单一值占比变量
- 使用 XGBoost 初筛变量重要性并选择 Top N 特征
- 训练最终模型并输出 Dev / Val KS
- 样本没有 `seg_v` 时，按 `person_uuid` 的 70%/30% 划分 Dev/Val；同一人的所有订单保持在同一组

## 安装依赖

项目需要 pandas、NumPy、SQLAlchemy、Presto SQLAlchemy 驱动和 XGBoost，并依赖运行环境中的 `titan.common.data_download.hdfs_to_local`。

## Presto 连接

默认连接配置位于 `risk_model_pipeline/config.py`：

```python
presto_user = "denghf10"
presto_host = "elacpresto.akulaku.com"
presto_port = 8443
presto_catalog = "hive"
```

不要将数据库密码写入代码或提交到 GitHub。`presto_password` 保持为 `None` 时，程序运行期间会通过隐藏输入提示读取密码，并自动进行 URL 编码。

## 使用示例

```python
from risk_model_pipeline import PipelineConfig, RiskModelPipeline

config = PipelineConfig(
    sample_table="hive_store.risk_model.your_sample_table",
    feature_table_list=[
        "hive_store.risk_model.your_feature_table_1",
        "hive_store.risk_model.your_feature_table_2",
    ],
    join_key_sample="po_id",
    join_key_feature="main_iou_id",
    person_uuid_col="person_uuid",
    target_col="fpd15",
    seg_col="seg_v",
    top_n_feature=300,
    num_boost_round=800,
    xgb_params={
        "booster": "gbtree",
        "objective": "binary:logistic",
        "learning_rate": 0.03,
        "gamma": 1,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "max_bin": 6,
        "min_child_weight": 100,
        "seed": 12345,
        "disable_default_eval_metric": 1,
        "validate_parameters": True,
    },
)

pipeline = RiskModelPipeline(config)
result = pipeline.run()

print("Dev KS:", result.ks_dev)
print("Val KS:", result.ks_val)
print("Selected vars:", result.selected_vars)
```

只使用指定变量：

```python
var_lst = ["mf_xxx_1", "mf_xxx_2"]
result = pipeline.run(var_lst)
```

## 数据拼接和样本划分

样本表字段 `po_id` 默认与特征表字段 `main_iou_id` 通过普通 `JOIN` 关联，因此只有成功匹配特征表的样本会保留。

如果查询结果中已经存在 `seg_v`，pipeline 会直接使用原始分组。如果不存在，则先对 `person_uuid` 去重，随机抽取约 70% 的用户作为 Dev，其余作为 Val。随机种子由 `random_seed` 控制，默认值为 `42`。