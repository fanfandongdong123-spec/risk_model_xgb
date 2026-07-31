# risk_model_xgb

XGBoost 风控模型自动建模 pipeline。

## 功能

- 读取特征表字段并自动匹配变量所在表
- 根据样本表和特征表动态拼接 Presto SQL
- 拉取样本与特征宽表
- 缺失率、单一值变量过滤
- XGBoost 初筛变量重要性
- 输出 TopN 特征并训练最终模型
- 输出 Dev / Val KS

## 使用示例

```python
from risk_model_pipeline import PipelineConfig, RiskModelPipeline

config = PipelineConfig(
    sample_table="hive_store.risk_model.xxx_sample_table",
    feature_table_list=[
        "hive_store.risk_model.xxx_feature_table_1",
        "hive_store.risk_model.xxx_feature_table_2",
    ],
    join_key_sample="po_id",
    join_key_feature="main_iou_id",
    target_col="fpd15",
    seg_col="seg_v",
)

pipeline = RiskModelPipeline(config)
result = pipeline.run()

xgb_model = result.model
selected_vars = result.selected_vars
importance_df = result.importance_df
```

如果只想使用指定变量：

```python
var_lst = ["mf_xxx_1", "mf_xxx_2"]
result = pipeline.run(var_lst)
```
