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

xgb_mod = result.model
top_var_list = result.selected_vars
imp_df = result.importance_df
ks_dev = result.ks_dev
ks_val = result.ks_val
sql = result.sql
lost_vars = result.lost_vars

print("Dev KS:", ks_dev)
print("Val KS:", ks_val)
print("Top变量数:", len(top_var_list))
print("未找到的变量:", lost_vars)

display(imp_df.head(20))
```

如需同时返回建模后的数据，设置 `return_data=True`：

```python
result = pipeline.run(return_data=True)
model_data = result.data
```

`model_data` 包含样本表的全部字段以及最终模型的 `result.model.feature_names`。Presto 模式会读取样本表结构并保留全部样本字段；手动 DataFrame 模式会把未列入候选特征的字段视为样本字段。默认 `return_data=False`，此时 `result.data` 为 `None`，可减少内存占用。手动输入 DataFrame 时同样适用：

```python
result = pipeline.run(df=df_input, return_data=True)
model_data = result.data
```

只使用指定变量：

```python
var_lst = ["mf_xxx_1", "mf_xxx_2"]
result = pipeline.run(var_lst)
```

直接使用已有的 pandas DataFrame 训练，不连接或下载 Presto 数据：

```python
# 自动使用 df_input 中所有以 feature_prefix（默认 mf_）开头的字段
result = pipeline.run(df=df_input)

# 或明确指定入模候选变量
result = pipeline.run(
    var_lst=["mf_xxx_1", "mf_xxx_2"],
    df=df_input,
)
```

输入的 DataFrame 至少需要包含目标字段（默认 `fpd15`）和特征字段。如果没有 `seg_v`，还需包含 `person_uuid`，pipeline 会保证同一个人的全部订单处于同一个 Dev/Val 分组。传入的 DataFrame 会先复制，原始对象不会被修改。此方式返回的 `result.sql` 是空字符串。

## 单独划分数据和训练模型

按照 `person_uuid` 单独划分数据。同一用户的全部 `po_id` 会进入同一个分组：

```python
split_result = pipeline.split_dataset(
    df_input,
    train_frac=0.7,
    random_state=42,
)

df_with_seg = split_result.data
train_df = split_result.train
val_df = split_result.val
```

使用已经带有 `seg_v` 的 DataFrame 单独训练模型：

```python
train_result = pipeline.train_model(
    df=df_with_seg,
    feature_names=var_in,
    target_col="delq_d30_cnt",
    weight_col="weight",
    params=config.xgb_params,
    num_boost_round=2000,
    early_stopping_rounds=70,
    verbose_eval=50,
)

xgb_mod = train_result.model
ks_dev = train_result.ks_dev
ks_val = train_result.ks_val
evals_result = train_result.evals_result
```

`weight_col` 只应用于 Dev 训练集，与 `xgb.DMatrix(..., weight=sample_weight)` 等价；验证集不加权。传入 `None` 表示不使用样本权重。独立训练使用 KS 作为自定义评估指标，并按验证集 KS 执行早停。

完整 pipeline 同样支持权重：

```python
result = pipeline.run(
    df=df_input,
    weight_col="weight",
)
```

## 数据拼接和样本划分

样本表字段 `po_id` 默认与特征表字段 `main_iou_id` 通过普通 `JOIN` 关联，因此只有成功匹配特征表的样本会保留。

如果查询结果中已经存在 `seg_v`，pipeline 会直接使用原始分组。如果不存在，则先对 `person_uuid` 去重，随机抽取约 70% 的用户作为 Dev，其余作为 Val。随机种子由 `random_seed` 控制，默认值为 `42`。