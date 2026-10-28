# risk_model_xgb

用于风控二分类场景的 XGBoost 自动建模 pipeline。

## 功能

- 读取多张特征表的字段，并自动定位变量所在表
- 以样本表为主表，使用 `JOIN` 动态拼接所需特征表
- 自动过滤高缺失率和高单一值占比变量
- 使用 XGBoost 初筛变量重要性并选择 Top N 特征
- 训练最终模型并输出 Dev / Val KS
- 样本没有 `seg_v` 时，默认按 `person_uuid` 的 70%/30% 划分 Dev/Val；设置 `split_by_person=False` 可按样本行随机划分
- 输出各阶段开始、完成和耗时，逐列预处理及模型训练会输出中间进度
- Top N 特征选出后，将样本字段、`seg_v` 和入选特征自动保存至 `./data` 的 Parquet 文件

## 安装依赖

项目需要 pandas、NumPy、SQLAlchemy、Presto SQLAlchemy 驱动和 XGBoost，并依赖运行环境中的 `titan.common.data_download.hdfs_to_local`。

默认会保存 Parquet 数据，请安装 `pyarrow`（或已有 `fastparquet`）：

```bash
pip install pyarrow
```

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
    split_by_person=True,  # False 表示按样本行随机划分
    top_n_feature=300,
    num_boost_round=800,
    output_dir="./data",
    save_selected_data=True,
    feature_log_interval=100,
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
result = pipeline.run(
    weight_col="weight",
)

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

## 运行进度和筛选后数据保存

完整流程会输出读取表结构、下载、Dev/Val 划分、类型转换、内存回收、缺失率过滤、单一值过滤、DMatrix 构建、初筛训练、Top N 选择、文件保存、正式训练及 KS 计算的阶段日志。日志包含时间与耗时，立即刷新，便于在 Notebook 中观察。

- 类型转换、缺失率和单一值统计默认每处理 100 个特征输出一次，并输出首个和最后一个特征的处理进度；通过 `feature_log_interval` 调整。
- 两次模型训练默认每 20 轮输出一次进度，并输出第 1 轮和最后一轮；通过 `verbose_eval` 调整。设置 `verbose_eval=False` 只关闭训练轮次输出，阶段日志仍会显示。
- 下载函数内部的查询、传输和读取仍由 `hdfs_to_local` 自己输出；外层记录整次下载调用的耗时。

`run()` 默认在初筛选出 Top N 后、正式训练前保存 Parquet，因此正式训练失败也不会丢失已成功保存的数据。默认路径示例：

```text
./data/selected_data_top300_20261028_153000_123456.parquet
```

`./data` 相对于 Python 当前工作目录（Notebook 可用 `Path.cwd()` 查看），目录不存在会自动创建；文件名包含实际入选特征数和时间戳。数据包含全部返回的样本行、样本字段、标签、自动生成或已有的 `seg_v` 及入选特征，不保存 DataFrame 索引。Presto 模式为此会读取样本表全部字段；DataFrame 模式把未列入候选特征的列作为样本字段。若可用特征不足 300 个，就保存实际选出的数量。

文件保存不要求 `return_data=True`。训练完成后可通过返回路径读取：

```python
import pandas as pd

result = pipeline.run()
print(result.selected_data_path)
selected_data = pd.read_parquet(result.selected_data_path)
```

如需调整路径或关闭文件保存：

```python
config.output_dir = "./data"
config.save_selected_data = True  # False：关闭保存，result.selected_data_path 为 None
config.feature_log_interval = 100
```

保存失败会显示失败阶段并抛出异常，正式训练不会继续。单独调用 `train_model()` 不执行 Top N 筛选，也不自动保存该文件。

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

输入的 DataFrame 至少需要包含目标字段（默认 `fpd15`）和特征字段。如果没有 `seg_v`，默认还需包含 `person_uuid`，pipeline 会保证同一个人的全部订单处于同一个 Dev/Val 分组。设置 `split_by_person=False` 后不需要 `person_uuid`。传入的 DataFrame 会先复制，原始对象不会被修改。此方式返回的 `result.sql` 是空字符串。

## 选择 Dev / Val 划分方式

`person_uuid_col` 指定用户字段名，`split_by_person` 决定自动划分时是否按该字段分组：

| 输入数据 | 配置 | 行为 |
| --- | --- | --- |
| 已有 `seg_v` | 任意 `split_by_person` | `run()` 和 `train_model()` 沿用已有分组 |
| 没有 `seg_v` | `split_by_person=True`（默认） | 按去重用户随机划分；同一用户的全部记录保持同组 |
| 没有 `seg_v` | `split_by_person=False` | 按样本行随机划分；不要求存在 `person_uuid` |

例如，直接按样本行随机划分后运行完整流程：

```python
config.split_by_person = False
config.random_seed = 42
pipeline = RiskModelPipeline(config)

# 从配置的样本表和特征表读取数据
result = pipeline.run()

# 或使用已有 DataFrame
# result = pipeline.run(df=df_input)
```

自动划分默认使用约 70% 作为 Dev、30% 作为 Val。按用户划分时该比例针对用户数，实际样本行比例可能不同；按行划分时该比例针对样本行数，不按标签分层。按行随机划分允许同一用户同时出现在 Dev 和 Val 中，重复客户或案件较多时，验证效果可能偏乐观。

数据内容和行顺序相同时，固定随机种子可以复现划分。Presto 查询没有固定排序时，不保证跨次下载的行顺序一致；如需固定分组，可保存首次生成的 `seg_v` 后重复使用。

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

如果希望不按用户分组，直接按样本行随机划分：

```python
config.split_by_person = False
split_result = pipeline.split_dataset(df_input, train_frac=0.7, random_state=42)
df_with_seg = split_result.data
```

也可以在创建 `PipelineConfig` 时设置 `split_by_person=False`，`run()` 和 `train_model()` 缺少 `seg_v` 时都会使用该设置。按行随机划分允许同一用户出现在 Dev 和 Val 中；如需覆盖已有 `seg_v`，显式调用 `split_dataset()`。

单独训练模型。如果输入 DataFrame 没有 `seg_v`，会根据 `config.split_by_person` 自动划分（默认按用户），并通过 `train_result.data` 返回带有 `seg_v` 的数据：

```python
train_result = pipeline.train_model(
    df=df_input,
    feature_names=var_in,
    target_col="delq_d30_cnt",
    weight_col="weight",
    params=config.xgb_params,
    num_boost_round=2000,
    early_stopping_rounds=70,
    verbose_eval=50,
    auto_split=True,
    train_frac=0.7,
    random_state=42,
)

df_with_seg = train_result.data
xgb_mod = train_result.model
ks_dev = train_result.ks_dev
ks_val = train_result.ks_val
evals_result = train_result.evals_result
```

传入的数据已经包含 `seg_v` 时会直接使用现有分组；缺少时，`auto_split=True` 会自动划分。原始 DataFrame 不会被修改，带分组字段的数据从 `train_result.data` 获取。设置 `auto_split=False` 可以在缺少 `seg_v` 时直接报错。

`train_frac` 和 `random_state` 可在 `split_dataset()` 或 `train_model()` 中设置；省略 `random_state` 时使用 `config.random_seed`。`run()` 默认按 70%/30% 自动划分，如需其他比例，可先调用 `split_dataset()`，再把 `split_result.data` 传给 `run()`。

`weight_col` 只应用于 Dev 训练集，与 `xgb.DMatrix(..., weight=sample_weight)` 等价；验证集不加权。传入 `None` 表示不使用样本权重。独立训练使用 KS 作为自定义评估指标，并按验证集 KS 执行早停。

## 完整 Pipeline 使用样本权重

从 Presto 拉取数据时，`weight_col` 填写样本表中的权重字段名：

```python
result = pipeline.run(
    weight_col="weight",
)
```

此时 `hive_store.risk_model.your_sample_table` 必须包含 `weight` 字段。该字段会随样本表一起下载，并应用于变量初筛模型和最终模型的 Dev 训练集。验证集不加权。

手动传入 DataFrame 时同样支持：

```python
result = pipeline.run(
    df=df_input,
    weight_col="weight",
)
```

不需要样本权重时，省略该参数或传入 `None`：

```python
result = pipeline.run()
```

## 数据拼接和样本划分

样本表字段 `po_id` 默认与特征表字段 `main_iou_id` 通过普通 `JOIN` 关联，因此只有成功匹配特征表的样本会保留。

如果查询结果中已经存在 `seg_v`，pipeline 会直接使用原始分组。如果不存在，默认先对 `person_uuid` 去重，随机抽取约 70% 的用户作为 Dev，其余作为 Val；设置 `split_by_person=False` 时直接随机抽取约 70% 的样本行作为 Dev，其余作为 Val。随机种子由 `random_seed` 控制，默认值为 `42`。Presto 查询会根据样本表结构跳过不存在的 `seg_v` 和用户字段，再按配置执行自动划分。

## 本地验证

安装上述依赖后，在项目目录运行以下小样本测试。测试使用真实 XGBoost 和 Parquet 读写，内部下载接口使用模拟数据，不连接 Presto：

```bash
python -m unittest discover -s tests -v
```
