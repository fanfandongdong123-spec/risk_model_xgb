from risk_model_pipeline import PipelineConfig, RiskModelPipeline


config = PipelineConfig(
    sample_table="hive_store.risk_model.your_sample_table",
    feature_table_list=[
        "hive_store.risk_model.your_feature_table_1",
        "hive_store.risk_model.your_feature_table_2",
        "hive_store.risk_model.your_feature_table_3",
    ],
    presto_user="denghf10",
    presto_host="elacpresto.akulaku.com",
    join_key_sample="po_id",
    join_key_feature="main_iou_id",
    target_col="fpd15",
    seg_col="seg_v",
    top_n_feature=300,
    miss_threshold=0.95,
    single_value_threshold=0.95,
    num_boost_round=800,
    xgb_params={
        "booster": "gbtree",
        "objective": "binary:logistic",
        "learning_rate": 0.03,
        "disable_default_eval_metric": 1,
        "validate_parameters": True,
        "gamma": 1,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "max_bin": 6,
        "min_child_weight": 100,
        "seed": 12345,
    },
)

pipeline = RiskModelPipeline(config)

# Option 1: use all features with the configured prefix, such as mf_
result = pipeline.run()

# Option 2: use selected features only
# var_lst = ["mf_xxx_1", "mf_xxx_2"]
# result = pipeline.run(var_lst)

# Option 3: train directly from an existing pandas DataFrame (skip Presto)
# result = pipeline.run(df=df_input)
# result = pipeline.run(var_lst=["mf_xxx_1", "mf_xxx_2"], df=df_input)

xgb_model = result.model
selected_vars = result.selected_vars
importance_df = result.importance_df

print("Dev KS:", result.ks_dev)
print("Val KS:", result.ks_val)
print("Selected vars:", selected_vars[:20])
