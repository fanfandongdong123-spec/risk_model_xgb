import pandas as pd

from .config import PipelineConfig


def get_table_columns(table_name: str, engine) -> list[str]:
    return pd.read_sql(f"desc {table_name}", engine)["Column"].tolist()


def build_feature_table_mapping(feature_tables: list[str], engine):
    """Build feature -> source table mapping. Duplicate feature names keep first table."""
    feat2table, table_cols = {}, {}

    for tbl in feature_tables:
        print(f"读取表结构: {tbl}")
        cols = get_table_columns(tbl, engine)
        table_cols[tbl] = cols
        for col in cols:
            feat2table.setdefault(col, tbl)

    return feat2table, table_cols


def group_vars_by_table(var_lst: list[str], feat2table: dict) -> tuple[dict[str, list[str]], list[str]]:
    table_vars, lost_vars, seen = {}, [], set()

    for var in var_lst:
        if not var or var in seen:
            continue
        seen.add(var)

        table = feat2table.get(var)
        if table is None:
            lost_vars.append(var)
            continue
        table_vars.setdefault(table, []).append(var)

    if lost_vars:
        print(f"警告: {len(lost_vars)} 个变量不在特征表中，已丢弃，示例: {lost_vars[:20]}")

    print(f"需要关联的特征表数量: {len(table_vars)}")
    return table_vars, lost_vars


def build_join_sql(config: PipelineConfig, table_vars: dict[str, list[str]]) -> str:
    select_cols = [
        f"s.{config.join_key_sample}",
        f"s.{config.person_uuid_col}",
        f"s.{config.target_col}",
        f"s.{config.seg_col}",
    ]
    join_sql = []

    for i, (table, vars_) in enumerate(table_vars.items()):
        alias = f"f{i}"
        select_cols.extend([f"{alias}.{var}" for var in vars_])
        join_sql.append(
            f"join {table} {alias} "
            f"on s.{config.join_key_sample} = {alias}.{config.join_key_feature}"
        )

    return f"""
select
    {", ".join(select_cols)}
from {config.sample_table} s
{" ".join(join_sql)}
"""
