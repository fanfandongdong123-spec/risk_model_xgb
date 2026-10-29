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


def build_join_sql(
    config: PipelineConfig,
    table_vars: dict[str, list[str]],
    include_all_sample_cols: bool = False,
    sample_cols: list[str] | None = None,
) -> str:
    """仅关联有所选特征的表；默认先用去重样本键预过滤。"""
    table_vars = {table: vars_ for table, vars_ in table_vars.items() if vars_}

    def quote_column(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    sample_key = quote_column(config.join_key_sample)
    feature_key = quote_column(config.join_key_feature)

    if include_all_sample_cols:
        select_cols = ["s.*"]
    else:
        sample_fields = [config.join_key_sample]
        # seg 可在下载后生成；按行划分也不要求样本表存在用户字段。
        if (
            sample_cols is not None and config.person_uuid_col in sample_cols
        ) or (sample_cols is None and config.split_by_person):
            sample_fields.append(config.person_uuid_col)
        sample_fields.append(config.target_col)
        if sample_cols is None or config.seg_col in sample_cols:
            sample_fields.append(config.seg_col)
        select_cols = [f"s.{quote_column(col)}" for col in dict.fromkeys(sample_fields)]
    join_sql = []
    ctes = []
    if config.prefilter_features and table_vars:
        ctes.append(
            f"__dlf_sample_ids AS (\n"
            f"    SELECT DISTINCT s.{sample_key}\n"
            f"    FROM {config.sample_table} s\n"
            f")"
        )

    for i, (table, vars_) in enumerate(table_vars.items()):
        alias = f"f{i}"
        select_cols.extend([f"{alias}.{quote_column(var)}" for var in vars_])
        join_source = table
        if config.prefilter_features:
            required_columns = list(dict.fromkeys([config.join_key_feature, *vars_]))
            projection = ", ".join(f"{alias}.{quote_column(col)}" for col in required_columns)
            join_source = f"__dlf_feature_{i}"
            ctes.append(
                f"{join_source} AS (\n"
                f"    SELECT {projection}\n"
                f"    FROM {table} {alias}\n"
                f"    INNER JOIN __dlf_sample_ids ids\n"
                f"        ON {alias}.{feature_key} = ids.{sample_key}\n"
                f")"
            )
        join_sql.append(
            f"INNER JOIN {join_source} {alias}\n"
            f"    ON s.{sample_key} = {alias}.{feature_key}"
        )

    with_sql = "WITH\n" + ",\n".join(ctes) + "\n" if ctes else ""
    return (
        with_sql
        + "SELECT\n    " + ",\n    ".join(select_cols)
        + f"\nFROM {config.sample_table} s\n"
        + "\n".join(join_sql) + "\n"
    )
