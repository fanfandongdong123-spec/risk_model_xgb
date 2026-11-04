"""Standalone numeric XGBoost feature PSI/IV using the local Pysam report rules.

Only numpy and pandas are imported. Pass an already loaded XGBoost tree Booster
or sklearn estimator. No Pysam installation, report generation or model fitting
is required. Input data and the model are never modified.
"""

import numpy as np
import pandas as pd


def get_xgb_feature_bins(xgb_mod, var_lst=None):
    """Return {feature: edges}, using all unique float32 model split values.

    Edges include -inf and inf. Intervals are [left, right), matching Pysam's
    Xgboost transform. An unused model feature has one numeric bin. Native
    categorical split models are deliberately rejected.
    """
    model = xgb_mod.get_booster() if hasattr(xgb_mod, "get_booster") else xgb_mod
    if not hasattr(model, "trees_to_dataframe"):
        raise TypeError("xgb_mod 必须是 XGBoost 树模型 Booster 或 sklearn 接口模型")
    names = model.feature_names
    if names is None:
        raise ValueError("模型未保存 feature_names，请先按训练时顺序补齐模型特征名")
    variables = list(names) if var_lst is None else list(dict.fromkeys(var_lst))
    if not variables:
        raise ValueError("var_lst 不能为空")
    unknown = set(variables) - set(names)
    if unknown:
        raise ValueError(f"以下变量不在模型特征中: {sorted(unknown)}")

    types = dict(zip(names, model.feature_types or []))
    if any(types.get(var) == "c" for var in variables):
        raise ValueError("暂不支持原生类别分裂；类别集合不能当作连续数值阈值分箱")
    tree = model.trees_to_dataframe()
    tree = tree.loc[tree["Feature"].isin(variables)]
    if "Category" in tree and tree["Category"].notna().any():
        raise ValueError("暂不支持包含原生类别分裂的模型")
    split_values = {}
    for name, group in tree.groupby("Feature", sort=False):
        values = group["Split"].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} 的模型阈值存在非有限值，无法按数值阈值分箱")
        split_values[name] = np.unique(values)
    return {
        name: np.concatenate(([-np.inf], split_values.get(name, []), [np.inf]))
        for name in variables
    }


def calc_psi_iv_by_group(
    df,
    xgb_mod,
    target,
    group_var,
    date_base,
    var_lst=None,
    *,
    return_detail=False,
    verbose=True,
):
    """Calculate one row per feature/group, without depending on Pysam.

    Parameters
    ----------
    df : pandas.DataFrame
        Numeric model features, a group column and a binary 0/1 target column.
        Numeric strings are accepted; other nonnumeric feature values raise.
    xgb_mod : XGBoost Booster or sklearn estimator
        Fixed tree model; all its split thresholds are reused in every group.
    target : str
        Binary target. 1 is called Y1, without assuming it means bad/customer
        repayment. Missing targets remain in PSI but are excluded from IV.
    group_var : str
        Group values are converted to strings, without changing granularity.
        For monthly analysis, create a YYYY-MM column before calling.
    date_base : str or scalar
        Exact group value used as the PSI reference, e.g. '2025-01'.
        IV is computed within each group and does not use the base group.
    var_lst : list[str] or None
        None uses all model feature_names, including unused features.
    return_detail : bool
        False returns summary DataFrame; True returns (summary, bin_detail).
    verbose : bool
        Print current feature and split count.

    Notes
    -----
    Pysam rules: float32 data/cuts, left-closed intervals, separate missing bin,
    EPS=1e-30, WOE=clip(log(Y1_pct/(Y0_pct+EPS)+EPS), -3, 3).
    PSI/IV agree with the local Pysam get_metrics for numeric finite data,
    complete 0/1 labels and groups containing both classes. Single-class IV
    is deliberately NaN rather than returning a misleading finite value.
    Missing groups and infinite feature values raise instead of losing rows.
    This computes feature PSI/IV, not PSI of predicted scores (PSI_phat).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df 必须是非空 DataFrame")
    if not df.columns.is_unique:
        raise ValueError("df 存在重复列名，请先处理")
    if group_var == "Variable":
        raise ValueError("group_var 不能使用输出保留名称 Variable")
    for column in (group_var, target):
        if column not in df:
            raise ValueError(f"df 缺少字段: {column}")
    if df[group_var].isna().any():
        raise ValueError(f"{group_var} 含空值，请先补齐或筛除这些行")
    groups_raw = df[group_var].astype(str)
    group_codes, groups = pd.factorize(groups_raw, sort=True)
    base = str(date_base)
    if base not in groups:
        raise ValueError(
            f"基准组 {base!r} 不存在于 {group_var} 中。示例: {groups[:5].tolist()}。"
            "不会自动把日粒度合并为月份，请确保 date_base 与分组值完全一致。"
        )
    if group_var in {"PSI", "IV", "N", "N_labeled", "Y1", "Y0", "Bin", "Bin_no",
                     "LBound", "UBound", "date_base", "target_rate", "missing_rate",
                     "n_bins", "split_count", "iv_status", "N_pct", "base_pct",
                     "Y1_pct", "Y0_pct", "WOE"}:
        raise ValueError("group_var 与输出指标重名，请重命名分组列")
    base_index = groups.get_loc(base)

    try:
        y = pd.to_numeric(df[target], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{target} 必须为0/1标签") from exc
    valid_y = ~np.isnan(y)
    if not np.isin(y[valid_y], [0, 1]).all():
        raise ValueError(f"{target} 存在非0/1值，不能计算标准二分类IV")

    feature_bins = get_xgb_feature_bins(xgb_mod, var_lst)
    missing_columns = set(feature_bins) - set(df.columns)
    if missing_columns:
        raise ValueError(f"df 缺少模型字段: {sorted(missing_columns)}")

    eps = 1e-30
    n_groups = len(groups)
    group_n = np.bincount(group_codes, minlength=n_groups)
    group_labeled = np.bincount(group_codes[valid_y], minlength=n_groups)
    group_y1 = np.bincount(group_codes[valid_y & (y == 1)], minlength=n_groups)
    group_y0 = group_labeled - group_y1
    iv_valid = (group_y1 > 0) & (group_y0 > 0)
    statuses = np.where(group_labeled == 0, "no_labeled_rows",
                        np.where(iv_valid, "ok", "single_class"))
    summary_parts, detail_parts = [], []

    for feature_no, (var, edges) in enumerate(feature_bins.items(), 1):
        if verbose:
            print(f"[{feature_no}/{len(feature_bins)}] {var}: "
                  f"{len(edges) - 2}个模型阈值，{n_groups}组", flush=True)
        try:
            numeric = pd.to_numeric(df[var], errors="raise")
            with np.errstate(over="ignore"):
                values = numeric.to_numpy(dtype=np.float32, na_value=np.nan)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{var} 含非数值内容，请先处理") from exc
        if np.isinf(values).any():
            raise ValueError(f"{var} 含无穷值或超出float32范围，请先处理")
        missing = np.isnan(values)
        cuts = edges[1:-1].astype(np.float32)
        numeric_bins = len(cuts) + 1
        n_bins = numeric_bins + 1  # Last bin is always missing.
        codes = np.searchsorted(cuts, values, side="right")
        codes[missing] = numeric_bins
        flat = group_codes * n_bins + codes
        shape = (n_groups, n_bins)
        counts = np.bincount(flat, minlength=n_groups * n_bins).reshape(shape)
        n_labeled = np.bincount(flat[valid_y], minlength=n_groups * n_bins).reshape(shape)
        y1 = np.bincount(flat[valid_y & (y == 1)], minlength=n_groups * n_bins).reshape(shape)
        y0 = n_labeled - y1

        pct = counts / (group_n[:, None] + eps)
        base_pct = pct[base_index]
        psi = (pct - base_pct) * np.log((pct + eps) / (base_pct + eps))
        y1_pct = y1 / (group_y1[:, None] + eps)
        y0_pct = y0 / (group_y0[:, None] + eps)
        woe = np.clip(np.log(y1_pct / (y0_pct + eps) + eps), -3, 3)
        iv = (y1_pct - y0_pct) * woe
        woe[~iv_valid] = np.nan
        iv[~iv_valid] = np.nan

        summary_parts.append(pd.DataFrame({
            "Variable": var, group_var: groups, "date_base": base,
            "PSI": psi.sum(axis=1), "IV": iv.sum(axis=1),
            "N": group_n, "N_labeled": group_labeled,
            "Y1": group_y1, "Y0": group_y0,
            "target_rate": np.divide(group_y1, group_labeled,
                                     out=np.full(n_groups, np.nan), where=group_labeled > 0),
            "missing_rate": counts[:, -1] / group_n,
            "split_count": len(cuts), "n_bins": n_bins,
            "iv_status": statuses,
        }))

        if return_detail:
            labels = [f"[{str(np.float32(left))}, {str(np.float32(right))})"
                      for left, right in zip(edges[:-1], edges[1:])] + ["_MISSING_"]
            detail_parts.append(pd.DataFrame({
                "Variable": var, group_var: np.repeat(groups.to_numpy(), n_bins),
                "Bin_no": np.tile(np.arange(n_bins), n_groups),
                "Bin": np.tile(labels, n_groups),
                "LBound": np.tile(np.r_[edges[:-1], np.nan], n_groups),
                "UBound": np.tile(np.r_[edges[1:], np.nan], n_groups),
                "N": counts.ravel(), "N_pct": pct.ravel(),
                "base_pct": np.tile(base_pct, n_groups), "PSI": psi.ravel(),
                "N_labeled": n_labeled.ravel(), "Y1": y1.ravel(), "Y0": y0.ravel(),
                "Y1_pct": y1_pct.ravel(), "Y0_pct": y0_pct.ravel(),
                "WOE": woe.ravel(), "IV": iv.ravel(),
            }))

    summary = pd.concat(summary_parts, ignore_index=True)
    if return_detail:
        return summary, pd.concat(detail_parts, ignore_index=True)
    return summary
