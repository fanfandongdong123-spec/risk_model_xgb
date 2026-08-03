from dataclasses import dataclass, field
from getpass import getpass
from urllib.parse import quote


@dataclass
class PipelineConfig:
    sample_table: str
    feature_table_list: list[str]

    presto_user: str = "denghf10"
    presto_password: str | None = None
    presto_host: str = "elacpresto.akulaku.com"
    presto_port: int = 8443
    presto_catalog: str = "hive"

    join_key_sample: str = "po_id"
    join_key_feature: str = "main_iou_id"
    person_uuid_col: str = "person_uuid"
    target_col: str = "fpd15"
    seg_col: str = "seg_v"

    dev_value: str = "dev"
    val_value: str = "val"
    random_seed: int = 42

    top_n_feature: int = 300
    miss_threshold: float = 0.95
    single_value_threshold: float = 0.95
    feature_prefix: str = "mf_"

    num_boost_round: int = 800
    verbose_eval: int | bool = 20
    xgb_params: dict = field(default_factory=lambda: {
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
    })

    def presto_url(self) -> str:
        password = self.presto_password
        if password is None:
            password = getpass("Presto password: ")

        encoded_password = quote(password, safe="")
        return (
            f"presto://{self.presto_user}:{encoded_password}"
            f"@{self.presto_host}:{self.presto_port}/{self.presto_catalog}"
        )
