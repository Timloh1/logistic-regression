from pathlib import Path
import json

import joblib
import pandas as pd

from src.config import (
    ARTIFACTS_DIR, 
    REPORTS_DIR, 
    INTERMEDIATE_DIR, 
    PREPARED_P2P_PATH, 
    TRANS_DATA_PATH, 
    FEATURES_PATH, 
    EDA_SUMMARY_PATH)
from src.data_loading import load_p2p_data, load_trans_data
from src.eda import run_eda
from src.model import build_model, calculate_metrics, find_best_threshold, plot_model_curves
from src.preprocessing import add_features, assign_split_column, get_feature_columns
from src.utils import ensure_directories, save_json


def prepare_datasets() -> dict[str, str]:
    ensure_directories()
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    p2p_df = assign_split_column(load_p2p_data())
    trans_df = load_trans_data()

    p2p_df.to_pickle(PREPARED_P2P_PATH)
    trans_df.to_pickle(TRANS_DATA_PATH)
    return {"p2p": str(PREPARED_P2P_PATH), "trans": str(TRANS_DATA_PATH)}


def run_eda_stage(paths: dict[str, str]) -> str:
    p2p_df = pd.read_pickle(paths["p2p"])
    trans_df = pd.read_pickle(paths["trans"])

    eda_summary = run_eda(p2p_df, trans_df)
    save_json(EDA_SUMMARY_PATH, eda_summary)
    return str(EDA_SUMMARY_PATH)


def build_features_stage(paths: dict[str, str]) -> str:
    p2p_df = pd.read_pickle(paths["p2p"])
    trans_df = pd.read_pickle(paths["trans"])

    feature_df = add_features(p2p_df, trans_df)
    feature_df = feature_df.loc[feature_df["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    feature_df.to_pickle(FEATURES_PATH)

    feature_columns = get_feature_columns(feature_df)
    save_json(ARTIFACTS_DIR / "feature_columns.json", {"feature_columns": feature_columns})
    return str(FEATURES_PATH)


def train_and_evaluate_stage(features_path: str, eda_summary_path: str | None = None) -> str:
    feature_df = pd.read_pickle(features_path)
    feature_columns = get_feature_columns(feature_df)

    x_train = feature_df.loc[feature_df["split"] == "train", feature_columns]
    y_train = feature_df.loc[feature_df["split"] == "train", "IsFraud"]
    x_val = feature_df.loc[feature_df["split"] == "val", feature_columns]
    y_val = feature_df.loc[feature_df["split"] == "val", "IsFraud"]
    x_test = feature_df.loc[feature_df["split"] == "test", feature_columns]
    y_test = feature_df.loc[feature_df["split"] == "test", "IsFraud"]

    split_summary = {}
    for split_name in ["train", "val", "test"]:
        split_df = feature_df.loc[feature_df["split"] == split_name]
        split_summary[split_name] = {
            "rows": int(len(split_df)),
            "fraud_count": int(split_df["IsFraud"].sum()),
            "fraud_rate": float(split_df["IsFraud"].mean()),
        }

    model = build_model()
    model.fit(x_train, y_train)

    y_val_score = model.predict_proba(x_val)[:, 1]
    threshold, threshold_info = find_best_threshold(y_val.to_numpy(), y_val_score)
    validation_metrics = calculate_metrics(y_val.to_numpy(), y_val_score, threshold)

    final_model = build_model()
    final_model.fit(pd.concat([x_train, x_val], axis=0), pd.concat([y_train, y_val], axis=0))

    y_test_score = final_model.predict_proba(x_test)[:, 1]
    test_metrics = calculate_metrics(y_test.to_numpy(), y_test_score, threshold)
    plot_model_curves(y_test.to_numpy(), y_test_score)

    eda_summary = {}
    if eda_summary_path:
        eda_summary = json.loads(Path(eda_summary_path).read_text(encoding="utf-8"))

    metrics_payload = {
        "model_name": "logistic_regression",
        "eda_summary": eda_summary,
        "split_summary": split_summary,
        "selected_threshold": float(threshold),
        "threshold_selection": threshold_info,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "feature_columns": feature_columns,
    }
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    save_json(metrics_path, metrics_payload)
    joblib.dump(final_model, ARTIFACTS_DIR / "fraud_model.joblib")
    
    summary_lines = [
        "# Summary",
        "",
        "- Model: `LogisticRegression`",
        f"- Selected threshold: `{threshold:.6f}`",
        f"- Validation Average Precision: `{validation_metrics['average_precision']:.6f}`",
        f"- Validation ROC-AUC: `{validation_metrics['roc_auc']:.6f}`",
        f"- Test Average Precision: `{test_metrics['average_precision']:.6f}`",
        f"- Test ROC-AUC: `{test_metrics['roc_auc']:.6f}`",
        f"- Test precision at threshold: `{test_metrics['precision_at_threshold']:.6f}`",
        f"- Test recall at threshold: `{test_metrics['recall_at_threshold']:.6f}`",
    ]
    (REPORTS_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    return str(metrics_path)


def run_pipeline() -> Path:
    paths = prepare_datasets()
    eda_summary_path = run_eda_stage(paths)
    features_path = build_features_stage(paths)
    return Path(train_and_evaluate_stage(features_path, eda_summary_path))
