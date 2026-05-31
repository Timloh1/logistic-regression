import pandas as pd
import joblib
from pathlib import Path
import json

from src.config import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR
from src.data_loading import load_p2p_data, load_trans_data
from src.preprocessing import assign_split_column, add_features, get_feature_columns
from src.eda import run_eda
from src.model import build_model, find_best_threshold, calculate_metrics, plot_model_curves
from src.utils import ensure_directories, save_json

def main() -> None:
    # Создание папок артефактов
    ensure_directories()

    # Загрузка данных
    print("Loading datasets...")
    p2p_df = load_p2p_data()
    trans_df = load_trans_data()

    # Временные периоды
    p2p_df = assign_split_column(p2p_df)

    # Разведочный анализ
    print("Running EDA...")
    eda_summary = run_eda(p2p_df, trans_df)

    # Генерация признаков
    print("Building features...")
    feature_df = add_features(p2p_df, trans_df)

    # Исключение warmup-периода
    feature_df = feature_df.loc[feature_df["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    feature_columns = get_feature_columns(feature_df)

    # Разбиение на выборки
    x_train = feature_df.loc[feature_df["split"] == "train", feature_columns]
    y_train = feature_df.loc[feature_df["split"] == "train", "IsFraud"]
    x_val = feature_df.loc[feature_df["split"] == "val", feature_columns]
    y_val = feature_df.loc[feature_df["split"] == "val", "IsFraud"]
    x_test = feature_df.loc[feature_df["split"] == "test", feature_columns]
    y_test = feature_df.loc[feature_df["split"] == "test", "IsFraud"]

    # Сводка по периодам
    split_summary = {}
    for split_name in ["train", "val", "test"]:
        split_df = feature_df.loc[feature_df["split"] == split_name]
        split_summary[split_name] = {
            "rows": int(len(split_df)),
            "fraud_count": int(split_df["IsFraud"].sum()),
            "fraud_rate": float(split_df["IsFraud"].mean()),
        }

    # Обучение базовой модели
    model = build_model()
    print("Training Logistic Regression...")
    model.fit(x_train, y_train)

    # Подбор порога на валидации
    y_val_score = model.predict_proba(x_val)[:, 1]
    threshold, threshold_info = find_best_threshold(y_val.to_numpy(), y_val_score)

    # Метрики на валидации
    validation_metrics = calculate_metrics(y_val.to_numpy(), y_val_score, threshold)

    # Финальное обучение на train+val
    x_train_full = pd.concat([x_train, x_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    final_model = build_model()
    final_model.fit(x_train_full, y_train_full)

    # Оценка на тесте
    y_test_score = final_model.predict_proba(x_test)[:, 1]
    test_metrics = calculate_metrics(y_test.to_numpy(), y_test_score, threshold)

    # Кривые
    plot_model_curves(y_test.to_numpy(), y_test_score)

    # Сохранение результатов
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
    save_json(ARTIFACTS_DIR / "metrics.json", metrics_payload)
    save_json(ARTIFACTS_DIR / "feature_columns.json", {"feature_columns": feature_columns})
    joblib.dump(final_model, ARTIFACTS_DIR / "fraud_model.joblib")

    # Markdown-сводка
    summary_lines = [
        "# Stage 1 Summary",
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
    (ARTIFACTS_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("Artifacts saved to:", ARTIFACTS_DIR)

if __name__ == "__main__":
    main()