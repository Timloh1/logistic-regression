import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from src.config import RANDOM_STATE, MIN_RECALL, FIGURES_DIR
from src.utils import save_figure

def build_model() -> Pipeline:
    """Создание пайплайна: заполнение пропусков, масштабирование, логистическая регрессия"""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict]:
    """Подбор порога по максимальной precision при recall >= MIN_RECALL"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    best_threshold = 0.5
    best_precision = -1.0
    best_recall = 0.0

    for idx, thr in enumerate(thresholds):
        p = float(precision[idx + 1])
        r = float(recall[idx + 1])
        if r >= MIN_RECALL and p > best_precision:
            best_threshold = float(thr)
            best_precision = p
            best_recall = r

    if best_precision < 0:
        fallback_idx = int(np.argmax(recall[:-1]))
        best_threshold = float(thresholds[fallback_idx])
        best_precision = float(precision[fallback_idx + 1])
        best_recall = float(recall[fallback_idx + 1])

    return best_threshold, {"precision": best_precision, "recall": best_recall}

def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Расчёт основных метрик качества с фиксированным порогом"""
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate_at_threshold": float(y_pred.mean()),
    }

def plot_model_curves(y_test: np.ndarray, y_test_score: np.ndarray) -> None:
    """Построение Precision-Recall и ROC кривых для тестовой выборки"""
    precision, recall, _ = precision_recall_curve(y_test, y_test_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#4c78a8")
    ax.set_title("Precision-Recall кривая")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    save_figure(fig, FIGURES_DIR / "logistic_precision_recall_curve.png")

    fpr, tpr, _ = roc_curve(y_test, y_test_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#e45756")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC кривая")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    save_figure(fig, FIGURES_DIR / "logistic_roc_curve.png")