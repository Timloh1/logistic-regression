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


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray, method: str = "distance_to_line") -> tuple[float, dict]:
    """
    Подбор порога различными методами.
    method: 
        'max_precision_at_recall' - максимизация precision при recall >= MIN_RECALL
        'distance_to_line' - максимальное расстояние от прямой от (0,1) до (1,0) на PR-кривой до точки
        'distance_to_ideal' - минимальное расстояние до идеальной точки (recall=1, precision=1)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    # Корректировка длин (precision и recall на 1 элемент длиннее thresholds)
    if len(thresholds) == len(precision) - 1:
        precision = precision[:-1]
        recall = recall[:-1]
    
    # Удаляем возможные NaN и inf
    mask = np.isfinite(precision) & np.isfinite(recall) & np.isfinite(thresholds)
    precision = precision[mask]
    recall = recall[mask]
    thresholds = thresholds[mask]
    
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    
    if method == "max_precision_at_recall":
        for idx, thr in enumerate(thresholds):
            p = float(precision[idx])
            r = float(recall[idx])
            if r >= MIN_RECALL and p > best_precision:
                best_threshold = float(thr)
                best_precision = p
                best_recall = r
        
        # если ни один порог не дал нужного recall
        if best_precision == 0.0:
            # Находим порог с максимальным recall
            best_idx = np.argmax(recall)
            best_threshold = float(thresholds[best_idx])
            best_precision = float(precision[best_idx])
            best_recall = float(recall[best_idx])
        
        info = {
            "method": "max_precision_at_recall",
            "min_recall_constraint": MIN_RECALL,
            "precision": best_precision,
            "recall": best_recall,
        }        

    elif method == "distance_to_line":
        """
        Метод максимального расстояния до прямой от (recall=0, precision=1) до (recall=1, precision=0).
        Прямая: p = 1 - r  =>  r + p - 1 = 0
        Расстояние: |r + p - 1| / sqrt(2)
        """
        distances = np.abs(recall + precision - 1) / np.sqrt(2)
        best_idx = np.argmax(distances)
        best_threshold = float(thresholds[best_idx])
        best_precision = float(precision[best_idx])
        best_recall = float(recall[best_idx])
        
        info = {
            "method": "distance_to_line",
            "threshold": best_threshold,
            "precision": best_precision,
            "recall": best_recall,
        }     

    elif method == "distance_to_ideal":
        distances = np.sqrt((1 - recall)**2 + (1 - precision)**2)
        best_idx = np.argmin(distances)
        best_threshold = float(thresholds[best_idx])
        best_precision = float(precision[best_idx])
        best_recall = float(recall[best_idx])
        
        info = {
            "method": "distance_to_ideal",
            "threshold": best_threshold,
            "precision": best_precision,
            "recall": best_recall,
        }  

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: 'max_precision_at_recall', 'distance_to_line', 'distance_to_ideal'")
    
    return best_threshold, info


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

