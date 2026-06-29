import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.config import FIGURES_DIR, REPORTS_DIR
from src.utils import save_figure


def run_eda(p2p_df: pd.DataFrame, trans_df: pd.DataFrame) -> dict:
    """Расчёт EDA-статистик, построение графиков, запись сводки"""
    fraud_rate = float(p2p_df["IsFraud"].mean())
    fraud_count = int(p2p_df["IsFraud"].sum())
    unique_senders = int(p2p_df["UserID"].nunique())
    unique_recipients = int(p2p_df["RecipientID"].nunique())

    # Распределение классов
    class_counts = p2p_df["IsFraud"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Legit", "Fraud"], class_counts.values, color=["#4c78a8", "#e45756"])
    ax.set_title("Распределение классов")
    ax.set_ylabel("Количество операций")
    save_figure(fig, FIGURES_DIR / "class_distribution.png")

    # Гистограммы логарифма суммы по классам
    legit_amount = np.log1p(p2p_df.loc[p2p_df["IsFraud"] == 0, "Amount"])
    fraud_amount = np.log1p(p2p_df.loc[p2p_df["IsFraud"] == 1, "Amount"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(legit_amount, bins=50, alpha=0.7, density=True, label="Legit", color="#4c78a8")
    ax.hist(fraud_amount, bins=50, alpha=0.7, density=True, label="Fraud", color="#e45756")
    ax.set_title("Распределение log1p(Amount)")
    ax.set_xlabel("log1p(Amount)")
    ax.set_ylabel("Плотность")
    ax.legend()
    save_figure(fig, FIGURES_DIR / "amount_log_hist_by_class.png")

    # Доля мошенничества по часам и дням недели
    eda_time_df = p2p_df[["EventTime", "IsFraud"]].copy()
    eda_time_df["hour"] = eda_time_df["EventTime"].dt.hour
    eda_time_df["dayofweek"] = eda_time_df["EventTime"].dt.dayofweek

    fraud_by_hour = eda_time_df.groupby("hour")["IsFraud"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(fraud_by_hour.index, fraud_by_hour.values, marker="o", color="#e45756")
    ax.set_title("Доля мошенничества по часам")
    ax.set_xlabel("Час")
    ax.set_ylabel("Fraud rate")
    ax.set_xticks(range(24))
    save_figure(fig, FIGURES_DIR / "fraud_rate_by_hour.png")

    fraud_by_day = eda_time_df.groupby("dayofweek")["IsFraud"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(fraud_by_day.index.astype(str), fraud_by_day.values, color="#f28e2b")
    ax.set_title("Доля мошенничества по дням недели")
    ax.set_xlabel("День недели")
    ax.set_ylabel("Fraud rate")
    save_figure(fig, FIGURES_DIR / "fraud_rate_by_dayofweek.png")

    # Топ категорий мерчантов из общего журнала
    top_merchant_types = trans_df["MerchantType"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_merchant_types.index[::-1], top_merchant_types.values[::-1], color="#59a14f")
    ax.set_title("Топ MerchantType по числу транзакций")
    ax.set_xlabel("Количество транзакций")
    save_figure(fig, FIGURES_DIR / "top_merchant_types.png")

    eda_summary = {
        "p2p_rows": int(len(p2p_df)),
        "trans_rows": int(len(trans_df)),
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
        "unique_senders": unique_senders,
        "unique_recipients": unique_recipients,
        "mean_amount": float(p2p_df["Amount"].mean()),
        "median_amount": float(p2p_df["Amount"].median()),
        "fraud_mean_amount": float(p2p_df.loc[p2p_df["IsFraud"] == 1, "Amount"].mean()),
        "nonfraud_mean_amount": float(p2p_df.loc[p2p_df["IsFraud"] == 0, "Amount"].mean()),
        "top_merchant_types": top_merchant_types.to_dict(),
    }

    # Markdown-отчёт
    report_lines = [
        "# EDA Summary",
        "",
        f"- Размер P2P-журнала: `{eda_summary['p2p_rows']}`",
        f"- Размер общего журнала транзакций: `{eda_summary['trans_rows']}`",
        f"- Количество мошеннических операций: `{eda_summary['fraud_count']}`",
        f"- Доля мошенничества: `{eda_summary['fraud_rate']:.6f}`",
        f"- Уникальных отправителей: `{eda_summary['unique_senders']}`",
        f"- Уникальных получателей: `{eda_summary['unique_recipients']}`",
        f"- Средняя сумма обычного перевода: `{eda_summary['nonfraud_mean_amount']:.2f}`",
        f"- Средняя сумма мошеннического перевода: `{eda_summary['fraud_mean_amount']:.2f}`",
    ]
    (REPORTS_DIR / "eda_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    return eda_summary