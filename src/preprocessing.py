import numpy as np
import pandas as pd
from src.config import WARMUP_END, TRAIN_END, VAL_END

def assign_split_column(p2p_df: pd.DataFrame) -> pd.DataFrame:
    """Добавление колонки split (warmup/train/val/test) по времени"""
    result = p2p_df.copy()
    result["split"] = "unknown"
    result.loc[result["EventTime"] < WARMUP_END, "split"] = "warmup"
    result.loc[
        (result["EventTime"] >= WARMUP_END) & (result["EventTime"] < TRAIN_END),
        "split",
    ] = "train"
    result.loc[
        (result["EventTime"] >= TRAIN_END) & (result["EventTime"] < VAL_END),
        "split",
    ] = "val"
    result.loc[result["EventTime"] >= VAL_END, "split"] = "test"
    return result

def add_features(p2p_df: pd.DataFrame, trans_df: pd.DataFrame) -> pd.DataFrame:
    """Генерация признаков: временные, исторические, профили отправителя/получателя"""
    result = p2p_df.copy()

    # Час транзакции
    result["event_hour"] = result["EventTime"].dt.hour
    # День недели (0 = понедельник)
    result["event_dayofweek"] = result["EventTime"].dt.dayofweek
    # День месяца
    result["event_day"] = result["EventTime"].dt.day
    # Флаг выходного дня
    result["is_weekend"] = (result["event_dayofweek"] >= 5).astype(np.int8)

    # Логарифм суммы перевода
    result["amount_log1p"] = np.log1p(result["Amount"])

    # Порядковый номер перевода отправителя (начиная с 0)
    result["sender_prev_count"] = result.groupby("UserID").cumcount()
    # Накопленная сумма предыдущих переводов отправителя (без текущего)
    sender_cumsum = result.groupby("UserID")["Amount"].cumsum()
    result["sender_prev_amount_sum"] = sender_cumsum - result["Amount"]
    # Средняя сумма предыдущих переводов отправителя
    result["sender_prev_amount_mean"] = (
        result["sender_prev_amount_sum"] / result["sender_prev_count"].replace(0, np.nan)
    )

    # Порядковый номер перевода получателю (начиная с 0)
    result["recipient_prev_count"] = result.groupby("RecipientID").cumcount()
    # Накопленная сумма предыдущих переводов получателю (без текущего)
    recipient_cumsum = result.groupby("RecipientID")["Amount"].cumsum()
    result["recipient_prev_amount_sum"] = recipient_cumsum - result["Amount"]
    # Средняя сумма предыдущих переводов получателю
    result["recipient_prev_amount_mean"] = (
        result["recipient_prev_amount_sum"] / result["recipient_prev_count"].replace(0, np.nan)
    )

    # Часы с предыдущего перевода отправителя
    sender_prev_time = result.groupby("UserID")["EventTime"].shift(1)
    result["sender_hours_since_prev"] = (
        (result["EventTime"] - sender_prev_time).dt.total_seconds() / 3600.0
    )

    # Часы с предыдущего перевода получателю
    recipient_prev_time = result.groupby("RecipientID")["EventTime"].shift(1)
    result["recipient_hours_since_prev"] = (
        (result["EventTime"] - recipient_prev_time).dt.total_seconds() / 3600.0
    )

    # Число уникальных получателей отправителя до текущей транзакции
    sender_pair_order = result.groupby(["UserID", "RecipientID"]).cumcount()
    sender_new_recipient = sender_pair_order.eq(0).astype(np.int8)
    result["sender_prev_unique_recipients"] = (
        sender_new_recipient.groupby(result["UserID"]).cumsum() - sender_new_recipient
    )

    # Число уникальных отправителей получателя до текущей транзакции
    recipient_pair_order = result.groupby(["RecipientID", "UserID"]).cumcount()
    recipient_new_sender = recipient_pair_order.eq(0).astype(np.int8)
    result["recipient_prev_unique_senders"] = (
        recipient_new_sender.groupby(result["RecipientID"]).cumsum() - recipient_new_sender
    )

    # Профиль пользователя по общему журналу до warmup-периода
    trans_history = trans_df.loc[trans_df["EventTime"] < WARMUP_END].copy()
    user_profile = trans_history.groupby("UserID").agg(
        # Число транзакций пользователя
        trans_count=("Amount", "size"),
        # Средняя сумма транзакции
        trans_amount_mean=("Amount", "mean"),
        # Общая сумма транзакций
        trans_amount_sum=("Amount", "sum"),
        # Число уникальных мерчантов
        trans_unique_merchants=("MerchantID", "nunique"),
        # Число уникальных типов мерчантов
        trans_unique_merchant_types=("MerchantType", "nunique"),
        # Доля успешных транзакций
        trans_success_rate=("TransactionSuccessful", "mean"),
    ).reset_index()

    # Присоединение профиля отправителя
    result = result.merge(
        user_profile.add_prefix("sender_trans_").rename(columns={"sender_trans_UserID": "UserID"}),
        on="UserID",
        how="left",
    )

    # Присоединение профиля получателя
    result = result.merge(
        user_profile.add_prefix("recipient_trans_").rename(
            columns={"recipient_trans_UserID": "RecipientID"}
        ),
        on="RecipientID",
        how="left",
    )

    # Отношение суммы к средней прошлой сумме отправителя
    result["amount_to_sender_mean_ratio"] = result["Amount"] / result["sender_prev_amount_mean"]
    # Отношение суммы к средней прошлой сумме получателя
    result["amount_to_recipient_mean_ratio"] = (
        result["Amount"] / result["recipient_prev_amount_mean"]
    )

    return result

def get_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Список колонок-признаков (исключая мета-информацию и целевую переменную)"""
    excluded_columns = {
        "EventTime",
        "UserID",
        "RecipientID",
        "Currency",
        "IsFraud",
        "split",
    }
    return [col for col in feature_df.columns if col not in excluded_columns]