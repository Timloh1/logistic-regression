import pandas as pd
from src.config import P2P_PATH, TRANS_PATH

def load_p2p_data() -> pd.DataFrame:
    """Загрузка и сортировка журнала P2P-переводов"""
    p2p_df = pd.read_csv(P2P_PATH, parse_dates=["EventTime"])
    if "Unnamed: 0" in p2p_df.columns:
        p2p_df = p2p_df.drop(columns=["Unnamed: 0"])
    p2p_df = p2p_df.sort_values(["EventTime", "UserID", "RecipientID"]).reset_index(drop=True)
    return p2p_df

def load_trans_data() -> pd.DataFrame:
    """Загрузка общего журнала транзакций (только нужные колонки)"""
    # Колонки общего журнала, не нужные для признаков
    usecols = [
        "UserID",
        "EventTime",
        "MerchantID",
        "MerchantType",
        "Amount",
        "Country",
        "TransactionSuccessful",
    ]
    trans_df = pd.read_csv(TRANS_PATH, usecols=usecols, parse_dates=["EventTime"])
    trans_df = trans_df.sort_values(["EventTime", "UserID"]).reset_index(drop=True)
    return trans_df