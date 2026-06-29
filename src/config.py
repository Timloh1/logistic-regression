from pathlib import Path
import pandas as pd

# Корневая папка проекта
ROOT_DIR = Path(__file__).resolve().parent.parent

# Путь к журналу P2P-переводов
P2P_PATH = ROOT_DIR / "data" / "final_p2p_log.csv"

# Путь к журналу банковских транзакций
TRANS_PATH = ROOT_DIR / "data" / "final_trans_log.csv"

# Папка для артефактов
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Подпапка для графиков
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Подпапка для текстовых отчётов
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Сохранение промежуточных файлов пайплайна airflow
INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"
PREPARED_P2P_PATH = INTERMEDIATE_DIR / "p2p_prepared.pkl"
TRANS_DATA_PATH = INTERMEDIATE_DIR / "trans.pkl"
FEATURES_PATH = INTERMEDIATE_DIR / "features.pkl"
EDA_SUMMARY_PATH = INTERMEDIATE_DIR / "eda_summary.json"

# Фиксированный random seed для воспроизводимости
RANDOM_STATE = 123

# Границы временных периодов
WARMUP_END = pd.Timestamp("2021-02-01 00:00:00+03:00")
TRAIN_END = pd.Timestamp("2021-03-01 00:00:00+03:00")
VAL_END = pd.Timestamp("2021-03-16 00:00:00+03:00")

# Минимально допустимый recall
MIN_RECALL = 0.7