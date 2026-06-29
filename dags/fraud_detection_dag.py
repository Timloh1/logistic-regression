import sys
from datetime import datetime
from pathlib import Path
from airflow.decorators import dag, task


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


from src.pipeline import prepare_datasets, run_eda_stage, build_features_stage, train_and_evaluate_stage


@dag(
    dag_id="fraud_detection_pipeline",
    description="Train and evaluate the P2P fraud detection model.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "fraud-detection"],
)
def fraud_detection_pipeline():
    @task
    def prepare():
        return prepare_datasets()

    @task
    def eda(paths):
        return run_eda_stage(paths)

    @task
    def features(paths):
        return build_features_stage(paths)

    @task
    def train(features_path, eda_summary_path):
        return train_and_evaluate_stage(features_path, eda_summary_path)

    prepared_paths = prepare()
    eda_summary_path = eda(prepared_paths)
    features_path = features(prepared_paths)
    train(features_path, eda_summary_path)


fraud_detection_pipeline()
