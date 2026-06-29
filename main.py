from src.pipeline import run_pipeline


def main() -> None:
    metrics_path = run_pipeline()
    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()
