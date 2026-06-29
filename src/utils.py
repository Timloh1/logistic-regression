import json
from pathlib import Path
import matplotlib.pyplot as plt
from src.config import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR


def ensure_directories() -> None:
    """Создание папок для артефактов"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    """Сохранение словаря в JSON"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Сохранение matplotlib-фигуры в PNG"""
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)