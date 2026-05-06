import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from td_feature import compute_actuator_adjacency, compute_sensor_adjacency


PAPER_PARAMS = {
    "T-1": {"n_seconds": 8, "atd_threshold": 3800},
    "T-2": {"n_seconds": 3, "atd_threshold": 700},
    "T-3": {"n_seconds": 5, "atd_threshold": 780},
    "T-4": {"n_seconds": 4, "atd_threshold": 3800},
    "T-5": {"n_seconds": 6, "atd_threshold": 800},
}

DEFAULT_DATASETS = [
    "dataset/apartment-3_sensor_dataset_filtered.csv",
    "dataset/lab_sensor_actuator_dataset.csv",
    "dataset/3w_apartment-3_sensor_dataset_filtered.csv",
]

TECHNIQUES = ("base", "atd", "sri", "atd-sri")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LSTM/Transformer actuator predictors with selectable dependency technique."
    )
    parser.add_argument("--technique", choices=TECHNIQUES, default="atd")
    parser.add_argument("--data", type=str, default=None, help="Path to event CSV.")
    parser.add_argument("--testbed", choices=sorted(PAPER_PARAMS), default="T-5")
    parser.add_argument("--n-seconds", type=float, default=None)
    parser.add_argument("--atd-threshold", type=float, default=None)
    parser.add_argument("--sri-threshold", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def safe_name(value):
    return str(value).replace("/", "-").replace(" ", "_").replace(".", "p")


def make_run_paths(args, n_seconds, atd_threshold):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"{timestamp}_{args.testbed.lower()}_{args.technique}_"
        f"tau{safe_name(n_seconds)}_atd{safe_name(atd_threshold)}_"
        f"sri{safe_name(args.sri_threshold)}_alpha{safe_name(args.alpha)}"
    )
    output_dir = Path(args.output_dir)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def setup_logger(log_file):
    logger = logging.getLogger("atd_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def json_float(value):
    return None if value is None else float(value)


def json_metrics(metrics):
    return {key: json_float(value) for key, value in metrics.items()}


def resolve_dataset_path(user_path):
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    for candidate in DEFAULT_DATASETS:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        "No dataset found. Pass --data path/to/file.csv. Expected columns: "
        "seconds, sensor_name, state."
    )


def prepare_dataframe(path):
    df = pd.read_csv(path)
    required = {"seconds", "sensor_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if "state" not in df.columns:
        df["state"] = 1.0

    if not np.issubdtype(df["seconds"].dtype, np.number):
        df["seconds"] = pd.to_datetime(df["seconds"])
        df["seconds"] = (df["seconds"] - df["seconds"].min()).dt.total_seconds()

    df["state"] = pd.to_numeric(df["state"], errors="coerce").fillna(0)
    df["state"] = (df["state"] > 0).astype(float)
    return df.sort_values("seconds").reset_index(drop=True)


def chronological_split(df, n_seconds):
    times = df["seconds"].to_numpy(dtype=np.float64)
    train_end = np.quantile(times, 0.70)
    val_end = np.quantile(times, 0.85)
    gap = float(n_seconds)

    train_df = df[df["seconds"] <= train_end].copy()
    val_df = df[(df["seconds"] > train_end + gap) & (df["seconds"] <= val_end)].copy()
    test_df = df[df["seconds"] > val_end + gap].copy()
    return train_df, val_df, test_df, gap


def make_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def build_matrices(train_df, entity_to_id, technique, atd_threshold, sri_threshold, logger):
    use_atd = technique in {"atd", "atd-sri"}
    use_sri = technique in {"sri", "atd-sri"}

    atd_matrix = None
    sensor_matrix = None

    if use_atd:
        logger.info("Computing ATD matrix from training data only")
        atd_matrix = compute_actuator_adjacency(
            train_df, entity_to_id, th_seconds=atd_threshold, symmetric=False
        )
        logger.info(
            "ATD matrix ready: shape=%s, nonzero=%s",
            atd_matrix.shape,
            int(np.count_nonzero(atd_matrix)),
        )

    if use_sri:
        logger.info("Computing SRI matrix from training data only")
        sensor_matrix = compute_sensor_adjacency(
            train_df, entity_to_id, th_seconds=sri_threshold, symmetric=True
        )
        logger.info(
            "SRI matrix ready: shape=%s, nonzero=%s",
            sensor_matrix.shape,
            int(np.count_nonzero(sensor_matrix)),
        )

    return atd_matrix, sensor_matrix, use_atd, use_sri
