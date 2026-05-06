import numpy as np
import pandas as pd


def ensure_numeric_time(df, time_col):
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    if not np.issubdtype(df[time_col].dtype, np.number):
        df[time_col] = pd.to_datetime(df[time_col])
        df[time_col] = (df[time_col] - df[time_col].min()).dt.total_seconds()
    return df


def row_minmax_normalize(matrix):
    normalized = matrix.astype(np.float32, copy=True)
    for i in range(normalized.shape[0]):
        row = normalized[i]
        row_min = row.min()
        row_max = row.max()
        if row_max > row_min:
            normalized[i] = (row - row_min) / (row_max - row_min)
    return normalized


def compute_actuator_adjacency(
    df,
    entity_to_id,
    time_col="seconds",
    name_col="sensor_name",
    th_seconds=5.0,
    symmetric=False,
):
    """
    Build the paper's Actuator Transition Dependency (ATD) matrix.

    Consecutive actuator events create a directed edge when the second event
    occurs within th_seconds of the first event. Rows are min-max normalized.
    """
    df = ensure_numeric_time(df, time_col)
    actuator_df = df[df[name_col].str.contains("Actuator", na=False)].reset_index(drop=True)

    size = len(entity_to_id)
    adjacency = np.zeros((size, size), dtype=np.float32)

    for i in range(len(actuator_df) - 1):
        a_name, a_time = actuator_df.loc[i, name_col], actuator_df.loc[i, time_col]
        b_name, b_time = actuator_df.loc[i + 1, name_col], actuator_df.loc[i + 1, time_col]
        delta = float(b_time - a_time)

        if 0 < delta <= float(th_seconds):
            a_id = entity_to_id.get(a_name)
            b_id = entity_to_id.get(b_name)
            if a_id is not None and b_id is not None:
                adjacency[a_id, b_id] += 1.0
                if symmetric:
                    adjacency[b_id, a_id] += 1.0

    return row_minmax_normalize(adjacency)


def compute_sensor_adjacency(
    df,
    entity_to_id,
    time_col="seconds",
    name_col="sensor_name",
    th_seconds=5.0,
    symmetric=True,
):
    """
    Optional Sensor Relation Inference (SRI) matrix for the paper's baseline.

    This is kept available for ATD-SRI experiments, but training.py defaults
    to ATD because the paper identifies it as the practical trade-off.
    """
    df = ensure_numeric_time(df, time_col)
    sensor_df = df[~df[name_col].str.contains("Actuator", na=False)].reset_index(drop=True)

    size = len(entity_to_id)
    adjacency = np.zeros((size, size), dtype=np.float32)

    for i in range(len(sensor_df) - 1):
        a_name, a_time = sensor_df.loc[i, name_col], sensor_df.loc[i, time_col]
        b_name, b_time = sensor_df.loc[i + 1, name_col], sensor_df.loc[i + 1, time_col]
        delta = float(b_time - a_time)

        if 0 < delta <= float(th_seconds):
            a_id = entity_to_id.get(a_name)
            b_id = entity_to_id.get(b_name)
            if a_id is not None and b_id is not None:
                adjacency[a_id, b_id] += 1.0
                if symmetric:
                    adjacency[b_id, a_id] += 1.0

    return row_minmax_normalize(adjacency)
