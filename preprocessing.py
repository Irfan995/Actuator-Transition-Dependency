from collections import deque

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SmartHomeDataset(Dataset):
    """
    Creates paper-style actuator prediction samples.

    For every actuator state change, the previous n_seconds of sensor and
    actuator events are collected, padded/truncated to seq_len, and augmented
    with per-event time-gap and ATD transition-strength features.
    """

    def __init__(
        self,
        df,
        entity_to_id,
        A_matrix_actuator=None,
        A_matrix_sensor=None,
        n_seconds=5,
        seq_len=20,
        alpha=0.1,
        use_atd=False,
        use_sri=False,
    ):
        self.df = df.sort_values("seconds").reset_index(drop=True).copy()
        self.entity_to_id = entity_to_id
        self.A_matrix_actuator = A_matrix_actuator
        self.A_matrix_sensor = A_matrix_sensor
        self.n_seconds = float(n_seconds)
        self.seq_len = int(seq_len)
        self.alpha = float(alpha)
        self.use_atd = bool(use_atd)
        self.use_sri = bool(use_sri)
        self.samples = self.build_samples()

    def build_samples(self):
        df = self.df.copy()
        if not np.issubdtype(df["seconds"].dtype, np.number):
            df["seconds"] = pd.to_datetime(df["seconds"])
            df["seconds"] = (df["seconds"] - df["seconds"].min()).dt.total_seconds()

        names = df["sensor_name"].to_numpy()
        times = df["seconds"].to_numpy(dtype=np.float64)
        raw_states = df["state"].to_numpy() if "state" in df.columns else np.ones(len(df))
        ids = df["sensor_name"].map(lambda name: self.entity_to_id.get(name, 0)).to_numpy()

        is_actuator_by_id = {
            idx: ("Actuator" in name) for name, idx in self.entity_to_id.items()
        }

        window = deque()
        samples = []
        pad_id = len(self.entity_to_id)
        pad_state = 0.0
        pad_delta = 0.0
        pad_atd = 0.0

        for i in range(len(df)):
            current_time = float(times[i])
            current_name = names[i]
            current_id = int(ids[i])
            current_state = 1.0 if float(raw_states[i]) > 0 else 0.0

            cutoff = current_time - self.n_seconds
            while window and window[0][0] < cutoff:
                window.popleft()

            recent_events = list(window)[-self.seq_len :]
            seq_ids = [event_id for _, event_id, _ in recent_events]
            seq_states = [state for _, _, state in recent_events]
            seq_times = [event_time for event_time, _, _ in recent_events]

            seq_deltas = [0.0]
            for left, right in zip(seq_times, seq_times[1:]):
                seq_deltas.append(max(0.0, float(right - left)))
            if not seq_times:
                seq_deltas = []

            seq_atd = [0.0] * len(seq_ids)
            previous_actuator_id = None
            if self.use_atd and self.A_matrix_actuator is not None:
                for j, event_id in enumerate(seq_ids):
                    if not is_actuator_by_id.get(event_id, False):
                        continue
                    if previous_actuator_id is not None:
                        seq_atd[j] = float(self.A_matrix_actuator[previous_actuator_id, event_id])
                    previous_actuator_id = event_id

            pad_len = self.seq_len - len(seq_ids)
            if pad_len > 0:
                seq_ids = ([pad_id] * pad_len) + seq_ids
                seq_states = ([pad_state] * pad_len) + seq_states
                seq_deltas = ([pad_delta] * pad_len) + seq_deltas
                seq_atd = ([pad_atd] * pad_len) + seq_atd

            if "Actuator" in current_name:
                previous_actuator_id = None
                for event_id in reversed(seq_ids):
                    if event_id != pad_id and is_actuator_by_id.get(event_id, False):
                        previous_actuator_id = event_id
                        break

                strength = 0.0
                if (
                    self.use_atd
                    and previous_actuator_id is not None
                    and self.A_matrix_actuator is not None
                ):
                    strength += float(self.A_matrix_actuator[previous_actuator_id, current_id])

                if self.use_sri and self.A_matrix_sensor is not None:
                    sensor_ids = [
                        event_id
                        for event_id in seq_ids
                        if event_id != pad_id and not is_actuator_by_id.get(event_id, False)
                    ]
                    for left_id, right_id in zip(sensor_ids, sensor_ids[1:]):
                        strength += float(self.A_matrix_sensor[left_id, right_id])

                sample_weight = 1.0 + self.alpha * strength
                samples.append(
                    (
                        seq_ids,
                        seq_states,
                        seq_deltas,
                        seq_atd,
                        current_id,
                        current_state,
                        sample_weight,
                    )
                )

            window.append((current_time, current_id, current_state))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_ids, seq_states, seq_deltas, seq_atd, actuator, state, weight = self.samples[idx]
        return (
            torch.tensor(seq_ids, dtype=torch.long),
            torch.tensor(seq_states, dtype=torch.float32),
            torch.tensor(seq_deltas, dtype=torch.float32),
            torch.tensor(seq_atd, dtype=torch.float32),
            torch.tensor(actuator, dtype=torch.long),
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )
