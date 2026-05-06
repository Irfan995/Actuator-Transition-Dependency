# Actuator Transition Dependency Aware Deep Learning Approach for Actuator State Prediction

This repository implements an Actuator Transition Dependency (ATD) aware deep learning pipeline for smart-home and smart-office actuator prediction. It trains LSTM and Transformer models on time-ordered sensor-actuator event streams and supports four experiment variants:

- `base`: no dependency-aware weighting.
- `atd`: Actuator Transition Dependency feature and ATD-informed state-loss weighting.
- `sri`: Sensor Relation Inference informed state-loss weighting.
- `atd-sri`: combined ATD and SRI weighting, with ATD input features.

The training script follows the paper-style experimental setup: chronological train/validation/test split, dependency matrices built from the training portion only, and timestamped result exports for each run.

## Repository Layout

```text
training.py        Main training/evaluation entry point
utils.py           CLI, logging, dataset loading, splitting, and run helpers
preprocessing.py   Dataset windowing and sample-weight construction
td_feature.py      ATD and SRI matrix construction
lstm.py            LSTM model
transformer.py     Transformer model
model.py           Training and validation loop
evaluation.py      Test-set metrics
predict.py         Interactive LSTM prediction utility
```

## Requirements

Python 3.8+ is recommended. This repository includes a pyenv version file for
Python 3.8.18 and a `requirements.txt` file for the runtime dependencies.

Install Python with pyenv:

```bash
pyenv install 3.8.18
pyenv local 3.8.18
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you use a GPU, install the PyTorch build that matches your CUDA version from the official PyTorch instructions.

## Dataset Format

Training expects a CSV file with at least these columns:

```text
seconds,sensor_name,state
```

- `seconds`: numeric timestamp in seconds, or a parseable datetime value.
- `sensor_name`: device name. Actuator rows must contain the substring `Actuator`.
- `state`: binary or numeric device state. Values greater than `0` are treated as active/on.

Example:

```csv
seconds,sensor_name,state
0,Kitchen Motion Sensor,1
2,Kitchen Light Actuator,1
5,Door Sensor,0
8,Washroom Fan Actuator,1
```

## Running Experiments

Use `--informed-weighting` to choose the experiment variant.

```bash
python3 training.py \
  --informed-weighting atd \
  --data dataset/apartment-3_sensor_dataset_filtered.csv \
  --testbed T-5
```

Run all four variants:

```bash
python3 training.py --informed-weighting base --data dataset/apartment-3_sensor_dataset_filtered.csv --testbed T-5
python3 training.py --informed-weighting atd --data dataset/apartment-3_sensor_dataset_filtered.csv --testbed T-5
python3 training.py --informed-weighting sri --data dataset/apartment-3_sensor_dataset_filtered.csv --testbed T-5
python3 training.py --informed-weighting atd-sri --data dataset/apartment-3_sensor_dataset_filtered.csv --testbed T-5
```

Common options:

```bash
python3 training.py \
  --informed-weighting atd-sri \
  --data dataset/apartment-3_sensor_dataset_filtered.csv \
  --testbed T-5 \
  --n-seconds 6 \
  --atd-threshold 800 \
  --sri-threshold 3 \
  --alpha 0.1 \
  --seq-len 20 \
  --batch-size 128 \
  --epochs 100 \
  --patience 10 \
  --output-dir results
```

## Testbed Defaults

If `--n-seconds` or `--atd-threshold` are not provided, `training.py` uses these paper-style defaults:

| Testbed | Temporal Window (`tau`) | ATD Threshold (`delta_t`) |
| --- | ---: | ---: |
| `T-1` | 8s | 3800s |
| `T-2` | 3s | 700s |
| `T-3` | 5s | 780s |
| `T-4` | 4s | 3800s |
| `T-5` | 6s | 800s |

## Outputs

Every run is saved in a unique timestamped directory:

```text
results/YYYYMMDD_HHMMSS_t-5_atd_tau6_atd800_sri3p0_alpha0p1/
```

Each run directory contains:

```text
run_log.txt              Timestamped training/status log
evaluation_results.txt   Human-readable LSTM and Transformer metrics
run_summary.json         Structured metadata and metrics
lstm_*.pth               Best LSTM checkpoint
transformer_*.pth        Best Transformer checkpoint
entity_map.json          Entity ID mapping for this run
actuator_ids.json        Actuator IDs for this run
```

The root-level `entity_map.json` and `actuator_ids.json` are also updated for compatibility with `predict.py`.

## Prediction

After training an LSTM model, run:

```bash
python3 predict.py --model results/<run_id>/lstm_<run_id>.pth
```

Enter sensor events as comma-separated names. Optional state can be supplied with `name:state`.

Example:

```text
Kitchen Motion Sensor:1, Door Sensor:0
```
