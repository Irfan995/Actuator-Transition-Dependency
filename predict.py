import argparse
import json
from pathlib import Path

import torch

from lstm import LSTMModel


def predict_top_k_actuators(
    event_names,
    event_states,
    model,
    entity_to_id,
    id_to_entity,
    actuator_ids,
    seq_len=20,
    k=5,
):
    model.eval()
    device = next(model.parameters()).device
    pad_id = len(entity_to_id)

    ids = [entity_to_id.get(name, pad_id) for name in event_names][-seq_len:]
    states = [1.0 if float(state) > 0 else 0.0 for state in event_states][-seq_len:]
    deltas = [0.0] * len(ids)
    atd = [0.0] * len(ids)

    pad_len = seq_len - len(ids)
    if pad_len > 0:
        ids = ([pad_id] * pad_len) + ids
        states = ([0.0] * pad_len) + states
        deltas = ([0.0] * pad_len) + deltas
        atd = ([0.0] * pad_len) + atd

    with torch.no_grad():
        pred_logits, pred_state_prob = model(
            torch.tensor([ids], dtype=torch.long, device=device),
            torch.tensor([states], dtype=torch.float32, device=device),
            torch.tensor([deltas], dtype=torch.float32, device=device),
            torch.tensor([atd], dtype=torch.float32, device=device),
        )

    pred_probs = torch.softmax(pred_logits, dim=1).squeeze()
    actuator_mask = torch.zeros_like(pred_probs)
    actuator_mask[actuator_ids] = 1
    masked_probs = pred_probs * actuator_mask

    top_k_probs, top_k_indices = torch.topk(masked_probs, min(k, len(actuator_ids)))
    predicted_state = "On" if pred_state_prob.item() > 0.5 else "Off"

    results = []
    for prob, idx in zip(top_k_probs, top_k_indices):
        if prob.item() == 0.0:
            continue
        results.append((id_to_entity[str(idx.item())], predicted_state, round(prob.item(), 4)))
    return results or [("No actuator prediction", "Unknown", 0.0)]


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive LSTM actuator prediction.")
    parser.add_argument("--model", required=True, help="Path to a trained lstm_*.pth file.")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    with open("entity_map.json", "r", encoding="utf-8") as f:
        entity_to_id = json.load(f)
    id_to_entity = {str(v): k for k, v in entity_to_id.items()}
    with open("actuator_ids.json", "r", encoding="utf-8") as f:
        actuator_ids = json.load(f)

    input_dim = len(entity_to_id) + 1
    model = LSTMModel(
        input_dim=input_dim,
        embed_dim=32,
        state_embed_dim=8,
        time_embed_dim=8,
        atd_embed_dim=4,
        hidden_dim=64,
        output_dim=len(entity_to_id),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    valid_names = [name for name in entity_to_id if "Actuator" not in name]
    print("\nValid sensor names:")
    for name in valid_names:
        print(f"- {name}")

    while True:
        raw = input("\nEnter sensor events as name[:state], comma-separated, or 'exit':\n> ")
        if raw.lower() in {"exit", "quit"}:
            break
        if not raw.strip():
            continue

        event_names = []
        event_states = []
        for item in raw.split(","):
            parts = [part.strip() for part in item.split(":", 1)]
            event_names.append(parts[0])
            event_states.append(float(parts[1]) if len(parts) == 2 else 1.0)

        invalid = [name for name in event_names if name not in valid_names]
        if invalid:
            print(f"Invalid sensor name(s): {', '.join(invalid)}")
            continue

        predictions = predict_top_k_actuators(
            event_names,
            event_states,
            model,
            entity_to_id,
            id_to_entity,
            actuator_ids,
            seq_len=args.seq_len,
            k=args.k,
        )
        for actuator, state, prob in predictions:
            print(f"- Actuator: {actuator}, State: {state}, Confidence: {prob * 100:.2f}%")
