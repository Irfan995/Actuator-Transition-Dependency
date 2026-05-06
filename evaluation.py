import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(model, dataloader, log_file, model_name="Model", results_file=None):
    device = next(model.parameters()).device
    model.eval()
    all_act_preds, all_act_true = [], []
    all_state_preds, all_state_true = [], []

    with torch.no_grad():
        for seq_ids, seq_states, seq_deltas, seq_atd, actuator, state, _ in dataloader:
            seq_ids = seq_ids.long().to(device, non_blocking=True)
            seq_states = seq_states.float().to(device, non_blocking=True)
            seq_deltas = seq_deltas.float().to(device, non_blocking=True)
            seq_atd = seq_atd.float().to(device, non_blocking=True)
            actuator = actuator.long().to(device, non_blocking=True)
            state = state.float().to(device, non_blocking=True)

            pred_actuator, pred_state = model(seq_ids, seq_states, seq_deltas, seq_atd)

            all_act_preds.extend(pred_actuator.argmax(dim=1).cpu().numpy())
            all_act_true.extend(actuator.cpu().numpy())
            all_state_preds.extend((pred_state.view(-1) > 0.5).int().cpu().numpy())
            all_state_true.extend(state.view(-1).int().cpu().numpy())

    act_accuracy = accuracy_score(all_act_true, all_act_preds)
    act_f1 = f1_score(all_act_true, all_act_preds, average="weighted", zero_division=0)
    act_precision = precision_score(
        all_act_true, all_act_preds, average="weighted", zero_division=0
    )
    act_recall = recall_score(all_act_true, all_act_preds, average="weighted", zero_division=0)

    state_accuracy = accuracy_score(all_state_true, all_state_preds)
    state_f1 = f1_score(all_state_true, all_state_preds, zero_division=0)
    state_precision = precision_score(all_state_true, all_state_preds, zero_division=0)
    state_recall = recall_score(all_state_true, all_state_preds, zero_division=0)

    metrics = {
        "actuator_accuracy": act_accuracy,
        "actuator_f1": act_f1,
        "actuator_precision": act_precision,
        "actuator_recall": act_recall,
        "state_accuracy": state_accuracy,
        "state_f1": state_f1,
        "state_precision": state_precision,
        "state_recall": state_recall,
    }

    report = (
        f"\n--- Evaluation Results: {model_name} ---\n"
        "\nActuator Prediction:\n"
        f"Accuracy: {act_accuracy}\n"
        f"F1 Score: {act_f1}\n"
        f"Precision: {act_precision}\n"
        f"Recall: {act_recall}\n"
        "\nState Prediction:\n"
        f"Accuracy: {state_accuracy}\n"
        f"F1 Score: {state_f1}\n"
        f"Precision: {state_precision}\n"
        f"Recall: {state_recall}\n"
    )
    print(report)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(report)
    if results_file is not None:
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(report)
    return metrics
