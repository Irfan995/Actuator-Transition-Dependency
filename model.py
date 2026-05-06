import copy
import time

import numpy as np
import torch
import torch.nn as nn


def move_batch(batch, device):
    seq_ids, seq_states, seq_deltas, seq_atd, actuator, state, weights = batch
    return (
        seq_ids.long().to(device, non_blocking=True),
        seq_states.float().to(device, non_blocking=True),
        seq_deltas.float().to(device, non_blocking=True),
        seq_atd.float().to(device, non_blocking=True),
        actuator.long().to(device, non_blocking=True),
        state.float().view(-1).to(device, non_blocking=True),
        weights.float().view(-1).to(device, non_blocking=True),
    )


def compute_loss(
    pred_actuator,
    pred_state,
    actuator,
    state,
    weights,
    criterion_actuator,
    criterion_state,
):
    loss_actuator = criterion_actuator(pred_actuator, actuator)
    state_loss_per_sample = criterion_state(pred_state, state)
    loss_state = (state_loss_per_sample * weights).mean()
    return loss_actuator + loss_state


def validate_model(model, val_loader):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    criterion_actuator = nn.CrossEntropyLoss()
    criterion_state = nn.BCELoss(reduction="none")

    with torch.no_grad():
        for batch in val_loader:
            seq_ids, seq_states, seq_deltas, seq_atd, actuator, state, weights = move_batch(
                batch, device
            )
            pred_actuator, pred_state = model(seq_ids, seq_states, seq_deltas, seq_atd)
            loss = compute_loss(
                pred_actuator,
                pred_state,
                actuator,
                state,
                weights,
                criterion_actuator,
                criterion_state,
            )
            total_loss += loss.item()

    return total_loss / max(1, len(val_loader))


def train_model(
    model,
    train_loader,
    val_loader,
    log_file,
    save_path,
    epochs=100,
    patience=10,
    learning_rate=0.001,
    weight_decay=1e-5,
    model_name="Model",
):
    device = next(model.parameters()).device
    criterion_actuator = nn.CrossEntropyLoss()
    criterion_state = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_val_loss = np.inf
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        start = time.time()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            seq_ids, seq_states, seq_deltas, seq_atd, actuator, state, weights = move_batch(
                batch, device
            )
            pred_actuator, pred_state = model(seq_ids, seq_states, seq_deltas, seq_atd)
            loss = compute_loss(
                pred_actuator,
                pred_state,
                actuator,
                state,
                weights,
                criterion_actuator,
                criterion_state,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (pred_actuator.argmax(dim=1) == actuator).sum().item()
            total_samples += actuator.size(0)

        val_loss = validate_model(model, val_loader)
        duration = time.time() - start
        train_loss = total_loss / max(1, len(train_loader))
        accuracy = 100.0 * total_correct / max(1, total_samples)
        line = (
            f"[{model_name}] Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Actuator Acc: {accuracy:.2f}% | "
            f"Dur: {duration:.2f}s"
        )
        print(line)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"[{model_name}] Early stopping triggered after {epoch + 1} epochs. "
                f"Best val loss: {best_val_loss:.4f}"
            )
            break

    if best_model_state is None:
        best_model_state = model.state_dict()
    torch.save(best_model_state, save_path)
    line = f"[{model_name}] Saved best model to {save_path}"
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return best_val_loss
