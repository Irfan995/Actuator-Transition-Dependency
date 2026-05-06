import json

import torch

from evaluation import evaluate_model
from lstm import LSTMModel
from model import train_model
from preprocessing import SmartHomeDataset
from transformer import TransformerModel
from utils import (
    PAPER_PARAMS,
    build_matrices,
    chronological_split,
    json_metrics,
    make_loader,
    make_run_paths,
    parse_args,
    prepare_dataframe,
    resolve_dataset_path,
    setup_logger,
)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    defaults = PAPER_PARAMS[args.testbed]
    n_seconds = args.n_seconds if args.n_seconds is not None else defaults["n_seconds"]
    atd_threshold = (
        args.atd_threshold if args.atd_threshold is not None else defaults["atd_threshold"]
    )
    run_id, run_dir = make_run_paths(args, n_seconds, atd_threshold)
    log_file = run_dir / "run_log.txt"
    results_file = run_dir / "evaluation_results.txt"
    summary_file = run_dir / "run_summary.json"
    logger = setup_logger(log_file)

    data_path = resolve_dataset_path(args.data)
    logger.info("Run ID: %s", run_id)
    logger.info("Output directory: %s", run_dir)
    logger.info("Starting run")
    logger.info("Device: %s", device)
    logger.info("Technique: %s", args.technique)
    logger.info("Loading dataset: %s", data_path)
    df = prepare_dataframe(data_path)
    entity_to_id = {name: i for i, name in enumerate(df["sensor_name"].unique())}
    logger.info("Loaded %s events", len(df))
    logger.info("Entity count: %s", len(entity_to_id))

    train_df, val_df, test_df, gap = chronological_split(df, n_seconds)
    logger.info(
        "Chronological split: train=%s, val=%s, test=%s, gap=%ss",
        len(train_df),
        len(val_df),
        len(test_df),
        gap,
    )

    atd_matrix, sensor_matrix, use_atd, use_sri = build_matrices(
        train_df,
        entity_to_id,
        args.technique,
        atd_threshold,
        args.sri_threshold,
        logger,
    )

    logger.info("Building datasets and samples")
    dataset_kwargs = {
        "entity_to_id": entity_to_id,
        "A_matrix_actuator": atd_matrix,
        "A_matrix_sensor": sensor_matrix,
        "n_seconds": n_seconds,
        "seq_len": args.seq_len,
        "alpha": args.alpha,
        "use_atd": use_atd,
        "use_sri": use_sri,
    }
    train_dataset = SmartHomeDataset(train_df, **dataset_kwargs)
    val_dataset = SmartHomeDataset(val_df, **dataset_kwargs)
    test_dataset = SmartHomeDataset(test_df, **dataset_kwargs)
    logger.info(
        "Samples: train=%s, val=%s, test=%s",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
    if min(len(train_dataset), len(val_dataset), len(test_dataset)) == 0:
        raise ValueError("One split produced zero actuator samples. Adjust the data/window.")

    logger.info(
        "Creating DataLoaders: batch_size=%s, num_workers=%s",
        args.batch_size,
        args.num_workers,
    )
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers)

    num_entities = len(entity_to_id)
    input_dim = num_entities + 1
    embed_dim = 32
    hidden_dim = 64
    state_embed_dim = 8
    time_embed_dim = 8
    atd_embed_dim = 4
    transformer_heads = 2
    transformer_layers = 2

    logger.info("Building LSTM and Transformer models")
    lstm_model = LSTMModel(
        input_dim,
        embed_dim,
        state_embed_dim,
        time_embed_dim,
        atd_embed_dim,
        hidden_dim,
        num_entities,
    ).to(device)
    transformer_model = TransformerModel(
        input_dim,
        embed_dim,
        state_embed_dim,
        time_embed_dim,
        atd_embed_dim,
        num_heads=transformer_heads,
        num_layers=transformer_layers,
        output_dim=num_entities,
        max_len=args.seq_len,
        hidden_dim=hidden_dim,
    ).to(device)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n================ {run_id} ================\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Technique: {args.technique}\n")
        f.write(f"Training on {device}\n")
        f.write("Split: chronological 70/15/15, dependency matrices from train only\n")
        f.write(f"n_seconds/tau: {n_seconds}, ATD threshold: {atd_threshold}\n")
        f.write(f"SRI threshold: {args.sri_threshold}, alpha: {args.alpha}\n")
        f.write(f"seq_len: {args.seq_len}, batch_size: {args.batch_size}\n")
        f.write(
            f"Events: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}\n"
        )
        f.write(
            f"Samples: train={len(train_dataset)}, val={len(val_dataset)}, "
            f"test={len(test_dataset)}\n"
        )

    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Technique: {args.technique}\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Output directory: {run_dir}\n")
        f.write(f"n_seconds/tau: {n_seconds}\n")
        f.write(f"ATD threshold: {atd_threshold}\n")
        f.write(f"SRI threshold: {args.sri_threshold}\n")
        f.write(f"alpha: {args.alpha}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")

    logger.info("Starting LSTM training")
    lstm_path = run_dir / f"lstm_{run_id}.pth"
    lstm_best_val_loss = train_model(
        lstm_model,
        train_loader,
        val_loader,
        log_file,
        save_path=lstm_path,
        epochs=args.epochs,
        patience=args.patience,
        model_name="LSTM",
    )
    logger.info("Loading best LSTM checkpoint: %s", lstm_path)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    logger.info("Evaluating LSTM on test split")
    lstm_metrics = evaluate_model(
        lstm_model,
        test_loader,
        log_file=log_file,
        model_name="LSTM",
        results_file=results_file,
    )

    logger.info("Starting Transformer training")
    transformer_path = run_dir / f"transformer_{run_id}.pth"
    transformer_best_val_loss = train_model(
        transformer_model,
        train_loader,
        val_loader,
        log_file,
        save_path=transformer_path,
        epochs=args.epochs,
        patience=args.patience,
        model_name="Transformer",
    )
    logger.info("Loading best Transformer checkpoint: %s", transformer_path)
    transformer_model.load_state_dict(torch.load(transformer_path, map_location=device))
    logger.info("Evaluating Transformer on test split")
    transformer_metrics = evaluate_model(
        transformer_model,
        test_loader,
        log_file=log_file,
        model_name="Transformer",
        results_file=results_file,
    )

    logger.info("Saving artifacts")
    with open(run_dir / "entity_map.json", "w", encoding="utf-8") as f:
        json.dump(entity_to_id, f)
    actuator_ids = [
        entity_id for name, entity_id in entity_to_id.items() if "Actuator" in name
    ]
    with open(run_dir / "actuator_ids.json", "w", encoding="utf-8") as f:
        json.dump(actuator_ids, f)
    with open("entity_map.json", "w", encoding="utf-8") as f:
        json.dump(entity_to_id, f)
    with open("actuator_ids.json", "w", encoding="utf-8") as f:
        json.dump(actuator_ids, f)

    summary = {
        "run_id": run_id,
        "output_dir": str(run_dir),
        "data": str(data_path),
        "testbed": args.testbed,
        "technique": args.technique,
        "device": str(device),
        "n_seconds": n_seconds,
        "atd_threshold": atd_threshold,
        "sri_threshold": args.sri_threshold,
        "alpha": args.alpha,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "events": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "samples": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "model_paths": {
            "lstm": str(lstm_path),
            "transformer": str(transformer_path),
        },
        "best_val_loss": {
            "lstm": json_float(lstm_best_val_loss),
            "transformer": json_float(transformer_best_val_loss),
        },
        "metrics": {
            "lstm": json_metrics(lstm_metrics),
            "transformer": json_metrics(transformer_metrics),
        },
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Run complete")
    logger.info("Log file: %s", log_file)
    logger.info("Evaluation results: %s", results_file)
    logger.info("Summary JSON: %s", summary_file)


if __name__ == "__main__":
    main()
