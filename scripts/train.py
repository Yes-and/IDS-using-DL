"""Train a model from a YAML config.

Usage:
    python scripts/train.py --config configs/xiiotid_dnn.yaml
    python scripts/train.py --config configs/cicids2019_cnn.yaml --run-name my_exp
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]
    proc_dir = ROOT / "data" / "processed" / dataset

    X_train = np.load(proc_dir / "X_train.npy")
    X_val = np.load(proc_dir / "X_val.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    y_val = np.load(proc_dir / "y_val.npy")
    with open(proc_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    cfg["input_dim"] = X_train.shape[1]
    cfg["num_classes"] = len(le.classes_)

    from src.data.dataset import IDSDataset
    from src.models.build import build_model
    from src.training.losses import FocalLoss
    from src.training.trainer import fit

    cnn_input = cfg["arch"] == "cnn1d"
    train_ds = IDSDataset(X_train, y_train, cnn_input=cnn_input)
    val_ds = IDSDataset(X_val, y_val, cnn_input=cnn_input)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg["batch_size"] * 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    # Class-weighted loss
    if cfg.get("use_class_weights"):
        counts = np.bincount(y_train)
        weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    else:
        weights = None

    if cfg.get("loss") == "focal":
        criterion = FocalLoss(gamma=cfg.get("focal_gamma", 2.0), weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5) if cfg.get("scheduler") == "plateau" else None

    run_name = args.run_name or f"{dataset}_{cfg['arch']}"
    fit(model, train_loader, val_loader, criterion, optimizer, scheduler,
        cfg["epochs"], device, ROOT / "results", run_name)


if __name__ == "__main__":
    main()
