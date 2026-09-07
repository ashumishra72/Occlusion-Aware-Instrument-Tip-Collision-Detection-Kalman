"""
Temporal (sequence) models: LSTM, GRU, and a small TCN (Temporal
Convolutional Network), all sharing one training loop and differing only in
their core layer, selected via --arch.

Frame-level models (XGBoost etc.) look at one frame's features at a time.
These look at a short window of the last --seq-len frames, so they can
directly learn motion patterns (closing speed, sustained proximity) from
the raw sequence rather than relying on hand-engineered rolling features.

Sequence construction and leakage: a sequence for frame t uses frames
[t-seq_len+1 .. t]. To keep this leak-free under the same StratifiedGroupKFold
scheme as the other models (5-second time chunks = "groups"), a sequence is
only built when all seq_len frames fall inside the SAME group - so no
sequence ever straddles a fold boundary. This means frames within the first
(seq_len-1) positions of each 5-second group don't get a prediction (not
enough history yet); with seq_len=10 at 5fps that drops ~9 frames per
~25-frame group, so total evaluated frames is smaller than the frame-level
models' - this is reported explicitly, not hidden.

Usage:
    python scripts/11_train_temporal_models.py --arch lstm
    python scripts/11_train_temporal_models.py --arch gru
    python scripts/11_train_temporal_models.py --arch tcn
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_cv_utils import best_fbeta_threshold, load_features, save_model_results

torch.manual_seed(42)


class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return self.fc(h[-1]).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1]).squeeze(-1)


class TCNBlock(nn.Module):
    """Causal 1D conv block: pads left only, so output at position i never
    sees input at position > i."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return self.relu(self.conv(x))


class TCNClassifier(nn.Module):
    def __init__(self, n_features, channels=32):
        super().__init__()
        self.block1 = TCNBlock(n_features, channels, kernel_size=3, dilation=1)
        self.block2 = TCNBlock(channels, channels, kernel_size=3, dilation=2)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, seq_len) for Conv1d
        x = self.block1(x)
        x = self.block2(x)
        return self.fc(x[:, :, -1]).squeeze(-1)


ARCHS = {"lstm": LSTMClassifier, "gru": GRUClassifier, "tcn": TCNClassifier}


def build_sequences(df, feature_cols, seq_len, group_seconds):
    """Build (X, y, frame_idx, group) arrays, one sequence per eligible
    frame, never crossing a group boundary."""
    groups = (df["time_s"] // group_seconds).astype(int)
    X_list, y_list, frame_idx_list, group_list, time_list = [], [], [], [], []
    for g, sub in df.groupby(groups):
        if len(sub) < seq_len:
            continue
        feats = sub[feature_cols].to_numpy(dtype=np.float32)
        labels = sub["label_collision"].to_numpy()
        frame_idxs = sub["frame_idx"].to_numpy()
        times = sub["time_s"].to_numpy()
        for i in range(seq_len - 1, len(sub)):
            X_list.append(feats[i - seq_len + 1: i + 1])
            y_list.append(labels[i])
            frame_idx_list.append(frame_idxs[i])
            time_list.append(times[i])
            group_list.append(g)
    return (np.stack(X_list), np.array(y_list), np.array(frame_idx_list),
            np.array(time_list), np.array(group_list))


def train_torch_model(arch, X_tr, y_tr, X_val, y_val, n_features, epochs=80, lr=1e-3, patience=12):
    model_cls = ARCHS[arch]
    model = model_cls(n_features)
    n_pos, n_neg = float((y_tr == 1).sum()), float((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss, best_state, patience_ctr = float("inf"), None, 0
    batch_size = 64
    n = len(X_tr_t)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        if val_loss < best_val_loss - 1e-4:
            best_val_loss, best_state, patience_ctr = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=list(ARCHS), required=True)
    parser.add_argument("--seq-len", type=int, default=10, help="Frames of history per sequence (~2s at 5fps)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--group-seconds", type=float, default=5.0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--fbeta", type=float, default=1.3)
    args = parser.parse_args()

    df, feature_cols = load_features()
    X, y, frame_idx, time_s, groups = build_sequences(df, feature_cols, args.seq_len, args.group_seconds)
    n_features = X.shape[2]
    print(f"[{args.arch}] Built {len(X)} sequences (seq_len={args.seq_len}) from {len(df)} frames "
          f"across {len(np.unique(groups))} groups; {int(y.sum())} positive ({100 * y.mean():.2f}%)")

    kf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof_proba = np.full(len(X), np.nan)
    fold_test_idx = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X, y, groups), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]
        split = int(len(X_train) * (1 - args.val_frac))
        X_tr, y_tr = X_train[:split], y_train[:split]
        X_val, y_val = X_train[split:], y_train[split:]

        scaler = StandardScaler().fit(X_tr.reshape(-1, n_features))
        def scale(a):
            shape = a.shape
            return scaler.transform(a.reshape(-1, n_features)).reshape(shape).astype(np.float32)
        X_tr_s, X_val_s, X_test_s = scale(X_tr), scale(X_val), scale(X_test)

        model = train_torch_model(args.arch, X_tr_s, y_tr, X_val_s, y_val, n_features)
        with torch.no_grad():
            proba = torch.sigmoid(model(torch.tensor(X_test_s, dtype=torch.float32))).numpy()
        oof_proba[test_idx] = proba
        fold_test_idx.append(test_idx)
        print(f"[{args.arch}] Fold {fold_i}/{args.n_folds}: n={len(test_idx)} trained")

    oof = pd.DataFrame({
        "frame_idx": frame_idx, "time_s": time_s, "label_collision": y,
        "predicted_proba": oof_proba,
    })
    oof["predicted_proba_raw"] = oof["predicted_proba"]
    if args.smooth_window > 1:
        oof = oof.sort_values("time_s").reset_index(drop=True)
        oof["predicted_proba"] = oof["predicted_proba"].rolling(
            args.smooth_window, center=True, min_periods=1).mean()

    global_threshold, _ = best_fbeta_threshold(oof["label_collision"], oof["predicted_proba"], beta=args.fbeta)
    oof["predicted_label"] = (oof["predicted_proba"] >= global_threshold).astype(int)

    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    fold_metrics = []
    for fold_i, test_idx in enumerate(fold_test_idx, start=1):
        test_frame_idx = frame_idx[test_idx]
        sub = oof[oof["frame_idx"].isin(test_frame_idx)]
        fold_metrics.append({
            "fold": fold_i, "n_test": int(len(sub)), "n_positive": int(sub["label_collision"].sum()),
            "precision": float(precision_score(sub["label_collision"], sub["predicted_label"], zero_division=0)),
            "recall": float(recall_score(sub["label_collision"], sub["predicted_label"], zero_division=0)),
            "f1": float(f1_score(sub["label_collision"], sub["predicted_label"], zero_division=0)),
            "roc_auc": float(roc_auc_score(sub["label_collision"], sub["predicted_proba"]))
                       if sub["label_collision"].nunique() > 1 else None,
        })

    model_name = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN"}[args.arch]
    save_model_results(model_name, oof, fold_metrics, global_threshold,
                        extra={"seq_len": args.seq_len,
                               "n_sequences": len(X),
                               "note": f"Sequence model: evaluates {len(oof)} of {len(df)} frames "
                                       f"(frames needing {args.seq_len - 1} preceding frames of context "
                                       "within the same 5s group are excluded, not imputed)."})


if __name__ == "__main__":
    main()
