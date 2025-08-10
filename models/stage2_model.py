# ============================ IMPORTS & SETUP ============================
import os
import math
import time
import random
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


# ---------------------------- Reproducibility ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(123)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================ DATA INTERFACES ============================
@dataclass
class Config:
    # Sequence branch
    f_seq: int = 10  # [O,H,L,C,V,SMA50,SMA200,MACD,RSI,VIX]
    T: int = 64  # lookback window
    # GBDT (tabular) branch
    n_estimators: int = 600
    num_leaves: int = 128
    learning_rate: float = 0.01
    max_depth: int = -1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 10
    # Leaf embedding
    leaf_emb_dim: int = 8
    # Head / fusion
    mlp_hidden: Tuple[int, int] = (512, 256, 64)
    mlp_dropout: float = 0.1
    # Training
    batch_size: int = 256
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    # CV / splitting
    n_splits: int = 5
    purge: int = 64  # purge window (>= T)
    embargo: int = 5  # holdout buffer after val
    # Targeting
    multitask_cls: bool = False
    cls_weight: float = 0.2


# Purged + Embargoed time-series split (indices must be time-ordered)
class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int, purge: int = 0, embargo: int = 0):
        self.n_splits = n_splits
        self.purge = max(0, purge)
        self.embargo = max(0, embargo)

    def split(self, n_samples: int):
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        indices = np.arange(n_samples)
        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append((start, stop))
            current = stop

        for i, (val_start, val_stop) in enumerate(folds):
            # Train indices exclude [val_start - purge, val_stop + embargo)
            left_stop = max(0, val_start - self.purge)
            right_start = min(n_samples, val_stop + self.embargo)
            train_idx = np.concatenate(
                [indices[:left_stop], indices[right_start:]], axis=0
            )
            val_idx = indices[val_start:val_stop]
            yield train_idx, val_idx


# ============================ LIGHTGBM OOF PIPE ===========================
def fit_lgbm_oof(
    X_tab: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, lgb.LGBMRegressor]:
    """
    Returns:
      oof_pred  : [N, 1] float OOF predictions
      oof_leaves: [N, n_estimators] int leaf IDs (OOF)
      gbdt_full : refit model on full training (for test-time preds/leaves)
    """
    N = X_tab.shape[0]
    oof_pred = np.full((N, 1), np.nan, dtype=np.float32)
    oof_leaves = np.full((N, cfg.n_estimators), -1, dtype=np.int32)

    splitter = PurgedTimeSeriesSplit(cfg.n_splits, purge=cfg.purge, embargo=cfg.embargo)

    lgb_params = dict(
        n_estimators=cfg.n_estimators,
        num_leaves=cfg.num_leaves,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_samples=cfg.min_child_samples,
        objective="regression",
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_jobs=-1,
        verbose=-1,
    )

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(N), 1):
        X_tr, y_tr = X_tab[tr_idx], y[tr_idx]
        X_va, y_va = X_tab[va_idx], y[va_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

        # OOF predictions and leaf IDs for validation block
        pred = model.predict(X_va, raw_score=False).reshape(-1, 1)
        leaves = model.predict(X_va, pred_leaf=True)
        # leaves shape: [len(va_idx), n_trees]
        oof_pred[va_idx, :] = pred.astype(np.float32)
        oof_leaves[va_idx, :] = leaves.astype(np.int32)

        print(
            f"[GBDT] Fold {fold}/{cfg.n_splits}: val MSE = {mean_squared_error(y_va, pred):.6f}"
        )

    assert not np.isnan(oof_pred).any(), "OOF predictions contain NaNs (check splits)."
    assert (
        oof_leaves >= 0
    ).all(), "OOF leaves contain invalid IDs. Check LightGBM params."

    # Refit on full training for test-time inference
    gbdt_full = lgb.LGBMRegressor(**lgb_params)
    gbdt_full.fit(X_tab, y)

    return oof_pred, oof_leaves, gbdt_full


# ============================ TORCH MODEL ================================
class SeqEncoder(nn.Module):
    def __init__(
        self,
        f_seq: int,
        h: int = 128,
        n_layers: int = 1,
        bidir: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=f_seq,
            hidden_size=h,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.out_dim = h * (2 if bidir else 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, F]
        _, hN = self.gru(x_seq)  # hN: [layers*dir, B, H]
        h_last = hN[-1]  # [B, H] last layer’s hidden
        return h_last


class LeafEmbedder(nn.Module):
    """
    Embeds LightGBM leaf IDs per tree and concatenates.
    Assumes fixed (n_estimators, num_leaves) across folds by config.
    """

    def __init__(self, n_trees: int, num_leaves: int, emb_dim: int):
        super().__init__()
        self.n_trees = n_trees
        self.num_leaves = num_leaves
        self.emb_dim = emb_dim
        self.tables = nn.ModuleList(
            [nn.Embedding(num_leaves, emb_dim) for _ in range(n_trees)]
        )

        # Simple init
        for emb in self.tables:
            nn.init.xavier_uniform_(emb.weight)

    @property
    def out_dim(self) -> int:
        return self.n_trees * self.emb_dim

    def forward(self, leaves: torch.Tensor) -> torch.Tensor:
        # leaves: [B, n_trees] (long)
        zs = [
            self.tables[t](leaves[:, t]) for t in range(self.n_trees)
        ]  # list of [B, emb_dim]
        return torch.cat(zs, dim=-1)  # [B, n_trees*emb_dim]


class FusionHead(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden: Tuple[int, ...] = (256, 64),
        dropout: float = 0.2,
        multitask_cls: bool = False,
    ):
        super().__init__()
        layers = []
        last = d_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.mlp = nn.Sequential(*layers)
        self.out_reg = nn.Linear(last, 1)
        self.mt_cls = multitask_cls
        if self.mt_cls:
            self.out_cls = nn.Linear(last, 2)

    def forward(self, u: torch.Tensor):
        z = self.mlp(u)
        y_reg = self.out_reg(z).squeeze(-1)
        if hasattr(self, "out_cls"):
            y_cls = self.out_cls(z)
            return y_reg, y_cls
        return y_reg, None


class LateFusionModel(nn.Module):
    def __init__(
        self,
        f_seq: int,
        leaf_embedder: LeafEmbedder,
        d_gbdt: int = 1,
        d_aux: int = 1,
        seq_hidden: int = 128,
        head_hidden: Tuple[int, int] = (256, 64),
        head_dropout: float = 0.2,
        multitask_cls: bool = False,
    ):
        super().__init__()
        self.seq = SeqEncoder(f_seq=f_seq, h=seq_hidden, n_layers=4)
        self.leaf_emb = leaf_embedder
        d_in = self.seq.out_dim + self.leaf_emb.out_dim + d_gbdt + d_aux
        self.head = FusionHead(
            d_in, hidden=head_hidden, dropout=head_dropout, multitask_cls=multitask_cls
        )

    def forward(self, x_seq, y_gbdt, leaf_ids, x_aux):
        h_seq = self.seq(x_seq)  # [B, Hs]
        z_leaf = self.leaf_emb(leaf_ids)  # [B, D_leaf]
        u = torch.cat([h_seq, y_gbdt, z_leaf, x_aux], dim=-1)
        return self.head(u)  # y_reg, (opt) y_cls


# ============================ DATASET / DATALOADER =======================
class FusionDataset(Dataset):
    """
    Expects pre-aligned arrays (same N):
      - X_seq      : [N, T, F_seq], float32
      - y_gbdt     : [N, 1], float32  (OOF or test-time GBDT preds)
      - leaf_ids   : [N, n_trees], int64
      - x_aux      : [N, d_aux], float32
      - y          : [N], float32
    """

    def __init__(
        self,
        X_seq: np.ndarray,
        y_gbdt: np.ndarray,
        leaf_ids: np.ndarray,
        x_aux: np.ndarray,
        y: np.ndarray,
    ):
        assert (
            X_seq.shape[0]
            == y_gbdt.shape[0]
            == leaf_ids.shape[0]
            == x_aux.shape[0]
            == y.shape[0]
        )
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_gbdt = torch.tensor(y_gbdt, dtype=torch.float32)
        self.leaf_ids = torch.tensor(leaf_ids, dtype=torch.long)
        self.x_aux = torch.tensor(x_aux, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (
            self.X_seq[idx],
            self.y_gbdt[idx],
            self.leaf_ids[idx],
            self.x_aux[idx],
            self.y[idx],
        )


# ============================ TRAIN / EVAL LOOPS =========================
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion_reg,
    criterion_cls=None,
    cls_weight=0.2,
    grad_clip=1.0,
):
    model.train()
    total, total_reg = 0.0, 0.0
    for x_seq, y_gbdt, leaf_ids, x_aux, y in loader:
        x_seq = x_seq.to(DEVICE)
        y_gbdt = y_gbdt.to(DEVICE)
        leaf_ids = leaf_ids.to(DEVICE)
        x_aux = x_aux.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        y_reg, y_cls = model(x_seq, y_gbdt, leaf_ids, x_aux)
        loss = criterion_reg(y_reg, y)
        if (y_cls is not None) and (criterion_cls is not None):
            # Binary up/down from y (optional): here we threshold at 0
            y_bin = (y > 0).long()
            loss = loss + cls_weight * criterion_cls(y_cls, y_bin)

        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total += loss.item() * y.size(0)
        total_reg += criterion_reg(y_reg, y).item() * y.size(0)
    return total / len(loader.dataset), total_reg / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion_reg):
    model.eval()
    total, preds, targs = 0.0, [], []
    for x_seq, y_gbdt, leaf_ids, x_aux, y in loader:
        x_seq = x_seq.to(DEVICE)
        y_gbdt = y_gbdt.to(DEVICE)
        leaf_ids = leaf_ids.to(DEVICE)
        x_aux = x_aux.to(DEVICE)
        y = y.to(DEVICE)

        y_reg, _ = model(x_seq, y_gbdt, leaf_ids, x_aux)
        loss = criterion_reg(y_reg, y)
        total += loss.item() * y.size(0)
        preds.append(y_reg.detach().cpu().numpy())
        targs.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    mse = ((preds - targs) ** 2).mean()
    return total / len(loader.dataset), mse, preds


# ============================ DRIVER (TRAIN + INFER) =====================
def run_training(
    X_seq: np.ndarray,  # [N, T, F_seq]
    X_tab: np.ndarray,  # [N, F_tab]
    x_aux: np.ndarray,  # [N, d_aux]
    y: np.ndarray,  # [N]
    cfg: Config,
):
    N = X_tab.shape[0]
    assert X_seq.shape[0] == N == x_aux.shape[0] == y.shape[0]
    print(
        f"[Data] N={N}, X_seq={X_seq.shape}, X_tab={X_tab.shape}, x_aux={x_aux.shape}"
    )

    # 1) Compute GBDT OOF features
    oof_pred, oof_leaves, gbdt_full = fit_lgbm_oof(X_tab, y, cfg)
    print(
        f"[GBDT] OOF pred shape={oof_pred.shape}, OOF leaves shape={oof_leaves.shape}"
    )

    # 2) Build leaf embedder (fixed by config)
    leaf_emb = LeafEmbedder(
        n_trees=cfg.n_splits and cfg.n_estimators,  # always cfg.n_estimators
        num_leaves=cfg.num_leaves,
        emb_dim=cfg.leaf_emb_dim,
    )

    # 3) Split again for fusion training (time-aware, same purge/embargo)
    splitter = PurgedTimeSeriesSplit(cfg.n_splits, purge=cfg.purge, embargo=cfg.embargo)

    best_models = []
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(N), 1):
        ds_tr = FusionDataset(
            X_seq[tr_idx],
            oof_pred[tr_idx],
            oof_leaves[tr_idx],
            x_aux[tr_idx],
            y[tr_idx],
        )
        ds_va = FusionDataset(
            X_seq[va_idx],
            oof_pred[va_idx],
            oof_leaves[va_idx],
            x_aux[va_idx],
            y[va_idx],
        )
        dl_tr = DataLoader(
            ds_tr,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        model = LateFusionModel(
            f_seq=cfg.f_seq,
            leaf_embedder=leaf_emb,
            d_gbdt=1,
            d_aux=x_aux.shape[1],
            seq_hidden=256,  # increased from 128
            head_hidden=cfg.mlp_hidden,
            head_dropout=cfg.mlp_dropout,
            multitask_cls=cfg.multitask_cls,
        ).to(DEVICE)

        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        criterion_reg = nn.SmoothL1Loss()

        best_val = float("inf")
        best_state = None

        for epoch in range(1, cfg.epochs + 1):
            tr_loss, tr_reg = train_one_epoch(
                model,
                dl_tr,
                optimizer,
                criterion_reg,
                criterion_cls=(nn.CrossEntropyLoss() if cfg.multitask_cls else None),
                cls_weight=cfg.cls_weight,
                grad_clip=cfg.grad_clip,
            )
            va_loss, va_mse, _ = evaluate(model, dl_va, criterion_reg)
            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 2 == 0 or epoch == 1:
                print(
                    f"[Fold {fold}] Epoch {epoch:02d} | tr={tr_loss:.5f} (reg {tr_reg:.5f}) | va={va_loss:.5f} | va_mse={va_mse:.5f}"
                )

        # Save best
        model.load_state_dict(best_state)
        best_models.append(model.state_dict())
        fold_metrics.append(best_val)
        print(f"[Fold {fold}] Best val loss={best_val:.6f}")

    print(
        f"[CV] Mean best val loss: {np.mean(fold_metrics):.6f} ± {np.std(fold_metrics):.6f}"
    )

    # 4) Refit fusion on ALL data (optional, recommended)
    ds_all = FusionDataset(X_seq, oof_pred, oof_leaves, x_aux, y)
    dl_all = DataLoader(
        ds_all, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=0
    )

    final_model = LateFusionModel(
        f_seq=cfg.f_seq,
        leaf_embedder=leaf_emb,
        d_gbdt=1,
        d_aux=x_aux.shape[1],
        seq_hidden=256,
        head_hidden=cfg.mlp_hidden,
        head_dropout=cfg.mlp_dropout,
        multitask_cls=cfg.multitask_cls,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        final_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion_reg = nn.SmoothL1Loss()
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_reg = train_one_epoch(
            final_model,
            dl_all,
            optimizer,
            criterion_reg,
            criterion_cls=(nn.CrossEntropyLoss() if cfg.multitask_cls else None),
            cls_weight=cfg.cls_weight,
            grad_clip=cfg.grad_clip,
        )
        if epoch % 2 == 0 or epoch == 1:
            print(f"[Refit] Epoch {epoch:02d} | loss={tr_loss:.5f}")

    # Return artifacts
    artifacts = {
        "cfg": cfg,
        "leaf_embedder_state": {
            k: v.cpu().clone() for k, v in leaf_emb.state_dict().items()
        },
        "fusion_state": {
            k: v.cpu().clone() for k, v in final_model.state_dict().items()
        },
        "gbdt_full": gbdt_full,
    }
    return artifacts


@torch.no_grad()
def predict(
    artifacts: dict,
    X_seq_test: np.ndarray,  # [M, T, F_seq]
    X_tab_test: np.ndarray,  # [M, F_tab]
    x_aux_test: np.ndarray,  # [M, d_aux]
) -> np.ndarray:
    cfg: Config = artifacts["cfg"]
    gbdt_full: lgb.LGBMRegressor = artifacts["gbdt_full"]

    # Build test-time GBDT features
    pred_test = (
        gbdt_full.predict(X_tab_test, raw_score=False).reshape(-1, 1).astype(np.float32)
    )
    leaves_test = gbdt_full.predict(X_tab_test, pred_leaf=True).astype(np.int64)
    assert leaves_test.shape[1] == cfg.n_estimators, "n_trees mismatch at test-time."

    # Recreate modules and load weights
    leaf_emb = LeafEmbedder(
        n_trees=cfg.n_estimators, num_leaves=cfg.num_leaves, emb_dim=cfg.leaf_emb_dim
    ).to(DEVICE)
    leaf_emb.load_state_dict(artifacts["leaf_embedder_state"])

    model = LateFusionModel(
        f_seq=cfg.f_seq,
        leaf_embedder=leaf_emb,
        d_gbdt=1,
        d_aux=x_aux_test.shape[1],
        seq_hidden=256,
        head_hidden=cfg.mlp_hidden,
        head_dropout=cfg.mlp_dropout,
        multitask_cls=cfg.multitask_cls,
    ).to(DEVICE)
    model.load_state_dict(artifacts["fusion_state"])
    model.eval()

    ds_te = FusionDataset(
        X_seq_test,
        pred_test,
        leaves_test,
        x_aux_test,
        y=np.zeros(X_seq_test.shape[0], dtype=np.float32),
    )
    dl_te = DataLoader(
        ds_te, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0
    )

    preds = []
    for x_seq, y_gbdt, leaf_ids, x_aux, _ in dl_te:
        x_seq = x_seq.to(DEVICE)
        y_gbdt = y_gbdt.to(DEVICE)
        leaf_ids = leaf_ids.to(DEVICE)
        x_aux = x_aux.to(DEVICE)
        y_reg, _ = model(x_seq, y_gbdt, leaf_ids, x_aux)
        preds.append(y_reg.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


# ----- Test -----
if __name__ == "__main__":
    cfg = Config(
        mlp_hidden=(512, 256, 64),  # more/larger MLP layers
        mlp_dropout=0.2,  # you can also try lowering dropout if underfitting
    )

    # --- Load real data (from the scraper) ---
    dat = np.load("stage2_dataset.npz")
    X_seq = dat["X_seq"].astype(np.float32)  # [N, T, 10]
    X_tab = dat["X_tab"].astype(np.float32)  # [N, F_tab]
    x_aux = dat["x_aux"].astype(np.float32)  # [N, d_aux]
    y = dat["y"].astype(np.float32)  # [N]

    print("NaNs in X_seq:", np.isnan(X_seq).sum())
    print("NaNs in X_tab:", np.isnan(X_tab).sum())
    print("NaNs in x_aux:", np.isnan(x_aux).sum())
    print("NaNs in y:", np.isnan(y).sum())

    # Remove rows with any NaNs
    mask = (
        ~np.isnan(X_seq).any(axis=(1, 2))
        & ~np.isnan(X_tab).any(axis=1)
        & ~np.isnan(x_aux).any(axis=1)
        & ~np.isnan(y)
    )
    X_seq = X_seq[mask]
    X_tab = X_tab[mask]
    x_aux = x_aux[mask]
    y = y[mask]

    # --- Derive shapes + sync config ---
    N, T, F_SEQ = X_seq.shape
    F_TAB = X_tab.shape[1]
    D_AUX = x_aux.shape[1]
    cfg.T = T
    cfg.f_seq = F_SEQ
    cfg.purge = max(cfg.purge, cfg.T)

    print(f"[Loaded] N={N} T={T} F_SEQ={F_SEQ} F_TAB={F_TAB} D_AUX={D_AUX}")

    # --- Chronological split: last 20% as holdout ---
    split = int(0.8 * N)

    artifacts = run_training(
        X_seq[:split], X_tab[:split], x_aux[:split], y[:split], cfg
    )

    y_hat = predict(artifacts, X_seq[split:], X_tab[split:], x_aux[split:])

    # Optional: quick metric
    mse = ((y_hat - y[split:]) ** 2).mean()
    print(f"[Holdout] MSE={mse:.6f} | preds[:5]={np.round(y_hat[:5], 6)}")
    print("Predicted min/max/mean:", y_hat.min(), y_hat.max(), y_hat.mean())
    print("Actual min/max/mean:", y[split:].min(), y[split:].max(), y[split:].mean())

    import matplotlib.pyplot as plt

    plt.scatter(y[split:], y_hat, alpha=0.3)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Holdout: Actual vs Predicted")
    plt.show()

    importances = artifacts["gbdt_full"].feature_importances_
    print("Feature importances:", importances)
