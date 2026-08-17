from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CFG
from .dataset import HHDataset, load_data
from .model import HHSelector

METHODS = ["resolved", "semiresolved", "merged"]
METHOD_LABELS = ["Resolved", "Semi-resolved", "Merged"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def load_model(ckpt_path: Path, cfg: CFG, device: str = "cpu") -> HHSelector:
    model = HHSelector(
        ak4_feat_dim=cfg.ak4_feat_dim,
        ak8_feat_dim=cfg.ak8_feat_dim,
        event_feat_dim=cfg.event_feat_dim,
        embed_dim=cfg.embed_dim,
        mlp_hidden=cfg.mlp_hidden,
        dropout=cfg.dropout,
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded model from {ckpt_path} (epoch {state['epoch']}, val_loss={state['val_loss']:.6f})")
    return model


@torch.no_grad()
def predict(model, loader, device="cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a DataLoader.

    Returns
    -------
    preds : (N, 3)  predicted scores
    targets : (N, 3)  ground truth quality
    mask : (N, 3)  availability mask
    """
    all_pred, all_tgt, all_mask = [], [], []
    model.eval()
    for ak4, ak4_m, ak8, ak8_m, ev, tgt, msk, hg, res_idx, semi_idx, mrg_idx in loader:
        ak4 = ak4.to(device)
        ak4_m = ak4_m.to(device)
        ak8 = ak8.to(device)
        ak8_m = ak8_m.to(device)
        ev = ev.to(device)
        hg = hg.to(device)
        res_idx = res_idx.to(device)
        semi_idx = semi_idx.to(device)
        mrg_idx = mrg_idx.to(device)
        pred = model(ak4, ak4_m, ak8, ak8_m, ev, res_idx, semi_idx, mrg_idx, hg).cpu().numpy()
        all_pred.append(pred)
        all_tgt.append(tgt.numpy())
        all_mask.append(msk.numpy())
    return np.concatenate(all_pred), np.concatenate(all_tgt), np.concatenate(all_mask)


def plot_score_distributions(preds, targets, mask, plot_dir: Path):
    """Histogram of predicted scores for each method, split by 'best' vs not."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j, (ax, label) in enumerate(zip(axes, METHOD_LABELS)):
        avail = mask[:, j]
        if not avail.any():
            ax.set_title(f"{label} (no events)")
            continue
        scores_j = preds[avail, j]
        targets_j = targets[avail, j]

        # best = highest target among available methods
        targets_avail = targets.copy()
        targets_avail[~mask] = -1
        best_per_event = targets_avail.argmax(axis=1)
        is_best = best_per_event[avail] == j

        ax.hist(scores_j[is_best], bins=50, range=(0, 1), alpha=0.7, label="best", color=COLORS[j], density=True)
        ax.hist(scores_j[~is_best], bins=50, range=(0, 1), alpha=0.4, label="not best", color="gray", density=True)
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend()
    fig.suptitle("Score distributions (best vs not-best method)")
    fig.tight_layout()
    fig.savefig(plot_dir / "score_distributions.png", dpi=150)
    fig.savefig(plot_dir / "score_distributions.pdf")
    plt.close(fig)
    print(f"  Saved score_distributions.png/pdf")


def plot_argmax_accuracy(preds, targets, mask, plot_dir: Path):
    """Fraction of events where argmax(pred) == argmax(target), for multi-method events."""
    targets_avail = targets.copy()
    targets_avail[~mask] = -1.0

    preds_avail = preds.copy()
    preds_avail[~mask] = -1.0

    multi = mask.sum(axis=1) >= 2
    if multi.sum() == 0:
        print("  No multi-method events for accuracy plot")
        return

    pred_best = preds_avail[multi].argmax(axis=1)
    true_best = targets_avail[multi].argmax(axis=1)
    acc = (pred_best == true_best).mean()
    print(f"  Multi-method accuracy (argmax): {acc:.4f} ({multi.sum()} events)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j, (ax, label) in enumerate(zip(axes, METHOD_LABELS)):
        sel = true_best == j
        if sel.sum() == 0:
            ax.set_title(f"{label} (no events)")
            continue
        correct = (pred_best[sel] == j).mean()
        ax.text(0.5, 0.5, f"acc={correct:.3f}\n(n={sel.sum()})", transform=ax.transAxes, ha="center", va="center", fontsize=16)
        ax.set_title(f"Best = {label}")
        ax.axis("off")
    fig.suptitle(f"Per-method accuracy (overall: {acc:.4f})")
    fig.tight_layout()
    fig.savefig(plot_dir / "argmax_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  Saved argmax_accuracy.png")


def plot_score_vs_error(preds, targets, mask, plot_dir: Path):
    """Scatter: predicted score vs reconstruction error, for available methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j, (ax, label) in enumerate(zip(axes, METHOD_LABELS)):
        avail = mask[:, j]
        if not avail.any():
            ax.set_title(f"{label} (no events)")
            continue
        scores = preds[avail, j]
        tgts = targets[avail, j]
        # reconstruct error from target: error = -scale * ln(target)
        error = -50.0 * np.log(np.clip(tgts, 1e-10, None))
        ax.scatter(scores, error, s=1, alpha=0.3, color=COLORS[j])
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Reconstruction error |mH1-125| + |mH2-125| [GeV]")
        ax.set_title(label)
    fig.suptitle("Predicted score vs reconstruction error")
    fig.tight_layout()
    fig.savefig(plot_dir / "score_vs_error.png", dpi=150)
    plt.close(fig)
    print(f"  Saved score_vs_error.png")


def plot_threshold_efficiency(preds, targets, mask, plot_dir: Path):
    """For increasing score threshold, show average error of retained events."""
    thresholds = np.linspace(0, 1, 51)
    targets_avail = targets.copy()
    targets_avail[~mask] = -1

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, (label, color) in enumerate(zip(METHOD_LABELS, COLORS)):
        avail = mask[:, j]
        if not avail.any():
            continue
        scores = preds[avail, j]
        tgts = targets_avail[avail, j]
        error = -50.0 * np.log(np.clip(tgts, 1e-10, None))

        avg_err = []
        frac_retained = []
        for thr in thresholds:
            keep = scores >= thr
            frac_retained.append(keep.mean())
            avg_err.append(error[keep].mean() if keep.any() else np.nan)

        ax.plot(thresholds, avg_err, color=color, label=f"{label} (error)")
        ax2 = ax.twinx()
        ax2.plot(thresholds, frac_retained, color=color, linestyle="--", alpha=0.5)
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Avg reconstruction error [GeV]")
    ax2.set_ylabel("Fraction of events retained (dashed)")
    ax.set_title("Threshold vs avg error & efficiency")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_dir / "threshold_efficiency.png", dpi=150)
    plt.close(fig)
    print(f"  Saved threshold_efficiency.png")


def main(cfg: CFG, checkpoint: str = "best", max_events: int | None = None, device: str | None = None):
    cfg.plot_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    ak4_pad, ak4_mask, ak8_pad, ak8_mask, event_np, targets, mask, higgses, resolved_idx, semi_idx, merged_idx = load_data(cfg, max_events=max_events)
    ds = HHDataset(ak4_pad, ak4_mask, ak8_pad, ak8_mask,
                   event_np, targets, mask, higgses,
                   resolved_idx, semi_idx, merged_idx)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers,
                        persistent_workers=True, prefetch_factor=2)

    # load model
    ckpt_path = cfg.checkpoint_dir / f"{checkpoint}.pt"
    model = load_model(ckpt_path, cfg, device)

    # inference — model outputs raw logits; apply sigmoid for [0,1] scores
    logits, tgt, msk = predict(model, loader, device)
    preds = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    # filter to multi-method events only (same population as training)
    multi = msk.sum(axis=1) >= 2
    preds, tgt, msk = preds[multi], tgt[multi], msk[multi]
    print(f"Predictions shape: {preds.shape} (multi-method events only)")

    # plots
    print("Generating plots...")
    plot_score_distributions(preds, tgt, msk, cfg.plot_dir)
    plot_argmax_accuracy(preds, tgt, msk, cfg.plot_dir)
    plot_score_vs_error(preds, tgt, msk, cfg.plot_dir)
    plot_threshold_efficiency(preds, tgt, msk, cfg.plot_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HH topology selector")
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = CFG()
    main(cfg, checkpoint=args.checkpoint, max_events=args.max_events, device=args.device)
