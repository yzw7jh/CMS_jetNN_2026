from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .config import CFG
from .dataset import HHDataset, load_data
from .model import HHSelector

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as mh

# ── loss ──────────────────────────────────────────────────────────────────

def ranking_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 class_weight: torch.Tensor | None = None,
                 temperature: float = 1.0,
                 mse_weight: float = 0.0) -> torch.Tensor:
    """CE on best-method argmax + optional regression on quality scores.

    Unavailable methods are masked out (logits set to -inf) for the CE term.
    The MSE term selects only available methods, so the -inf masking cannot
    leak into it.
    """
    multi = mask.sum(dim=1) >= 2
    if multi.sum() == 0:
        return pred.sum() * 0

    p = pred[multi]    # (B_m, 3) raw logits
    t = target[multi]  # (B_m, 3) quality scores
    m = mask[multi]    # (B_m, 3) bool

    # ── CE on the best available method ─────────────────────────────
    p_masked = p.masked_fill(~m, float("-inf"))
    true_best = t.argmax(dim=1)
    ce = F.cross_entropy(p_masked / temperature, true_best, weight=class_weight)

    # ── auxiliary regression: calibrate sigmoid score vs quality ────
    mse = torch.zeros((), device=p.device)
    if mse_weight > 0:
        p_avail = torch.sigmoid(p)[m]  # available methods only
        t_avail = t[m]
        mse = ((p_avail - t_avail) ** 2).mean()

    return ce + mse_weight * mse


# ── training ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, class_weight=None,
                    temperature=1.0, mse_weight=0.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for ak4, ak4_m, ak8, ak8_m, ev, tgt, msk, hg, res_idx, semi_idx, mrg_idx in loader:
        ak4 = ak4.to(device)
        ak4_m = ak4_m.to(device)
        ak8 = ak8.to(device)
        ak8_m = ak8_m.to(device)
        ev = ev.to(device)
        tgt = tgt.to(device)
        msk = msk.to(device)
        hg = hg.to(device)
        res_idx = res_idx.to(device)
        semi_idx = semi_idx.to(device)
        mrg_idx = mrg_idx.to(device)

        optimizer.zero_grad()
        pred = model(ak4, ak4_m, ak8, ak8_m, ev, res_idx, semi_idx, mrg_idx, hg)
        loss = ranking_loss(pred, tgt, msk, class_weight=class_weight,
                            temperature=temperature, mse_weight=mse_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device, temperature=1.0, mse_weight=0.0):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for ak4, ak4_m, ak8, ak8_m, ev, tgt, msk, hg, res_idx, semi_idx, mrg_idx in loader:
        ak4 = ak4.to(device)
        ak4_m = ak4_m.to(device)
        ak8 = ak8.to(device)
        ak8_m = ak8_m.to(device)
        ev = ev.to(device)
        tgt = tgt.to(device)
        msk = msk.to(device)
        hg = hg.to(device)
        res_idx = res_idx.to(device)
        semi_idx = semi_idx.to(device)
        mrg_idx = mrg_idx.to(device)

        pred = model(ak4, ak4_m, ak8, ak8_m, ev, res_idx, semi_idx, mrg_idx, hg)
        loss = ranking_loss(pred, tgt, msk, temperature=temperature, mse_weight=mse_weight)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


# ── plotter ──────────────────────────────────────────────────────────────────
class Plotter():
    def __init__(self, plot_dir):
        self.plot_dir = plot_dir
        self.train_loss_arr = []
        self.val_loss_arr = []
        mh.style.use('CMS')

    def plot(self, epoch):
        epochs = list(range(1, epoch + 1))
        fig, ax = plt.subplots()
        ax.plot(epochs, self.train_loss_arr, color='b', label="train_loss")
        ax.plot(epochs, self.val_loss_arr, color='r', label="val_loss")
        ax.legend()
        mh.cms.label(ax=ax, data=False, lumi=26.67, com=13.6)
        fig.savefig(self.plot_dir / "training_progress.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def update(self, epoch, train_loss, val_loss):
        self.train_loss_arr.append(train_loss)
        self.val_loss_arr.append(val_loss)
        self.plot(epoch)

# ── main ──────────────────────────────────────────────────────────────────

def main(cfg: CFG, max_events: int | None = None, device: str | None = None):
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.plot_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ─────────────────────────────────────────────────────────────
    ak4_pad, ak4_mask, ak8_pad, ak8_mask, event_np, targets, mask, higgses, resolved_idx, semi_idx, merged_idx = load_data(cfg, max_events=max_events)

    n_total = len(targets)
    n_val = int(n_total * cfg.val_frac)
    n_test = int(n_total * cfg.val_frac)
    n_train = n_total - n_val - n_test
    gen = torch.Generator().manual_seed(cfg.seed)

    full_ds = HHDataset(ak4_pad, ak4_mask, ak8_pad, ak8_mask,
                        event_np, targets, mask, higgses,
                        resolved_idx, semi_idx, merged_idx)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=gen)

    print(f"Split — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # ── class weights (inverse frequency) for balanced loss ────────────
    train_indices = train_ds.indices
    train_tgt = torch.tensor(targets[train_indices])
    train_msk = torch.tensor(mask[train_indices])
    train_avail = train_tgt.clone()
    train_avail[~train_msk] = -1.0
    true_best_all = train_avail.argmax(dim=1)
    counts = true_best_all.bincount().float()
    class_weight = (counts.sum() / (len(counts) * counts)).to(device)
    print(f"  class counts: {counts.tolist()}")
    print(f"  class weights: {class_weight.tolist()}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    # ── model ────────────────────────────────────────────────────────────
    model = HHSelector(
        ak4_feat_dim=cfg.ak4_feat_dim,
        ak8_feat_dim=cfg.ak8_feat_dim,
        event_feat_dim=cfg.event_feat_dim,
        embed_dim=cfg.embed_dim,
        mlp_hidden=cfg.mlp_hidden,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=cfg.max_epochs, pct_start=0.3)

    # ── training loop ────────────────────────────────────────────────────
    best_val = float("inf")
    wait = 0

    progress_plotter = Plotter(cfg.plot_dir)
    
    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     class_weight=class_weight,
                                     temperature=cfg.loss_temperature,
                                     mse_weight=cfg.mse_weight)
        val_loss = validate(model, val_loader, device,
                            temperature=cfg.loss_temperature,
                            mse_weight=cfg.mse_weight)
        scheduler.step()

        print(f"Epoch {epoch:3d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")
        
        progress_plotter.update(epoch,train_loss,val_loss)
        
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            ckpt = cfg.checkpoint_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, ckpt)
            print(f"  → saved checkpoint ({ckpt})")
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── final checkpoint (always save) ───────────────────────────────────
    final_ckpt = cfg.checkpoint_dir / "last.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
    }, final_ckpt)
    print(f"Training complete. Best val loss: {best_val:.6f}")

    return model, test_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HH topology selector")
    parser.add_argument("--max-events", type=int, default=None, help="Limit number of events")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = CFG()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.epochs is not None:
        cfg.max_epochs = args.epochs

    main(cfg, max_events=args.max_events, device=args.device)
