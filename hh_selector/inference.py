"""
Inference utilities for HHSelector.

Apply a trained model to raw awkward arrays without using generator-level
or truth-level information.  Filters events by topology feasibility and
algorithm success, then returns model predictions.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import awkward as ak
import torch

from .config import CFG
from .model import HHSelector
from .dataset import _topology_possible, _algo_succeeded, compute_higgses


__all__ = ["InferenceResult", "apply_model", "load_model"]


# ── result container ─────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Container for model inference results.

    Attributes
    ----------
    logits : np.ndarray  — (N, 3) raw model output
    scores : np.ndarray  — (N, 3) sigmoid-transformed scores in [0, 1]
    method_mask : np.ndarray — (N, 3) bool, True where topology_possible & algo_succeeded
    event_indices : np.ndarray — (N,) int, indices into the original awkward array
    """
    logits: np.ndarray
    scores: np.ndarray
    method_mask: np.ndarray
    event_indices: np.ndarray


# ── helpers ──────────────────────────────────────────────────────────────

def _required_branches(cfg: CFG) -> list[str]:
    """List of ROOT branches required for inference (no truth / gen-level)."""
    brs: list[str] = []
    for f in cfg.ak4_features:
        brs.append(f"ak4_{f}")
    for f in cfg.ak8_features:
        brs.append(f"ak8_{f}")
    brs += list(cfg.event_features)
    brs += list(cfg.resolved_reco_branches)
    brs += list(cfg.semiresolved_reco_branches)
    brs += list(cfg.merged_reco_branches)
    return brs


def _validate_branches(arrays: ak.Array, cfg: CFG) -> list[str]:
    """Return sorted list of missing branches (empty if all present)."""
    return sorted(set(_required_branches(cfg)) - set(arrays.fields))


def _prepare_arrays(arrays: ak.Array, cfg: CFG):
    """Convert a raw awkward array into padded numpy arrays for the model.

    Returns None when no events have at least one valid method.

    Returns
    -------
    ak4_pad, ak4_mask, ak8_pad, ak8_mask : np.ndarray — padded jet features
    event_np : np.ndarray — (N, 13) event-level features
    method_mask : np.ndarray — (N, 3) bool
    higgses : np.ndarray — (N, 3, 9) reconstructed Higgs 4-vectors + dR per method
    event_indices : np.ndarray — (N,) int — mapping back to original array
    resolved_idx, semi_idx, merged_idx : np.ndarray — index assignments
    """
    METHODS = ["resolved", "semiresolved", "merged"]

    # ── cache topology + algo results (reused 3× below) ──────────
    topo_cache = {}
    algo_cache = {}
    for method in METHODS:
        topo_cache[method] = _topology_possible(arrays, method, cfg)
        algo_cache[method] = _algo_succeeded(arrays, method, cfg)

    # ── method availability (reco-level only, NO truth) ──────────
    method_mask = np.zeros((len(arrays), 3), dtype=bool)
    for j, method in enumerate(METHODS):
        method_mask[:, j] = topo_cache[method] & algo_cache[method]

    event_valid = method_mask.any(axis=1)
    event_indices = np.where(event_valid)[0]
    if event_indices.size == 0:
        return None

    arrays_v = arrays[event_valid]
    method_mask = method_mask[event_valid]

    # Slice cached results to match filtered events
    topo_v = {m: arr[event_valid] for m, arr in topo_cache.items()}
    algo_v = {m: arr[event_valid] for m, arr in algo_cache.items()}

    # ── per-method reconstructed Higgs features (4-vectors + dR) ─
    higgses = compute_higgses(arrays_v, cfg, topo=topo_v, algo=algo_v)

    # ── jet feature arrays ───────────────────────────────────────
    ak4_feats = ak.concatenate(
        [arrays_v[f"ak4_{f}"][..., np.newaxis] for f in cfg.ak4_features],
        axis=-1,
    )
    ak8_feats = ak.concatenate(
        [arrays_v[f"ak8_{f}"][..., np.newaxis] for f in cfg.ak8_features],
        axis=-1,
    )

    # ── event-level features (base scalars + topo flags + algo flags) ─
    event_cols = [
        ak.to_numpy(arrays_v[f]).astype(np.float32) for f in cfg.event_features
    ]
    event_np = np.stack(event_cols, axis=-1)

    avail_flags = np.stack(
        [topo_v[m].astype(np.float32) for m in METHODS], axis=-1,
    )
    algo_flags = np.stack(
        [algo_v[m].astype(np.float32) for m in METHODS], axis=-1,
    )
    event_np = np.concatenate([event_np, avail_flags, algo_flags], axis=-1)

    # ── algorithm index assignments ──────────────────────────────
    resolved_idx = np.stack(
        [ak.to_numpy(arrays_v[b]) for b in cfg.resolved_reco_branches], axis=-1
    ).astype(np.int64)

    semi_idx = np.stack(
        [ak.to_numpy(arrays_v[b]) for b in cfg.semiresolved_reco_branches], axis=-1
    ).astype(np.int64)

    merged_idx = np.stack(
        [ak.to_numpy(arrays_v[b]) for b in cfg.merged_reco_branches], axis=-1
    ).astype(np.int64)

    # ── pad jet arrays to global max length (vectorized) ─────────
    N = len(method_mask)
    D4, D8 = len(cfg.ak4_features), len(cfg.ak8_features)
    max_ak4 = max(len(a) for a in ak4_feats)
    max_ak8 = max(len(a) for a in ak8_feats)

    ak4_masked = ak.to_numpy(
        ak.pad_none(ak4_feats, max_ak4, axis=1), allow_missing=True,
    )
    ak4_pad = np.where(ak4_masked.mask, 0, ak4_masked.data).astype(np.float32)
    ak4_mask = ak4_masked.mask.any(axis=-1)

    ak8_masked = ak.to_numpy(
        ak.pad_none(ak8_feats, max_ak8, axis=1), allow_missing=True,
    )
    ak8_pad = np.where(ak8_masked.mask, 0, ak8_masked.data).astype(np.float32)
    ak8_mask = ak8_masked.mask.any(axis=-1)

    return (ak4_pad, ak4_mask, ak8_pad, ak8_mask, event_np,
            method_mask, higgses, event_indices, resolved_idx, semi_idx, merged_idx)


# ── model loading ────────────────────────────────────────────────────────

def load_model(checkpoint_path: str | Path, cfg: CFG,
               device: str = "cpu") -> HHSelector:
    """Load a trained HHSelector from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : path to a ``.pt`` checkpoint saved during training.
    cfg : the same :class:`CFG` used for training (determines architecture).
    device : torch device string.
    """
    model = HHSelector(
        ak4_feat_dim=cfg.ak4_feat_dim,
        ak8_feat_dim=cfg.ak8_feat_dim,
        event_feat_dim=cfg.event_feat_dim,
        embed_dim=cfg.embed_dim,
        mlp_hidden=cfg.mlp_hidden,
        dropout=cfg.dropout,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model


# ── main entry point ─────────────────────────────────────────────────────

@torch.inference_mode()
def apply_model(
    arrays: ak.Array,
    checkpoint_path: str | Path,
    cfg: CFG | None = None,
    *,
    min_methods: int = 1,
    device: str | None = None,
    batch_size: int = 4096,
    max_events: int | None = None,
    compile: bool = True,
) -> InferenceResult | None:
    """Apply a trained HHSelector to a raw awkward array.

    Only **topology feasibility** and **algorithm success** are used to
    decide which methods are available — no generator-level or truth-level
    information is accessed.

    Parameters
    ----------
    arrays : ak.Array
        Raw awkward array from ``uproot`` (must contain the required
        branches listed in :func:`_required_branches`).
    checkpoint_path : str | Path
        Path to a ``.pt`` checkpoint file.
    cfg : CFG, optional
        Configuration (defaults to :class:`CFG` if *None*).
    min_methods : int
        Only keep events where at least this many methods are valid
        (default ``1``).
    device : str, optional
        Torch device; auto-detected when *None*.
    batch_size : int
        Inference batch size (default ``256``).
    max_events : int, optional
        If given, only process this many events from the front of the
        array.
    compile : bool
        Whether to JIT-compile the model via ``torch.compile``
        (default ``True``, requires PyTorch ≥ 2.0).

    Returns
    -------
    InferenceResult or None
        Predictions for every event that passed the filter, or *None*
        when no events qualify.
    """
    if cfg is None:
        cfg = CFG()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── validate branches ────────────────────────────────────────────
    missing = _validate_branches(arrays, cfg)
    if missing:
        raise ValueError(f"Missing required branches: {missing}")

    if max_events is not None:
        arrays = arrays[:max_events]

    # ── prepare padded arrays ────────────────────────────────────────
    result = _prepare_arrays(arrays, cfg)
    if result is None:
        print("No events with valid methods found.")
        return None

    (ak4_pad, ak4_mask, ak8_pad, ak8_mask, event_np,
     method_mask, higgses, event_indices,
     resolved_idx, semi_idx, merged_idx) = result

    # ── filter by min_methods ────────────────────────────────────────
    keep = method_mask.sum(axis=1) >= min_methods
    if not keep.any():
        print(f"No events with >= {min_methods} valid method(s).")
        return None

    ak4_pad    = ak4_pad[keep]
    ak4_mask   = ak4_mask[keep]
    ak8_pad    = ak8_pad[keep]
    ak8_mask   = ak8_mask[keep]
    event_np   = event_np[keep]
    method_mask = method_mask[keep]
    higgses    = higgses[keep]
    event_indices = event_indices[keep]
    resolved_idx  = resolved_idx[keep]
    semi_idx      = semi_idx[keep]
    merged_idx    = merged_idx[keep]

    N = len(event_indices)
    n_methods = int(method_mask.sum())
    print(f"Inference: {N} events, {n_methods} valid method predictions  "
          f"(device={device})")

    # ── load model ───────────────────────────────────────────────────
    model = load_model(checkpoint_path, cfg, device)
    if compile:
        model = torch.compile(model, mode="reduce-overhead")

    # ── batched forward pass ─────────────────────────────────────────
    all_logits = np.empty((N, 3), dtype=np.float32)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda")
        if (isinstance(device, str) and "cuda" in device)
        else nullcontext()
    )

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        with autocast_ctx:
            logits = model(
                torch.tensor(ak4_pad[s:e],   device=device),
                torch.tensor(ak4_mask[s:e],  device=device),
                torch.tensor(ak8_pad[s:e],   device=device),
                torch.tensor(ak8_mask[s:e],  device=device),
                torch.tensor(event_np[s:e],  device=device),
                torch.tensor(resolved_idx[s:e], device=device),
                torch.tensor(semi_idx[s:e],     device=device),
                torch.tensor(merged_idx[s:e],   device=device),
                torch.tensor(higgses[s:e],      device=device),
            )
        all_logits[s:e] = logits.cpu().numpy()

    scores = 1.0 / (1.0 + np.exp(-all_logits))

    return InferenceResult(
        logits=all_logits,
        scores=scores,
        method_mask=method_mask,
        event_indices=event_indices,
    )
