from __future__ import annotations

import numpy as np
import uproot
import awkward as ak
import vector
from torch.utils.data import Dataset
from .config import CFG

vector.register_awkward()

# ── ROOT branch list ─────────────────────────────────────────────────────

def _build_branches(cfg: CFG) -> list[str]:
    """All branches we need from the ROOT file."""
    brs = []
    # jet kinematics
    for f in cfg.ak4_features:
        brs.append(f"ak4_{f}")
    for f in cfg.ak8_features:
        brs.append(f"ak8_{f}")
    # jet counts
    brs += ["n_ak4", "n_ak8"]
    # event-level scalars
    brs += list(cfg.event_features)
    # algorithm assignment indices
    brs += list(cfg.resolved_reco_branches)
    brs += list(cfg.semiresolved_reco_branches)
    brs += list(cfg.merged_reco_branches)
    # truth-matching indices
    brs += list(cfg.truth_ak4_branches)
    brs += list(cfg.truth_ak8_branches)
    # gen-level Higgs masses (used as reference)
    brs += [
        "genp_isHHto4b",
        "genp_H1_FC_m", "genp_H2_FC_m",
        "genp_H1_FC_pt","genp_H2_FC_pt",
        "genp_H1_FC_eta", "genp_H2_FC_eta",
        "genp_H1_FC_phi", "genp_H2_FC_phi"
    ]
    return brs


# ── 4-vector helpers ─────────────────────────────────────────────────────

def _build_4vec(pt, eta, phi, mass):
    """Zip kinematic branches into a vector-aware awkward array."""
    return vector.awk(ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass}))


def _reco_h1h2_for_method(
    ak4_4vec, ak8_4vec, event_idx_ak4, event_idx_ak8, arrays, method: str, cfg: CFG,
):
    """Return (H1, H2) arrays for one reconstruction method.

    Only valid for events where the algorithm succeeded (caller must mask).
    """
    if method == "resolved":
        h1b1 = ak4_4vec[event_idx_ak4, arrays.ak4_h1b1_index]
        h1b2 = ak4_4vec[event_idx_ak4, arrays.ak4_h1b2_index]
        h2b1 = ak4_4vec[event_idx_ak4, arrays.ak4_h2b1_index]
        h2b2 = ak4_4vec[event_idx_ak4, arrays.ak4_h2b2_index]
        return (h1b1 + h1b2), (h2b1 + h2b2)

    if method == "semiresolved":
        h1 = ak8_4vec[event_idx_ak8, arrays.semiHH_fatjet_index]
        h2b1 = ak4_4vec[event_idx_ak4, arrays.semiHH_resb1_index]
        h2b2 = ak4_4vec[event_idx_ak4, arrays.semiHH_resb2_index]
        return h1, (h2b1 + h2b2)

    if method == "merged":
        h1 = ak8_4vec[event_idx_ak8, arrays.mergedHH_H1_index]
        h2 = ak8_4vec[event_idx_ak8, arrays.mergedHH_H2_index]
        return h1, h2

    raise ValueError(f"Unknown method: {method}")


# ── availability masks ────────────────────────────────────────────────────

def _topology_possible(arrays, method: str, cfg: CFG) -> np.ndarray:
    """Layer 1: does the event have enough jets for this topology?"""
    n_ak4 = ak.to_numpy(arrays.n_ak4)
    n_ak8 = ak.to_numpy(arrays.n_ak8)
    if method == "resolved":
        return n_ak4 >= cfg.resolved_min_ak4
    if method == "semiresolved":
        return (n_ak8 >= cfg.semiresolved_min_ak8) & (n_ak4 >= cfg.semiresolved_min_ak4)
    if method == "merged":
        return n_ak8 >= cfg.merged_min_ak8
    raise ValueError(method)


def _algo_succeeded(arrays, method: str, cfg: CFG) -> np.ndarray:
    """Layer 2: did the pairing algorithm produce a valid assignment?"""
    sentinel = cfg.reco_sentinel
    if method == "resolved":
        ok = np.ones(len(arrays), dtype=bool)
        for br in cfg.resolved_reco_branches:
            ok &= ak.to_numpy(arrays[br]) != sentinel
        # all four indices must be distinct
        i0 = ak.to_numpy(arrays.ak4_h1b1_index)
        i1 = ak.to_numpy(arrays.ak4_h1b2_index)
        i2 = ak.to_numpy(arrays.ak4_h2b1_index)
        i3 = ak.to_numpy(arrays.ak4_h2b2_index)
        ok &= (i0 != i1) & (i0 != i2) & (i0 != i3)
        ok &= (i1 != i2) & (i1 != i3)
        ok &= (i2 != i3)
        return ok
    if method == "semiresolved":
        ok = np.ones(len(arrays), dtype=bool)
        for br in cfg.semiresolved_reco_branches:
            ok &= ak.to_numpy(arrays[br]) != sentinel
        ok &= (
            ak.to_numpy(arrays.semiHH_resb1_index)
            != ak.to_numpy(arrays.semiHH_resb2_index)
        )
        ok[ok] = ok[ok] & (ak.to_numpy(arrays[ok].ak8_msoftdrop[ak.Array(np.arange(len(arrays[ok]))),arrays[ok].semiHH_fatjet_index]) > 50)
        return ok
    if method == "merged":
        ok = np.ones(len(arrays), dtype=bool)
        for br in cfg.merged_reco_branches:
            ok &= ak.to_numpy(arrays[br]) != sentinel
            ok[ok] = ok[ok] & (ak.to_numpy(arrays[ok].ak8_msoftdrop[ak.Array(np.arange(len(arrays[ok]))),arrays[ok][br]]) > 0)
        ok &= (
            ak.to_numpy(arrays.mergedHH_H1_index)
            != ak.to_numpy(arrays.mergedHH_H2_index)
        )
        
        return ok
    raise ValueError(method)

def _algo_truth_correct(arrays, method: str, cfg: CFG) -> np.ndarray:
    """Layer 2: did the pairing algorithm produce a valid assignment?"""
    sentinel = cfg.truth_sentinel
    if method == "resolved":
        ok = np.ones(len(arrays), dtype=bool)
        for br in cfg.resolved_truth_branches:
            ok &= ak.to_numpy(arrays[br]) != sentinel
        # all four indices must be distinct
        i0 = ak.to_numpy(arrays.truerecoj_H1b1_index)
        i1 = ak.to_numpy(arrays.truerecoj_H1b2_index)
        i2 = ak.to_numpy(arrays.truerecoj_H2b1_index)
        i3 = ak.to_numpy(arrays.truerecoj_H2b2_index)
        ok &= (i0 != i1) & (i0 != i2) & (i0 != i3)
        ok &= (i1 != i2) & (i1 != i3)
        ok &= (i2 != i3)
        return ok
    if method == "semiresolved":
        ok = np.ones(len(arrays), dtype=bool)
        for br in cfg.semiresolved_truth_branches:
            ok &= ak.to_numpy(arrays[br]) != sentinel
        ok &= (
            ak.to_numpy(arrays.truerecoj_H2b1_index)
            != ak.to_numpy(arrays.truerecoj_H2b2_index)
        )
        return ok
    if method == "merged":
        ok = np.ones(len(arrays), dtype=bool)
        for br in cfg.merged_truth_branches:
            ok &= ak.to_numpy(arrays[br]) != sentinel
        ok &= (
            ak.to_numpy(arrays.truerecofj_H1b1_index)
            != ak.to_numpy(arrays.truerecofj_H2b1_index)
        )
        
        return ok
    raise ValueError(method)

# ── reco accuracy ─────────────────────────────────────────────────

def _reco_accuracy_for_method(arrays, method: str, cfg: CFG):
    """Return (mH1, mH2) arrays for one reconstruction method.

    Only valid for events where the algorithm succeeded (caller must mask).
    """
    if method == "resolved":
        correct = np.array([
            arrays[recoidx].to_numpy() == arrays[truthidx].to_numpy() for recoidx, truthidx in zip(cfg.resolved_reco_branches ,cfg.resolved_truth_branches)
        ])
        return correct.sum(axis=0) / correct.shape[0]

    if method == "semiresolved":
        correct_frac = np.array([
            .5 * (arrays.semiHH_fatjet_index == arrays.truerecofj_H1b1_index).to_numpy().astype(float),
            .25 * (arrays.semiHH_resb1_index == arrays.truerecoj_H2b1_index).to_numpy().astype(float),
            .25 * (arrays.semiHH_resb2_index == arrays.truerecoj_H2b2_index).to_numpy().astype(float)
        ])
        return correct_frac.sum(axis=0)

    if method == "merged":
        correct = np.array([
            arrays[recoidx].to_numpy() == arrays[truthidx].to_numpy() for recoidx, truthidx in zip(cfg.merged_reco_branches ,cfg.merged_truth_branches)
        ])
        return correct.sum(axis=0) / correct.shape[0]

# ── reconstructed Higgs features ───────────────────────────────────────────

def compute_higgses(arrays, cfg: CFG,
                    topo: dict[str, np.ndarray] | None = None,
                    algo: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Compute reconstructed (m, pt, eta, phi) of H1 and H2 + delta R(H1, H2) for each method.

    Parameters
    ----------
    topo, algo : dict[str, np.ndarray] or None
        Pre-computed topology_possible / algo_succeeded per method.
        Avoids redundant computation when called from inference.

    Returns (N, 3, 9) array with last dim = [mH1, ptH1, etaH1, phiH1,
                                                mH2, ptH2, etaH2, phiH2, dR],
    all -1 where the method is not available.
    """
    ak4_4vec = _build_4vec(arrays.ak4_pt, arrays.ak4_eta, arrays.ak4_phi, arrays.ak4_mass)
    ak8_4vec = _build_4vec(arrays.ak8_pt, arrays.ak8_eta, arrays.ak8_phi, arrays.ak8_msoftdrop)

    methods = ["resolved", "semiresolved", "merged"]
    N = len(arrays)
    higgses = np.full((N, 3, 9), -1.0, dtype=np.float32)

    _cached = topo is not None and algo is not None
    for j, method in enumerate(methods):
        if _cached:
            t = topo[method]
            a = algo[method]
        else:
            t = _topology_possible(arrays, method, cfg)
            a = _algo_succeeded(arrays, method, cfg)
        avail = t & a
        if not avail.any():
            continue

        sub = arrays[avail]
        event_idx_ak4 = ak.Array(np.arange(len(sub)))
        event_idx_ak8 = ak.Array(np.arange(len(sub)))
        sub_ak4 = ak4_4vec[avail]
        sub_ak8 = ak8_4vec[avail]
        reco_H1, reco_H2 = _reco_h1h2_for_method(
            sub_ak4, sub_ak8, event_idx_ak4, event_idx_ak8, sub, method, cfg,
        )
        higgses[avail, j, 0] = ak.to_numpy(reco_H1.mass).astype(np.float32)
        higgses[avail, j, 1] = ak.to_numpy(reco_H1.pt).astype(np.float32)
        higgses[avail, j, 2] = ak.to_numpy(reco_H1.eta).astype(np.float32)
        higgses[avail, j, 3] = ak.to_numpy(reco_H1.phi).astype(np.float32)
        higgses[avail, j, 4] = ak.to_numpy(reco_H2.mass).astype(np.float32)
        higgses[avail, j, 5] = ak.to_numpy(reco_H2.pt).astype(np.float32)
        higgses[avail, j, 6] = ak.to_numpy(reco_H2.eta).astype(np.float32)
        higgses[avail, j, 7] = ak.to_numpy(reco_H2.phi).astype(np.float32)
        dr = ak.to_numpy(reco_H1.deltaR(reco_H2)).astype(np.float32)
        higgses[avail, j, 8] = dr
    return higgses


# ── target quality scores ─────────────────────────────────────────────────

def _compute_targets(
    ak4_4vec, ak8_4vec, arrays, cfg: CFG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (targets, errors, mask, higgses) arrays.

    targets, errors : (N, 3)
    mask : (N, 3) bool
    higgses : (N, 3, 9)  [mH1, ptH1, etaH1, phiH1, mH2, ptH2, etaH2, phiH2, dR]
    """
    methods = ["resolved", "semiresolved", "merged"]
    N = len(arrays)

    targets = np.full((N, 3), -1.0, dtype=np.float32)
    errors = np.full((N, 3), -1.0, dtype=np.float32)
    mask = np.zeros((N, 3), dtype=bool)
    higgses = np.full((N, 3, 9), -1.0, dtype=np.float32)

    # gen-level Higgs vectors (mass always 125 for this signal)
    gen_H1 = _build_4vec(
        arrays.genp_H1_FC_pt, 
        arrays.genp_H1_FC_eta, 
        arrays.genp_H1_FC_phi, 
        arrays.genp_H1_FC_m
    )
    
    gen_H2 = _build_4vec(
        arrays.genp_H2_FC_pt, 
        arrays.genp_H2_FC_eta, 
        arrays.genp_H2_FC_phi, 
        arrays.genp_H2_FC_m
    )

    for j, method in enumerate(methods):
        topo = _topology_possible(arrays, method, cfg)
        algo = _algo_succeeded(arrays, method, cfg)
        truth = _algo_truth_correct(arrays, method, cfg)
        avail = topo & algo & truth

        if not avail.any():
            continue

        # subset arrays to available events only (avoids sentinel indexing)
        sub = arrays[avail]
        event_idx_ak4 = ak.Array(np.arange(len(sub)))
        event_idx_ak8 = ak.Array(np.arange(len(sub)))

        sub_ak4_4vec = ak4_4vec[avail]
        sub_ak8_4vec = ak8_4vec[avail]

        reco_H1, reco_H2 = _reco_h1h2_for_method(
            sub_ak4_4vec, sub_ak8_4vec, event_idx_ak4, event_idx_ak8, sub, method, cfg,
        )
        reco_mH1= ak.to_numpy(reco_H1.mass).astype(np.float64)
        reco_mH2 = ak.to_numpy(reco_H2.mass).astype(np.float64)
        reco_ptH1 = ak.to_numpy(reco_H1.pt).astype(np.float64)
        reco_ptH2 = ak.to_numpy(reco_H2.pt).astype(np.float64)
        
        sub_gen_mH1 = ak.to_numpy(gen_H1.mass).astype(np.float64)[avail]
        sub_gen_mH2 = ak.to_numpy(gen_H2.mass).astype(np.float64)[avail]
        sub_gen_ptH1 = ak.to_numpy(gen_H1.pt).astype(np.float64)[avail]
        sub_gen_ptH2 = ak.to_numpy(gen_H2.pt).astype(np.float64)[avail]

        deltaRH1 = ak.to_numpy(gen_H1[avail].deltaR(reco_H1)).astype(np.float64)
        deltaRH2 = ak.to_numpy(gen_H2[avail].deltaR(reco_H2)).astype(np.float64)
        
        reco_accuracy = _reco_accuracy_for_method(arrays[avail], method, cfg)
        
        error =  np.sqrt(np.sum(np.array([
            np.square(cfg.mH1_best_weight * ((reco_mH1 - sub_gen_mH1) / np.sqrt(np.square(reco_mH1) + np.square(sub_gen_mH1)))) +
            np.square(cfg.mH2_best_weight * ((reco_mH2 - sub_gen_mH2) / np.sqrt(np.square(reco_mH2) + np.square(sub_gen_mH2)))) +
            np.square(cfg.ptH1_best_weight * ((reco_ptH1 - sub_gen_ptH1) / np.sqrt(np.square(reco_ptH1) + np.square(sub_gen_ptH1)))) +
            np.square(cfg.ptH2_best_weight * ((reco_ptH2 - sub_gen_ptH2) / np.sqrt(np.square(reco_ptH2) + np.square(sub_gen_ptH2)))) +
            np.square(cfg.deltaRH1_best_weight * (deltaRH1 / np.pi)) +
            np.square(cfg.deltaRH2_best_weight * (deltaRH2 / np.pi)) +
            cfg.reco_accuracy_best_weight * (1.0 - reco_accuracy)
        ]), axis = 0))
        quality = np.exp(-error)

        targets[avail, j] = quality.astype(np.float32)
        errors[avail, j] = error.astype(np.float32)
        mask[avail, j] = True

        # reconstructed (mass, pt, eta, phi) of H1 and H2 + delta R(H1, H2)
        dr = ak.to_numpy(reco_H1.deltaR(reco_H2)).astype(np.float32)
        higgses[avail, j, 0] = ak.to_numpy(reco_H1.mass).astype(np.float32)
        higgses[avail, j, 1] = ak.to_numpy(reco_H1.pt).astype(np.float32)
        higgses[avail, j, 2] = ak.to_numpy(reco_H1.eta).astype(np.float32)
        higgses[avail, j, 3] = ak.to_numpy(reco_H1.phi).astype(np.float32)
        higgses[avail, j, 4] = ak.to_numpy(reco_H2.mass).astype(np.float32)
        higgses[avail, j, 5] = ak.to_numpy(reco_H2.pt).astype(np.float32)
        higgses[avail, j, 6] = ak.to_numpy(reco_H2.eta).astype(np.float32)
        higgses[avail, j, 7] = ak.to_numpy(reco_H2.phi).astype(np.float32)
        higgses[avail, j, 8] = dr
    return targets, errors, mask, higgses


# ── data loading ──────────────────────────────────────────────────────────

def load_data(cfg: CFG, max_events: int | None = None):
    """Load ROOT file and return preprocessed arrays ready for the Dataset.

    Returns
    -------
    ak4_pad  : np.ndarray  — (N, max_ak4, feat_dim)  zero-padded AK4 features
    ak4_mask : np.ndarray  — (N, max_ak4) bool  True = padded position
    ak8_pad  : np.ndarray  — (N, max_ak8, feat_dim)  zero-padded AK8 features
    ak8_mask : np.ndarray  — (N, max_ak8) bool  True = padded position
    event_np : np.ndarray  — (N, event_feat_dim)  [scalars + availability flags]
    targets  : np.ndarray  — (N, 3)
    mask     : np.ndarray  — (N, 3) bool
    higgses  : np.ndarray  — (N, 3, 9)  [mH1, ptH1, etaH1, phiH1, mH2, ptH2, etaH2, phiH2, dR] per method
    resolved_idx : np.ndarray  — (N, 4) int64  [h1b1, h1b2, h2b1, h2b2] into ak4
    semi_idx     : np.ndarray  — (N, 3) int64  [fatjet(ak8), resb1(ak4), resb2(ak4)]
    merged_idx   : np.ndarray  — (N, 2) int64  [H1(ak8), H2(ak8)]
    """
    fpath = cfg.data_dir / cfg.signal_file
    branches = _build_branches(cfg)

    print(f"Loading {fpath} ...")
    with uproot.open(fpath) as f:
        arrays = f[cfg.tree_name].arrays(branches, library="ak")
    arrays = arrays[arrays.genp_isHHto4b == True]

    if max_events is not None:
        arrays = arrays[:max_events]
    N = len(arrays)
    print(f"  {N} events loaded")

    # ── build 4-vectors ──────────────────────────────────────────────────
    ak4_4vec = _build_4vec(arrays.ak4_pt, arrays.ak4_eta, arrays.ak4_phi, arrays.ak4_mass)
    ak8_4vec = _build_4vec(arrays.ak8_pt, arrays.ak8_eta, arrays.ak8_phi, arrays.ak8_msoftdrop)

    # ── per-event jet features (vectorized via awkward) ──────────────────
    ak4_jags = [arrays[f"ak4_{f}"] for f in cfg.ak4_features]  # list of jagged arrays
    ak8_jags = [arrays[f"ak8_{f}"] for f in cfg.ak8_features]

    ak4_feats = ak.concatenate([f[..., np.newaxis] for f in ak4_jags], axis=-1)  # (N, var, feat_dim)
    ak8_feats = ak.concatenate([f[..., np.newaxis] for f in ak8_jags], axis=-1)

    # ── event-level features ─────────────────────────────────────────────
    event_cols = []
    for feat in cfg.event_features:
        event_cols.append(ak.to_numpy(arrays[feat]).astype(np.float32))
    event_np = np.stack(event_cols, axis=-1)  # (N, n_event_feats)

    # ── availability flags (topology_possible + algo_succeeded) ──────────
    avail_flags = np.stack(
        [_topology_possible(arrays, m, cfg).astype(np.float32) for m in
         ["resolved", "semiresolved", "merged"]],
        axis=-1,
    )  # (N, 3)
    algo_flags = np.stack(
        [_algo_succeeded(arrays, m, cfg).astype(np.float32) for m in
         ["resolved", "semiresolved", "merged"]],
        axis=-1,
    )  # (N, 3)
    event_np = np.concatenate([event_np, avail_flags, algo_flags], axis=-1)  # (N, 7+3+3)

    # ── method index assignments ──────────────────────────────────────────
    resolved_idx = np.stack([
        ak.to_numpy(arrays["ak4_h1b1_index"]),
        ak.to_numpy(arrays["ak4_h1b2_index"]),
        ak.to_numpy(arrays["ak4_h2b1_index"]),
        ak.to_numpy(arrays["ak4_h2b2_index"]),
    ], axis=-1).astype(np.int64)  # (N, 4) — indices into ak4 jets

    semi_idx = np.stack([
        ak.to_numpy(arrays["semiHH_fatjet_index"]),   # ak8
        ak.to_numpy(arrays["semiHH_resb1_index"]),     # ak4
        ak.to_numpy(arrays["semiHH_resb2_index"]),     # ak4
    ], axis=-1).astype(np.int64)  # (N, 3) — [ak8_idx, ak4_idx, ak4_idx]

    merged_idx = np.stack([
        ak.to_numpy(arrays["mergedHH_H1_index"]),      # ak8
        ak.to_numpy(arrays["mergedHH_H2_index"]),      # ak8
    ], axis=-1).astype(np.int64)  # (N, 2) — indices into ak8 jets

    # ── targets ──────────────────────────────────────────────────────────
    targets, errors, mask, higgses = _compute_targets(ak4_4vec, ak8_4vec, arrays, cfg)
    print(f"  events with ≥1 valid method (topo+algo+truth): {(mask.any(axis=1)).sum()}")
    print(f"  valid methods — resolved: {mask[:,0].sum()}, semi: {mask[:,1].sum()}, merged: {mask[:,2].sum()}")

    # keep only events where at least 2 methods are available
    keep = mask.sum(axis=1) >= 2
    ak4_kept = ak4_feats[keep]
    ak8_kept = ak8_feats[keep]
    event_np = event_np[keep]
    targets = targets[keep]
    errors = errors[keep]
    mask = mask[keep]
    higgses = higgses[keep]
    resolved_idx = resolved_idx[keep]
    semi_idx = semi_idx[keep]
    merged_idx = merged_idx[keep]

    # pre-pad jet arrays to global max length
    N = len(targets)
    D4 = len(cfg.ak4_features)
    D8 = len(cfg.ak8_features)
    max_ak4 = max(len(a) for a in ak4_kept)
    max_ak8 = max(len(a) for a in ak8_kept)

    ak4_pad = np.zeros((N, max_ak4, D4), dtype=np.float32)
    ak8_pad = np.zeros((N, max_ak8, D8), dtype=np.float32)
    ak4_mask = np.ones((N, max_ak4), dtype=bool)
    ak8_mask = np.ones((N, max_ak8), dtype=bool)
    for i in range(N):
        n4 = len(ak4_kept[i])
        ak4_pad[i, :n4] = np.asarray(ak4_kept[i], dtype=np.float32)
        ak4_mask[i, :n4] = False
        n8 = len(ak8_kept[i])
        ak8_pad[i, :n8] = np.asarray(ak8_kept[i], dtype=np.float32)
        ak8_mask[i, :n8] = False

    print(f"  after filtering: {N} events with ≥2 valid methods")
    print(f"  jet padding: max_ak4={max_ak4}, max_ak8={max_ak8}")
    print(f"  valid methods (events with ≥2 methods) — resolved: {mask[:,0].sum()}, semi: {mask[:,1].sum()}, merged: {mask[:,2].sum()}")
    return ak4_pad, ak4_mask, ak8_pad, ak8_mask, event_np, targets, mask, higgses, resolved_idx, semi_idx, merged_idx


# ── PyTorch Dataset ───────────────────────────────────────────────────────

class HHDataset(Dataset):
    """PyTorch Dataset for HH→4b topology selector."""

    def __init__(self, ak4_pad, ak4_mask, ak8_pad, ak8_mask, event_np, targets, mask, higgses,
                 resolved_idx, semi_idx, merged_idx):
        self.ak4_pad = ak4_pad
        self.ak4_mask = ak4_mask
        self.ak8_pad = ak8_pad
        self.ak8_mask = ak8_mask
        self.event_np = event_np
        self.targets = targets
        self.mask = mask
        self.higgses = higgses
        self.resolved_idx = resolved_idx
        self.semi_idx = semi_idx
        self.merged_idx = merged_idx

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.ak4_pad[idx],
            self.ak4_mask[idx],
            self.ak8_pad[idx],
            self.ak8_mask[idx],
            self.event_np[idx],
            self.targets[idx],
            self.mask[idx],
            self.higgses[idx],
            self.resolved_idx[idx],
            self.semi_idx[idx],
            self.merged_idx[idx],
        )
