from dataclasses import dataclass
from pathlib import Path


@dataclass
class CFG:
    # ── paths ────────────────────────────────────────────────────────────
    data_dir: Path = Path("/home/kirchee/ATLAS/more_signal")
    signal_file: str = "ggHH4b-kl1-kt1-c20_tree.root"
    tree_name: str = "Events"
    checkpoint_dir: Path = Path("/home/kirchee/ATLAS/hh_selector/checkpoints")
    plot_dir: Path = Path("/home/kirchee/ATLAS/hh_selector/plots")

    # ── jet features ─────────────────────────────────────────────────────
    ak4_features: tuple = ("pt", "eta", "phi", "mass", "bdisc", "btag_M")
    ak8_features: tuple = ("pt", "eta", "phi", "mass", "msoftdrop", "tau32", "XbbvsQCD")

    # ── event-level features ─────────────────────────────────────────────
    event_features: tuple = (
        "n_ak4",
        "n_ak8",
        "alljets_ht",
        "avgbdisc_twoldgbdiscjets",
        "minDR_b",
        "mass_minDR_b",
        "maxMass_b",
    )

    # ── topology definitions ─────────────────────────────────────────────
    # each topology knows: (min_ak4, min_ak8, reco_index_branches)
    resolved_min_ak4: int = 4
    resolved_min_ak8: int = 0
    semiresolved_min_ak4: int = 2
    semiresolved_min_ak8: int = 1
    merged_min_ak4: int = 0
    merged_min_ak8: int = 2

    # branch names for algorithm assignment indices (sentinel = -1000)
    resolved_reco_branches: tuple = (
        "ak4_h1b1_index",
        "ak4_h1b2_index",
        "ak4_h2b1_index",
        "ak4_h2b2_index",
    )
    semiresolved_reco_branches: tuple = (
        "semiHH_fatjet_index",
        "semiHH_resb1_index",
        "semiHH_resb2_index",
    )
    merged_reco_branches: tuple = (
        "mergedHH_H1_index",
        "mergedHH_H2_index",
    )

    # truth-matching branches (sentinel = -1)
    truth_ak4_branches: tuple = (
        "truerecoj_H1b1_index",
        "truerecoj_H1b2_index",
        "truerecoj_H2b1_index",
        "truerecoj_H2b2_index",
    )
    truth_ak8_branches: tuple = (
        "truerecofj_H1b1_index",
        "truerecofj_H2b1_index",
    )
    
    resolved_truth_branches: tuple = truth_ak4_branches
    
    semiresolved_truth_branches: tuple = (
        "truerecofj_H1b1_index",
        "truerecoj_H2b1_index",
        "truerecoj_H2b2_index",
    )
    merged_truth_branches: tuple = truth_ak8_branches
    
    higgs_mass: float = 125.0  # gen-level mH target [GeV]

    # ── training hyperparameters ─────────────────────────────────────────
    batch_size: int = 64
    max_epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-2
    patience: int = 15
    dropout: float = 0.25
    embed_dim: int = 128
    mlp_hidden: tuple = (256, 128, 64)

    # ── loss hyperparameters ─────────────────────────────────────────
    loss_temperature: float = 1.0   # softmax temperature on CE (logits / τ); 1.0 = unchanged
    mse_weight: float = 1.0         # weight on auxiliary quality regression; 0.0 = CE only

    #── best-method parameters ─────────────────────────────────────────
    mH1_best_weight = 1
    mH2_best_weight = 1
    ptH1_best_weight = .8
    ptH2_best_weight = .8
    deltaRH1_best_weight = .8
    deltaRH2_best_weight = .8
    reco_accuracy_best_weight = 1
    
    best_param_list = {
        'mH1_best_weight' : mH1_best_weight,
        'mH2_best_weight' : mH2_best_weight,
        'ptH1_best_weight' : ptH1_best_weight,
        'ptH2_best_weight' : ptH2_best_weight,
        'deltaRH1_best_weight' : deltaRH1_best_weight,
        'deltaRH2_best_weight' : deltaRH2_best_weight,
        'reco_accuracy_best_weight' : reco_accuracy_best_weight
    }

    # ── data split ───────────────────────────────────────────────────────
    val_frac: float = 0.15
    seed: int = 42
    num_workers: int = 4

    # ── sentinel values in ROOT ──────────────────────────────────────────
    reco_sentinel: int = -1000   # algorithm index sentinel
    truth_sentinel: int = -1     # truth-matching index sentinel

    def __post_init__(self):
        self.ak4_feat_dim = len(self.ak4_features)
        self.ak8_feat_dim = len(self.ak8_features)
        self.event_feat_dim = len(self.event_features) + 3 + 3  # +3 topology +3 algo-success
