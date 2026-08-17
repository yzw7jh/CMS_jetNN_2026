
from dataclasses import dataclass
from .config import CFG
from .dataset import _build_4vec, _topology_possible, _algo_succeeded, _reco_h1h2_for_method
import numpy as np
import awkward as ak
from scipy.special import softmax

@dataclass
class SorterResult:
    """Container for model inference results.

    Attributes
    ----------
    scores : np.ndarray  — (N, 3) sigmoid-transformed scores in [0, 1]
    method_mask : np.ndarray — (N, 3) bool, True where topology_possible & algo_succeeded
    event_indices : np.ndarray — (N,) int, indices into the original awkward array
    """
    scores: np.ndarray
    method_mask: np.ndarray
    event_indices: np.ndarray


# def resVSsemi(
#         ev: Ak.array, 
#         method_combination: int, 
#         cfg: CFG = CFG(),
#         ak4_4vec, 
#         ak8_4vec
#     ):
#     """
#     Sorts events that can be reconstructed by both the resolved and semi-resolved algorithms.
#     Accepts an array of events that can all be reconstructed by both algorithms. 
#     """
#     assert method_combination in range(8), "method_combination must be an integer between 0 and 7"
#     if method_combination == 0:
#         return np.full((len(ev), 3), False, dtype=bool)
#     if method_combination == 1:
#         scores = np.full((len(ev), 3), False, dtype=bool)
#         scores[:,2] = True
#         return scores
#     if method_combination == 2:
#         scores = np.full((len(ev), 3), False, dtype=bool)
#         scores[:,1] = True
#         return scores
#     if method_combination == 4:
#         scores = np.full((len(ev), 3), False, dtype=bool)
#         scores[:,0] = True
#         return scores
#     if method_combination == 3:
#         scores = np.full((len(ev), 3), False, dtype=bool)
#         semi_xbb = ev.XbbvsQCD[range(len(ev)),ev.semiHH_fatjet_index].to_numpy()
        
#         merged_xbb = np.array(
#             ev.XbbvsQCD[range(len(ev)),ev.mergedHH_H1_index].to_numpy(), 
#             ev.XbbvsQCD[range(len(ev)),ev.mergedHH_H2_index].to_numpy()
#         ).mean(axis=0)
        
#         scores[semi_xbb > merged_xbb, 1] = True
#         scores[merged_xbb >= semi_xbb, 1] = True
        
        
#         return scores
#     if method_combination == 5:
def get_btag_counts(arrays, event_indices, method_mask, method):    
    if method == 'resolved':
        ev = arrays[method_mask[:,0]]
        return np.stack([
            ev.ak4_btag_M[ak.Array(range(len(ev))), ev[k]].to_numpy() 
            for k in CFG().resolved_reco_branches
        ]).sum(axis=0)
    if method == 'semiresolved':
        ev = arrays[method_mask[:,1]]
        return np.stack([
            ev.ak4_btag_M[ak.Array(range(len(ev))), ev[k]].to_numpy() 
            for k in CFG().semiresolved_reco_branches[1:]
        ]).sum(axis=0)
    if method == 'merged':
        return np.zeros(len(arrays[method_mask[:,2]]),dtype=float)
        

def get_scores(method_mask, H1mass, H2mass, H1pt, H2pt, btags, basic_params_list : dict):
    """
    Accepts 
        method_mask : (N, 3) bool, True where topology_possible & algo_succeeded, 
        H1mass : (N, 3) float, 
        H2mass : (N, 3) float, 
        H1pt : (N, 3) float, 
        H2pt : (N, 3) float
    
    Returns a (N, 3) array of scores for each method, where N is the number of events and 3 is the number of methods (resolved, semi-resolved, merged).
    The score is based on the mass and pt of the reconstructed Higgs bosons, with the goal of minimizing the difference between the reconstructed pt and the expected mass of 125 GeV.

    """
    H1m_w = basic_params_list['basic_mH1_weight']
    H2m_w = basic_params_list['basic_mH2_weight']

    dpt_w_arr = np.array(basic_params_list['basic_dpt_weight'])
    btag_weight = np.array(basic_params_list['basic_btag_weight'])
    
    error = np.abs(2 * 125.0 - H1mass * H1m_w - H2mass * H2m_w)
    error += np.abs(H1pt - H2pt) * dpt_w_arr[np.newaxis, :]
    error += (np.array([4.0,2.0,0.0])[np.newaxis, :] - btags) * btag_weight[np.newaxis, :]
    
    for j in range(3):
        error[~method_mask[:,j], j] = float('inf') #set unavailable methods to inf so they get 0 score after softmax of -error
    
    return softmax(-error, axis=1)
    

    
def apply_basicsort(ev: ak.Array,
                cfg: CFG = CFG(),
                min_methods: int = 1,
                basic_params_list: dict = CFG().basic_params_list
    ):
    methods = ["resolved", "semiresolved", "merged"]
    d = {
    'None' :	0,
    "merged":	1,
    "semiresolved":	2,
    "semiresolved and merged":    3,
    "resolved":	4,
    "resolved and merged":    5,
    "resolved and semiresolved":    6,
    "all":	7
    }

    N = len(ev)
    
    avail = np.full((N, 3), False, dtype=bool)
    for j,method in enumerate(methods):
        avail[:,j] = _topology_possible(ev, method, cfg) & _algo_succeeded(ev, method, cfg)
        
    event_indices = np.array(range(len(ev)))[np.sum(avail, axis=1) >= min_methods]
    method_mask = avail[event_indices]
    arrays = ev[event_indices]
    
    ak4_4vec = _build_4vec(arrays.ak4_pt, arrays.ak4_eta, arrays.ak4_phi, arrays.ak4_mass)
    ak8_4vec = _build_4vec(arrays.ak8_pt, arrays.ak8_eta, arrays.ak8_phi, arrays.ak8_msoftdrop)
    # method_combo_filter = {
    #     0 : np.array([]),
    #     1 : ~method_mask[:,0] & ~method_mask[:,1] & method_mask[:,2],
    #     2 : ~method_mask[:,0] & method_mask[:,1] & ~method_mask[:,2],
    #     3 : ~method_mask[:,0] & method_mask[:,1] & method_mask[:,2],
    #     4 : method_mask[:,0] & ~method_mask[:,1] & ~method_mask[:,2],
    #     5 : method_mask[:,0] & ~method_mask[:,1] & method_mask[:,2],
    #     6 : method_mask[:,0] & method_mask[:,1] & ~method_mask[:,2],
    #     7 : method_mask[:,0] & method_mask[:,1] & method_mask[:,2]
    # }
    shape = (len(event_indices), 3)
    ## Initialize arrays
    H1mass = np.full(shape, -1000.0, dtype=float)
    H2mass = np.full(shape, -1000.0, dtype=float)
    H1pt = np.full(shape, -1000.0, dtype=float)
    H2pt = np.full(shape, -1000.0, dtype=float)
    btags = np.full(shape, -1000.0, dtype=float)
    # Fill arrays
    for j,method in enumerate(methods):
        h1,h2 = _reco_h1h2_for_method(
            ak4_4vec[method_mask[:,j]],
            ak8_4vec[method_mask[:,j]], 
            ak.Array(np.arange(len(event_indices[method_mask[:,j]]))), 
            ak.Array(np.arange(len(event_indices[method_mask[:,j]]))), 
            arrays[method_mask[:,j]], 
            method, 
            cfg)
        H1mass[method_mask[:,j], j] = h1.mass.to_numpy()
        H2mass[method_mask[:,j], j] = h2.mass.to_numpy()
        H1pt[method_mask[:,j], j] = h1.pt.to_numpy()
        H2pt[method_mask[:,j], j] = h2.pt.to_numpy()
        btags[method_mask[:,j], j] = get_btag_counts(arrays, event_indices, method_mask, method)

    scores = get_scores(method_mask, H1mass, H2mass, H1pt, H2pt, btags, basic_params_list)

    return SorterResult(
        scores=scores,
        method_mask=method_mask,
        event_indices=event_indices,
    )