import re
import numpy as np

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    tot = np.sum(union)
    return float(np.sum(intersection)) / float(tot) if tot > 0 else 1.0

def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(text.split())

def compute_token_f1(pred: str, gt: str) -> float:
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gt).split()
    if not p_tokens or not g_tokens: return 1.0 if p_tokens == g_tokens else 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common: return 0.0
    prec = len(common) / len(p_tokens)
    rec = len(common) / len(g_tokens)
    return 2 * (prec * rec) / (prec + rec)

def compute_bleu_1(ref: str, hyp: str) -> float:
    r_tokens = normalize_text(ref).split()
    h_tokens = normalize_text(hyp).split()
    if not h_tokens: return 0.0
    return sum(1 for t in h_tokens if t in r_tokens) / len(h_tokens)
