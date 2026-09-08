"""Evaluation utilities for SatQuery AI benchmark tasks.

Covers VQA accuracy, captioning metrics (BLEU-1, BLEU-4, METEOR, token F1),
and spatial segmentation/change metrics (IoU, confusion matrix, F1).
All functions are pure Python + NumPy with zero heavy external dependencies.
"""

from __future__ import annotations

import collections
import math
import re
from typing import Any

import numpy as np


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation/extra whitespace."""
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", "", str(text).lower())
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Captioning / Text Metrics
# ---------------------------------------------------------------------------


def compute_token_f1(pred: str, gt: str) -> float:
    """Compute token-level F1 score between two text strings."""
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gt).split()
    if not p_tokens or not g_tokens:
        return 1.0 if p_tokens == g_tokens else 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(p_tokens)
    rec = len(common) / len(g_tokens)
    if prec + rec == 0.0:
        return 0.0
    return 2.0 * (prec * rec) / (prec + rec)


def compute_bleu_1(ref: str, hyp: str) -> float:
    """Compute unigram precision (BLEU-1)."""
    r_tokens = normalize_text(ref).split()
    h_tokens = normalize_text(hyp).split()
    if not h_tokens:
        return 0.0
    if not r_tokens:
        return 0.0
    return sum(1 for t in h_tokens if t in r_tokens) / len(h_tokens)


def _get_ngrams(tokens: list[str], n: int) -> collections.Counter[tuple[str, ...]]:
    """Extract n-grams from a token list."""
    return collections.Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu_4(reference: str, hypothesis: str) -> float:
    """Compute sentence-level smoothed BLEU-4 score.

    Uses standard modified n-gram precision with Laplace/Lin smoothing and brevity penalty.
    """
    r_tokens = normalize_text(reference).split()
    h_tokens = normalize_text(hypothesis).split()

    if not h_tokens or not r_tokens:
        return 0.0

    r_len = len(r_tokens)
    h_len = len(h_tokens)

    # Brevity penalty
    if h_len > r_len:
        bp = 1.0
    elif h_len == 0:
        return 0.0
    else:
        bp = math.exp(1.0 - (r_len / h_len))

    weights = [0.25, 0.25, 0.25, 0.25]
    log_precisions = 0.0

    for n in range(1, 5):
        hyp_ngrams = _get_ngrams(h_tokens, n)
        ref_ngrams = _get_ngrams(r_tokens, n)

        total_hyp = sum(hyp_ngrams.values())
        if total_hyp == 0:
            # Smoothing for higher-order ngrams when sentence is short
            clipped_matches = 1.0
            total_hyp = 1.0 + n
        else:
            clipped_matches = sum(
                min(count, ref_ngrams.get(ngram, 0)) for ngram, count in hyp_ngrams.items()
            )
            # Add-1 smoothing to avoid log(0)
            if clipped_matches == 0:
                clipped_matches = 0.1
                total_hyp += 1.0

        p_n = clipped_matches / total_hyp
        log_precisions += weights[n - 1] * math.log(max(p_n, 1e-9))

    bleu = bp * math.exp(log_precisions)
    return float(np.clip(bleu, 0.0, 1.0))


def compute_meteor(reference: str, hypothesis: str) -> float:
    """Compute unigram METEOR score with harmonic mean of precision and recall.

    Pure-Python implementation avoiding any external Java/NLTK download dependency.
    """
    r_tokens = normalize_text(reference).split()
    h_tokens = normalize_text(hypothesis).split()

    if not h_tokens or not r_tokens:
        return 0.0

    # Unigram matches (exact surface match)
    matches = 0
    ref_matched = [False] * len(r_tokens)
    for h_token in h_tokens:
        for i, r_token in enumerate(r_tokens):
            if not ref_matched[i] and h_token == r_token:
                ref_matched[i] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    precision = matches / len(h_tokens)
    recall = matches / len(r_tokens)

    # METEOR weights recall heavily (alpha = 0.9)
    # F_mean = (10 * P * R) / (R + 9 * P)
    alpha = 0.9
    denom = (1.0 - alpha) * precision + alpha * recall
    if denom <= 0:
        return 0.0
    f_mean = (precision * recall) / denom

    return float(np.clip(f_mean, 0.0, 1.0))


# ---------------------------------------------------------------------------
# VQA Accuracy
# ---------------------------------------------------------------------------


def compute_accuracy(predictions: list[Any], ground_truth: list[Any]) -> float:
    """Compute exact and normalized match accuracy across question-answer pairs."""
    if not predictions or not ground_truth:
        return 0.0
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(ground_truth)} ground truth."
        )

    correct = 0
    for p, g in zip(predictions, ground_truth):
        # Normalize and compare
        norm_p = normalize_text(str(p))
        norm_g = normalize_text(str(g))
        if norm_p == norm_g:
            correct += 1
        elif norm_g in norm_p or norm_p in norm_g:
            # Partial substring tolerance for conversational VQA answers
            correct += 1

    return float(correct / len(predictions))


# ---------------------------------------------------------------------------
# Spatial / Mask Metrics (IoU, Confusion Matrix, F1)
# ---------------------------------------------------------------------------


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Intersection over Union for binary or thresholded masks."""
    p = np.asarray(pred_mask) > 0
    g = np.asarray(gt_mask) > 0
    intersection = np.logical_and(p, g)
    union = np.logical_or(p, g)
    total_union = int(np.sum(union))
    if total_union == 0:
        # Both masks are completely empty -> perfect agreement
        return 1.0
    return float(np.sum(intersection)) / float(total_union)


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Alias for compute_iou to match standard semantic segmentation conventions."""
    return compute_iou(pred_mask, gt_mask)


def compute_confusion_matrix(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, int]:
    """Compute True Positives, False Positives, True Negatives, False Negatives."""
    p = (np.asarray(pred_mask) > 0).ravel()
    g = (np.asarray(gt_mask) > 0).ravel()

    tp = int(np.sum(np.logical_and(p, g)))
    fp = int(np.sum(np.logical_and(p, ~g)))
    fn = int(np.sum(np.logical_and(~p, g)))
    tn = int(np.sum(np.logical_and(~p, ~g)))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_f1_score(precision: float, recall: float) -> float:
    """Compute harmonic mean of precision and recall."""
    if precision <= 0 or recall <= 0:
        return 0.0
    denom = precision + recall
    if denom == 0:
        return 0.0
    return float(2.0 * (precision * recall) / denom)
