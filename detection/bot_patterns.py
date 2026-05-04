"""
Bot Pattern Detector — temporal and behavioural signals for automated accounts.

Analyses posting patterns, inter-event timing regularity,
content similarity clusters, and engagement anomalies.
"""

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def detect_regular_intervals(timestamps: List[float], tolerance: float = 0.1) -> float:
    """
    Returns a 'regularity score' [0, 1].
    1.0 = posts arrive on a perfectly fixed interval (cron-bot).
    0.0 = completely irregular (human-like).
    tolerance = fractional CV threshold below which we flag as bot-like.
    """
    if len(timestamps) < 4:
        return 0.0
    iets = np.diff(sorted(timestamps))
    iets = iets[iets > 0]
    if len(iets) < 2:
        return 0.0
    cv = iets.std() / iets.mean() if iets.mean() > 0 else 0.0
    # Low CV → high regularity → high bot score
    return float(max(0.0, 1.0 - cv / 1.5))


def detect_content_cloning(texts: List[str], threshold: float = 0.85) -> float:
    """
    Jaccard similarity between all text pairs (sampled for speed).
    Returns the fraction of pairs that exceed the similarity threshold.
    High fraction = likely bot farm with templated content.
    """
    if len(texts) < 2:
        return 0.0

    def jaccard(a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # Sample up to 200 pairs for performance
    import random
    pairs = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    sample = random.sample(pairs, min(200, len(pairs)))

    above = sum(1 for i, j in sample if jaccard(texts[i], texts[j]) > threshold)
    return above / len(sample) if sample else 0.0


def detect_engagement_anomaly(scores: List[float], num_comments: List[int]) -> float:
    """
    Flags suspiciously high engagement patterns (fake upvotes / engagement pods).
    Returns anomaly score [0, 1]; 1 = highly anomalous.
    """
    if len(scores) < 5:
        return 0.0

    score_arr = np.array(scores, dtype=float)
    comment_arr = np.array(num_comments, dtype=float)

    # Z-score outlier detection
    score_mean, score_std = score_arr.mean(), score_arr.std()
    if score_std == 0:
        return 0.0

    z_scores = np.abs((score_arr - score_mean) / score_std)
    anomalous = (z_scores > 3.0).mean()
    return float(anomalous)


def compute_account_velocity(
    post_timestamps: List[float],
    account_created_ts: Optional[float] = None,
) -> float:
    """
    Posts-per-day velocity. Very high velocity on new accounts = bot signal.
    Returns velocity (posts/day). No normalisation — caller interprets.
    """
    if not post_timestamps:
        return 0.0
    ts = sorted(post_timestamps)
    if account_created_ts and account_created_ts < ts[0]:
        window_days = max(1.0, (ts[-1] - account_created_ts) / 86400.0)
    else:
        window_days = max(1.0, (ts[-1] - ts[0]) / 86400.0)
    return len(ts) / window_days


def aggregate_bot_score(
    timestamps: Optional[List[float]] = None,
    texts: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    num_comments: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Composite bot likelihood score [0, 1] from all available signals.
    Returns individual signal values and a combined bot_probability.
    """
    signals: Dict[str, float] = {}

    if timestamps and len(timestamps) >= 4:
        signals["interval_regularity"] = detect_regular_intervals(timestamps)

    if texts and len(texts) >= 2:
        signals["content_cloning"] = detect_content_cloning(texts)

    if scores and num_comments and len(scores) >= 5:
        signals["engagement_anomaly"] = detect_engagement_anomaly(scores, num_comments)

    if not signals:
        return {"bot_probability": 0.0}

    # Equal-weight average of available signals
    bot_prob = sum(signals.values()) / len(signals)
    return {**signals, "bot_probability": round(float(np.clip(bot_prob, 0.0, 1.0)), 4)}
