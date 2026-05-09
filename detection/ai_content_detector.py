"""
AI Content Detector — applies the feature extractor to a corpus
and produces per-document aliveness scores and a corpus-level summary.

Works on pandas DataFrames from the silver layer.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .linguistic_features import extract_features, features_to_scores
from .perplexity_scorer import is_enabled as perplexity_enabled
from .perplexity_scorer import score_perplexity


# ── Weights (must match config/config.yaml detection.weights) ────────────────

DEFAULT_WEIGHTS: Dict[str, float] = {
    "ttr": 0.18,
    "mtld": 0.12,
    "entropy": 0.15,
    "sentence_variance": 0.15,
    "repetition": 0.15,
    "burstiness": 0.15,
    "zipf_deviation": 0.10,
}

PERPLEXITY_WEIGHT = 0.15


def _get_weights(include_perplexity: bool = False) -> dict:
    if not include_perplexity:
        return DEFAULT_WEIGHTS
    scale = 1.0 - PERPLEXITY_WEIGHT
    return {k: round(v * scale, 4) for k, v in DEFAULT_WEIGHTS.items()} | {"perplexity": PERPLEXITY_WEIGHT}


def score_document(text: str, timestamps: Optional[List[float]] = None,
                   weights: Optional[Dict[str, float]] = None) -> Dict:
    """
    Score a single document.
    Returns dict with raw features, sub-scores, and composite aliveness_score [0, 100].
    """
    use_perplexity = perplexity_enabled()
    w = weights or _get_weights(include_perplexity=use_perplexity)
    features = extract_features(text, timestamps)
    sub_scores = features_to_scores(features)

    if use_perplexity:
        sub_scores["perplexity"] = score_perplexity(text)

    composite = sum(sub_scores.get(k, 0.5) * v for k, v in w.items()) * 100.0
    composite = round(float(np.clip(composite, 0.0, 100.0)), 2)

    result = {
        "aliveness_score": composite,
        **{f"feat_{k}": v for k, v in features.items()},
        **{f"score_{k}": round(v, 4) for k, v in sub_scores.items()},
    }

    if use_perplexity:
        result["score_perplexity"] = round(sub_scores["perplexity"], 4)

    return result


def score_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    weights: Optional[Dict[str, float]] = None,
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    Vectorised scoring of a silver-layer DataFrame.
    Adds aliveness_score + all feature/sub-score columns in place.
    """
    print(f"[DETECTOR] Scoring {len(df):,} documents …")
    print(f"[DETECTOR] Perplexity scoring: {'ON' if perplexity_enabled() else 'OFF'}")
    w = weights or _get_weights(include_perplexity=perplexity_enabled())

    results: List[Dict] = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        for _, row in batch.iterrows():
            text = str(row.get(text_col, "") or "")
            scored = score_document(text, weights=w)
            results.append(scored)
        if (i // batch_size) % 10 == 0:
            print(f"  … {min(i + batch_size, len(df)):,}/{len(df):,}")

    score_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), score_df], axis=1)


def corpus_summary(scored_df: pd.DataFrame) -> Dict:
    """Aggregate corpus-level statistics from a scored DataFrame."""
    if scored_df.empty or "aliveness_score" not in scored_df.columns:
        return {}

    scores = scored_df["aliveness_score"].dropna()
    return {
        "n_documents": len(scored_df),
        "mean_aliveness": round(float(scores.mean()), 2),
        "median_aliveness": round(float(scores.median()), 2),
        "std_aliveness": round(float(scores.std()), 2),
        "pct_below_50": round(float((scores < 50).mean() * 100), 1),
        "pct_below_30": round(float((scores < 30).mean() * 100), 1),
        "per_source": (
            scored_df.groupby("source")["aliveness_score"]
            .agg(["mean", "median", "count"])
            .round(2)
            .rename(columns={"mean": "mean_score", "median": "median_score", "count": "n_docs"})
            .to_dict(orient="index")
        ) if "source" in scored_df.columns else {},
        "per_category": (
            scored_df.groupby("category")["aliveness_score"]
            .agg(["mean", "median", "count"])
            .round(2)
            .rename(columns={"mean": "mean_score", "median": "median_score", "count": "n_docs"})
            .to_dict(orient="index")
        ) if "category" in scored_df.columns else {},
    }
