"""
Detection Scientist — linguistic feature extractor.

Computes a battery of statistical signals that distinguish
human-authored text from AI-generated or bot-produced content.

No large LLMs required. All methods are O(n) to O(n log n)
and run fast on CPU with standard Python/NumPy.

Signals implemented:
  ttr              — Type-Token Ratio (vocabulary richness)
  mtld             — Measure of Textual Lexical Diversity (McCarthy 2005)
  entropy          — Shannon entropy of word unigrams
  sentence_variance — CV of sentence lengths (AI text is suspiciously uniform)
  repetition       — Ratio of repeated bigrams to total bigrams
  zipf_deviation   — Deviation from Zipf's power-law for natural language
  burstiness       — Goh-Barabasi burstiness of inter-event times (for temporal data)
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Text tokenisation ─────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\b[a-z]{2,}\b")
_SENT_RE = re.compile(r"[^.!?]+[.!?]+", re.MULTILINE)


def tokenise(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


# ── Individual feature functions ──────────────────────────────────────────────

def compute_ttr(tokens: List[str]) -> float:
    """
    Type-Token Ratio: unique_words / total_words.
    Range [0, 1]. Higher = more diverse vocabulary.
    Naive TTR is length-sensitive; we cap at 200 tokens for comparability.
    """
    if not tokens:
        return 0.0
    sample = tokens[:200]
    return len(set(sample)) / len(sample)


def compute_mtld(tokens: List[str], threshold: float = 0.72) -> float:
    """
    Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).
    Scans forward until TTR drops below threshold, counts each as a 'factor'.
    MTLD = len(tokens) / factor_count.
    Higher = more lexically diverse. Typical human prose: 70-120.
    """
    if len(tokens) < 10:
        return 0.0

    def _mtld_forward(toks: List[str]) -> float:
        factors = 0.0
        types: set = set()
        token_count = 0
        start = 0
        for i, tok in enumerate(toks):
            types.add(tok)
            token_count += 1
            current_ttr = len(types) / token_count
            if current_ttr <= threshold:
                factors += 1.0
                types = set()
                token_count = 0
                start = i + 1
        # Partial factor
        if token_count > 0:
            partial_ttr = len(types) / token_count
            if partial_ttr < 1.0:
                completion = (1.0 - partial_ttr) / (1.0 - threshold)
                factors += completion
        return len(toks) / factors if factors > 0 else len(toks)

    # Average forward + backward for stability
    fwd = _mtld_forward(tokens)
    bwd = _mtld_forward(tokens[::-1])
    return (fwd + bwd) / 2.0


def compute_entropy(tokens: List[str]) -> float:
    """
    Shannon entropy of the unigram word distribution.
    H = -Σ p(w) log2 p(w).
    Higher = more uniform / information-dense text.
    AI text tends to be slightly lower entropy (repetitive phrases).
    """
    if not tokens:
        return 0.0
    freq: Dict[str, int] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1
    n = len(tokens)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def compute_sentence_variance(text: str) -> float:
    """
    Coefficient of Variation (CV) of sentence lengths in words.
    Human writing has high CV (mix of short and long sentences).
    AI text is suspiciously uniform: CV << 0.5.
    Returns CV directly; we'll invert/normalise in the scorer.
    """
    sents = sentences(text)
    if len(sents) < 3:
        return 0.5   # neutral default for very short texts
    lengths = [len(tokenise(s)) for s in sents]
    arr = np.array(lengths, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return 0.0
    return float(arr.std() / mean)   # CV


def compute_repetition(tokens: List[str]) -> float:
    """
    Ratio of repeated bigrams to total bigrams.
    AI text frequently repeats stock phrases → higher repetition.
    Returns [0, 1]; 0 = no repetition, 1 = all bigrams repeated.
    """
    if len(tokens) < 4:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    bigram_freq: Dict[tuple, int] = {}
    for bg in bigrams:
        bigram_freq[bg] = bigram_freq.get(bg, 0) + 1
    repeated = sum(v - 1 for v in bigram_freq.values() if v > 1)
    return repeated / len(bigrams)


def compute_zipf_deviation(tokens: List[str]) -> float:
    """
    Measures how well the word frequency distribution fits Zipf's law.
    Natural language obeys: freq(rank r) ∝ 1/r^α  (α ≈ 1).
    We fit log-log slope; deviation from -1 indicates unnatural text.
    Returns 'alignment score' [0, 1]; 1 = perfect Zipf.
    """
    if len(tokens) < 30:
        return 0.5

    freq: Dict[str, int] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1

    sorted_freqs = sorted(freq.values(), reverse=True)[:50]  # top 50
    if len(sorted_freqs) < 5:
        return 0.5

    ranks = np.arange(1, len(sorted_freqs) + 1, dtype=float)
    freqs_arr = np.array(sorted_freqs, dtype=float)

    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs_arr)

    # Linear regression of log-log
    coeffs = np.polyfit(log_ranks, log_freqs, 1)
    slope = coeffs[0]

    # Ideal slope ≈ -1.0; we score based on distance from ideal
    deviation = abs(slope - (-1.0))
    # Map to [0, 1]: 0 deviation → 1.0 score; 2.0 deviation → 0.0
    score = max(0.0, 1.0 - deviation / 2.0)
    return float(score)


def compute_burstiness(timestamps_seconds: List[float]) -> float:
    """
    Goh-Barabasi (2008) burstiness parameter B from inter-event times.
    B = (σ - μ) / (σ + μ) ∈ [-1, 1].
    B ≈ +1 → very bursty (human bursts of activity).
    B ≈ -1 → perfectly regular (bot-like scheduling).
    B ≈  0 → Poisson / random.
    Returns a normalised aliveness score [0, 1].
    """
    if len(timestamps_seconds) < 3:
        return 0.5   # neutral

    ts = sorted(timestamps_seconds)
    iets = np.diff(ts)
    iets = iets[iets > 0]

    if len(iets) < 2:
        return 0.5

    mu = iets.mean()
    sigma = iets.std()

    if mu + sigma == 0:
        return 0.5

    B = (sigma - mu) / (sigma + mu)
    # Map from [-1, 1] to [0, 1]: bursty (positive) = more alive
    return float((B + 1.0) / 2.0)


# ── Combined feature vector ───────────────────────────────────────────────────

def extract_features(
    text: str,
    timestamps: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute all linguistic features for one text document.
    Returns a dict of raw feature values (not yet normalised to scores).
    """
    tokens = tokenise(text)

    features = {
        "ttr": compute_ttr(tokens),
        "mtld": compute_mtld(tokens),
        "entropy": compute_entropy(tokens),
        "sentence_variance": compute_sentence_variance(text),
        "repetition": compute_repetition(tokens),
        "zipf_deviation": compute_zipf_deviation(tokens),
        "token_count": len(tokens),
        "sentence_count": len(sentences(text)),
        "avg_word_length": (
            sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0
        ),
    }

    if timestamps:
        features["burstiness"] = compute_burstiness(timestamps)
    else:
        features["burstiness"] = 0.5   # neutral when no temporal data

    return features


def features_to_scores(features: Dict[str, float]) -> Dict[str, float]:
    """
    Normalise raw features to [0, 1] aliveness sub-scores.
    Higher sub-score always means "more alive / human-like".
    """
    scores: Dict[str, float] = {}

    # TTR: 0.3–0.7 is typical human; cap and scale
    ttr = features.get("ttr", 0.0)
    scores["ttr"] = min(1.0, max(0.0, (ttr - 0.2) / 0.5))

    # MTLD: 40 = low diversity, 120 = high. Cap at 120.
    mtld = features.get("mtld", 0.0)
    scores["mtld"] = min(1.0, max(0.0, (mtld - 20.0) / 100.0))

    # Entropy: typical English ≈ 8–11 bits. Scale.
    ent = features.get("entropy", 0.0)
    scores["entropy"] = min(1.0, max(0.0, (ent - 5.0) / 8.0))

    # Sentence variance: human CV ≈ 0.5–1.5. Low CV → low score.
    sv = features.get("sentence_variance", 0.0)
    scores["sentence_variance"] = min(1.0, max(0.0, sv / 1.2))

    # Repetition: lower is better (more human). Invert.
    rep = features.get("repetition", 0.0)
    scores["repetition"] = max(0.0, 1.0 - (rep / 0.3))

    # Zipf: already [0, 1]
    scores["zipf_deviation"] = features.get("zipf_deviation", 0.5)

    # Burstiness: already [0, 1]
    scores["burstiness"] = features.get("burstiness", 0.5)

    return scores
