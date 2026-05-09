"""
Perplexity scorer using distilgpt2.

Low perplexity = predictable text = likely AI-generated.
High perplexity = surprising text = likely human.

The score is normalised to [0, 1] where 1 = most human (high perplexity).
Only activated when ENABLE_PERPLEXITY=1 env var is set, to keep the
standard pipeline fast on runners without GPU.
"""

import math
import os
from typing import Optional

import numpy as np
import torch

_model = None
_tokenizer = None
MAX_LENGTH = 512        # truncate inputs to this many tokens
PERPLEXITY_FLOOR = 1.0
PERPLEXITY_CAP = 1000.0


def _load_model():
    """Lazy-load distilgpt2 once per process."""
    global _model, _tokenizer
    if _model is None:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
        _model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        _model.eval()
    return _model, _tokenizer


def is_enabled() -> bool:
    """Returns True when ENABLE_PERPLEXITY=1 is set in the environment."""
    return os.environ.get("ENABLE_PERPLEXITY", "0") == "1"


def score_perplexity(text: str) -> float:
    """
    Returns normalised perplexity score in [0, 1].
    1.0 = very high perplexity (human-like).
    0.0 = very low perplexity (AI-like).
    Returns 0.5 on error/empty text.
    """
    if not text or not text.strip():
        return 0.5

    try:
        model, tokenizer = _load_model()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = math.exp(loss.item())

        # Clamp to [PERPLEXITY_FLOOR, PERPLEXITY_CAP]
        perplexity = max(PERPLEXITY_FLOOR, min(PERPLEXITY_CAP, perplexity))

        # Normalise to [0, 1]
        score = (math.log(perplexity) - math.log(PERPLEXITY_FLOOR)) / (
            math.log(PERPLEXITY_CAP) - math.log(PERPLEXITY_FLOOR)
        )

        # Clamp final score to [0.0, 1.0]
        return float(np.clip(score, 0.0, 1.0))

    except Exception:
        return 0.5
