"""Independent quantitative judges, perplexity calculation, and comparative evaluation."""

from __future__ import annotations
import ast
import re
import math
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Independent Judges (External Evaluators)
# ==============================================================================

class PythonSyntaxJudge:
    """Independent judge evaluating Python code syntax, keywords, and structural validity."""

    PYTHON_KEYWORDS = {
        "def", "class", "return", "import", "from", "for", "while", "if", "elif",
        "else", "try", "except", "with", "as", "pass", "yield", "lambda", "print"
    }

    @classmethod
    def score(cls, text: str) -> Dict[str, float]:
        """Scores text for Python syntax presence, AST validity, and keyword density.
        
        Returns a normalized score between 0.0 and 1.0.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        total_words = max(len(words), 1)

        # Keyword density
        kw_count = sum(1 for w in words if w in cls.PYTHON_KEYWORDS)
        keyword_density = kw_count / total_words

        # Code syntax symbols: ':', '()', '[]', '{}', '=', '->'
        symbol_matches = len(re.findall(r"[:\(\)\[\]\{\}=><\+\-\*\/]", text))
        symbol_density = min(symbol_matches / total_words, 1.0)

        # AST parse attempt (check if any multi-line or block parse succeeds)
        ast_valid = 0.0
        try:
            ast.parse(text)
            ast_valid = 1.0
        except SyntaxError:
            # Check if at least individual lines or trimmed blocks parse
            lines = [l for l in text.split("\n") if l.strip()]
            valid_lines = 0
            for l in lines:
                try:
                    ast.parse(l)
                    valid_lines += 1
                except SyntaxError:
                    pass
            if lines:
                ast_valid = 0.5 * (valid_lines / len(lines))

        # Composite score
        composite = 0.5 * min(keyword_density * 4.0, 1.0) + 0.3 * symbol_density + 0.2 * ast_valid
        composite_clipped = min(max(composite, 0.0), 1.0)
        return {
            "composite_score": round(float(composite_clipped), 4),
            "keyword_density": round(float(keyword_density), 4),
            "symbol_density": round(float(symbol_density), 4),
            "ast_validity": round(float(ast_valid), 4),
        }


class SentimentJudge:
    """Independent judge evaluating sentiment polarity and positive emotional markers."""

    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic", "love",
        "beautiful", "superb", "brilliant", "delightful", "outstanding", "joy",
        "happy", "pleased", "perfect", "favorite", "helpful", "kind", "friendly"
    }
    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "worst", "hate", "ugly", "poor",
        "dreadful", "disappointing", "sad", "unhappy", "angry", "annoying", "cruel"
    }

    @classmethod
    def score(cls, text: str) -> Dict[str, float]:
        words = re.findall(r"\b\w+\b", text.lower())
        total_words = max(len(words), 1)

        pos_count = sum(1 for w in words if w in cls.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in cls.NEGATIVE_WORDS)

        pos_density = pos_count / total_words
        neg_density = neg_count / total_words

        # Polarity in [-1.0, 1.0], normalized to [0.0, 1.0]
        net_polarity = (pos_count - neg_count) / max(pos_count + neg_count, 1)
        normalized_score = 0.5 + 0.5 * net_polarity if (pos_count + neg_count) > 0 else 0.5

        return {
            "composite_score": round(float(normalized_score), 4),
            "positive_density": round(float(pos_density), 4),
            "negative_density": round(float(neg_density), 4),
        }


class FormalityJudge:
    """Independent judge evaluating formality, academic tone, and vocabulary complexity."""

    FORMAL_MARKERS = {
        "furthermore", "moreover", "consequently", "therefore", "subsequently",
        "specifically", "demonstrates", "indicates", "establishing", "methodology",
        "analysis", "empirical", "constitutes", "fundamental", "significantly",
        "investigation", "phenomenon", "hypothesis", "theoretical", "accordingly"
    }

    @classmethod
    def score(cls, text: str) -> Dict[str, float]:
        words = re.findall(r"\b\w+\b", text.lower())
        total_words = max(len(words), 1)

        formal_count = sum(1 for w in words if w in cls.FORMAL_MARKERS)
        formal_density = formal_count / total_words

        # Average word length as proxy for lexical complexity
        avg_word_len = sum(len(w) for w in words) / total_words if words else 0.0
        len_score = min(max((avg_word_len - 4.0) / 4.0, 0.0), 1.0)

        composite = 0.6 * min(formal_density * 5.0, 1.0) + 0.4 * len_score
        composite_clipped = min(max(composite, 0.0), 1.0)
        return {
            "composite_score": round(float(composite_clipped), 4),
            "formal_marker_density": round(float(formal_density), 4),
            "avg_word_length": round(float(avg_word_len), 2),
        }


JUDGES = {
    "python_code": PythonSyntaxJudge,
    "sentiment": SentimentJudge,
    "formality": FormalityJudge,
}


# ==============================================================================
# 2. Collateral Damage: Perplexity Evaluator
# ==============================================================================

def compute_perplexity(
    model: Any,
    text: str,
    device: Optional[Any] = None,
) -> float:
    """Computes language model perplexity of text under unsteered base model.
    
    Higher perplexity indicates output degradation / coherence loss.
    """
    import torch
    import torch.nn.functional as F

    resolved_device = device or next(model.parameters()).device
    tokens = model.tokenizer.encode(text, return_tensors="pt").to(resolved_device)

    if tokens.shape[1] < 2:
        return 1.0

    with torch.no_grad():
        # Standard unhooked forward pass
        logits = model(tokens)  # [1, seq_len, vocab_size]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="mean",
        )
        ppl = math.exp(min(loss.item(), 20.0))  # guard against overflow
        return float(ppl)


# ==============================================================================
# 3. Comparative Evaluation Suite: SAE vs Difference-of-Means
# ==============================================================================

def evaluate_steered_generations(
    model: Any,
    concept_name: str,
    steered_samples: List[Dict[str, Any]],
    method_name: str,
    baseline_ppl: Optional[float] = None,
    device: Optional[Any] = None,
) -> Any:
    """Evaluates generated outputs across alpha values for efficacy and collateral damage."""
    import numpy as np
    import pandas as pd
    judge = JUDGES.get(concept_name, PythonSyntaxJudge)
    rows = []

    for entry in steered_samples:
        alpha = entry["alpha"]
        prompt = entry["prompt"]
        samples = entry["samples"]

        judge_scores = []
        perplexities = []

        for text in samples:
            # Independent judge score
            j_res = judge.score(text)
            judge_scores.append(j_res["composite_score"])

            # Perplexity under unsteered model
            ppl = compute_perplexity(model, text, device=device)
            perplexities.append(ppl)

        mean_judge = float(np.mean(judge_scores))
        mean_ppl = float(np.mean(perplexities))

        ppl_delta = (mean_ppl - baseline_ppl) if baseline_ppl is not None else 0.0

        rows.append({
            "concept": concept_name,
            "method": method_name,
            "alpha": alpha,
            "prompt": prompt,
            "judge_score": round(mean_judge, 4),
            "perplexity": round(mean_ppl, 2),
            "perplexity_delta": round(ppl_delta, 2),
            "num_samples": len(samples),
        })

    return pd.DataFrame(rows)


def run_full_comparative_evaluation(
    model: Any,
    concept_name: str,
    sae_samples: List[Dict[str, Any]],
    diff_of_means_samples: List[Dict[str, Any]],
    device: Optional[Any] = None,
) -> Any:
    """Generates direct side-by-side comparison between SAE steering and Diff-of-Means."""
    import numpy as np
    import pandas as pd
    # Find unsteered baseline perplexity (alpha == 0.0)
    baseline_ppls = []
    for entry in sae_samples:
        if entry["alpha"] == 0.0:
            for s in entry["samples"]:
                baseline_ppls.append(compute_perplexity(model, s, device=device))

    baseline_ppl = float(np.mean(baseline_ppls)) if baseline_ppls else 30.0

    df_sae = evaluate_steered_generations(
        model=model,
        concept_name=concept_name,
        steered_samples=sae_samples,
        method_name="SAE_Feature",
        baseline_ppl=baseline_ppl,
        device=device,
    )

    df_diff = evaluate_steered_generations(
        model=model,
        concept_name=concept_name,
        steered_samples=diff_of_means_samples,
        method_name="Diff_of_Means_Baseline",
        baseline_ppl=baseline_ppl,
        device=device,
    )

    combined = pd.concat([df_sae, df_diff], ignore_index=True)
    return combined
