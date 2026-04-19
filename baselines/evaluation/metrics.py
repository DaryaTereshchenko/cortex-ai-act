"""Comprehensive evaluation metrics for RAG pipeline evaluation.

Metric groups:
  1. Retrieval Source Identification — doc/chapter/article/paragraph accuracy
  2. Retrieval Text Quality — correct answer vs retrieved context overlap
  3. Retrieval Multiple Choice — discrimination between correct and alternatives
  4. Retrieval Golden ID — precision/recall/ranking over document IDs
  5. Generator Quality — expected answer vs generated response (statistical + semantic)
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "and", "or", "on", "it", "its", "this", "that",
    "with", "as", "at", "by", "from", "but", "not", "no", "has", "have", "had",
    "do", "does", "did", "will", "shall", "should", "would", "could", "may",
    "might", "must", "can", "if", "so", "than", "then", "very", "just", "about",
}


def _normalize(text: str) -> str:
    lowered = str(text).lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in no_punct.split() if t and t not in STOPWORDS]
    return " ".join(tokens)


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def _sanitize(text: Any) -> str:
    return re.sub(r"Name:.*dtype:.*", "", str(text), flags=re.DOTALL).strip()


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# 1. Retrieval Source Identification Metrics
# ---------------------------------------------------------------------------

def _parse_doc_from_id(retrieved_id: str) -> str:
    rid = retrieved_id.lower().strip()
    if rid.startswith("eu_dsa") or rid.startswith("dsa"):
        return "dsa"
    if rid.startswith("eu_ai") or rid.startswith("ai"):
        return "eu_ai_act"
    return "unknown"


def _parse_article_from_id(retrieved_id: str) -> str | None:
    m = re.search(r"art[_\s]*(\d+)", retrieved_id.lower())
    return m.group(1) if m else None


def _parse_paragraph_from_id(retrieved_id: str) -> str | None:
    m = re.search(r"par[_\s]*(\d+)", retrieved_id.lower())
    return m.group(1) if m else None


def _normalize_doc(doc_value: str) -> str:
    doc = str(doc_value).strip().lower()
    if "dsa" in doc:
        return "dsa"
    if "ai" in doc:
        return "eu_ai_act"
    return doc


def _normalize_article(art_value: Any) -> str:
    return re.sub(r"[^0-9]", "", str(art_value).split(".")[0])


def _normalize_paragraph(par_value: Any) -> str:
    digits = re.sub(r"[^0-9]", "", str(par_value).split(".")[0])
    return digits


def source_identification_metrics(
    retrieved_ids: list[str],
    gold_doc: str,
    gold_chapter: str,
    gold_article: Any,
    gold_paragraph: Any,
) -> dict[str, float]:
    """Check if retrieved IDs come from the correct doc/article/paragraph.

    Returns accuracy scores (fraction of retrieved IDs matching each level).
    """
    if not retrieved_ids:
        return {
            "doc_accuracy": 0.0,
            "article_accuracy": 0.0,
            "paragraph_accuracy": 0.0,
            "doc_hit": 0.0,
            "article_hit": 0.0,
            "paragraph_hit": 0.0,
        }

    expected_doc = _normalize_doc(gold_doc)
    expected_article = _normalize_article(gold_article)
    expected_paragraph = _normalize_paragraph(gold_paragraph)

    doc_matches = 0
    article_matches = 0
    paragraph_matches = 0

    doc_hit = False
    article_hit = False
    paragraph_hit = False

    for rid in retrieved_ids:
        parsed_doc = _parse_doc_from_id(rid)
        parsed_art = _parse_article_from_id(rid)
        parsed_par = _parse_paragraph_from_id(rid)

        if parsed_doc == expected_doc:
            doc_matches += 1
            doc_hit = True
        if expected_article and parsed_art == expected_article:
            article_matches += 1
            article_hit = True
        if expected_paragraph and parsed_par == expected_paragraph:
            paragraph_matches += 1
            paragraph_hit = True

    n = len(retrieved_ids)
    return {
        "doc_accuracy": doc_matches / n,
        "article_accuracy": article_matches / n if expected_article else float("nan"),
        "paragraph_accuracy": paragraph_matches / n if expected_paragraph else float("nan"),
        "doc_hit": 1.0 if doc_hit else 0.0,
        "article_hit": 1.0 if article_hit else 0.0,
        "paragraph_hit": 1.0 if paragraph_hit else 0.0,
    }


# ---------------------------------------------------------------------------
# 2. Retrieval Text Quality Metrics (Correct Answer vs Retrieved Context)
# ---------------------------------------------------------------------------

def _token_overlap(pred_tokens: list[str], gold_tokens: list[str]) -> tuple[float, float, float]:
    """Return (precision, recall, f1) based on token overlap."""
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _rouge_n(pred_tokens: list[str], gold_tokens: list[str], n: int) -> dict[str, float]:
    """Compute ROUGE-N (precision, recall, F1) for n-grams."""
    pred_ngrams = _ngrams(pred_tokens, n)
    gold_ngrams = _ngrams(gold_tokens, n)

    if not pred_ngrams and not gold_ngrams:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_ngrams or not gold_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(pred_ngrams)
    gold_counter = Counter(gold_ngrams)
    overlap = sum((pred_counter & gold_counter).values())

    precision = overlap / len(pred_ngrams)
    recall = overlap / len(gold_ngrams)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimized LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _rouge_l(pred_tokens: list[str], gold_tokens: list[str]) -> dict[str, float]:
    """Compute ROUGE-L using longest common subsequence."""
    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _substring_containment(context: str, correct_answer: str) -> float:
    """Fraction of correct answer that appears as a contiguous substring in context."""
    norm_ctx = _normalize(context)
    norm_ans = _normalize(correct_answer)
    if not norm_ans:
        return 1.0
    if not norm_ctx:
        return 0.0
    if norm_ans in norm_ctx:
        return 1.0
    # Find longest contiguous match ratio
    ans_tokens = norm_ans.split()
    max_match = 0
    for length in range(len(ans_tokens), 0, -1):
        for start in range(len(ans_tokens) - length + 1):
            sub = " ".join(ans_tokens[start : start + length])
            if sub in norm_ctx:
                max_match = max(max_match, length)
                break
        if max_match >= length:
            break
    return max_match / len(ans_tokens) if ans_tokens else 0.0


def retrieval_text_metrics(
    retrieved_context: str,
    correct_answer: str,
) -> dict[str, float]:
    """Compute text overlap metrics between retrieved context and the correct answer.

    The correct answer is the verbatim paragraph from the regulation.
    """
    correct_answer = _sanitize(correct_answer)
    context = _sanitize(retrieved_context)

    ctx_tokens = _tokenize(context)
    ans_tokens = _tokenize(correct_answer)

    prec, rec, f1 = _token_overlap(ctx_tokens, ans_tokens)

    rouge1 = _rouge_n(ctx_tokens, ans_tokens, 1)
    rouge2 = _rouge_n(ctx_tokens, ans_tokens, 2)
    rouge_l = _rouge_l(ctx_tokens, ans_tokens)

    containment = _substring_containment(context, correct_answer)

    exact = 1.0 if _normalize(correct_answer) in _normalize(context) else 0.0

    return {
        "retrieval_token_precision": prec,
        "retrieval_token_recall": rec,
        "retrieval_token_f1": f1,
        "retrieval_rouge1_f1": rouge1["f1"],
        "retrieval_rouge1_recall": rouge1["recall"],
        "retrieval_rouge2_f1": rouge2["f1"],
        "retrieval_rouge2_recall": rouge2["recall"],
        "retrieval_rouge_l_f1": rouge_l["f1"],
        "retrieval_rouge_l_recall": rouge_l["recall"],
        "retrieval_exact_containment": exact,
        "retrieval_substring_containment": containment,
    }


# ---------------------------------------------------------------------------
# 3. Retrieval Multiple-Choice Metrics
# ---------------------------------------------------------------------------

def _jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two texts."""
    toks_a = set(_tokenize(a))
    toks_b = set(_tokenize(b))
    if not toks_a and not toks_b:
        return 1.0
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def _overlap_coefficient(context: str, option: str) -> float:
    """Token overlap coefficient: |intersection| / min(|A|, |B|)."""
    toks_ctx = set(_tokenize(context))
    toks_opt = set(_tokenize(option))
    if not toks_ctx or not toks_opt:
        return 0.0
    return len(toks_ctx & toks_opt) / min(len(toks_ctx), len(toks_opt))


def mcq_retrieval_metrics(
    retrieved_context: str,
    correct_answer: str,
    alternatives: list[str],
) -> dict[str, float]:
    """Evaluate a retriever's ability to distinguish correct from incorrect answers.

    Uses token-level overlap to score each option against the retrieved context,
    then computes ranking-based metrics.
    """
    import random as _rng

    context = _sanitize(retrieved_context)
    correct = _sanitize(correct_answer)

    if not context.strip():
        return {
            "mcq_accuracy": 0.0,
            "mcq_correct_rank": float(len(alternatives) + 1),
            "mcq_mrr": 0.0,
            "mcq_margin": 0.0,
            "mcq_normalized_margin": 0.0,
            "mcq_discrimination_score": 0.0,
        }

    all_options = [correct] + [_sanitize(a) for a in alternatives if _sanitize(a).strip()]

    # Score each option against retrieved context using overlap coefficient
    scores = []
    for opt in all_options:
        score = _overlap_coefficient(context, opt)
        scores.append(score)

    correct_score = scores[0]
    alt_scores = scores[1:]

    # Rank (1-based; rank=1 means correct answer has highest similarity)
    # Use deterministic random tie-breaking to avoid positional bias when
    # all options score identically (e.g. zero overlap with citation-only context).
    tie_breaker = _rng.Random(42)
    sorted_indices = sorted(
        range(len(scores)),
        key=lambda i: (scores[i], tie_breaker.random()),
        reverse=True,
    )
    rank = sorted_indices.index(0) + 1

    best_alt_score = max(alt_scores) if alt_scores else 0.0
    margin = correct_score - best_alt_score
    max_possible = max(correct_score, best_alt_score) if max(correct_score, best_alt_score) > 0 else 1.0
    normalized_margin = margin / max_possible

    mean_alt_score = float(np.mean(alt_scores)) if alt_scores else 0.0
    discrimination = correct_score - mean_alt_score

    return {
        "mcq_accuracy": 1.0 if rank == 1 else 0.0,
        "mcq_correct_rank": float(rank),
        "mcq_mrr": 1.0 / rank,
        "mcq_margin": margin,
        "mcq_normalized_margin": normalized_margin,
        "mcq_discrimination_score": discrimination,
    }


# ---------------------------------------------------------------------------
# 4. Golden ID Retrieval Metrics
# ---------------------------------------------------------------------------

def _average_precision(retrieved: list[str], gold_set: set[str]) -> float:
    """Compute Average Precision for a single query."""
    if not gold_set:
        return 1.0 if not retrieved else 0.0

    hits = 0
    sum_precision = 0.0
    for i, rid in enumerate(retrieved):
        if rid in gold_set:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / len(gold_set) if gold_set else 0.0


def _dcg(relevances: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    dcg_val = 0.0
    for i in range(min(k, len(relevances))):
        dcg_val += relevances[i] / math.log2(i + 2)
    return dcg_val


def _ndcg_at_k(retrieved: list[str], gold_set: set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    relevances = [1.0 if rid in gold_set else 0.0 for rid in retrieved[:k]]
    dcg = _dcg(relevances, k)

    ideal = sorted(relevances, reverse=True)
    n_gold = min(k, len(gold_set))
    ideal_relevances = [1.0] * n_gold + [0.0] * (k - n_gold)
    idcg = _dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def golden_id_metrics(
    retrieved_ids: list[str],
    golden_ids: list[str],
) -> dict[str, float]:
    """Comprehensive retrieval metrics over document IDs."""
    gold_set = set(golden_ids)
    ret_set = set(retrieved_ids)

    if not ret_set and not gold_set:
        precision, recall = 1.0, 1.0
    elif not ret_set:
        precision, recall = 0.0, 0.0
    elif not gold_set:
        precision, recall = 0.0, 1.0
    else:
        tp = len(ret_set & gold_set)
        precision = tp / len(ret_set)
        recall = tp / len(gold_set)

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Jaccard
    if not ret_set and not gold_set:
        jaccard = 1.0
    elif not ret_set or not gold_set:
        jaccard = 0.0
    else:
        jaccard = len(ret_set & gold_set) / len(ret_set | gold_set)

    # Hit@K
    hit_1 = 1.0 if any(rid in gold_set for rid in retrieved_ids[:1]) else 0.0
    hit_3 = 1.0 if any(rid in gold_set for rid in retrieved_ids[:3]) else 0.0
    hit_5 = 1.0 if any(rid in gold_set for rid in retrieved_ids[:5]) else 0.0

    # MRR
    mrr = 0.0
    for i, rid in enumerate(retrieved_ids):
        if rid in gold_set:
            mrr = 1.0 / (i + 1)
            break

    # MAP (single query = AP)
    ap = _average_precision(retrieved_ids, gold_set)

    # NDCG@K
    ndcg_3 = _ndcg_at_k(retrieved_ids, gold_set, 3)
    ndcg_5 = _ndcg_at_k(retrieved_ids, gold_set, 5)
    ndcg_10 = _ndcg_at_k(retrieved_ids, gold_set, 10)

    return {
        "id_precision": precision,
        "id_recall": recall,
        "id_f1": f1,
        "id_jaccard": jaccard,
        "id_hit_at_1": hit_1,
        "id_hit_at_3": hit_3,
        "id_hit_at_5": hit_5,
        "id_mrr": mrr,
        "id_map": ap,
        "id_ndcg_at_3": ndcg_3,
        "id_ndcg_at_5": ndcg_5,
        "id_ndcg_at_10": ndcg_10,
    }


# ---------------------------------------------------------------------------
# 5. Generator Quality Metrics (expected answer vs generated response)
# ---------------------------------------------------------------------------

def _bleu_score(prediction: str, reference: str, max_n: int = 4) -> float:
    """Compute corpus-level BLEU (single reference) up to max_n-grams."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ng = _ngrams(pred_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)

        if not pred_ng:
            precisions.append(0.0)
            continue

        pred_counter = Counter(pred_ng)
        ref_counter = Counter(ref_ng)
        clipped = sum((pred_counter & ref_counter).values())
        precisions.append(clipped / len(pred_ng))

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if pred_tokens else 0.0

    # Geometric mean of precisions (with smoothing for zero)
    log_avg = 0.0
    valid_n = 0
    for p in precisions:
        if p > 0:
            log_avg += math.log(p)
            valid_n += 1
        else:
            log_avg += math.log(1e-10)
            valid_n += 1

    if valid_n == 0:
        return 0.0
    return bp * math.exp(log_avg / valid_n)


def _meteor_score(prediction: str, reference: str) -> float:
    """Simplified METEOR: unigram precision/recall with harmonic mean (alpha=0.9)."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_set = Counter(pred_tokens)
    ref_set = Counter(ref_tokens)
    matches = sum((pred_set & ref_set).values())

    precision = matches / len(pred_tokens) if pred_tokens else 0.0
    recall = matches / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    # METEOR uses alpha=0.9 to weight recall more heavily
    alpha = 0.9
    score = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    return score


def generator_statistical_metrics(
    generated: str,
    expected: str,
) -> dict[str, float]:
    """Statistical/lexical metrics for generator evaluation."""
    generated = _sanitize(generated)
    expected = _sanitize(expected)

    gen_tokens = _tokenize(generated)
    exp_tokens = _tokenize(expected)

    # Token F1
    prec, rec, f1 = _token_overlap(gen_tokens, exp_tokens)

    # ROUGE
    rouge1 = _rouge_n(gen_tokens, exp_tokens, 1)
    rouge2 = _rouge_n(gen_tokens, exp_tokens, 2)
    rouge_l = _rouge_l(gen_tokens, exp_tokens)

    # BLEU
    bleu = _bleu_score(generated, expected)

    # METEOR
    meteor = _meteor_score(generated, expected)

    # Length ratio (verbosity)
    length_ratio = len(gen_tokens) / len(exp_tokens) if exp_tokens else 0.0

    return {
        "gen_token_precision": prec,
        "gen_token_recall": rec,
        "gen_token_f1": f1,
        "gen_rouge1_precision": rouge1["precision"],
        "gen_rouge1_recall": rouge1["recall"],
        "gen_rouge1_f1": rouge1["f1"],
        "gen_rouge2_precision": rouge2["precision"],
        "gen_rouge2_recall": rouge2["recall"],
        "gen_rouge2_f1": rouge2["f1"],
        "gen_rouge_l_precision": rouge_l["precision"],
        "gen_rouge_l_recall": rouge_l["recall"],
        "gen_rouge_l_f1": rouge_l["f1"],
        "gen_bleu": bleu,
        "gen_meteor": meteor,
        "gen_length_ratio": length_ratio,
    }


def generator_semantic_metrics(
    generated: str,
    expected: str,
    encoder: Any | None = None,
) -> dict[str, float]:
    """Semantic metrics comparing generated vs expected answer using embeddings.

    If encoder is None, tries to load sentence-transformers.
    """
    generated = _sanitize(generated)
    expected = _sanitize(expected)

    if not generated.strip() or not expected.strip():
        return {
            "gen_semantic_similarity": 0.0,
            "gen_bertscore_precision": 0.0,
            "gen_bertscore_recall": 0.0,
            "gen_bertscore_f1": 0.0,
        }

    if encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        except ImportError:
            return {
                "gen_semantic_similarity": float("nan"),
                "gen_bertscore_precision": float("nan"),
                "gen_bertscore_recall": float("nan"),
                "gen_bertscore_f1": float("nan"),
            }

    # Sentence-level cosine similarity
    embs = encoder.encode([generated, expected], convert_to_numpy=True)
    gen_emb, exp_emb = embs[0], embs[1]
    dot = float(np.dot(gen_emb, exp_emb))
    norm_g = float(np.linalg.norm(gen_emb))
    norm_e = float(np.linalg.norm(exp_emb))
    sentence_sim = dot / (norm_g * norm_e) if norm_g > 0 and norm_e > 0 else 0.0

    # BERTScore approximation: token-level embedding matching
    # Split into sentences for finer granularity
    gen_sents = [s.strip() for s in re.split(r'[.!?]+', generated) if s.strip()]
    exp_sents = [s.strip() for s in re.split(r'[.!?]+', expected) if s.strip()]

    if not gen_sents:
        gen_sents = [generated]
    if not exp_sents:
        exp_sents = [expected]

    gen_embs = encoder.encode(gen_sents, convert_to_numpy=True)
    exp_embs = encoder.encode(exp_sents, convert_to_numpy=True)

    # Cosine similarity matrix
    gen_norms = np.linalg.norm(gen_embs, axis=1, keepdims=True)
    exp_norms = np.linalg.norm(exp_embs, axis=1, keepdims=True)
    gen_normed = gen_embs / np.maximum(gen_norms, 1e-9)
    exp_normed = exp_embs / np.maximum(exp_norms, 1e-9)
    sim_matrix = gen_normed @ exp_normed.T

    # Precision: for each gen sentence, max similarity to any exp sentence
    bert_precision = float(np.mean(np.max(sim_matrix, axis=1)))
    # Recall: for each exp sentence, max similarity to any gen sentence
    bert_recall = float(np.mean(np.max(sim_matrix, axis=0)))
    bert_f1 = (
        2 * bert_precision * bert_recall / (bert_precision + bert_recall)
        if (bert_precision + bert_recall) > 0
        else 0.0
    )

    return {
        "gen_semantic_similarity": sentence_sim,
        "gen_bertscore_precision": bert_precision,
        "gen_bertscore_recall": bert_recall,
        "gen_bertscore_f1": bert_f1,
    }
