# Empirical Analysis of RAG Architectures for EU Regulatory Compliance QA

## Abstract

We evaluate eight retrieval-augmented generation (RAG) pipeline configurations on a 200-question benchmark spanning the EU AI Act and Digital Services Act. Systems range from sparse lexical baselines (BM25, Neo4j full-text) through dense semantic retrieval (all-MiniLM-L6-v2), to graph-based agentic reasoning (CORTEX) and an LLM-as-judge self-refinement pipeline (Judge RAG). Our analysis reveals several critical findings: (1) lexical retrieval consistently outperforms a general-purpose dense encoder on domain-specific legal text; (2) the CORTEX pipeline's reported context efficiency metrics are artifacts of measuring citation strings rather than actual retrieved content; (3) MCQ-based retrieval evaluation is vulnerable to degenerate tie-breaking when context is minimal; and (4) aggressive graph pruning destroys paragraph-level retrieval granularity that the critic component alone preserves. We detail these findings and flag methodological concerns that must be addressed before publication.

---

## 1. Experimental Setup

**Dataset.** 200 questions drawn from the EU AI Act and Digital Services Act, each annotated with golden document IDs, correct article/paragraph references, a ground-truth answer, and four distractor alternatives for multiple-choice evaluation.

**Systems evaluated.** Eight configurations forming a controlled ablation study:

| System | Retrieval | Pruning | Critic | Generation |
|--------|-----------|---------|--------|------------|
| Naive | Neo4j full-text | — | — | None |
| BM25 | rank_bm25 | — | — | None |
| Dense | all-MiniLM-L6-v2 | — | — | None |
| Advanced | KG + graph traversal | Off | Off | LLM synthesis |
| Cortex Pruner Only | KG + graph traversal | On | Off | LLM synthesis |
| Cortex Critic Only | KG + graph traversal | Off | On | LLM synthesis |
| Cortex (full) | KG + graph traversal | On | On | LLM synthesis |
| Judge RAG | BM25 + semantic hybrid | — | LLM judge | llama3.1:8b + phi-4 judge |

**Models.** Dense retrieval uses `all-MiniLM-L6-v2`. Judge RAG uses `llama3.1:8b` for retrieval/generation and `phi4:latest` for the judge. CORTEX variants use semantic pruning (threshold = 0.45) and graph re-traversal.

---

## 2. Results

### 2.1 Retrieval Source Identification

How accurately each pipeline identifies the correct regulation, article, and paragraph.

| System | Doc Acc. ↑ | Art. Acc. ↑ | Doc Hit ↑ | Art. Hit ↑ | Para. Hit ↑ |
|--------|:----------:|:-----------:|:---------:|:----------:|:-----------:|
| Naive | 0.889 | 0.171 | 1.000 | 0.850 | 0.000 |
| BM25 | 0.877 | 0.163 | 1.000 | 0.810 | 0.000 |
| Dense | 0.923 | 0.148 | 0.980 | 0.740 | 0.000 |
| Advanced | 0.903 | 0.209 | 1.000 | 0.850 | 0.055 |
| Cortex Pruner Only | 0.867 | 0.284 | 0.885 | 0.730 | 0.050 |
| Cortex Critic Only | 0.923 | 0.237 | 1.000 | 0.875 | **0.840** |
| Cortex (full) | 0.870 | 0.246 | 0.885 | 0.765 | 0.095 |
| Judge RAG | **1.000** | 0.154 | 1.000 | 0.635 | 0.565 |

### 2.2 Golden ID Retrieval

Precision, recall, and ranking metrics over ground-truth document IDs.

| System | Precision ↑ | Recall ↑ | F1 ↑ | Hit@5 ↑ | MRR ↑ | MAP ↑ |
|--------|:-----------:|:--------:|:----:|:-------:|:-----:|:-----:|
| Naive | 0.124 | 0.243 | 0.154 | 0.545 | 0.453 | 0.200 |
| BM25 | 0.122 | 0.238 | 0.151 | 0.535 | 0.450 | 0.201 |
| Dense | 0.106 | 0.205 | 0.130 | 0.485 | 0.378 | 0.159 |
| Advanced | 0.124 | 0.243 | 0.154 | 0.545 | 0.452 | 0.199 |
| Cortex Pruner Only | **0.167** | 0.214 | **0.166** | 0.490 | 0.376 | 0.163 |
| Cortex Critic Only | 0.045 | **0.285** | 0.074 | 0.545 | **0.454** | **0.205** |
| Cortex (full) | 0.114 | 0.231 | 0.134 | 0.475 | 0.336 | 0.150 |
| Judge RAG | 0.070 | 0.251 | 0.104 | 0.355 | 0.248 | 0.135 |

### 2.3 Retrieval Text Quality

Overlap between retrieved context and the ground-truth correct answer.

| System | Token Recall ↑ | ROUGE-L F1 ↑ | Exact Contain. ↑ | Substr. Contain. ↑ |
|--------|:--------------:|:------------:|:-----------------:|:-------------------:|
| Naive | 0.824 | 0.022 | 0.065 | 0.311 |
| BM25 | **0.828** | 0.019 | **0.070** | **0.317** |
| Dense | 0.770 | **0.026** | 0.060 | 0.297 |
| Advanced | 0.002 | 0.002 | 0.000 | 0.023 |
| Cortex Pruner Only | 0.002 | 0.002 | 0.000 | 0.021 |
| Cortex Critic Only | 0.005 | 0.003 | 0.000 | 0.024 |
| Cortex (full) | 0.005 | 0.005 | 0.000 | 0.021 |
| Judge RAG | 0.759 | 0.023 | 0.040 | 0.270 |

### 2.4 MCQ Retrieval Discrimination

Multiple-choice accuracy: can the retrieved context distinguish the correct answer from distractors?

| System | MCQ Accuracy ↑ | MCQ MRR ↑ |
|--------|:--------------:|:---------:|
| Naive | 0.940 | 0.963 |
| BM25 | 0.925 | 0.953 |
| Dense | 0.880 | 0.923 |
| Advanced | **0.990** ⚠️ | **0.995** ⚠️ |
| Cortex Pruner Only | 0.885 | 0.888 |
| Cortex Critic Only | 0.900 | 0.946 |
| Cortex (full) | 0.825 | 0.855 |
| Judge RAG | 0.905 | 0.941 |

### 2.5 Context Efficiency

| System | Avg Context Tokens | Token Efficiency Ratio ↓ |
|--------|:------------------:|:------------------------:|
| Naive | 2669 | — (reference) |
| BM25 | 3065 | 1.215 |
| Dense | 2105 | 0.872 |
| Advanced | **10** ⚠️ | 0.004 ⚠️ |
| Cortex Pruner Only | **7** ⚠️ | 0.003 ⚠️ |
| Cortex Critic Only | **63** ⚠️ | 0.026 ⚠️ |
| Cortex (full) | **16** ⚠️ | 0.007 ⚠️ |
| Judge RAG | 2163 | 0.927 |

### 2.6 Generator Quality (Generative Systems Only)

| System | Token F1 ↑ | ROUGE-L F1 ↑ | BLEU ↑ | METEOR ↑ | Sem. Sim. ↑ | BERTScore F1 ↑ | Len. Ratio |
|--------|:----------:|:------------:|:------:|:--------:|:-----------:|:--------------:|:----------:|
| Advanced | 0.086 | 0.062 | 0.006 | 0.230 | 0.547 | 0.431 | 14.2 |
| Cortex Pruner Only | 0.089 | 0.067 | 0.008 | 0.217 | 0.535 | 0.420 | 10.5 |
| Cortex Critic Only | 0.086 | 0.062 | 0.006 | 0.230 | 0.547 | 0.431 | 14.2 |
| Cortex (full) | 0.099 | 0.080 | 0.008 | 0.197 | 0.521 | 0.407 | 6.1 |
| **Judge RAG** | **0.197** | **0.160** | **0.030** | **0.331** | **0.678** | **0.571** | 6.9 |

### 2.7 Judge RAG — LLM-as-Judge Self-Assessment

| Metric | Mean Score (1–10) |
|--------|:-----------------:|
| Overall LLM Score | 8.25 |
| Relevance | 8.75 |
| Faithfulness | 8.14 |
| Completeness | 6.79 |
| Precision (Citation Accuracy) | 9.41 |
| Avg Attempts Used | 1.21 |
| Retrieval Similarity | 0.536 |

**Uncertainty distribution** (n=200): Low: 147 (73.5%), Medium: 44 (22.0%), High: 9 (4.5%).

---

## 3. Critical Anomalies and Methodological Concerns

### 3.1 CORTEX Context Tokens Measure Citation Strings, Not Actual Context

**Severity: Critical — invalidates efficiency claims.**

The reported context token counts for all four CORTEX variants (Advanced, Pruner Only, Critic Only, Full) are measuring the *length of citation reference strings*, not the actual text content fed to the synthesis LLM. Inspection of the pipeline reveals that `run_cortex_engine()` returns a `citations` field containing formatted strings of the form `"{node_id} ({REGULATION})"` (e.g., `"eu_ai_act_art_5 (EU_AI_ACT)"`), but does not expose the full retrieved text under the keys checked by `_extract_retrieved_context()` (`retrieved_context`, `metrics.retrieved_context`, `context`). The fallback therefore concatenates only citation strings.

This explains the impossibly low context token counts:
- Advanced: **10.0 tokens** — approximately 2 citation strings
- Cortex (full): **16.3 tokens** — approximately 3 citation strings
- Cortex Pruner Only: **6.6 tokens** — approximately 1–2 citation strings

The actual context processed by the synthesis LLM (in `synthesis_node`) is drawn from `state["pruned_context"]`, which contains full article/recital text typically spanning hundreds to thousands of tokens. **The token efficiency ratio, context token counts, and all retrieval text quality metrics for CORTEX models are therefore invalid and should not be reported as-is.**

**Impact on other metrics:** This cascading failure also invalidates retrieval text quality metrics (Table 2.3) for CORTEX models, since token recall, ROUGE-L, and containment are computed against these citation strings rather than actual retrieved text.

### 3.2 MCQ Accuracy Is Artificially Inflated for CORTEX Models

**Severity: High — inflates a headline metric.**

Advanced RAG achieves 0.990 MCQ accuracy despite having only ~10 tokens of context (citation strings). Investigation of the MCQ metric computation reveals a degenerate tie-breaking scenario:

1. The MCQ metric computes an overlap coefficient between retrieved context tokens and each answer option.
2. Citation strings like `"eu_ai_act_art_5 (EU_AI_ACT)"` yield tokens `["euaiactart5", "euaiact"]` after punctuation removal (since `string.punctuation` includes underscores).
3. These concatenated tokens have **zero overlap** with all answer option tokens.
4. When all options score 0, Python's stable sort preserves the original order.
5. Since the correct answer is always placed at index 0 in the candidate list, it always receives rank 1.

Result: MCQ accuracy of ~1.0 is an artifact of index ordering, not retrieval quality. The 0.990 (not 1.0) arises from rare edge cases where a citation token coincidentally overlaps with an answer token. Similarly, Cortex Pruner Only's lower MCQ (0.885) reflects cases where pruning eliminates all context, triggering the empty-context early return (`mcq_accuracy = 0.0`).

**Recommendation:** MCQ accuracy should be excluded from the CORTEX evaluation, or the metric should add random shuffling of answer options before ranking.

### 3.3 Advanced and Cortex Critic Only Produce Identical Generator Outputs

**Severity: High — undermines ablation analysis.**

Advanced (pruning=off, critic=off) and Cortex Critic Only (pruning=off, critic=on) yield **byte-identical** generator metrics:

| Metric | Advanced | Cortex Critic Only |
|--------|:--------:|:------------------:|
| gen_token_f1 | 0.08613 | 0.08613 |
| gen_rouge_l_f1 | 0.06237 | 0.06237 |
| gen_bleu | 0.00632 | 0.00632 |
| gen_meteor | 0.22984 | 0.22984 |
| gen_length_ratio | 14.219 | 14.219 |
| gen_semantic_similarity | 0.54711 | 0.54711 |
| gen_bertscore_f1 | 0.43085 | 0.43085 |

These values match to the last decimal place across all 200 questions. Two possible explanations:

1. **The critic loop does not alter the final synthesis input.** If `critic_node()` marks `is_accurate = True` on the first pass (before triggering re-traversal), the synthesis LLM receives the same pruned context as the no-critic case. Since pruning is also off in both, the entire pipeline reduces to the same sequence of operations.
2. **Generator evaluation uses the same generated text.** If the synthesis LLM is deterministic (temperature=0) and receives the same context, outputs will be identical.

Examining the orchestrator code confirms explanation (1): when the critic calls `self_correction_router` and returns `"generate_final_answer"` on the first iteration, no re-traversal occurs, making the critic a no-op. The critic only triggers additional hops when it flags inaccuracies — which evidently never produces different *synthesis inputs* in the Advanced-equivalent case.

**Implication:** The critic component alone (without pruning) does not alter generation quality. The ablation column "Cortex Critic Only" adds no information beyond "Advanced" for generator metrics. However, note that critic *does* affect retrieval metrics (see Section 3.4), suggesting its value lies in expanding retrieval coverage rather than improving answer quality.

### 3.4 Pruning Destroys Paragraph-Level Retrieval Granularity

**Finding: Important for EMNLP discussion.**

Paragraph-level hit rates reveal a dramatic interaction between pruning and the critic:

| System | Pruning | Critic | Para. Hit |
|--------|:-------:|:------:|:---------:|
| Advanced | Off | Off | 0.055 |
| Cortex Pruner Only | On | Off | 0.050 |
| Cortex Critic Only | Off | On | **0.840** |
| Cortex (full) | On | On | 0.095 |

The critic component, through its re-traversal mechanism, discovers paragraph-level nodes (e.g., `eu_ai_act_art_5_par_1`) by traversing graph edges from article nodes to their paragraph children. Without pruning, these fine-grained nodes are retained, yielding 0.84 paragraph hit rate. However, when pruning is enabled (threshold=0.45), the semantic similarity between the user query and paragraph-level text frequently falls below the threshold, causing these nodes to be pruned.

This represents a **precision–granularity tradeoff**: pruning improves ID precision (0.167 for Pruner Only vs 0.045 for Critic Only) but destroys paragraph-level recall. The full Cortex pipeline inherits the worst of both: lower paragraph hit than either ablation alone.

### 3.5 Advanced and Naive Share Identical Retrieval ID Metrics

Advanced RAG (pruning=off, critic=off) produces retrieval ID metrics nearly identical to the Naive baseline:

| Metric | Naive | Advanced |
|--------|:-----:|:--------:|
| id_precision | 0.124 | 0.124 |
| id_recall | 0.243 | 0.243 |
| id_f1 | 0.154 | 0.154 |
| id_hit_at_5 | 0.545 | 0.545 |

This is expected: both use Neo4j full-text search as the initial retrieval step. Without pruning or critic-driven re-traversal, Advanced merely wraps the same retrieval in a graph traversal framework that adds no new document IDs. The only difference is that Advanced proceeds to generate an answer, while Naive is retrieval-only. This confirms that **the CORTEX framework adds no retrieval value without its pruning and critic components**.

---

## 4. Cross-System Analysis

### 4.1 Lexical Retrieval Outperforms Dense Embedding on Legal Text

Across nearly all retrieval metrics, BM25 and the Naive (Neo4j full-text) baseline outperform the Dense embedding baseline (all-MiniLM-L6-v2):

| Metric | Naive | BM25 | Dense |
|--------|:-----:|:----:|:-----:|
| id_recall | 0.243 | 0.238 | 0.205 |
| id_hit_at_5 | 0.545 | 0.535 | 0.485 |
| id_mrr | 0.453 | 0.450 | 0.378 |
| MCQ accuracy | 0.940 | 0.925 | 0.880 |
| Token recall | 0.824 | 0.828 | 0.770 |

The one exception is document-level accuracy (Dense: 0.923 > BM25: 0.877), where Dense's semantic understanding of document-level topics compensates for its weaker fine-grained matching. This finding is consistent with prior work showing that general-purpose sentence encoders trained on informal text (Reddit, Wikipedia) underperform term-matching methods on domain-specific legal language with precise technical vocabulary (Thakur et al., 2021; BEIR benchmark). Legal text features highly specific terminology (e.g., "placing on the market," "conformity assessment") where exact lexical match is more reliable than distributional semantics from a general-domain encoder.

**Recommendation for EMNLP:** This result motivates domain-adapted dense encoders (e.g., fine-tuned on legal corpora) as a potentially stronger baseline. The current Dense baseline is not a strong representative of semantic retrieval capabilities.

### 4.2 Judge RAG: Strong Generation, Weak Article-Level Retrieval

Judge RAG achieves the best generator quality across all metrics (Token F1: 0.197, Semantic Similarity: 0.678, BERTScore F1: 0.571) while maintaining perfect document-level accuracy (1.000). However, it has the **worst article-level accuracy** (0.154) and **lowest ID precision** (0.070) among all systems.

This pattern suggests a **granularity mismatch**: Judge RAG's chunking strategy produces chunks that reliably identify the correct regulation but frequently span or misalign with article boundaries. The hybrid BM25+semantic retrieval retrieves topically relevant passages without precise article-level provenance.

The LLM-as-judge self-refinement loop effectively compensates for retrieval imprecision: the judge's Precision score (9.41/10, measuring citation accuracy) indicates the LLM can infer correct article references from chunk content even when chunk IDs don't directly encode article boundaries.

**Key tension for EMNLP:** Judge RAG's architecture produces the most human-useful answers but the least structurally precise retrieval — a tradeoff between end-to-end answer quality and component-level transparency.

### 4.3 Generation Quality Is Uniformly Low Across CORTEX Variants

All CORTEX variants produce Token F1 scores between 0.086–0.099, BLEU scores below 0.01, and ROUGE-L F1 below 0.08. Even semantic similarity hovers around 0.52–0.55. These scores, while not strong in absolute terms, must be interpreted carefully:

1. **Length ratio inflation.** CORTEX variants produce answers 6–14× longer than reference answers, which depresses precision-oriented metrics (BLEU, token precision). The synthesis LLM generates expansive, explanatory answers rather than concise extracts.
2. **Semantic similarity is moderate.** Values around 0.52–0.55 indicate the generated answers are topically related but lexically divergent from reference answers. This may reflect a valid (but different) answer formulation rather than failure.
3. **Judge RAG significantly outperforms** (Token F1: 0.197, 2× the best CORTEX variant) owing to its iterative refinement loop that aligns outputs more closely with expected answer patterns.

### 4.4 Ablation Summary: Value of Each CORTEX Component

| Component | Effect on Retrieval | Effect on Generation |
|-----------|:-------------------:|:--------------------:|
| Graph traversal alone (Advanced) | No improvement over Naive baseline | Produces answers (F1=0.086) |
| + Pruning only | ↑ ID precision (0.124→0.167), ↓ recall, ↓ paragraph coverage | Marginal improvement (F1=0.089) |
| + Critic only | ↑ Paragraph hit (0.055→0.840), ↓ ID precision (0.124→0.045) | No improvement (F1=0.086)† |
| + Both (full Cortex) | Mixed: slight ↑ article accuracy, ↓ paragraph hit vs critic-only | Best CORTEX F1 (0.099) |

†Identical to Advanced due to critic being a no-op for generation when pruning is disabled (see Section 3.3).

---

## 5. Recommendations Before Publication

### 5.1 Must Fix (Blocking) — RESOLVED

1. **~~Fix CORTEX context extraction.~~** ✅ Fixed: `run_cortex_engine()` now builds `retrieved_context` from actual `state["pruned_context"]` node content and includes it in the return dict.

2. **~~Fix MCQ tie-breaking.~~** ✅ Fixed: `mcq_retrieval_metrics()` now uses deterministic random tie-breaking (`Random(42)`) to prevent positional bias when all overlap scores tie.

3. **Re-evaluate after fixes.** ⬜ Re-run the full evaluation to regenerate all metric CSVs with corrected context extraction and MCQ scoring:
   ```bash
   # Model 1: llama3.1:8b (baseline — reproduces prior results with bug fixes)
   python -m baselines.evaluation.run_eval \
     --models naive,bm25,dense,dense-legal,advanced,cortex-pruner-only,cortex-critic-only,cortex,judge \
     --cortex-synthesis-model llama3.1:8b \
     --judge-rag-generation-model llama3.1:8b \
     --judge-rag-judge-model phi4:latest \
     --artifact-dir artifacts/eval_llama31_8b

   # Model 2: phi4 (CORTEX + Judge generation)
   python -m baselines.evaluation.run_eval \
     --models naive,bm25,dense,dense-legal,advanced,cortex-pruner-only,cortex-critic-only,cortex,judge \
     --cortex-synthesis-model phi4:latest \
     --judge-rag-generation-model phi4:latest \
     --judge-rag-judge-model gemma3:latest \
     --artifact-dir artifacts/eval_phi4

   # Model 3: gemma4 (CORTEX + Judge generation)
   ollama pull gemma4:latest
   python -m baselines.evaluation.run_eval \
     --models naive,bm25,dense,dense-legal,advanced,cortex-pruner-only,cortex-critic-only,cortex,judge \
     --cortex-synthesis-model gemma4:latest \
     --judge-rag-generation-model gemma4:latest \
     --judge-rag-judge-model gemma3:latest \
     --artifact-dir artifacts/eval_gemma4
   ```

### 5.2 Should Fix (Strengthening)

4. **~~Add statistical significance tests.~~** ✅ Implemented: Paired bootstrap confidence intervals (10,000 iterations, seed=42) are now computed automatically and saved to `significance_tests.csv` for key metric comparisons (BM25 vs Dense, Cortex vs Naive, Cortex vs Judge, etc.).

5. **~~Include a domain-adapted dense baseline.~~** ✅ Implemented: `Dense_Legal` model added using `law-ai/InLegalBERT`. Run with `--models dense-legal` or include in the comma list. The `--dense-embedding-model` CLI arg also allows swapping the default Dense encoder.

6. **~~Evaluate CORTEX with a stronger synthesis LLM.~~** ✅ Implemented: CORTEX synthesis is now LLM-based via Ollama (configurable via `--cortex-synthesis-model`). The prior template-based synthesis (which concatenated snippets without an LLM) has been replaced. Three model configurations are planned: llama3.1:8b (baseline), phi4, and gemma4:latest.

7. **~~Report Cortex Critic Only generation metrics separately.~~** ✅ Resolved: The old template synthesis used only `top_nodes[:3]`, making the critic's re-traversal (which appends neighbor nodes at the end of the list) invisible to generation. The new LLM-based synthesis uses ALL pruned context, so critic-triggered re-traversals now produce different synthesis inputs — breaking the byte-identical equivalence with Advanced.

### 5.3 Presentation Suggestions for EMNLP

8. **Use radar/spider charts** for multi-metric comparison across systems, grouping metrics by retrieval precision, retrieval coverage, generation quality, and efficiency.

9. **Focus the narrative** on the precision–granularity tradeoff (Section 3.4) and the lexical-vs-dense finding (Section 4.1) — these are the most publishable insights.

10. **Frame Judge RAG's tradeoff** (best generation, worst article retrieval) as evidence for the hypothesis that end-to-end quality and component-level interpretability are in tension in RAG systems.

---

## 6. Conclusions

This evaluation reveals that the CORTEX knowledge-graph-based RAG pipeline, in its current implementation, does not consistently outperform simple lexical baselines on retrieval quality. Its primary contribution — graph-structured reasoning with semantic pruning — introduces a precision–granularity tradeoff where paragraph-level coverage is sacrificed for node-level precision. Meanwhile, the Judge RAG pipeline demonstrates that iterative self-refinement can compensate for imprecise retrieval, achieving the strongest generation quality despite mediocre article-level retrieval.

However, **three critical measurement errors** (context extraction, MCQ tie-breaking, and identical critic-only/advanced outputs) have been identified and resolved (see Section 5). The evaluation must be re-run to produce corrected numbers before they can be reported in a peer-reviewed venue.

The finding that BM25 outperforms all-MiniLM-L6-v2 on legal QA is consistent with prior BEIR benchmarks and motivates the use of domain-adapted encoders in future work. The paragraph-level precision–granularity tradeoff introduced by graph pruning represents a novel and publishable finding, particularly if contextualized within the legal AI literature on provenance and citation accuracy.

---

## Appendix A: Full Metric Definitions

| Metric | Definition | Range |
|--------|-----------|-------|
| Doc/Article Accuracy | Fraction of retrieved IDs matching the correct document/article | 0–1 |
| Doc/Article/Para Hit | Binary: at least one retrieved ID matches (at respective level) | 0 or 1 |
| ID Precision | \|retrieved ∩ gold\| / \|retrieved\| | 0–1 |
| ID Recall | \|retrieved ∩ gold\| / \|gold\| | 0–1 |
| ID F1 | Harmonic mean of ID Precision and ID Recall | 0–1 |
| Hit@5 | At least one gold ID in top-5 retrieved | 0 or 1 |
| MRR | Reciprocal rank of first relevant retrieved ID | 0–1 |
| MAP | Mean Average Precision (per-query AP averaged over dataset) | 0–1 |
| Token Recall | Token overlap recall between retrieved context and gold answer | 0–1 |
| ROUGE-L F1 | F1 based on longest common subsequence | 0–1 |
| Exact Containment | Gold answer appears as complete substring in context | 0 or 1 |
| Substring Containment | Fraction of gold answer appearing as contiguous match in context | 0–1 |
| MCQ Accuracy | Retrieved context ranks correct answer first among 5 options | 0 or 1 |
| MCQ MRR | Reciprocal rank of correct answer in overlap-based ranking | 0–1 |
| Context Tokens | Whitespace-delimited token count of retrieved context | ≥ 0 |
| Token Efficiency Ratio | Model context tokens / Naive context tokens (per-question avg) | ≥ 0 |
| Gen Token F1 | Token-level F1 between generated and expected answer | 0–1 |
| Gen ROUGE-L F1 | Longest common subsequence F1 (generated vs expected) | 0–1 |
| Gen BLEU | Corpus BLEU-4 (generated vs expected) | 0–1 |
| Gen METEOR | Simplified METEOR (α=0.9 recall weighting) | 0–1 |
| Gen Semantic Similarity | Cosine similarity of sentence embeddings (generated vs expected) | 0–1 |
| Gen BERTScore F1 | Sentence-level embedding matching F1 (approximated via all-MiniLM-L6-v2) | 0–1 |
| Length Ratio | \|generated tokens\| / \|expected tokens\| | ≥ 0 |

## Appendix B: Evaluation Configuration

```json
{
  "total_questions": 200,
  "embedding_model": "all-MiniLM-L6-v2",
  "judge_rag": {
    "retrieval_model": "llama3.1:8b",
    "generation_model": "llama3.1:8b",
    "judge_model": "phi4:latest"
  },
  "cortex_pruning_threshold": 0.45,
  "neo4j_uri": "bolt://localhost:7687"
}
```

---

*Evaluation timestamp: 2026-04-05T01:04:52 UTC*
