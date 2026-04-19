"""
Enrich extracted EU regulation JSON files with chapter- and article-level
contextual metadata designed for downstream agent consumption.

Context-engineering approach
----------------------------
Each chapter and article receives a ``metadata`` dict with fields optimised for
three agent workflows:

  * **Retrieval** - ``key_topics``, ``applies_to``, ``regulatory_action`` let
    vector-search and graph-query agents narrow the search space fast.
  * **Reasoning** - ``summary`` and ``key_obligations`` give a compressed
    snapshot an LLM can ground its chain-of-thought on without re-reading entire
    articles.
  * **Navigation** - ``cross_references`` and structural counters
    (``article_count``, ``paragraph_count``) help agents traverse the document
    hierarchy.

Usage
-----
    # Best quality - uses local Ollama (needs `ollama serve` running)
    python enrich_metadata.py --provider ollama

    # Local / offline - extractive heuristics, no API needed
    python enrich_metadata.py --provider extractive

    # Specify a single file instead of all files in data/
    python enrich_metadata.py --provider ollama --file ../data/ai_act_extracted.json

    # Use a different Ollama model or host
    python enrich_metadata.py --provider ollama --model gemma3:4b --ollama-url http://localhost:11434
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _concat_paragraphs(paragraphs: list[dict]) -> str:
    """Join all paragraph texts into a single string."""
    return " ".join(p.get("text", "") for p in paragraphs if p.get("text"))


def _first_n_words(text: str, n: int = 300) -> str:
    words = text.split()
    return " ".join(words[:n])


def _extract_cross_refs(text: str) -> list[str]:
    """Pull explicit Article / Annex / Chapter references from text."""
    patterns = [
        r"Article\s+\d+(?:\(\d+\))?",
        r"Annex\s+[IVXLCDM]+",
        r"Chapter\s+[IVXLCDM]+",
        r"Section\s+\d+",
    ]
    refs: set[str] = set()
    for pat in patterns:
        refs.update(re.findall(pat, text, re.IGNORECASE))
    return sorted(refs)


_OBLIGATION_VERBS = re.compile(
    r"\bshall\b|\bmust\b|\bprohibited\b|\brequire[ds]?\b"
    r"|\boblige[ds]?\b|\bensure\b|\bprohibit\b",
    re.IGNORECASE,
)


def _extract_obligation_sentences(text: str, max_count: int = 5) -> list[str]:
    """Return sentences that contain obligation/prohibition language."""
    sentences = re.split(r"(?<=[.;])\s+", text)
    hits: list[str] = []
    for s in sentences:
        if _OBLIGATION_VERBS.search(s):
            hits.append(s.strip())
            if len(hits) >= max_count:
                break
    return hits


_ENTITY_PATTERNS = re.compile(
    r"\b(?:provider[s]?|deployer[s]?|importer[s]?|distributor[s]?"
    r"|authorised representative[s]?"
    r"|notified bod(?:y|ies)|market surveillance authorit(?:y|ies)"
    r"|national competent authorit(?:y|ies)|Commission"
    r"|Member State[s]?|AI Office|Board"
    r"|user[s]?|natural person[s]?|operator[s]?"
    r"|online platform[s]?|intermediary service[s]?"
    r"|hosting service[s]?|search engine[s]?"
    r"|very large online platform[s]?"
    r"|recipient[s]? of the service"
    r"|trader[s]?|consumer[s]?)\b",
    re.IGNORECASE,
)


def _extract_entities(text: str) -> list[str]:
    """Return de-duplicated entity mentions found in text."""
    found = _ENTITY_PATTERNS.findall(text)
    # Normalise to lower then title-case
    seen: set[str] = set()
    result: list[str] = []
    for e in found:
        key = e.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(key.title())
    return result


_ACTION_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bprohibit", re.I), "prohibits"),
    (re.compile(r"\bdefin(?:e[sd]?|ition)", re.I), "defines"),
    (re.compile(r"\bshall\b.*\bensure\b", re.I), "requires"),
    (re.compile(r"\bshall\b", re.I), "requires"),
    (re.compile(r"\bobligation", re.I), "establishes_obligations"),
    (re.compile(r"\bright[s]?\b", re.I), "establishes_rights"),
    (re.compile(r"\bpenalt|sanction|fine", re.I), "establishes_penalties"),
    (re.compile(r"\bexempt|derogate|except", re.I), "grants_exemptions"),
    (re.compile(r"\btransparenc", re.I), "transparency_requirement"),
    (re.compile(r"\bgovernance|supervision|oversight", re.I), "governance"),
]


def _classify_action(text: str) -> list[str]:
    actions: list[str] = []
    for pat, label in _ACTION_MAP:
        if pat.search(text):
            actions.append(label)
    return actions or ["general_provision"]


# ---------------------------------------------------------------------------
# Extractive summariser (no external API)
# ---------------------------------------------------------------------------


class ExtractiveSummariser:
    """Builds metadata using only regex heuristics - works offline."""

    def summarise_article(self, article: dict) -> dict:
        full_text = _concat_paragraphs(article.get("paragraphs", []))
        title = article.get("title", "")
        obligations = _extract_obligation_sentences(full_text)

        summary_source = full_text or title
        summary = _first_n_words(summary_source, 60)
        if len(summary_source.split()) > 60:
            summary += " …"

        return {
            "summary": summary,
            "key_obligations": obligations,
            "key_topics": _derive_topics(title, full_text),
            "applies_to": _extract_entities(full_text),
            "cross_references": _extract_cross_refs(full_text),
            "regulatory_action": _classify_action(full_text),
            "paragraph_count": len(article.get("paragraphs", [])),
        }

    def summarise_chapter(self, chapter: dict, article_metas: list[dict]) -> dict:
        title = chapter.get("title", "")
        all_topics: list[str] = []
        all_entities: list[str] = []
        all_actions: list[str] = []
        for m in article_metas:
            all_topics.extend(m.get("key_topics", []))
            all_entities.extend(m.get("applies_to", []))
            all_actions.extend(m.get("regulatory_action", []))

        return {
            "summary": f"Chapter {chapter.get('number', '?')} - {title}. "
            f"Contains {len(article_metas)} article(s) covering: "
            + ", ".join(_dedup(all_topics)[:8])
            + ".",
            "key_themes": _dedup(all_topics)[:12],
            "scope_description": "Applies to: " + ", ".join(_dedup(all_entities)[:8]),
            "regulatory_functions": _dedup(all_actions),
            "article_count": len(article_metas),
        }


def _derive_topics(title: str, text: str) -> list[str]:
    """Cheap keyword extraction: noun-phrase-ish tokens that recur."""
    blob = (title + " " + text).lower()
    # Remove very short / stop words
    stops = {
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "from",
        "which",
        "shall",
        "such",
        "not",
        "any",
        "are",
        "its",
        "has",
        "been",
        "may",
        "all",
        "can",
        "will",
        "but",
        "into",
        "than",
        "other",
        "those",
        "where",
        "their",
        "they",
        "also",
        "each",
        "have",
        "who",
        "when",
        "upon",
        "under",
        "about",
        "more",
        "only",
        "referred",
        "paragraph",
        "article",
        "regulation",
        "union",
        "member",
        "states",
        "pursuant",
        "accordance",
        "within",
        "respect",
        "point",
        "provided",
        "including",
        "without",
        "set",
        "laid",
        "down",
        "should",
        "would",
        "could",
        "does",
        "did",
        "being",
        "having",
        "before",
        "after",
        "between",
        "through",
        "during",
        "following",
        "applicable",
        "relevant",
    }
    words = re.findall(r"[a-z]{4,}", blob)
    freq: dict[str, int] = {}
    for w in words:
        if w not in stops:
            freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq, key=lambda w: freq[w], reverse=True)
    return ranked[:10]


def _dedup(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Ollama-powered summariser  (local LLM - no API key needed)
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434"
# qwen2.5:7b - best balance of quality / speed / VRAM for structured JSON
# extraction on a laptop GPU like the RTX 4070 (12 GB).  Alternatives:
#   gemma3:4b   - faster, slightly lower quality
#   llama3.1:8b - similar quality, slightly more VRAM
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


class OllamaSummariser:
    """Uses a local Ollama instance for high-quality summaries."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._extractive = ExtractiveSummariser()
        self._verify_server()

    def _verify_server(self) -> None:
        """Check that Ollama is reachable and the model is available."""
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            names = [m["name"] for m in data.get("models", [])]
            # Match with or without tag suffix (e.g. "qwen2.5:7b" or "qwen2.5:7b-instruct")
            base = self._model.split(":")[0]
            if not any(base in n for n in names):
                log.warning(
                    "Model '%s' not found locally. Available: %s. "
                    "Ollama will attempt to pull it on first use.",
                    self._model,
                    ", ".join(names[:10]),
                )
        except Exception as exc:
            log.error(
                "Cannot reach Ollama at %s (%s). Make sure 'ollama serve' is running.",
                self._base_url,
                exc,
            )
            sys.exit(1)

    # -- helpers --------------------------------------------------------
    def _call(self, system: str, user: str) -> str:
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1024,
                },
            }
        ).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read())
                return body["message"]["content"].strip()
            except Exception as exc:
                wait = 2**attempt
                log.warning("Ollama call failed (%s), retrying in %ss …", exc, wait)
                time.sleep(wait)
        log.error("Ollama call failed after retries - falling back to extractive.")
        return ""

    def _parse_json_response(self, raw: str) -> dict:
        """Best-effort parse of an LLM JSON response."""
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning("Could not parse LLM JSON - using extractive fallback.")
            return {}

    # -- public API -----------------------------------------------------
    def summarise_article(self, article: dict) -> dict:
        full_text = _concat_paragraphs(article.get("paragraphs", []))
        title = article.get("title", "")
        # Keep prompt within context window - truncate for 8k ctx models
        truncated = _first_n_words(full_text, 1200)

        system = textwrap.dedent("""\
            You are a specialist in EU regulatory law. You produce concise,
            structured metadata for legal articles that will be consumed by
            AI agents for retrieval-augmented generation.

            Return ONLY valid JSON with these keys:
            - "summary": 1-3 sentence plain-language summary
            - "key_obligations": list of the main obligations/requirements (max 5, short phrases)
            - "key_topics": list of 5-10 topic keywords
            - "applies_to": list of entities/stakeholders this article affects
            - "regulatory_action": list from [defines, requires, prohibits,
              establishes_obligations, establishes_rights, establishes_penalties,
              grants_exemptions, transparency_requirement, governance, general_provision]
        """)
        user = f"Article {article.get('number', '?')} - {title}\n\n{truncated}"

        raw = self._call(system, user)
        parsed = self._parse_json_response(raw) if raw else {}

        # Merge with extractive results for completeness
        extractive = self._extractive.summarise_article(article)
        merged = {
            "summary": parsed.get("summary") or extractive["summary"],
            "key_obligations": parsed.get("key_obligations") or extractive["key_obligations"],
            "key_topics": parsed.get("key_topics") or extractive["key_topics"],
            "applies_to": parsed.get("applies_to") or extractive["applies_to"],
            "cross_references": extractive["cross_references"],  # always heuristic
            "regulatory_action": parsed.get("regulatory_action") or extractive["regulatory_action"],
            "paragraph_count": extractive["paragraph_count"],
        }
        return merged

    def summarise_chapter(self, chapter: dict, article_metas: list[dict]) -> dict:
        title = chapter.get("title", "")
        number = chapter.get("number", "?")

        # Build a bullet list of article summaries for the LLM
        article_bullets = "\n".join(
            f"- Art. {a.get('number', '?')} ({a.get('title', '')}): {m.get('summary', 'N/A')}"
            for a, m in zip(_iter_articles(chapter), article_metas, strict=False)
        )
        article_bullets = _first_n_words(article_bullets, 1200)

        system = textwrap.dedent("""\
            You are a specialist in EU regulatory law. Summarise this chapter
            for use by AI agents doing retrieval and reasoning.

            Return ONLY valid JSON with these keys:
            - "summary": 2-4 sentence overview of the chapter's purpose and scope
            - "key_themes": list of 5-12 thematic keywords
            - "scope_description": one sentence on who/what this chapter applies to
            - "regulatory_functions": list from [defines, requires, prohibits,
              establishes_obligations, establishes_rights, establishes_penalties,
              grants_exemptions, transparency_requirement, governance, general_provision]
        """)
        user = (
            f"Chapter {number} - {title}\n"
            f"({len(article_metas)} articles)\n\n"
            f"Article summaries:\n{article_bullets}"
        )

        raw = self._call(system, user)
        parsed = self._parse_json_response(raw) if raw else {}

        extractive = self._extractive.summarise_chapter(chapter, article_metas)
        return {
            "summary": parsed.get("summary") or extractive["summary"],
            "key_themes": parsed.get("key_themes") or extractive["key_themes"],
            "scope_description": parsed.get("scope_description") or extractive["scope_description"],
            "regulatory_functions": (
                parsed.get("regulatory_functions") or extractive["regulatory_functions"]
            ),
            "article_count": extractive["article_count"],
        }


# ---------------------------------------------------------------------------
# Document traversal
# ---------------------------------------------------------------------------


def _iter_articles(chapter: dict):
    """Yield articles from a chapter, respecting optional sections."""
    for art in chapter.get("articles", []):
        yield art
    for sec in chapter.get("sections", []):
        for art in sec.get("articles", []):
            yield art


def _set_article_metadata(article: dict, meta: dict) -> None:
    article["metadata"] = meta


def _set_section_metadata(section: dict, article_metas: list[dict]) -> None:
    """Add lightweight metadata to sections."""
    all_topics: list[str] = []
    all_entities: list[str] = []
    for m in article_metas:
        all_topics.extend(m.get("key_topics", []))
        all_entities.extend(m.get("applies_to", []))
    section["metadata"] = {
        "key_themes": _dedup(all_topics)[:10],
        "applies_to": _dedup(all_entities)[:8],
        "article_count": len(article_metas),
    }


def enrich_document(doc: dict, summariser) -> dict:
    """Walk the document tree, enrich every chapter and article in-place."""
    doc_type = doc.get("document_type", "unknown")
    log.info("Enriching '%s' (%s) …", doc.get("title", "?")[:60], doc_type)

    for ch_idx, chapter in enumerate(doc.get("chapters", [])):
        ch_num = chapter.get("number", ch_idx)
        log.info("  Chapter %s - %s", ch_num, chapter.get("title", ""))

        article_metas: list[dict] = []

        # Articles directly under the chapter
        for art in chapter.get("articles", []):
            log.info("    Article %s - %s", art.get("number", "?"), art.get("title", ""))
            meta = summariser.summarise_article(art)
            _set_article_metadata(art, meta)
            article_metas.append(meta)

        # Articles nested inside sections
        for section in chapter.get("sections", []):
            section_metas: list[dict] = []
            for art in section.get("articles", []):
                log.info("    Article %s - %s", art.get("number", "?"), art.get("title", ""))
                meta = summariser.summarise_article(art)
                _set_article_metadata(art, meta)
                section_metas.append(meta)
                article_metas.append(meta)
            _set_section_metadata(section, section_metas)

        # Chapter-level summary (aggregates article metadata)
        chapter_meta = summariser.summarise_chapter(chapter, article_metas)
        chapter["metadata"] = chapter_meta
        log.info("  → Chapter %s metadata added (%d articles)", ch_num, len(article_metas))

    log.info("Done enriching '%s'.", doc_type)
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Add contextual metadata to EU regulation JSON extracts.",
    )
    p.add_argument(
        "--provider",
        choices=["ollama", "extractive"],
        default="extractive",
        help="Summarisation backend (default: extractive).",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL}).",
    )
    p.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL}).",
    )
    p.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to a single JSON file. If omitted, all *_extracted.json in data/ are processed.",
    )
    p.add_argument(
        "--output-suffix",
        default="_enriched",
        help="Suffix appended before .json for the output file (default: _enriched).",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the original file instead of creating a new one.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Pick summariser
    if args.provider == "ollama":
        summariser: Any = OllamaSummariser(
            model=args.model,
            base_url=args.ollama_url,
        )
    else:
        summariser = ExtractiveSummariser()

    # Collect files
    files = [args.file.resolve()] if args.file else sorted(DATA_DIR.glob("*_extracted.json"))

    if not files:
        log.error("No files found to process.")
        sys.exit(1)

    for fpath in files:
        log.info("Loading %s …", fpath.name)
        with open(fpath, encoding="utf-8") as f:
            doc = json.load(f)

        enrich_document(doc, summariser)

        if args.in_place:
            out_path = fpath
        else:
            out_path = fpath.with_name(fpath.stem + args.output_suffix + fpath.suffix)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        log.info("Written → %s", out_path.name)


if __name__ == "__main__":
    main()
