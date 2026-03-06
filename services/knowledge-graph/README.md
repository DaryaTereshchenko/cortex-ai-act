# Knowledge-Graph Service

REST API backed by **Neo4j** that stores EU regulatory documents (AI Act, Digital Services Act) as a richly connected knowledge graph. Designed to be queried by LLM agents via Cypher or the convenience REST endpoints.

---

## Table of contents

1. [Data overview](#data-overview)
2. [Node types & properties](#node-types--properties)
3. [Relationship types](#relationship-types)
4. [ID conventions](#id-conventions)
5. [Indexes & constraints](#indexes--constraints)
6. [REST API reference](#rest-api-reference)
7. [Cypher query examples](#cypher-query-examples)
8. [Environment variables](#environment-variables)
9. [Running locally](#running-locally)

---

## Data overview

The graph currently contains **two EU regulations**, both ingested from enriched JSON files produced by the scraping + LLM enrichment pipeline:

| Regulation | `regulation` ID | Chapters | Sections | Articles | Paragraphs | Points | Recitals | Annexes | Definitions | Cross-references |
|---|---|---|---|---|---|---|---|---|---|---|
| EU AI Act (2024/1689) | `eu_ai_act` | 13 | 16 | 113 | 593 | 345 | 180 | 13 | 69 | 348 |
| Digital Services Act (2022/2065) | `eu_dsa` | 5 | 12 | 93 | 363 | 295 | 156 | 0 | 5 | 193 |

Articles include LLM-generated metadata: `summary`, `key_obligations`, `key_topics`, `applies_to`, `cross_references`, `regulatory_action`.
Chapters and sections hold: `summary`, `key_themes`, `scope_description`, `regulatory_functions`.

---

## Node types & properties

### Regulation
Top-level node representing a single EU regulation.

| Property | Type | Example |
|---|---|---|
| `id` | string | `eu_ai_act` |
| `title` | string | Full title from the Official Journal |
| `short_name` | string | `EU AI Act` |
| `title_formal` | string | `Regulation (EU) 2024/1689` |
| `celex` | string | `32024R1689` |
| `document_type` | string | `ai_act` |
| `oj_reference` | string | `OJ L, 2024/1689, 12.7.2024` |
| `entry_into_force` | string | `2024-08-01` |
| `eur_lex_url` | string | EUR-Lex link |
| `recital_count` | int | 180 |
| `chapter_count` | int | 13 |
| `article_count` | int | 113 |
| `annex_count` | int | 13 |

### Chapter
| Property | Type | Description |
|---|---|---|
| `id` | string | `eu_ai_act_ch_I` |
| `number` | string | Roman numeral (e.g. `I`, `II`) |
| `title` | string | Chapter heading |
| `regulation` | string | Parent regulation ID |
| `summary` | string | LLM-generated summary |
| `key_themes` | string[] | Topics covered |
| `scope_description` | string | Scope overview |
| `regulatory_functions` | string[] | Functions performed |
| `article_count` | int | Number of articles |

### Section
| Property | Type | Description |
|---|---|---|
| `id` | string | `eu_ai_act_ch_III_sec_1` |
| `number` | string | Section number within chapter |
| `title` | string | Section heading |
| `regulation` | string | Parent regulation ID |
| `chapter` | string | Parent chapter number |
| `key_themes` | string[] | Topics covered |
| `applies_to` | string[] | Entities the section applies to |
| `article_count` | int | Number of articles |

### Article
| Property | Type | Description |
|---|---|---|
| `id` | string | `eu_ai_act_art_6` |
| `number` | int/string | Article number |
| `title` | string | Article heading |
| `regulation` | string | Parent regulation ID |
| `chapter` | string | Parent chapter number |
| `section` | string/null | Parent section number (if applicable) |
| `full_text` | string | Concatenated paragraph text |
| `summary` | string | LLM-generated summary |
| `key_obligations` | string[] | Obligations extracted by LLM |
| `key_topics` | string[] | Topic tags |
| `applies_to` | string[] | Target entities |
| `cross_references` | string[] | Textual cross-references |
| `regulatory_action` | string[] | Actions required |
| `paragraph_count` | int | Number of paragraphs |

### Paragraph
| Property | Type | Description |
|---|---|---|
| `id` | string | `eu_ai_act_art_6_par_1` |
| `number` | string | Paragraph number within article |
| `text` | string | Full paragraph text |
| `regulation` | string | Regulation ID |

### Point
Sub-paragraph lettered items such as `Article 6(1)(a)`.

| Property | Type | Description |
|---|---|---|
| `id` | string | `eu_ai_act_art_6_par_1_pt_a` |
| `number` | string | Letter (e.g. `a`, `b`) |
| `text` | string | Point text |
| `regulation` | string | Regulation ID |

### Recital
Preamble clauses that explain legislative intent.

| Property | Type | Description |
|---|---|---|
| `id` | string | `eu_ai_act_rec_1` |
| `number` | int | Sequential recital number |
| `text` | string | Full recital text |
| `regulation` | string | Regulation ID |

### Annex
| Property | Type | Description |
|---|---|---|
| `id` | string | `eu_ai_act_anx_I` |
| `number` | string | Roman numeral |
| `title` | string | Annex title |
| `content` | string | Raw content |
| `regulation` | string | Regulation ID |

### AnnexSection
Sections parsed from annex content.

| Property | Type | Description |
|---|---|---|
| `id` | string | `eu_ai_act_anx_I_sec_A` |
| `number` | string | Section letter |
| `title` | string | Section title |
| `content` | string | Section text (max 2000 chars) |
| `regulation` | string | Regulation ID |

### Definition
Terms defined in Article 3 of each regulation.

| Property | Type | Description |
|---|---|---|
| `id` | string | `eu_ai_act_def_ai_system` |
| `term` | string | Defined term |
| `definition_text` | string | Full definition |
| `source_article` | string | Article ID (always `*_art_3`) |
| `regulation` | string | Regulation ID |
| `paragraph_num` | string | Paragraph number in Article 3 |

---

## Relationship types

| Relationship | Direction | Description |
|---|---|---|
| `CONTAINS` | parent → child | Structural containment (Regulation→Chapter, Chapter→Section, Section/Chapter→Article, Article→Paragraph, Paragraph→Point, Regulation→Recital, Regulation→Annex, Annex→AnnexSection) |
| `PART_OF` | child → parent | Inverse of CONTAINS |
| `NEXT` | node → successor | Sequential ordering within siblings (chapters, articles, paragraphs, recitals, annexes) |
| `PREVIOUS` | node → predecessor | Inverse of NEXT |
| `REFERENCES` | source → target | Cross-reference between nodes. Properties: `anchor` (link text), `url` (target URL), or `context` (metadata reference text) |
| `DEFINES` | Article → Definition | Article 3 defines a term |

### Structural hierarchy

```
Regulation
├── CONTAINS → Chapter
│   ├── CONTAINS → Section
│   │   └── CONTAINS → Article
│   │       └── CONTAINS → Paragraph
│   │           └── CONTAINS → Point
│   └── CONTAINS → Article  (chapters without sections)
├── CONTAINS → Recital
├── CONTAINS → Annex
│   └── CONTAINS → AnnexSection
└── (Article 3) ─ DEFINES → Definition
```

---

## ID conventions

All node IDs follow a deterministic pattern:

| Node type | Pattern | Example |
|---|---|---|
| Regulation | `{reg_id}` | `eu_ai_act` |
| Chapter | `{reg_id}_ch_{number}` | `eu_ai_act_ch_III` |
| Section | `{reg_id}_ch_{ch}_sec_{sec}` | `eu_ai_act_ch_III_sec_1` |
| Article | `{reg_id}_art_{number}` | `eu_ai_act_art_6` |
| Paragraph | `{reg_id}_art_{art}_par_{par}` | `eu_ai_act_art_6_par_1` |
| Point | `{reg_id}_art_{art}_par_{par}_pt_{letter}` | `eu_ai_act_art_6_par_1_pt_a` |
| Recital | `{reg_id}_rec_{number}` | `eu_ai_act_rec_12` |
| Annex | `{reg_id}_anx_{number}` | `eu_ai_act_anx_III` |
| AnnexSection | `{reg_id}_anx_{anx}_sec_{letter}` | `eu_ai_act_anx_I_sec_A` |
| Definition | `{reg_id}_def_{safe_term}` | `eu_ai_act_def_ai_system` |

The two regulation IDs currently in the graph are **`eu_ai_act`** and **`eu_dsa`**.

---

## Indexes & constraints

### Uniqueness constraints (one per node type)
Every node type has a uniqueness constraint on its `id` property (`reg_id`, `rec_id`, `ch_id`, `sec_id`, `art_id`, `par_id`, `point_id`, `anx_id`, `anx_sec_id`, `def_id`).

### B-tree lookup indexes
| Index | Node | Property |
|---|---|---|
| `art_reg` | Article | `regulation` |
| `par_reg` | Paragraph | `regulation` |
| `rec_reg` | Recital | `regulation` |
| `ch_reg` | Chapter | `regulation` |
| `art_number` | Article | `number` |
| `rec_number` | Recital | `number` |

### Full-text search indexes
| Index name | Node | Properties |
|---|---|---|
| `article_fulltext` | Article | `title`, `summary`, `full_text` |
| `paragraph_fulltext` | Paragraph | `text` |
| `recital_fulltext` | Recital | `text` |

Use with `CALL db.index.fulltext.queryNodes('article_fulltext', 'high risk')`.

---

## REST API reference

Base URL: `http://localhost:8001/graph` (or via Nginx at `http://localhost/api/knowledge-graph/graph`).

### POST `/graph/ingest`
Ingest enriched JSON files into Neo4j.

```json
// Request body
{ "file": "ai_act_extracted_enriched.json", "clear": false }

// Omit "file" to ingest all enriched files
{ "clear": true }
```

### GET `/graph/stats/simple`
Node and relationship counts (no APOC needed).

```bash
curl http://localhost:8001/graph/stats/simple
```
```json
{
  "nodes": {"Regulation": 2, "Article": 206, "Chapter": 18, ...},
  "relationships": {"CONTAINS": 1834, "PART_OF": 1834, "NEXT": 1612, ...}
}
```

### GET `/graph/regulations`
List all regulations in the graph.

```bash
curl http://localhost:8001/graph/regulations
```

### GET `/graph/article/{regulation}/{number}`
Fetch a single article with its parent and children.

```bash
curl http://localhost:8001/graph/article/eu_ai_act/6
```
```json
{
  "article": {"id": "eu_ai_act_art_6", "title": "Classification rules...", "summary": "...", ...},
  "parent_label": "Section",
  "parent_id": "eu_ai_act_ch_III_sec_1",
  "children": [{"id": "eu_ai_act_art_6_par_1", "label": "Paragraph", "number": "1"}, ...]
}
```

### GET `/graph/chapter/{regulation}/{number}`
Fetch a chapter with its direct children (sections/articles).

```bash
curl http://localhost:8001/graph/chapter/eu_ai_act/III
```

### GET `/graph/recitals/{regulation}?skip=0&limit=20`
Paginated recitals.

```bash
curl "http://localhost:8001/graph/recitals/eu_ai_act?skip=0&limit=10"
```

### GET `/graph/annexes/{regulation}`
List annexes with their sections.

```bash
curl http://localhost:8001/graph/annexes/eu_ai_act
```

### GET `/graph/definitions/{regulation}`
List all definitions from Article 3.

```bash
curl http://localhost:8001/graph/definitions/eu_ai_act
```

### GET `/graph/search?q={query}&regulation={id}&limit=10`
Full-text search across articles and paragraphs.

```bash
curl "http://localhost:8001/graph/search?q=high+risk+ai+system&regulation=eu_ai_act&limit=5"
```

### GET `/graph/traverse/{node_id}?direction=both&depth=1`
Graph traversal from any node. `direction`: `in`, `out`, or `both`. `depth`: 1–3.

```bash
curl "http://localhost:8001/graph/traverse/eu_ai_act_art_6?direction=out&depth=2"
```

### POST `/graph/cypher`
Execute an arbitrary **read-only** Cypher query. Write operations (`CREATE`, `MERGE`, `DELETE`, etc.) are rejected.

```bash
curl -X POST http://localhost:8001/graph/cypher \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (a:Article {regulation: $reg}) RETURN a.number, a.title LIMIT 5", "parameters": {"reg": "eu_ai_act"}}'
```
```json
{
  "columns": ["a.number", "a.title"],
  "rows": [
    {"a.number": 1, "a.title": "Subject matter"},
    {"a.number": 2, "a.title": "Scope"}
  ]
}
```

---

## Cypher query examples

### 1. Find all articles about "high-risk AI systems" with their obligations and parent chapter

This uses the full-text index to find relevant articles, then traverses up to the chapter level and returns enriched metadata.

```cypher
CALL db.index.fulltext.queryNodes('article_fulltext', 'high-risk AI system')
YIELD node AS art, score
WHERE art.regulation = 'eu_ai_act' AND score > 1.0
MATCH (art)-[:PART_OF*1..2]->(ch:Chapter)
RETURN art.number        AS article,
       art.title         AS title,
       art.summary       AS summary,
       art.key_obligations AS obligations,
       art.applies_to    AS applies_to,
       ch.number         AS chapter,
       ch.title          AS chapter_title,
       score
ORDER BY score DESC
LIMIT 10
```

**Via API:**
```bash
curl -X POST http://localhost:8001/graph/cypher \
  -H "Content-Type: application/json" \
  -d '{
    "query": "CALL db.index.fulltext.queryNodes(\"article_fulltext\", $q) YIELD node AS art, score WHERE art.regulation = $reg AND score > 1.0 MATCH (art)-[:PART_OF*1..2]->(ch:Chapter) RETURN art.number AS article, art.title AS title, art.summary AS summary, art.key_obligations AS obligations, ch.title AS chapter_title, score ORDER BY score DESC LIMIT 10",
    "parameters": {"q": "high-risk AI system", "reg": "eu_ai_act"}
  }'
```

### 2. Compare how two regulations define similar concepts

Find definitions from both the AI Act and the DSA, then look for overlapping terms.

```cypher
MATCH (d1:Definition {regulation: 'eu_ai_act'})
MATCH (d2:Definition {regulation: 'eu_dsa'})
WHERE d1.term = d2.term
   OR d1.term CONTAINS d2.term
   OR d2.term CONTAINS d1.term
RETURN d1.term            AS ai_act_term,
       d1.definition_text AS ai_act_definition,
       d2.term            AS dsa_term,
       d2.definition_text AS dsa_definition
ORDER BY d1.term
```

**Via API:**
```bash
curl -X POST http://localhost:8001/graph/cypher \
  -H "Content-Type: application/json" \
  -d '{
    "query": "MATCH (d1:Definition {regulation: \"eu_ai_act\"}) MATCH (d2:Definition {regulation: \"eu_dsa\"}) WHERE d1.term = d2.term OR d1.term CONTAINS d2.term OR d2.term CONTAINS d1.term RETURN d1.term AS ai_act_term, d1.definition_text AS ai_act_definition, d2.term AS dsa_term, d2.definition_text AS dsa_definition ORDER BY d1.term"
  }'
```

### 3. Trace cross-references from a specific article to understand its regulatory context

Starting from Article 6 (classification rules for high-risk AI), follow all REFERENCES edges, collect the targets, and for each referenced article show what it regulates.

```cypher
MATCH (source:Article {id: 'eu_ai_act_art_6'})
MATCH (source)-[ref:REFERENCES]->(target)
OPTIONAL MATCH (target)-[:PART_OF*1..2]->(ch:Chapter)
RETURN source.title          AS source_title,
       ref.context           AS reference_context,
       labels(target)[0]     AS target_type,
       target.id             AS target_id,
       target.title          AS target_title,
       CASE labels(target)[0]
         WHEN 'Article' THEN target.summary
         WHEN 'Annex'   THEN left(target.content, 200)
         ELSE null
       END                   AS target_summary,
       ch.title              AS chapter_title
ORDER BY target.id
```

**Via API:**
```bash
curl -X POST http://localhost:8001/graph/cypher \
  -H "Content-Type: application/json" \
  -d '{
    "query": "MATCH (source:Article {id: $art_id}) MATCH (source)-[ref:REFERENCES]->(target) OPTIONAL MATCH (target)-[:PART_OF*1..2]->(ch:Chapter) RETURN source.title AS source_title, ref.context AS reference_context, labels(target)[0] AS target_type, target.id AS target_id, target.title AS target_title, CASE labels(target)[0] WHEN \"Article\" THEN target.summary WHEN \"Annex\" THEN left(target.content, 200) ELSE null END AS target_summary, ch.title AS chapter_title ORDER BY target.id",
    "parameters": {"art_id": "eu_ai_act_art_6"}
  }'
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `NEO4J_URI` | Yes | `bolt://neo4j:7687` | Neo4j connection URI |
| `NEO4J_USER` | Yes | `neo4j` | Database username |
| `NEO4J_PASSWORD` | Yes | `password` | Database password |
| `NEO4J_DATABASE` | No | `neo4j` | Database name (use instance ID for Aura) |
| `ENVIRONMENT` | No | `development` | Environment label |

**GitHub Secrets to configure:** `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`.

---

## Running locally

```bash
# Start the full stack (Neo4j + services)
docker compose up -d

# Or run just this service against an existing Neo4j
cd services/knowledge-graph
pip install -r requirements.txt
export NEO4J_URI="bolt://localhost:7687" NEO4J_USER="neo4j" NEO4J_PASSWORD="your-password"
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Ingest data
curl -X POST http://localhost:8001/graph/ingest -H "Content-Type: application/json" -d '{"clear": true}'

# Verify
curl http://localhost:8001/graph/stats/simple
```

Interactive API docs are available at `http://localhost:8001/docs` (Swagger UI).
