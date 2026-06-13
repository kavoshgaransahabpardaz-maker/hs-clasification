HS Code Classification Service — Implementation Spec

Audience: an implementing engineer/agent. This document is self-contained.
Build a Python microservice that classifies a product description into an EU or
UK commodity code with a calibrated confidence score, grounded in official
government data and binding rulings, escalating low-confidence cases to a
human.


Read sections 0–3 fully before writing code. Section 3 (government data
retrieval) is the part most implementations skip — do not skip it.




0. Goal & non-negotiables


Input: free-text product description + target market (EU or UK). Output: a
commodity code (UK 10-digit / EU CN8 or TARIC10), a calibrated confidence,
a status (auto_resolved / needs_review), and the supporting rulings.
Never emit a code that is not currently valid in the live nomenclature.
The confidence threshold must be calibrated against a labelled gold set
before it is trusted. An uncalibrated LLM "confidence" is not acceptable.
UK and EU are separate pipelines past the 6-digit level (codes diverge).
Every decision is persisted with its candidates and cited rulings (audit
trail + training data).
Position the service as decision support, not a legal determination.



1. Fixed tech stack

ConcernChoiceLanguagePython 3.12APIFastAPI + UvicornDBPostgreSQL + pgvector extension (single DB, no separate vector store)ORMSQLAlchemy 2.x (typed Mapped[...])ValidationPydantic v2 / pydantic-settingsMigrationsAlembic (production); create_all() acceptable only for first spikeEmbeddingsPluggable provider behind an Embedder protocolLLMPluggable; used for extraction and (later) GRI tie-breaking


2. Architecture

                ┌─────────────────────────────────────────────┐
 GOV SOURCES →  │  INGESTION (section 3)                       │
 (UK API,       │  loaders -> nomenclature_node, legal_note,   │
  EU TARIC,     │             ruling ; then embed -> embedding │
  EBTI, ATaR)   └─────────────────────────────────────────────┘
                                   │  (populates the DB)
                                   ▼
 POST /v1/classify
   → extraction   (LLM: text -> structured profile)
   → retrieval    (pgvector cosine over rulings, filtered by jurisdiction/validity)
   → rules        (chapter/section notes prune candidates; GRI tie-break)
   → validation   (winning code checked against live nomenclature)
   → ranking      (similarity + support -> one score per code)
   → confidence   (calibrated) → auto_resolve OR needs_review
   → persist ClassificationRequest (audit + feedback)

Two layers people conflate: ingestion pulls government data INTO the DB
(batch, scheduled); retrieval searches that local DB at request time. The
service does not call government APIs on the hot path except for final
code-validity validation (which is cacheable for 24h since sources update daily).


3. Government data retrieval (the ingestion layer)

Build one loader per source. All are batch jobs writing into the tables in
section 4. Schedule daily; cache 24h.

3.1 UK — nomenclature + legal notes + duties


Source: GOV.UK Trade Tariff Public API v2, base
https://www.trade-tariff.service.gov.uk/api/v2/ (also /uk/api/).
Auth: none for the public www host. (The managed api.trade-tariff…
host needs OAuth2 client-credentials + Accept: application/vnd.hmrc.2.0+json;
not required for ingestion — use the public host.)
Format: JSON:API — parse both data and the included array; entities are
included once and linked via relationships.
Endpoints to crawl:

/api/v2/sections and /api/v2/sections/{id} — sections + section notes.
/api/v2/chapters and /api/v2/chapters/{id} — chapters; response carries
chapter_note (this is your legal_note text) and the section relationship.
/api/v2/headings/{4-digit} — headings.
/api/v2/commodities/{10-digit} — full commodity page (description, parent
chain, measures/duties). URL mirrors the public tariff page with api/v2/
inserted.
/api/v2/search?q=... — free-text search (also usable as a UK candidate
generator later; optional).



Bulk alternative (faster cold start): Department for Business & Trade Data
API — https://data.api.trade.gov.uk/v1/datasets/uk-tariff-2021-01-01/versions/{version}/tables/commodities/data?format=csv
gives the whole commodities table (code, suffix, description, validity, parent).
Use bulk for the initial load, the v2 API for daily deltas + notes.
Licence: Open Government Licence v3.
Loader output: nomenclature_node (jurisdiction=UK, levels section→
commodity, path, validity) and legal_note (from chapter/section notes).


3.2 EU — nomenclature (CN/TARIC) + legal notes


Source: EU Customs Tariff TARIC open dataset on
https://data.europa.eu/ ("EU Customs Tariff (TARIC)"), updated daily; and the
TARIC Consultation interface https://ec.europa.eu/taxation_customs/dds2/taric/.
Section/chapter notes: from the Combined Nomenclature regulation (annual
Commission Implementing Regulation, via EUR-Lex) or the TARIC export.
Note: the EU has no official "classify from description" endpoint
(unlike the UK FPO tool). Candidate generation for EU therefore relies on your
own retrieval over rulings + nomenclature.
Loader output: nomenclature_node (jurisdiction=EU, CN8 + TARIC10) and
legal_note.


3.3 EU rulings — EBTI (corpus + gold labels)


Source: European Binding Tariff Information — public dataset on
https://data.europa.eu/ ("European Binding Tariff Information"); contains all
currently-valid BTI decisions (description → assigned code, validity ~3 years).
Reference: taxation-customs.ec.europa.eu/.../ebti-european-binding-tariff-information_en.
Loader output: ruling rows (source=EBTI, jurisdiction=EU,
product_description, assigned_code, validity). Reserve ~10–20% as is_eval=True.


3.4 UK rulings — ATaR (corpus + gold labels)


Source: UK Advance Tariff Rulings (ATaR), published/searchable on GOV.UK.
Verify the current access method (public search vs downloadable dataset)
before implementing; harvest the published rulings (description → code).
Loader output: ruling rows (source=ATaR, jurisdiction=UK).


3.5 Embedding the corpus

After rulings load, embed each product_description with the configured
Embedder and write an embedding row (object_type='ruling', object_id,
jurisdiction, model, vector). Optionally also embed nomenclature
descriptions. Re-embed whenever you change embedding model (model column tracks
which version produced each vector).

3.6 Ingestion acceptance criteria


 nomenclature_node populated for both UK and EU; spot-checks resolve
(e.g. UK 0702000007 → "tomatoes…" with a correct parent chain).
 legal_note rows exist for chapters that have notes (e.g. exclusion notes).
 ruling populated from EBTI (+ ATaR), with an is_eval split set aside.
 Every ruling has a matching embedding row.
 validate_code(code, jurisdiction) returns True only for codes present and
currently valid; False for expired/unknown codes.



4. Data model (six tables)

Single Postgres DB + pgvector. Enable with CREATE EXTENSION IF NOT EXISTS vector.

nomenclature_node — the code tree.
id PK · code varchar(12) · level enum(section,chapter,heading,subheading,cn8,commodity)
· jurisdiction enum(WCO,EU,UK) · description text · parent_id FK→self
· path text (materialized, e.g. 84.8471.847130) · valid_from/valid_to date.
Indexes: (jurisdiction,code), (jurisdiction,level), path with
text_pattern_ops (for LIKE 'prefix%' subtree queries).

legal_note — section/chapter notes (rules layer).
id · jurisdiction · scope enum(section,chapter) · scope_code varchar(4)
· note_type enum(exclusion,inclusion,definition,other) · text text.
Index (jurisdiction,scope,scope_code).

ruling — retrieval corpus AND gold labels.
id · source enum(EBTI,ATaR,INTERNAL) · reference · jurisdiction
· product_description text · assigned_code varchar(12) · justification
· keywords jsonb · valid_from/valid_to · is_eval bool · created_at.
Indexes (jurisdiction,assigned_code), is_eval, reference.

embedding — one polymorphic, HNSW-indexed vector store.
id · object_type enum(ruling,nomenclature) · object_id int
· jurisdiction (denormalized, so filtered ANN needs no join) · model
· vector vector(DIM). Indexes: HNSW on vector with vector_cosine_ops
(m=16, ef_construction=64); (object_type,object_id); (jurisdiction,object_type).

classification_request — audit trail + Phase-4 feedback.
id · input_text · target_jurisdiction · profile jsonb · predicted_code
· confidence float · status enum(auto_resolved,needs_review,reviewed)
· candidates jsonb · cited_ruling_ids jsonb · reviewed_code
· reviewer_note · pipeline_version · created_at · reviewed_at.

eval_run — benchmark history.
id · pipeline_version · n_samples · recall_at_k jsonb
· accuracy_by_digit jsonb · ece float · notes · created_at.


Reuse a single SQLAlchemy Enum(..., name="jurisdiction") instance across
tables so the PG enum type is created once.




5. Pipeline (request-time)


extraction — LLM turns raw text into a structured profile
(material, function, form_state, intended_use, components, processing_level)
plus a clean query string for retrieval. Validate JSON output.
retrieval — embed query; pgvector cosine search over embedding where
object_type='ruling' AND jurisdiction=target AND ruling still valid; return
top-K with similarity. Recall@K is the ceiling — optimize this first.
rules — load legal_note exclusions for the candidates' chapters; drop
legally-impossible candidates; for ambiguous cases apply GRIs via an LLM call
constrained to the retrieved candidate set (must state which GRI/note).
validation — confirm the winning code exists and is currently valid in
nomenclature_node (or via the live UK/EU source, cached 24h). Reject invalids.
ranking — collapse to one score per code (max similarity + support count;
later add LLM self-consistency across N samples).
confidence — produce a score, then calibrate (isotonic/Platt/binned
fitted on the gold set). >= threshold → auto_resolved, else needs_review.
persist — write one classification_request row.



6. Microservice API (contract — keep stable, version with /v1)


GET /health → {status, pipeline_version}
POST /v1/classify body {text, jurisdiction, max_candidates?, confidence_threshold?}
→ {request_id, predicted_code, confidence, status, profile, candidates[], cited_ruling_ids[], pipeline_version}
where candidates[] = {code, description, score, supporting_ruling_id, supporting_ruling_ref}.
GET /v1/classify/{id} → same response shape (audit re-fetch).
POST /v1/classify/{id}/review body {reviewed_code, reviewer_note?} → records
the human decision, sets status reviewed, and inserts an INTERNAL ruling
(then embed + index it so future retrieval improves).


The calling product depends only on these shapes, never on the DB or models.


7. Evaluation harness

python -m app.eval: run the pipeline over ruling rows with is_eval=True,
report recall@k (k=1,5,10) and accuracy at 2/4/6/8/10 digits, and write an
eval_run row. Add Expected Calibration Error (ECE) once a calibrator exists.
Run after every change; a change is only "done" if eval doesn't regress.


8. Build order (milestones with definition-of-done)


M1 — Schema. Tables + pgvector + indexes created; all DDL compiles.
Done when: init_db()/migrations succeed; validate_code works on seed rows.
M2 — Nomenclature ingestion (UK + EU). Section 3.1–3.2 loaders.
Done when: section 3.6 nomenclature/notes checks pass.
M3 — Rulings ingestion + embeddings. Section 3.3–3.5.
Done when: every ruling has an embedding; eval split exists.
M4 — Retrieval + service skeleton. Pipeline steps 1–2 + /v1/classify
returning candidates. Done when: recall@10 on gold meets target.
M5 — Rules + validation + ranking + confidence. Steps 3–6.
Done when: per-digit accuracy improves over M4; no invalid codes emitted.
M6 — Calibration. Fit calibrator on gold; ECE reported.
Done when: stated 0.90 ≈ 90% empirical accuracy on held-out data.
M7 — Review loop + hardening. /review feedback → INTERNAL rulings;
separate UK/EU pipelines; audit logging; Alembic; drift monitoring.



9. Guardrails (do NOT)


Do not return a code without validating it against the live nomenclature.
Do not trust the confidence threshold before calibration (M6).
Do not let the LLM invent codes — always constrain it to retrieved,
validated candidates.
Do not merge UK and EU codes past 6 digits.
Do not drop the audit trail; it is the compliance record.



10. Open questions for the human


EU-only, UK-only, or both in the first dataset, and target dataset size
(e.g. ~1,000 rulings to start)?
Which embedding model + dimension (sets embedding_dim and the vector size)?
Which LLM provider for extraction / GRI reasoning?
Confirm the current ATaR access method (search vs bulk) before M3.