# CV Tailoring Assistant

A small pipeline that takes a job description and a bank of role-agnostic
achievement bullets, then produces a tailored draft: matched bullets rewritten
in the JD's own terminology (without overclaiming), plus an honest list of
gaps where no strong match exists.

Built as a learning project covering: hybrid retrieval (BM25 + vector search),
cross-encoder reranking, and a Corrective-RAG-style self-check loop applied to
*rewriting* rather than plain Q&A.

## How it works

1. **Ingest** (`src/ingest.py`) -- loads achievements from `data/achievements/achievements.json`.
   Each achievement is already an atomic bullet (no chunking needed).
2. **Embed** (`src/embed.py`) -- embeds achievements with a local BGE model and
   upserts into Qdrant.
3. **Classify** (`src/classify_jd.py`) -- an LLM call classifies the JD into a
   role category (AI/ML Engineer, Data Engineer, BI/Analytics, etc.) and
   extracts its key terminology and requirements.
4. **Retrieve** (`src/retrieve.py`) -- for each JD requirement, runs hybrid
   search (BM25 + vector, merged via reciprocal rank fusion), boosts results
   whose tags match the classified role category, then reranks with a
   cross-encoder.
5. **Rewrite** (`src/rewrite.py`) -- rephrases the top matches using the JD's
   terminology, under a strict prompt constraint: only substitute vocabulary
   for concepts already present in the original, never introduce new tools,
   scope, or seniority.
6. **Verify** (`src/verify.py`) -- a second LLM call checks the rewrite against
   the original for overclaiming. If it finds unsupported claims, the pipeline
   reverts to the original wording instead of using the rewrite.
7. **Gaps** (`src/gaps.py`) -- for requirements with no strong retrieval match,
   outputs an honest gap statement (with the closest adjacent experience, if any)
   instead of forcing a weak bullet to fit.
8. **CLI** (`src/cli.py`) -- wires all of the above together into one command.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start Qdrant locally:
   ```
   docker compose up -d
   ```

3. Copy `.env.example` to `.env` and add your Anthropic API key:
   ```
   cp .env.example .env
   # edit .env and set ANTHROPIC_API_KEY
   ```

4. Add your own achievements to `data/achievements/achievements.json`
   (a starter set is already included -- edit/extend it with your real bullets).

5. Index your achievements into Qdrant:
   ```
   python src/embed.py
   ```

## Running it

Against a JD in a file:
```
python src/cli.py --jd data/jds/some_job.txt --out output.md
```

Against pasted JD text:
```
python src/cli.py --jd-text "paste the full JD text here"
```

## Evaluation

Fill in `eval/eval_set.json` with real JDs you've manually tailored for before,
plus your own ground-truth achievement picks and gap calls. Then run:
```
python eval/run_eval.py
```

This checks three things against your own judgment: did retrieval find the
achievements you'd pick manually, did the verify step behave sensibly, and did
gap detection correctly distinguish real gaps from real matches.

## Known limitations / explicitly out of scope

- No agent harness runtime, sandboxing, or multi-tenancy.
- No GraphRAG / knowledge graph.
- No frontend -- CLI/markdown output only.
- Output is a **draft for review**, not a finished CV -- always read the gap
  list and spot-check rewrites before using anything in a real application.
- The role-category tag boost table (`ROLE_TAG_BOOSTS` in `retrieve.py`) is a
  simple hand-maintained mapping, not learned -- extend it as you add tags.
