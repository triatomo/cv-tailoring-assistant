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

By default this runs entirely **free and local** -- no API subscription
required. LLM calls go to a local model via [Ollama](https://ollama.com)
instead of a paid API.

1. Install [Ollama](https://ollama.com/download), then pull a model:
  ```
  ollama pull llama3.1:8b
  ```
   Ollama runs a local server automatically once installed -- no signup, no
   API key, no per-token cost. Any tool-use-capable instruct model works;
   `llama3.1:8b` is a reasonable default if your machine can run it.
2. Install Python dependencies:
  ```
   pip install -r requirements.txt
  ```
3. Start Qdrant locally:
  ```
   docker compose up -d
  ```
4. Copy `.env.example` to `.env` (the defaults already point to the free
  local Ollama backend, no editing required unless you want to change the
   model):
5. Add your own achievements to `data/achievements/achievements.json`
  (a starter set is already included -- edit/extend it with your real bullets).
6. Index your achievements into Qdrant:
  ```
   python src/embed.py
  ```



### Switching to the paid Anthropic API later

If local model quality isn't good enough (smaller local models are weaker
at the structured-JSON classification and overclaim-checking steps than
Claude), you can switch backends without touching any pipeline code:

1. Uncomment `anthropic` in `requirements.txt` and run `pip install -r requirements.txt` again.
2. In `.env`, set `LLM_BACKEND=anthropic` and `ANTHROPIC_API_KEY=<your key>`.

Everything else (retrieval, embeddings, reranking) stays local and free
either way -- only the three LLM-reasoning steps (classify, rewrite, verify)
are affected by this switch.

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

