"""
Hybrid retrieval: vector search (Qdrant) + BM25 keyword search, merged with
reciprocal rank fusion, then reranked with a cross-encoder. Retrieval scoring
is biased using the JD's role classification: achievements whose tags overlap
with terms relevant to the classified role category get a small score boost.
"""
from typing import List, Dict
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from embed import get_qdrant_client, COLLECTION_NAME, EMBEDDING_MODEL
from classify_jd import JDClassification
from ingest import load_achievements, Achievement

RERANKER_MODEL = "BAAI/bge-reranker-base"

# Simple mapping from role category -> tags to up-weight. Adjust/extend as
# your tag vocabulary grows; this is intentionally a small, editable table
# rather than something the LLM infers, so retrieval behavior stays stable
# and debuggable.
ROLE_TAG_BOOSTS: Dict[str, List[str]] = {
    "AI/ML Engineer": ["llm", "agent", "prompt-engineering", "mlops", "llmops", "production-ai"],
    "Data Engineer": ["etl", "pipeline", "schema", "data-modeling", "airflow", "spark", "kafka", "iceberg"],
    "BI/Analytics": ["stakeholder", "reporting", "dashboards", "translation", "domain-expertise", "looker"],
    "Platform/Infra": ["kubernetes", "ci-cd", "infrastructure", "observability", "monitoring", "docker"],
    "Data Scientist": ["data-science", "experimentation", "modeling", "fraud", "risk"],
}

RRF_K = 60  # standard reciprocal rank fusion constant
TAG_BOOST_WEIGHT = 0.05  # small additive boost per matching tag, applied post-fusion


@dataclass
class RetrievedAchievement:
    achievement: Achievement
    score: float


def _bm25_search(query: str, achievements: List[Achievement], top_k: int) -> List[str]:
    """Return achievement ids ranked by BM25, best first."""
    tokenized_corpus = [a.text.lower().split() for a in achievements]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(achievements, scores), key=lambda x: x[1], reverse=True)
    return [a.id for a, _ in ranked[:top_k]]


def _vector_search(query: str, model: SentenceTransformer, top_k: int) -> List[str]:
    """Return achievement ids ranked by vector similarity, best first."""
    client = get_qdrant_client()
    # bge models recommend a query instruction prefix for retrieval queries
    query_vec = model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        normalize_embeddings=True,
    )
    hits = client.search(collection_name=COLLECTION_NAME, query_vector=query_vec.tolist(), limit=top_k)
    return [hit.payload["achievement_id"] for hit in hits]


def _reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = RRF_K) -> Dict[str, float]:
    """Combine multiple ranked id lists into one fused score per id."""
    fused_scores: Dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, doc_id in enumerate(ranked_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return fused_scores


def hybrid_search(
    query: str,
    achievements: List[Achievement],
    embedding_model: SentenceTransformer,
    top_k: int = 10,
) -> List[str]:
    """Return achievement ids ranked by fused BM25 + vector score."""
    bm25_ids = _bm25_search(query, achievements, top_k=top_k * 2)
    vector_ids = _vector_search(query, embedding_model, top_k=top_k * 2)
    fused = _reciprocal_rank_fusion([bm25_ids, vector_ids])
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:top_k]]


def apply_role_boost(
    ranked_ids: List[str],
    achievements_by_id: Dict[str, Achievement],
    role_category: str,
) -> List[str]:
    """Re-sort ranked ids, adding a small boost for tag overlap with the role category."""
    boost_tags = set(ROLE_TAG_BOOSTS.get(role_category, []))
    if not boost_tags:
        return ranked_ids  # unknown category ("Other") -> no boost, leave order as-is

    scored = []
    for rank, doc_id in enumerate(ranked_ids):
        base_score = 1.0 / (rank + 1)
        achievement = achievements_by_id[doc_id]
        overlap = len(boost_tags.intersection(achievement.tags))
        boosted_score = base_score + (overlap * TAG_BOOST_WEIGHT)
        scored.append((doc_id, boosted_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scored]


def rerank(query: str, candidate_ids: List[str], achievements_by_id: Dict[str, Achievement], top_k: int = 5) -> List[RetrievedAchievement]:
    """Cross-encoder rerank of candidates against the query."""
    cross_encoder = CrossEncoder(RERANKER_MODEL)
    pairs = [(query, achievements_by_id[doc_id].text) for doc_id in candidate_ids]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return [
        RetrievedAchievement(achievement=achievements_by_id[doc_id], score=float(score))
        for doc_id, score in ranked[:top_k]
    ]


def retrieve_for_requirement(
    requirement: str,
    classification: JDClassification,
    embedding_model: SentenceTransformer,
    top_k: int = 5,
) -> List[RetrievedAchievement]:
    """Full pipeline: hybrid search -> role-boost re-sort -> cross-encoder rerank."""
    achievements = load_achievements()
    achievements_by_id = {a.id: a for a in achievements}

    fused_ids = hybrid_search(requirement, achievements, embedding_model, top_k=15)
    boosted_ids = apply_role_boost(fused_ids, achievements_by_id, classification.role_category)
    return rerank(requirement, boosted_ids, achievements_by_id, top_k=top_k)


if __name__ == "__main__":
    from classify_jd import classify_jd

    model = SentenceTransformer(EMBEDDING_MODEL)
    sample_jd = "We need a Senior Data Engineer with dbt, LookML, and semantic layer experience."
    classification = classify_jd(sample_jd)
    print("Classification:", classification.model_dump_json(indent=2))

    results = retrieve_for_requirement("semantic layer / governed metrics", classification, model)
    for r in results:
        print(f"  [{r.score:.3f}] {r.achievement.id}: {r.achievement.text[:80]}...")
