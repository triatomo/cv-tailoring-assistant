"""
Embed achievements with a local BGE model and upsert into Qdrant, storing
tags/company/role/date_range as payload metadata for later filtering/boosting.
"""
import os
from typing import List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ingest import Achievement, load_achievements
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

COLLECTION_NAME = "achievements"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # dimension for bge-small-en-v1.5


def get_qdrant_client() -> QdrantClient:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")


def embed_and_index(achievements: List[Achievement], client: QdrantClient, model: SentenceTransformer):
    texts = [a.text for a in achievements]
    # bge models recommend no special prefix for passages, only for queries
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    points = []
    for a, vec in zip(achievements, vectors):
        # Qdrant point IDs must be an unsigned int or a UUID -- extract the
        # numeric suffix from ids like "ach_01" and store the original
        # string id in the payload so we can still reference it downstream.
        numeric_id = int("".join(ch for ch in a.id if ch.isdigit()) or "0")
        points.append(
            PointStruct(
                id=numeric_id,
                vector=vec.tolist(),
                payload={
                    "achievement_id": a.id,
                    "text": a.text,
                    "company": a.company,
                    "role": a.role,
                    "date_range": a.date_range,
                    "tags": a.tags,
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} achievements into '{COLLECTION_NAME}'.")


def run():
    logging.info("Loading achievements...")
    achievements = load_achievements()
    logging.info(f"Loaded {len(achievements)} achievements.")
    logging.info("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logging.info(f"Loaded embedding model '{EMBEDDING_MODEL}'.")
    logging.info("Getting Qdrant client...")
    client = get_qdrant_client()
    logging.info("Ensuring collection...")
    ensure_collection(client)
    logging.info("Embedding and indexing achievements...")
    embed_and_index(achievements, client, model)


if __name__ == "__main__":
    run()
