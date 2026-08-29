"""
Classify a JD into a role category and extract its key terminology/requirements
using the LLM. This output is used downstream to bias retrieval (e.g., up-weight
"stakeholder"/"reporting" tags for a BI JD, up-weight "llm"/"agent" tags for an
AI Engineer JD) and to drive the grounded-rewrite terminology substitution.
"""
import json
import os
from typing import List
from pydantic import BaseModel
from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

ROLE_CATEGORIES = [
    "AI/ML Engineer",
    "Data Engineer",
    "BI/Analytics",
    "Platform/Infra",
    "Data Scientist",
    "Other",
]


class JDClassification(BaseModel):
    role_category: str
    key_terminology: List[str]
    key_requirements: List[str]


CLASSIFY_PROMPT = """You are analyzing a job description to help tailor a candidate's CV.

Job description:
---
{jd_text}
---

Do two things:
1. Classify this JD into exactly one of these role categories: {categories}
2. Extract two lists:
   - key_terminology: the specific vocabulary/terms this JD uses that a candidate
     should mirror in their CV (e.g., "semantic layer", "grain", "drift detection",
     "governed metrics"). Focus on distinctive, role-specific terms, not generic
     words like "experience" or "team".
   - key_requirements: the concrete skills/requirements listed, as short phrases
     (e.g., "5+ years SQL", "Kubernetes", "dbt or LookML experience").

Respond with ONLY a JSON object, no other text, matching this exact shape:
{{
  "role_category": "<one of {categories}>",
  "key_terminology": ["...", "..."],
  "key_requirements": ["...", "..."]
}}
"""


def classify_jd(jd_text: str, client: Anthropic = None) -> JDClassification:
    client = client or Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = CLASSIFY_PROMPT.format(
        jd_text=jd_text,
        categories=", ".join(f'"{c}"' for c in ROLE_CATEGORIES),
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()
    # Defensive: strip markdown code fences if the model adds them anyway
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    data = json.loads(raw_text)
    return JDClassification(**data)


if __name__ == "__main__":
    sample_jd = """
    We are looking for a Senior Data Engineer with experience in semantic
    modeling, governed metrics, dbt or LookML, and Git-reviewed knowledge assets.
    """
    result = classify_jd(sample_jd)
    print(result.model_dump_json(indent=2))
