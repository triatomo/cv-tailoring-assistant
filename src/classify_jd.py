"""
Classify a JD into a role category and extract its key terminology/requirements
using the LLM. This output is used downstream to bias retrieval (e.g., up-weight
"stakeholder"/"reporting" tags for a BI JD, up-weight "llm"/"agent" tags for an
AI Engineer JD) and to drive the grounded-rewrite terminology substitution.
"""
import json
from typing import List
from pydantic import BaseModel

from llm_client import call_llm

ROLE_CATEGORIES = [
    "AI/ML Engineer",
    "Data Engineer",
    "BI/Analytics",
    "Platform/Infra",
    "Data Scientist",
    "Other",
]


class JDClassification(BaseModel):
    job_title: str = ""  # exactly as the JD writes it
    role_category: str
    key_terminology: List[str]
    key_requirements: List[str]


CLASSIFY_PROMPT = """You are analyzing a job description to help tailor a candidate's CV.

Job description:
---
{jd_text}
---

Do three things:
1. Copy the job title exactly as the JD writes it (e.g. "Senior AI Engineer - Growth").
2. Classify this JD into exactly one of these role categories: {categories}
3. Extract two lists:
   - key_terminology: the specific vocabulary/terms this JD uses that a candidate
     should mirror in their CV (e.g., "semantic layer", "grain", "drift detection",
     "governed metrics"). Focus on distinctive, role-specific terms, not generic
     words like "experience" or "team".
   - key_requirements: the concrete skills/requirements listed, as short phrases
     (e.g., "5+ years SQL", "Kubernetes", "dbt or LookML experience").

Respond with ONLY a JSON object, no other text, matching this exact shape:
{{
  "job_title": "<job title as written in the JD>",
  "role_category": "<one of {categories}>",
  "key_terminology": ["...", "..."],
  "key_requirements": ["...", "..."]
}}
"""


def classify_jd(jd_text: str) -> JDClassification:
    prompt = CLASSIFY_PROMPT.format(
        jd_text=jd_text,
        categories=", ".join(f'"{c}"' for c in ROLE_CATEGORIES),
    )

    raw_text = call_llm(prompt, max_tokens=1024)

    # Defensive: strip markdown code fences if the model adds them anyway,
    # and grab the first {...} block in case a local model adds chatter
    # before/after the JSON (free local models are less obedient about
    # "respond with ONLY JSON" than Claude tends to be).
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1:
        raw_text = raw_text[start : end + 1]

    data = json.loads(raw_text)
    return JDClassification(**data)


if __name__ == "__main__":
    sample_jd = """
    We are looking for a Senior Data Engineer with experience in semantic
    modeling, governed metrics, dbt or LookML, and Git-reviewed knowledge assets.
    """
    result = classify_jd(sample_jd)
    print(result.model_dump_json(indent=2))
