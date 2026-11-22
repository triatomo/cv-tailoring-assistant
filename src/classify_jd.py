"""
Classify a JD into a role category and extract its key terminology/requirements
using the LLM. This output is used downstream to bias retrieval (e.g., up-weight
"stakeholder"/"reporting" tags for a BI JD, up-weight "llm"/"agent" tags for an
AI Engineer JD) and to drive the grounded-rewrite terminology substitution.
"""
import json
import re
from typing import List
from pydantic import BaseModel

from llm_client import call_llm

ROLE_CATEGORIES = [
    "AI/ML Engineer",
    "Data Engineer",
    "AI Data Engineer",
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
   Use "AI Data Engineer" when the role is both data engineering and AI/LLM/ML
   engineering (e.g. "AI Data Engineer", data pipelines feeding LLM or ML systems).
3. Extract two lists:
   - key_terminology: the specific vocabulary/terms this JD uses that a candidate
     should mirror in their CV (e.g., "semantic layer", "grain", "drift detection",
     "governed metrics"). Focus on distinctive, role-specific terms, not generic
     words like "experience" or "team".
   - key_requirements: everything the JD asks the candidate to do or have, as short
     phrases in the order the JD lists them. Cover BOTH the responsibilities ("what
     you'll do", e.g. "build and maintain data infrastructure and pipelines",
     "collaborate with external partners to resolve problems") AND the skills and
     experience lines (e.g. "incident management and test automation", "Kubernetes",
     "dbt or LookML experience"). Keep every line -- do not reduce the JD to tool
     names. Leave out degrees, total years of experience on their own, languages,
     soft skills on their own ("strong communication"), and location or remote-work
     conditions: achievements can't demonstrate those.

The candidate's CV material is in English, so write key_terminology and
key_requirements in English, translating them if the JD is in another language
(keep tool and product names as written). Copy job_title as written, untranslated.

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
    classification = JDClassification(**data)
    classification.role_category = category_from_title(classification.job_title, classification.role_category)
    return classification


AI_TITLE = re.compile(r"\b(AI|ML|LLM|GenAI|Machine Learning|Artificial Intelligence)\b", re.IGNORECASE)
# Checked in order; the first matching title pattern decides the category.
TITLE_CATEGORIES = [
    (re.compile(r"\banalytics\b|\bBI\b|business intelligence", re.IGNORECASE), "BI/Analytics"),
    (re.compile(r"\bdata scien", re.IGNORECASE), "Data Scientist"),
    (re.compile(r"\bdata engineer", re.IGNORECASE), "Data Engineer"),
    (re.compile(r"\bplatform\b|\binfrastructure\b|\bdevops\b|\bSRE\b", re.IGNORECASE), "Platform/Infra"),
]


def category_from_title(job_title: str, llm_category: str) -> str:
    """The JD title outranks the LLM's guess when it is clear: "Analytics Engineer" is
    BI/Analytics, never an AI category. AI categories need an AI term in the title."""
    if AI_TITLE.search(job_title):
        return "AI Data Engineer" if re.search(r"\bdata\b", job_title, re.IGNORECASE) else "AI/ML Engineer"
    for pattern, category in TITLE_CATEGORIES:
        if pattern.search(job_title):
            return category
    if llm_category in ("AI Data Engineer", "AI/ML Engineer"):
        return "Other"  # no AI in the title: don't let the LLM call it an AI role
    return llm_category


if __name__ == "__main__":
    sample_jd = """
    We are looking for a Senior Data Engineer with experience in semantic
    modeling, governed metrics, dbt or LookML, and Git-reviewed knowledge assets.
    """
    result = classify_jd(sample_jd)
    print(result.model_dump_json(indent=2))
