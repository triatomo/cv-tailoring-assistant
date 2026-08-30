"""
Self-check (corrective) pass: compare a rewritten bullet against its original
source and flag whether the rewrite introduces any claim NOT supported by the
original. This is the CRAG-style correction step, applied to rewriting rather
than Q&A -- it's the guardrail against overclaiming (new tools, inflated scope,
inflated seniority) that got caught manually multiple times during CV tailoring.
"""
import json
from typing import List
from pydantic import BaseModel

from rewrite import RewriteResult
from llm_client import call_llm

VERIFY_PROMPT = """You are fact-checking a rewritten CV bullet against its original source
to catch overclaiming.

Original bullet:
"{original_text}"

Rewritten bullet:
"{rewritten_text}"

Check specifically for:
- New tools, frameworks, or technologies not mentioned in the original
- Inflated scope (e.g., "the whole platform" when original said "a component")
- Inflated seniority/ownership (e.g., "led" when original said "contributed to"
  or "supported")
- Invented metrics or numbers not present in the original
- Any claim a careful reader could not verify from the original alone

Respond with ONLY a JSON object, no other text:
{{
  "is_supported": true or false,
  "issues": ["<short description of each unsupported claim found>"],
  "recommendation": "use_rewrite" or "revert_to_original"
}}

If is_supported is true, issues should be an empty list and recommendation
should be "use_rewrite". If false, recommendation should be "revert_to_original".
"""


class VerificationResult(BaseModel):
    is_supported: bool
    issues: List[str]
    recommendation: str


def verify_rewrite(rewrite_result: RewriteResult) -> VerificationResult:
    # If the rewrite is byte-identical to the original, skip the LLM call --
    # nothing to verify, it's trivially supported.
    if rewrite_result.rewritten_text.strip() == rewrite_result.original_text.strip():
        return VerificationResult(is_supported=True, issues=[], recommendation="use_rewrite")

    prompt = VERIFY_PROMPT.format(
        original_text=rewrite_result.original_text,
        rewritten_text=rewrite_result.rewritten_text,
    )

    raw_text = call_llm(prompt, max_tokens=512).strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1:
        raw_text = raw_text[start : end + 1]

    data = json.loads(raw_text)
    return VerificationResult(**data)


def get_final_bullet(rewrite_result: RewriteResult, verification: VerificationResult) -> str:
    """Apply the verification recommendation to decide what text actually gets used."""
    if verification.recommendation == "use_rewrite":
        return rewrite_result.rewritten_text
    return rewrite_result.original_text


if __name__ == "__main__":
    sample = RewriteResult(
        original_text="Modeled schemas and data semantics in AWS Glue Data Catalog to standardize reporting logic.",
        rewritten_text="Owned end-to-end lineage and semantic layer architecture using dbt and Glue Data Catalog.",
    )
    result = verify_rewrite(sample)
    print(result.model_dump_json(indent=2))
