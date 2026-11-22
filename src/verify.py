"""
Self-check (corrective) pass: compare a rewritten bullet against its original
source and flag whether the rewrite introduces any claim NOT supported by the
original. This is the CRAG-style correction step, applied to rewriting rather
than Q&A -- it's the guardrail against overclaiming (new tools, inflated scope,
inflated seniority) that got caught manually multiple times during CV tailoring.
"""
import json
import re
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

Words the rewrite removed: {removed}
Words the rewrite added: {added}

Check specifically for:
- Replaced concepts: each added word or phrase must mean the SAME thing as what it
  replaced, not merely sound like the job description. Flag a swap that changes what
  was done, e.g. "observability" -> "operational KPIs", "model integration" ->
  "modeling layer integration", "datasets" -> "data lake".
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


def _word_diff(original: str, rewritten: str):
    """Words only in the original / only in the rewrite, in order of appearance.
    Spelling them out makes the checker judge each swap instead of skimming."""
    words = lambda t: re.findall(r"[\w'+#./-]+", t.lower())
    orig, new = words(original), words(rewritten)
    removed = list(dict.fromkeys(w for w in orig if w not in set(new)))
    added = list(dict.fromkeys(w for w in new if w not in set(orig)))
    return removed, added


def verify_rewrite(rewrite_result: RewriteResult) -> VerificationResult:
    # If the rewrite is byte-identical to the original, skip the LLM call --
    # nothing to verify, it's trivially supported.
    if rewrite_result.rewritten_text.strip() == rewrite_result.original_text.strip():
        return VerificationResult(is_supported=True, issues=[], recommendation="use_rewrite")

    removed, added = _word_diff(rewrite_result.original_text, rewrite_result.rewritten_text)
    prompt = VERIFY_PROMPT.format(
        original_text=rewrite_result.original_text,
        rewritten_text=rewrite_result.rewritten_text,
        removed=", ".join(removed) or "(none)",
        added=", ".join(added) or "(none)",
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

    try:
        return VerificationResult(**json.loads(raw_text))
    except (ValueError, TypeError):
        # Small local models sometimes emit broken JSON. An unverifiable rewrite
        # is treated as unsupported, so the original bullet is kept.
        return VerificationResult(
            is_supported=False,
            issues=["Could not parse the fact-check response"],
            recommendation="revert_to_original",
        )


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
