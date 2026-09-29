"""
Grounded rewrite step: rephrase a matched achievement using the JD's own
terminology, under a strict constraint against introducing unsupported claims.
This is the piece that solves "I have the experience but need to use the JD's
words" -- WITHOUT letting the model invent tools/scope that weren't there.
"""
from pydantic import BaseModel

from ingest import Achievement
from llm_client import call_llm

REWRITE_PROMPT = """You are helping tailor a CV bullet point to match a job description's
terminology, WITHOUT changing what actually happened.

Original bullet (ground truth -- do not contradict or exceed this):
"{original_text}"

Job description terminology to consider using, if genuinely applicable:
{terminology}

Job requirement this bullet is meant to address:
"{requirement}"

Rules (do not break these):
1. Only substitute vocabulary for concepts that are ALREADY present in the
   original bullet. Do not introduce tools, frameworks, metrics, scope, or
   seniority (e.g., "led" when the original says "contributed to") that
   aren't in the original.
2. If none of the provided terminology maps cleanly onto the original bullet,
   return the original bullet unchanged rather than forcing a fit.
3. Keep the rewrite roughly the same length as the original.
4. Keep these key phrases word-for-word if they appear: {highlights}
5. Return ONLY the rewritten bullet text, no preamble, no explanation, no quotes.
"""


class RewriteResult(BaseModel):
    original_text: str
    rewritten_text: str


def rewrite_achievement(
    achievement: Achievement,
    requirement: str,
    terminology: list,
) -> RewriteResult:
    prompt = REWRITE_PROMPT.format(
        original_text=achievement.text,
        terminology=", ".join(terminology) if terminology else "(none provided)",
        requirement=requirement,
        highlights="; ".join(f'"{h}"' for h in achievement.highlights) or "(none)",
    )

    rewritten = call_llm(prompt, max_tokens=512).strip().strip('"')
    return RewriteResult(original_text=achievement.text, rewritten_text=rewritten)


if __name__ == "__main__":
    sample = Achievement(
        id="ach_006",
        company="Klarna Bank AB",
        role="Data Engineer",
        date_range="12/2021 - 01/2026",
        text="Modeled schemas and data semantics in AWS Glue Data Catalog to standardize reporting logic and ensure consistent metric definitions across markets.",
        tags=["data-modeling", "schema", "glue", "governance"],
    )
    result = rewrite_achievement(
        sample,
        requirement="Model governed metrics, dimensions, joins, and grain for business domains",
        terminology=["governed metrics", "semantic layer", "grain"],
    )
    print("Original: ", result.original_text)
    print("Rewritten:", result.rewritten_text)
