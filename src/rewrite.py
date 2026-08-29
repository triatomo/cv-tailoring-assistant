"""
Grounded rewrite step: rephrase a matched achievement using the JD's own
terminology, under a strict constraint against introducing unsupported claims.
This is the piece that solves "I have the experience but need to use the JD's
words" -- WITHOUT letting the model invent tools/scope that weren't there.
"""
import os
from pydantic import BaseModel
from anthropic import Anthropic

from ingest import Achievement

MODEL = "claude-haiku-4-5-20251001"

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
4. Return ONLY the rewritten bullet text, no preamble, no explanation, no quotes.
"""


class RewriteResult(BaseModel):
    original_text: str
    rewritten_text: str


def rewrite_achievement(
    achievement: Achievement,
    requirement: str,
    terminology: list,
    client: Anthropic = None,
) -> RewriteResult:
    client = client or Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = REWRITE_PROMPT.format(
        original_text=achievement.text,
        terminology=", ".join(terminology) if terminology else "(none provided)",
        requirement=requirement,
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    rewritten = response.content[0].text.strip().strip('"')
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
