"""
Generate a per-JD headline + profile summary for the top of the CV, grounded
in the candidate's achievements, skills, and hand-written default summary.
The same overclaim guardrail as verify.py applies: if the check finds claims
the source material doesn't support, fall back to the default summary.
"""
import json
import re
from typing import List
from pydantic import BaseModel

from classify_jd import JDClassification
from cv_profile import Profile
from ingest import Achievement
from llm_client import call_llm

SUMMARY_PROMPT = """You are writing the profile summary at the top of a candidate's CV,
tailored to a specific job description.

Candidate's own default summary (ground truth, reuse its facts and tone):
"{default_summary}"

Candidate's skills: {skills}

Candidate's achievements (ground truth):
{achievements}

Target job:
- Role category: {role_category}
- Key requirements: {requirements}
- Job terminology you may mirror (the candidate's material supports it): {terminology}
- Job terminology the candidate has NO evidence for -- never mention these: {unsupported_terminology}

Write:
1. headline: a short CV headline (e.g. "Data Engineer - AI Platform") combining a job
   title the candidate actually holds or held with the focus of this job.
2. summary: 4-6 sentences (about 90-110 words) tailored to this job. Lead with the
   experience most relevant to the job, and mirror its terminology ONLY where the
   candidate's material genuinely supports it. Wrap the 8-14 most important skills,
   tools, and terms in **double asterisks** for bold.

Rules (do not break these):
- Do not mention any tool, technology, domain, metric, or years of experience that is
  not in the default summary, skills, or achievements above.
- Do not inflate seniority (no "led", "senior", "expert" unless the material says so).
- Keep the years-of-experience framing exactly as in the default summary: the 5+ years
  are in data engineering, and the AI/ML platform work started in 2026.
- Do not list the skills -- the CV has a separate skills section. Mention only the
  handful of tools most relevant to this job.
- Do not copy phrases from the job description's description of its ideal candidate;
  describe only what the candidate's material shows.

Respond with ONLY a JSON object, no other text:
{{"headline": "...", "summary": "..."}}
"""

CHECK_PROMPT = """You are fact-checking a CV summary against the candidate's source material
to catch overclaiming.

Source material:
---
{source}
---

Summary to check:
"{summary}"

Check for tools, technologies, domains, metrics, years of experience, or seniority
claims in the summary that the source material does not support. Rephrasing and
generalizing supported facts is fine.

Respond with ONLY a JSON object, no other text:
{{"is_supported": true or false, "issues": ["<short description of each unsupported claim>"]}}
"""


class CVSummary(BaseModel):
    headline: str
    summary: str  # may contain **bold** markers
    is_generated: bool  # False = fell back to the profile's default summary
    issues: List[str] = []


def _parse_json(raw_text: str) -> dict:
    raw_text = raw_text.strip()
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1:
        raw_text = raw_text[start : end + 1]
    return json.loads(raw_text)


MAX_ATTEMPTS = 2  # one retry, told which claims to avoid, before falling back


def _check_summary(summary: str, source: str, classification: JDClassification) -> List[str]:
    """Return the unsupported claims found in the summary (empty list = supported)."""
    plain_summary = re.sub(r"\*\*", "", summary)

    # Deterministic pre-check: small local models happily echo JD terms back and
    # then pass their own fact check, so any JD term used in the summary must
    # also appear somewhere in the source material.
    echoed_terms = [
        term for term in classification.key_terminology
        if term.lower() in plain_summary.lower() and term.lower() not in source.lower()
    ]
    if echoed_terms:
        return [f"JD term not supported by your material: '{t}'" for t in echoed_terms]

    check = _parse_json(call_llm(CHECK_PROMPT.format(source=source, summary=plain_summary), max_tokens=512))
    if check.get("is_supported", False):
        return []
    return check.get("issues", []) or ["Summary check flagged unsupported claims."]


def generate_summary(
    classification: JDClassification,
    profile: Profile,
    achievements: List[Achievement],
) -> CVSummary:
    achievements_text = "\n".join(f"- ({a.role}) {a.text}" for a in achievements)
    source = "\n".join([profile.default_summary, "Skills: " + ", ".join(profile.skills), achievements_text])
    supported_terms = [t for t in classification.key_terminology if t.lower() in source.lower()]
    unsupported_terms = [t for t in classification.key_terminology if t not in supported_terms]
    prompt = SUMMARY_PROMPT.format(
        default_summary=profile.default_summary,
        skills=", ".join(profile.skills),
        achievements=achievements_text,
        role_category=classification.role_category,
        requirements="; ".join(classification.key_requirements),
        terminology=", ".join(supported_terms) or "(none)",
        unsupported_terminology=", ".join(unsupported_terms) or "(none)",
    )

    issues: List[str] = []
    for _ in range(MAX_ATTEMPTS):
        attempt_prompt = prompt
        if issues:
            attempt_prompt += (
                "\nA previous draft was rejected for these unsupported claims -- do not repeat them:\n"
                + "\n".join(f"- {issue}" for issue in issues)
            )
        try:
            data = _parse_json(call_llm(attempt_prompt, max_tokens=1024))
            candidate = CVSummary(headline=data["headline"], summary=data["summary"], is_generated=True)
            issues = _check_summary(candidate.summary, source, classification)
        except (ValueError, KeyError) as e:
            issues = [f"Could not parse LLM response: {e}"]
            continue
        if not issues:
            return candidate

    return CVSummary(
        headline=profile.default_headline,
        summary=profile.default_summary,
        is_generated=False,
        issues=issues,
    )
