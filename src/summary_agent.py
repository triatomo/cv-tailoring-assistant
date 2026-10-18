"""
Agentic version of summary.py: a tool-calling agent writes the per-JD headline
and the middle sentences of the profile summary.

Instead of one prompt with pre-picked evidence, the agent reads the full JD,
decides which requirements matter most, looks up the candidate's achievements
for them with `search_achievements` (the pipeline's hybrid retrieval), and runs
its drafts through `check_draft` -- the same deterministic honesty, no-duplication,
and grounding checks summary.py uses -- revising until a draft passes. Only a
draft that passed `check_draft` is used; the agent's final reply is ignored.
The fixed opening/closing sentences from data/profile.json stay as they are.

Runs on Ollama tool calling with SUMMARY_AGENT_MODEL (default gemma4:31b-cloud).
With LLM_BACKEND=anthropic it falls back to the one-shot summary.generate_summary.
"""
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from classify_jd import JDClassification
from cv_profile import Profile
from ingest import Achievement
from llm_client import LLM_BACKEND, chat_ollama
from retrieve import retrieve_for_requirement
from summary import (
    HEADLINE_FROM_JD,
    HEADLINE_WITHOUT_JD_TITLE,
    CVSummary,
    _check_headline,
    _check_middle,
    ai_tenure,
    clean_job_title,
    fallback_summary,
    generate_summary,
    jd_headline,
)

SUMMARY_AGENT_MODEL = os.getenv("SUMMARY_AGENT_MODEL", "gemma4:31b-cloud")
MAX_TURNS = 15  # model replies (each may call tools) before giving up

AGENT_PROMPT = """You are writing the profile summary at the top of a candidate's CV,
tailored to the job description below. It must be honest: a recruiter should find every
claim backed up by the CV.

Job description:
---
{jd_text}
---

The summary starts with this fixed sentence (do not repeat or change it):
"{opening}"
and ends with this fixed sentence (do not repeat it):
"{closing}"

Candidate's job titles, most recent first: {titles}
The candidate has been an AI Engineer for only {ai_months} months, so describe AI work as
things they have built, never as expertise.

Your job is to write:
- HEADLINE: {headline_instruction}
- MIDDLE: 2-3 sentences (40-70 words) between the fixed sentences. Each covers a DIFFERENT
  thing this job cares most about -- what the role is for, not just tool names -- and
  summarizes the candidate's matching experience as a theme (the kind of work, domains, and
  who it served), drawing on several achievements where possible. The achievements are listed
  as bullets right below the summary, so a recruiter must not read the same thing twice:
  never copy their wording and never repeat their metrics or numbers.
  Wrap 3-6 key tools or terms in **double asterisks**.

How to work:
1. Read the job description and decide the 2-3 things it cares about most.
2. Call search_achievements for each of them to find the candidate's real achievements.
3. Write each MIDDLE sentence as a broader summary of the achievements you found, in your own
   words, using the job's wording only where those achievements support it.
4. Call check_draft with your headline and middle. If it reports problems, fix them and call
   check_draft again. Keep going until it returns OK, then reply with just: DONE

Rules (check_draft enforces them):
- CV style: no "I", "my", or "As an ..."; start sentences with a past-tense verb.
- No run of more than 5 consecutive words from any achievement, and no numbers.
- Only name tools and technologies that appear in the achievements. Never claim more
  ownership than they do: if they say "Co-designed" or "Contributed to", do not write
  "Designed", "Led", or "Owned". Keep counts: one platform stays one platform, not "platforms".
- Do not mention years of experience. Do not list skills.
- Do not use: expert, expertise, proven, seasoned, extensive, deep, leading, senior,
  specialist, results-driven, proficient, successfully.
- Job terminology the candidate has NO evidence for -- never use: {unsupported_terminology}
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_achievements",
            "description": "Search the candidate's CV achievements. Returns the best matches with id, role, and full text.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A job requirement or topic to find evidence for."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_draft",
            "description": "Check a headline + middle sentences for honesty and grounding. Returns OK or a list of problems to fix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "middle": {"type": "string", "description": "The 2-3 middle sentences, with **bold** markers."},
                },
                "required": ["headline", "middle"],
            },
        },
    },
]


def generate_summary_with_agent(
    jd_text: str,
    classification: JDClassification,
    profile: Profile,
    achievements: List[Achievement],
    embedding_model: SentenceTransformer,
    matched_requirements: Optional[Dict[str, str]] = None,
) -> CVSummary:
    if LLM_BACKEND != "ollama":
        return generate_summary(classification, profile, achievements, matched_requirements)

    achievements_text = "\n".join(f"- ({a.role}) {a.text}" for a in achievements)
    source = "\n".join([profile.default_summary, "Skills: " + ", ".join(profile.skills), achievements_text])
    unsupported_terms = [t for t in classification.key_terminology if t.lower() not in source.lower()]
    job_title = classification.job_title.strip()
    if job_title:
        headline_instruction = HEADLINE_FROM_JD.format(clean_title=jd_headline(job_title))
    else:
        headline_instruction = HEADLINE_WITHOUT_JD_TITLE
    _, ai_months = ai_tenure(profile)
    prompt = AGENT_PROMPT.format(
        jd_text=jd_text.strip(),
        opening=profile.summary_opening,
        closing=profile.summary_closing,
        titles=", ".join(dict.fromkeys(entry.role for entry in profile.experience)),
        ai_months=ai_months,
        headline_instruction=headline_instruction,
        unsupported_terminology=", ".join(unsupported_terms) or "(none)",
    )

    accepted: Optional[CVSummary] = None
    good_headline: Optional[str] = None
    issues: List[str] = []

    def search_achievements(query: str) -> str:
        cv_ids = {a.id for a in achievements}  # only roles that are on the CV
        found = [r for r in retrieve_for_requirement(query, classification, embedding_model, top_k=8) if r.achievement.id in cv_ids][:5]
        return "\n".join(f"[{r.achievement.id}] ({r.achievement.role}) {r.achievement.text}" for r in found)

    def check_draft(headline: str, middle: str) -> str:
        nonlocal accepted, good_headline, issues
        # With a JD title the headline is fixed to its role; the model only picks one without.
        headline = jd_headline(job_title) if job_title else clean_job_title(headline)
        middle = " ".join(middle.split())
        headline_issues = [] if job_title else _check_headline(headline, job_title, profile)
        if not headline_issues:
            good_headline = headline
        issues = headline_issues + _check_middle(headline, middle, source, profile, classification, achievements)
        if issues:
            return "Problems to fix:\n" + "\n".join(f"- {issue}" for issue in issues)
        summary = " ".join(p for p in [profile.summary_opening, middle, profile.summary_closing] if p)
        accepted = CVSummary(headline=headline, summary=summary, is_generated=True)
        return "OK"

    tools: Dict[str, Callable[..., str]] = {"search_achievements": search_achievements, "check_draft": check_draft}
    messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        for _ in range(MAX_TURNS):
            reply = chat_ollama(messages, TOOLS, SUMMARY_AGENT_MODEL)
            messages.append(reply)
            tool_calls = reply.get("tool_calls") or []
            if not tool_calls:
                if accepted:
                    break
                messages.append({"role": "user", "content": "No draft has passed check_draft yet. Call check_draft with a draft."})
                continue
            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args)
                print(f"  summary agent -> {name}({', '.join(f'{k}={str(v)[:60]!r}' for k, v in args.items())})", file=sys.stderr)
                try:
                    result = tools[name](**args) if name in tools else f"Unknown tool: {name}"
                except TypeError as e:
                    result = f"Bad arguments for {name}: {e}"
                messages.append({"role": "tool", "tool_name": name, "content": result})
            if accepted:
                break
    except Exception as e:  # network / model errors: fall back instead of failing the whole CV
        issues = issues + [f"Summary agent failed: {e}"]

    return accepted or fallback_summary(profile, job_title, good_headline, issues or ["Summary agent gave up"])
