"""
Writes the CV profile summary with the cv-summary skill from Claude Desktop, run
through the Claude Code CLI (`claude -p`) on the Claude Pro subscription.

The skill's own SKILL.md is the system prompt, so edits made to the skill in
Claude Desktop apply here too. The model gets the full JD plus the CV (roles,
achievements, education, skills) and returns the summary, the JD must-haves it
covers, and an alternative version as JSON.

Billing: the CLI is started without ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN and
with the API-provider switches removed, and the run is aborted unless the CLI
reports it authenticated with the logged-in subscription (apiKeySource "none").
Run `claude` once and /login with the Pro account if it isn't logged in.

Settings (env):
    CV_SUMMARY_SKILL  path to SKILL.md (default: newest ~/.claude/skills/**/cv-summary/SKILL.md)
    CV_SUMMARY_MODEL  model alias passed to `claude --model` (default: the CLI's default)
    CLAUDE_BIN        claude executable (default: "claude" on PATH)
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel

from classify_jd import JDClassification
from cv_profile import Profile
from ingest import Achievement

CLAUDE_BIN = os.getenv("CLAUDE_BIN", "claude")
CV_SUMMARY_MODEL = os.getenv("CV_SUMMARY_MODEL", "")
CLAUDE_TIMEOUT = 300
MAX_ATTEMPTS = 2
MIN_WORDS, MAX_WORDS = 45, 130  # the skill asks for ~60-100; leave some slack



class CVSummary(BaseModel):
    headline: str
    summary: str  # may contain **bold** markers (the default summary does)
    is_generated: bool  # False = fell back to the profile's default summary
    issues: List[str] = []
    review_notes: List[str] = []  # extra lines for the review notes (must-haves covered, alternative)


# Anything that would make the CLI bill the API (or another provider) instead of the subscription.
API_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
]

PIPELINE_INSTRUCTIONS = """

## Running inside the CV pipeline (overrides the chat Input/Output sections above)
You are called non-interactively: there is no chat and nobody to ask. The job description
and Tria's CV are in the user message; use the CV as her work experience. Do Step 1 and
Step 2 in your head, then reply with ONLY this JSON object (no prose, no code fences):
{
  "core_need": "what they're really hiring for, one line",
  "must_haves": ["the 4-6 JD must-haves"],
  "summary": "the summary paragraph, ready to paste",
  "covered": ["must-haves the summary covers"],
  "left_to_experience": ["must-haves deliberately left to the experience section"],
  "alternative": "one alternative version with a different emphasis",
  "alternative_emphasis": "what the alternative leans toward"
}
Plain text only inside the strings: no markdown, no bold markers.
"""


def jd_headline(job_title: str) -> str:
    """The CV headline for a JD with a title: the JD's role name, seniority included, e.g.
    "Lead Analytics Engineer (F/M/X)" -> "Lead Analytics Engineer", "Senior AI Engineer - Growth" -> "Senior AI Engineer"."""
    title = re.sub(r"\(.*?\)", " ", job_title)
    role = re.split(r"\s+[-–|]\s+|,", title)[0]  # drop a focus/team suffix
    return " ".join(role.split()).strip(" -–|,/")


def fallback_summary(profile: Profile, headline: str, issues: List[str]) -> CVSummary:
    """The profile's default summary, under the JD headline."""
    return CVSummary(headline=headline, summary=profile.default_summary, is_generated=False, issues=issues)


def find_skill() -> Path:
    override = os.getenv("CV_SUMMARY_SKILL")
    if override:
        return Path(override).expanduser()
    found = sorted(
        (Path.home() / ".claude" / "skills").glob("**/cv-summary/SKILL.md"),
        key=lambda p: p.stat().st_mtime,
    )
    if not found:
        raise FileNotFoundError("cv-summary skill not found under ~/.claude/skills; set CV_SUMMARY_SKILL")
    return found[-1]


def load_skill() -> str:
    text = find_skill().read_text(encoding="utf-8")
    return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()


def cv_text(profile: Profile, achievements: List[Achievement]) -> str:
    lines = [f"# CV of {profile.name}", ""]
    if profile.domain:
        lines.append(f"Industry: {profile.domain}")
    lines.append(f"Years of experience: {profile.years_of_experience}")
    if profile.languages:
        lines.append("Languages (fluent): " + ", ".join(profile.languages))
    lines += ["", "## Work experience"]
    for entry in profile.experience:
        lines.append(f"### {entry.role}, {entry.company} ({entry.location}), {entry.date_range}")
        lines += [
            f"- {a.text}" for a in achievements
            if a.company == entry.company and a.role == entry.role
        ]
    if profile.personal_projects:
        lines += ["", "## Personal projects"]
        lines += [f"- {p.name} ({p.subtitle}): {p.description}" for p in profile.personal_projects]
    lines += ["", "## Education"]
    lines += [f"- {e.degree}, {e.school}, {e.date_range}" for e in profile.education]
    lines += ["", "## Skills", ", ".join(profile.skills)]
    return "\n".join(lines)


def _subscription_env() -> Dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in API_ENV_VARS}


def call_claude(system_prompt: str, user_prompt: str) -> str:
    """One no-tools `claude -p` turn on the subscription; returns the reply text."""
    cmd = [
        CLAUDE_BIN, "-p",
        "--output-format", "stream-json", "--verbose",
        "--system-prompt", system_prompt,
        "--tools", "",
        "--strict-mcp-config",
        "--no-session-persistence",
    ]
    if CV_SUMMARY_MODEL:
        cmd += ["--model", CV_SUMMARY_MODEL]
    proc = subprocess.run(
        cmd, input=user_prompt, capture_output=True, text=True,
        env=_subscription_env(), timeout=CLAUDE_TIMEOUT,
    )
    events = []
    for line in proc.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
    if init is None:
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}")
    if init.get("apiKeySource") != "none":
        raise RuntimeError(f"claude CLI is using an API key ({init['apiKeySource']}), not the Pro subscription")
    result = next((e for e in events if e.get("type") == "result"), None)
    if result is None or result.get("is_error"):
        raise RuntimeError(f"claude CLI error: {(result or {}).get('result') or proc.stderr.strip()[-500:]}")
    return result["result"]


def _parse(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in reply")
    data = json.loads(match.group(0))
    if not isinstance(data.get("summary"), str) or not data["summary"].strip():
        raise ValueError("reply has no summary")
    return data


def _check(summary: str) -> List[str]:
    issues = []
    words = len(summary.split())
    if not MIN_WORDS <= words <= MAX_WORDS:
        issues.append(f"Summary is {words} words; it must be about 60-100")
    if re.search(r"\b(I|my|me)\b", summary):
        issues.append("Written in first person; use implied first person")
    if "**" in summary or "\n" in summary.strip():
        issues.append("Must be one plain paragraph without markdown")
    return issues


def generate_summary(
    jd_text: str,
    classification: JDClassification,
    profile: Profile,
    achievements: List[Achievement],
) -> CVSummary:
    job_title = classification.job_title.strip()
    headline = jd_headline(job_title) if job_title else profile.default_headline
    try:
        system_prompt = load_skill() + PIPELINE_INSTRUCTIONS
    except OSError as e:
        return fallback_summary(profile, headline, [str(e)])
    user_prompt = f"# Job description\n\n{jd_text.strip()}\n\n{cv_text(profile, achievements)}"

    issues: List[str] = []
    for _ in range(MAX_ATTEMPTS):
        prompt = user_prompt
        if issues:
            prompt += "\n\nA previous draft was rejected for: " + "; ".join(issues)
        try:
            data = _parse(call_claude(system_prompt, prompt))
        except (RuntimeError, ValueError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Claude summary failed: {e}", file=sys.stderr)
            issues = [str(e)]
            continue
        summary = " ".join(data["summary"].split())
        issues = _check(summary)
        if issues:
            continue
        notes = [
            f"- Core need: {data.get('core_need', '')}",
            f"- Must-haves covered: {', '.join(data.get('covered', []))}",
            f"- Left to experience section: {', '.join(data.get('left_to_experience', [])) or '(none)'}",
            f"- Alternative ({data.get('alternative_emphasis', 'different emphasis')}): {data.get('alternative', '')}",
        ]
        return CVSummary(headline=headline, summary=summary, is_generated=True, review_notes=notes)

    return fallback_summary(profile, headline, issues)
