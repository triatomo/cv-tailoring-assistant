"""
Generate a per-JD headline + profile summary for the top of the CV, grounded
in the candidate's achievements, skills, and hand-written default summary.

The summary's opening sentence (who the candidate is, years of experience) and
closing sentence (working style) are fixed text from data/profile.json, so the
seniority framing can never drift. The LLM only writes the headline and the
2-3 middle sentences that tailor the summary to the JD. Those go through
deterministic checks (no inflated seniority wording, no copied bullet wording
or metrics, every named tool and ownership claim backed by the achievements);
if every attempt fails, fall back to the default summary.
"""
import re
from datetime import date
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

from classify_jd import JDClassification
from cv_profile import Profile
from ingest import Achievement
from llm_client import call_llm

SUMMARY_PROMPT = """You are tailoring the profile summary at the top of a candidate's CV
to a specific job. It must be honest: a recruiter should find every claim backed up
by the CV below it.

The summary already starts with this sentence (do not repeat it):
"{opening}"

Candidate's job titles, most recent first: {titles}
The candidate has been an AI Engineer for only {ai_months} months, so describe AI work as
things they have built, never as expertise.

The job's requirements, each with the candidate's achievements that match it:
{evidence}

Job terminology you may reuse (the candidate's material supports it): {terminology}
Job terminology the candidate has NO evidence for -- never use these: {unsupported_terminology}

Write two things:

HEADLINE: {headline_instruction}

MIDDLE: 2-3 sentences (40-70 words), each covering a DIFFERENT job requirement above.
Summarize the THEME of the matching achievements (the kind of work, domains, and who it
served) in broader words -- the bullets appear right below the summary, so never copy
their wording, never reuse their metrics or numbers. Use the job's wording where the
achievements support it. Wrap 3-6 key tools or terms in **double asterisks**.
- CV style: no "I", "my", or "As an AI Engineer"; start sentences with a past-tense verb.
- Do not mention years of experience.
- Do not use: expert, expertise, proven, seasoned, extensive, deep, leading, senior,
  specialist, results-driven, proficient, successfully.
- Do not list skills.
- Do not add tools or technologies that are not in the achievements above.
- Never claim more ownership than the achievements: if they say "Co-designed" or
  "Contributed to", do not write "Designed", "Led", or "Owned".

Reply in exactly this format, with no other text:
HEADLINE: <headline>
MIDDLE: <sentences>
"""

HEADLINE_FROM_JD = """exactly "{clean_title}" -- the job's role title with seniority, company,
and any focus suffix removed. Write it as given."""

HEADLINE_WITHOUT_JD_TITLE = """one of the candidate's job titles exactly as written
above, then " - ", then the focus of this job in 2-5 words. Example:
"AI Engineer - Agent Workflows & Integrations"."""


class CVSummary(BaseModel):
    headline: str
    summary: str  # may contain **bold** markers
    is_generated: bool  # False = fell back to the profile's default summary
    issues: List[str] = []


MAX_ATTEMPTS = 3  # retries are told which problems to avoid before falling back
MAX_BOLD_TERMS = 8  # more than this and the middle is a skills dump
MAX_HEADLINE_WORDS = 6

# Seniority levels stripped from the JD title: the candidate isn't senior.
SENIORITY_PATTERN = re.compile(r"\b(senior|sr\.?|lead|principal|staff)(?=\s|$|[^\w])", re.IGNORECASE)
TITLE_STOPWORDS = {"and", "of", "the", "for", "in", "&", "-"}
# Words too generic to decide whether a headline is "the same role" as the JD title.
GENERIC_TITLE_WORDS = {"engineer", "developer", "software", "specialist"}

# Seniority words the generated text may not use about the candidate: under a
# year of AI engineering doesn't make anyone an expert, and small local models
# reach for these by default.
INFLATED_WORDS = [
    "expert", "expertise", "proven", "seasoned", "results-driven", "extensive", "mastery",
    "specialist", "spearheaded", "world-class", "senior", "leading", "proficient",
    "successfully", "deep knowledge", "deep experience", "thought leader",
]


def _parse_month(date_str: str) -> date:
    month, year = date_str.strip().split("/")
    return date(int(year), int(month), 1)


def ai_tenure(profile: Profile, today: Optional[date] = None) -> Tuple[str, int]:
    """(start "MM/YYYY", months) of the candidate's AI engineering experience."""
    today = today or date.today()
    starts = [
        entry.date_range.split("-")[0].strip()
        for entry in profile.experience
        if re.search(r"\bAI\b", entry.role)
    ]
    if not starts:
        return "never", 0
    start = min(starts, key=_parse_month)
    start_date = _parse_month(start)
    return start, max(1, (today.year - start_date.year) * 12 + today.month - start_date.month)


def _parse_response(raw_text: str) -> Tuple[str, str]:
    """Pull HEADLINE/MIDDLE out of the reply; plain text survives small models better than JSON."""
    headline = re.search(r"HEADLINE:\s*(.+)", raw_text)
    middle = re.search(r"MIDDLE:\s*(.+)", raw_text, re.DOTALL)
    if not headline or not middle:
        raise ValueError("reply did not contain HEADLINE: and MIDDLE: lines")
    clean = lambda t: " ".join(t.split()).strip().strip('"')
    return clean(headline.group(1)), clean(middle.group(1))


def clean_job_title(title: str) -> str:
    """Drop seniority levels and parentheticals from a JD title: "Senior AI Engineer (m/f/d)" -> "AI Engineer"."""
    title = re.sub(r"\(.*?\)", " ", title)
    title = SENIORITY_PATTERN.sub(" ", title)
    title = re.sub(r"\s+(I{1,3}|IV)\s*$", " ", title)  # level suffixes like "Engineer II"
    title = re.sub(r"\s*([-–|,/])\s*(?=[-–|,/]|$)", " ", title)  # separators left dangling
    title = re.sub(r"\s+,", ",", title)
    return " ".join(title.split()).strip(" -–|,/")


def _title_words(title: str) -> set:
    return {w for w in re.findall(r"[a-z0-9/&+-]+", title.lower()) if w not in TITLE_STOPWORDS}


def _role_part(title: str) -> str:
    """The role before any focus/team suffix: "AI Engineer - Growth" -> "AI Engineer"."""
    return re.split(r"\s+[-–|]\s+|,", title)[0]


def jd_headline(job_title: str) -> str:
    """The CV headline for a JD with a title: just the role, e.g.
    "Lead Analytics Engineer (F/M/X)" -> "Analytics Engineer", "Senior AI Engineer - Growth" -> "AI Engineer"."""
    return _role_part(clean_job_title(job_title))


def _check_headline(headline: str, job_title: str, profile: Profile) -> List[str]:
    """The headline should be the JD title made to fit: short, not senior, recognisably the same role."""
    issues = []
    if len(_title_words(headline)) > MAX_HEADLINE_WORDS:
        issues.append(f"Headline is too long; keep it to at most {MAX_HEADLINE_WORDS} words")
    if job_title:
        # The role part ("AI Engineer" in "AI Engineer - Growth") must keep the JD's
        # role words; a long JD role is free to be shortened.
        jd_role = _title_words(_role_part(clean_job_title(job_title))) - GENERIC_TITLE_WORDS
        missing = jd_role - _title_words(_role_part(headline))
        if len(jd_role) <= 2 and missing:
            issues.append(
                f"Headline should keep the job's role '{_role_part(clean_job_title(job_title))}' "
                f"before any ' - ' (missing: {', '.join(sorted(missing))})"
            )
    else:
        titles = {entry.role.lower() for entry in profile.experience}
        if not any(headline.lower().startswith(title) for title in titles):
            issues.append(f"Headline must start with a job title the candidate held: {', '.join(sorted(titles))}")
    return issues


def _check_honesty(headline: str, middle: str, profile: Profile) -> List[str]:
    """Deterministic checks for inflated seniority, first-person wording, and skill dumps."""
    issues = []
    plain_middle = re.sub(r"\*\*", "", middle)
    for word in INFLATED_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", f"{headline} {plain_middle}", re.IGNORECASE):
            issues.append(f"Inflated seniority wording: '{word}'")
    if re.search(r"\b(I|my|me)\b|\bAs an?\b", plain_middle):
        issues.append("Written in first person; use CV style starting with past-tense verbs")
    if re.search(r"\d+\+?\s*(years|yrs)", plain_middle, re.IGNORECASE):
        issues.append("Mentions years of experience; the opening sentence already covers that")
    if len(re.findall(r"\*\*(.+?)\*\*", middle)) > MAX_BOLD_TERMS:
        issues.append("Lists too many skills; name what was built instead")
    return issues


# The summary sits right above the bullets, so it must not repeat them: no run of
# this many consecutive words from any one achievement, and no metrics.
MAX_COPIED_WORDS = 5
# Ownership words the summary may only use if the candidate's achievements do.
OWNERSHIP_WORDS = ["led", "lead", "owned", "own", "owns", "managed", "headed", "directed", "drove", "architected"]


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9+#/-]*", text.lower())


def _longest_shared_run(sentence: str, text: str) -> int:
    """Length of the longest run of consecutive words the sentence shares with text."""
    a, b = _words(sentence), _words(text)
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def _check_grounding(middle: str, source: str, achievements: List[Achievement]) -> List[str]:
    """The middle must summarize, not copy: no bullet wording or metrics, and every
    named tool/term and ownership claim must exist in the candidate's material."""
    issues = []
    plain_middle = re.sub(r"\*\*", "", middle)
    achievements_text = " ".join(a.text for a in achievements)
    ach_words = set(_words(achievements_text))
    for sentence in re.split(r"(?<=[.!?])\s+", plain_middle.strip()):
        if not sentence:
            continue
        copied = max(achievements, key=lambda a: _longest_shared_run(sentence, a.text))
        if _longest_shared_run(sentence, copied.text) > MAX_COPIED_WORDS:
            issues.append(
                f"Copies the wording of {copied.id} -- the recruiter reads that bullet right below; "
                f"summarize the theme in different words: '{sentence}'"
            )
        if re.search(r"\d", sentence):
            issues.append(f"Contains a number; leave metrics to the bullets: '{sentence}'")
        # "platforms" when every achievement says "platform" inflates scope.
        for word in set(_words(sentence)):
            if len(word) > 4 and word.endswith("s") and word[:-1] in ach_words and word not in ach_words:
                issues.append(f"'{word}' is plural, but the achievements only mention one {word[:-1]}")
        for word in OWNERSHIP_WORDS:
            if re.search(rf"\b{word}\b", sentence, re.IGNORECASE) and not re.search(rf"\b{word}\b", achievements_text, re.IGNORECASE):
                issues.append(f"Ownership word '{word}' is not backed by any achievement")
    # Named things (bold terms, capitalized names mid-sentence) must come from the material.
    terms = set(re.findall(r"\*\*(.+?)\*\*", middle))
    terms |= set(re.findall(r"(?<=[a-z,] )([A-Z][\w./+-]*(?: [A-Z][\w./+-]*)*)", plain_middle))
    # Achievement tags count as material too ("data-modeling" backs "data modeling").
    known = (source + " " + " ".join(t.replace("-", " ") for a in achievements for t in a.tags)).lower()
    for term in sorted({t.strip(".,;:") for t in terms}):
        if term.lower() not in known:
            issues.append(f"'{term}' is not in the candidate's material")
    return issues


def _check_middle(
    headline: str, middle: str, source: str, profile: Profile,
    classification: JDClassification, achievements: List[Achievement],
) -> List[str]:
    """Return the problems found in the generated text (empty list = OK).

    All checks are deterministic: an 8B model judging its own output rejected
    good drafts with made-up complaints and let small inflations through.
    """
    honesty_issues = _check_honesty(headline, middle, profile)
    if honesty_issues:
        return honesty_issues

    # Small local models happily echo JD terms back, so any JD term used must
    # also appear in the source material.
    plain_middle = re.sub(r"\*\*", "", middle)
    echoed_terms = [
        term for term in classification.key_terminology
        if term.lower() in plain_middle.lower() and term.lower() not in source.lower()
    ]
    if echoed_terms:
        return [f"JD term not supported by your material: '{t}'" for t in echoed_terms]

    return _check_grounding(middle, source, achievements)


def generate_summary(
    classification: JDClassification,
    profile: Profile,
    achievements: List[Achievement],
    matched_requirements: Optional[Dict[str, str]] = None,
) -> CVSummary:
    """matched_requirements: achievement id -> the JD requirement it matched best."""
    matched_requirements = matched_requirements or {}
    achievements_text = "\n".join(f"- ({a.role}) {a.text}" for a in achievements)
    source = "\n".join([profile.default_summary, "Skills: " + ", ".join(profile.skills), achievements_text])
    supported_terms = [t for t in classification.key_terminology if t.lower() in source.lower()]
    unsupported_terms = [t for t in classification.key_terminology if t not in supported_terms]
    by_requirement: Dict[str, List[Achievement]] = {}
    for a in achievements:
        if a.id in matched_requirements:
            by_requirement.setdefault(matched_requirements[a.id], []).append(a)
    evidence = "\n".join(
        f"* {requirement}\n" + "\n".join(f"  - ({a.role}) {a.text}" for a in matched)
        for requirement, matched in by_requirement.items()
    ) or achievements_text
    titles = list(dict.fromkeys(entry.role for entry in profile.experience))
    ai_start, ai_months = ai_tenure(profile)
    job_title = classification.job_title.strip()
    if job_title:
        headline_instruction = HEADLINE_FROM_JD.format(clean_title=jd_headline(job_title))
    else:
        headline_instruction = HEADLINE_WITHOUT_JD_TITLE
    prompt = SUMMARY_PROMPT.format(
        headline_instruction=headline_instruction,
        opening=profile.summary_opening,
        titles=", ".join(titles),
        ai_months=ai_months,
        evidence=evidence,
        terminology=", ".join(supported_terms) or "(none)",
        unsupported_terminology=", ".join(unsupported_terms) or "(none)",
    )

    issues: List[str] = []
    # Even if every summary draft fails, a headline that passed its own checks is kept.
    good_headline: Optional[str] = None
    for _ in range(MAX_ATTEMPTS):
        attempt_prompt = prompt
        if issues:
            attempt_prompt += (
                "\nA previous draft was rejected for these problems -- do not repeat them:\n"
                + "\n".join(f"- {issue}" for issue in issues)
            )
        try:
            headline, middle = _parse_response(call_llm(attempt_prompt, max_tokens=512))
            # With a JD title the headline is fixed to its role; the model only picks one without.
            headline = jd_headline(job_title) if job_title else clean_job_title(headline)
            headline_issues = [] if job_title else _check_headline(headline, job_title, profile)
            if not headline_issues:
                good_headline = headline
            issues = headline_issues + _check_middle(headline, middle, source, profile, classification, achievements)
        except (ValueError, KeyError) as e:
            issues = [f"Could not parse LLM response: {e}"]
            continue
        if not issues:
            summary = " ".join(p for p in [profile.summary_opening, middle, profile.summary_closing] if p)
            return CVSummary(headline=headline, summary=summary, is_generated=True)

    return fallback_summary(profile, job_title, good_headline, issues)


def fallback_summary(profile: Profile, job_title: str, good_headline: Optional[str], issues: List[str]) -> CVSummary:
    """The profile's default summary, under the best headline available."""
    fallback_headline = jd_headline(job_title) if job_title else good_headline
    return CVSummary(
        headline=fallback_headline or profile.default_headline,
        summary=profile.default_summary,
        is_generated=False,
        issues=issues,
    )
