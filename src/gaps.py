"""
Gap detection: for a JD requirement where retrieval found no strong match
(below a similarity/rerank threshold), output an honest gap statement instead
of forcing a weak bullet to fit -- mirroring the "I haven't worked with X,
though my Y experience is adjacent" pattern used throughout manual CV tailoring.
"""
import re
from typing import Dict, Iterable, List, Optional
from pydantic import BaseModel

from retrieve import RetrievedAchievement

# bge-reranker-base scores are probabilities (0-1). Against the saved JDs, clear
# matches scored 0.05-0.7 ("Apache Spark, Kafka, or Flink" vs the Flink/Spark
# platform bullet: 0.70) while noise reached 0.03 ("Bachelor degree": 0.030,
# "Git/GitHub": 0.028). But short tool requirements score low even against a
# bullet naming that tool ("Apache Airflow" vs an Airflow bullet: 0.02), so a
# shared named tool counts as a match too (see is_match).
MATCH_THRESHOLD = 0.04


class GapAssessment(BaseModel):
    requirement: str
    has_strong_match: bool
    best_candidate: Optional[str] = None  # closest adjacent achievement text, if any
    gap_statement: Optional[str] = None


# Tags naming an activity rather than a skill: "automation" on a file-transfer API
# bullet must not answer "test automation".
GENERIC_TAGS = {
    "automation", "batch", "infrastructure", "leadership", "migration", "modeling", "monitoring",
    "performance", "pipeline", "platform", "reporting", "risk", "service", "standardization",
    "testing", "translate", "translation",
}


def tool_terms(skills: Iterable[str], tags: Iterable[str]) -> List[str]:
    """Named tools/topics to match literally: the candidate's skills (also without an
    "Apache " prefix, plus acronyms in multi-word skills: "LLM orchestration" -> "LLM")
    and achievement tags, except GENERIC_TAGS."""
    terms = {t for t in tags if t not in GENERIC_TAGS}
    for skill in skills:
        terms.add(skill)
        terms.add(re.sub(r"^Apache\s+", "", skill))
        words = skill.split()
        if len(words) > 1:
            terms.update(w for w in words if re.fullmatch(r"[A-Z]{2,}", w))
    return sorted(terms, key=str.lower)


def _names(term: str, text: str) -> bool:
    """Whole-word match allowing simple endings ("Dockerized", "agentic", "pipelines");
    a hyphen in a tag also matches a space or slash ("ci-cd" -> "CI/CD")."""
    pattern = r"[-\s/]".join(re.escape(part) for part in term.split("-"))
    return re.search(rf"(?<![\w-]){pattern}(?:s|es|ed|ic|ing|ized|ised)?(?![\w-])", text, re.IGNORECASE) is not None


def is_match(
    requirement: str,
    candidate: RetrievedAchievement,
    terms: Iterable[str] = (),
    threshold: float = MATCH_THRESHOLD,
    evidence: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """A strong rerank score, a term the requirement names that the bullet names or is
    tagged with, or a profile.term_evidence term the requirement names whose evidence
    phrase is in the bullet ("incident management" <- "incident response")."""
    if candidate.score >= threshold:
        return True
    achievement = candidate.achievement
    if any(
        _names(t, requirement) and (_names(t, achievement.text) or t in achievement.tags)
        for t in terms
    ):
        return True
    return any(
        term.lower() in requirement.lower() and any(p.lower() in achievement.text.lower() for p in phrases)
        for term, phrases in (evidence or {}).items()
    )


# The reranker can't judge responsibility requirements ("influence scope and
# requirements", "communicate risks to executives"): against the saved JDs the right
# bullet scored 0.002-0.008 while unrelated ones scored up to 0.011, so no threshold
# separates them. The LLM reads the best hybrid-search candidates and names the ones
# that really show the requirement; only ids from that list are accepted.
JUDGE_CANDIDATES = 8

JUDGE_PROMPT = """A job requirement and some achievements from a candidate's CV.

Requirement: {requirement}

Achievements:
{achievements}

Which achievements directly show that the candidate has done what the requirement asks?
Be strict: count one only if a recruiter reading it would say "yes, this person has done
that". A shared topic or a vague similarity does not count, and a requirement naming a
tool or product needs an achievement that uses it.

Reply with only the matching ids, comma-separated (e.g. "ach_03, ach_12"), or NONE.
"""


def judge_matches(requirement: str, candidates: List[RetrievedAchievement]) -> List[str]:
    """Ids of the candidates the LLM judges to really show the requirement, best first."""
    from llm_client import call_llm  # imported here so the rule-based matching needs no LLM

    candidates = candidates[:JUDGE_CANDIDATES]
    if not candidates:
        return []
    prompt = JUDGE_PROMPT.format(
        requirement=requirement,
        achievements="\n".join(f"[{c.achievement.id}] {c.achievement.text}" for c in candidates),
    )
    reply = call_llm(prompt, max_tokens=64)
    named = set(re.findall(r"\bach_\d+\b", reply))
    return [c.achievement.id for c in candidates if c.achievement.id in named]


def assess_gap(
    requirement: str,
    retrieved: List[RetrievedAchievement],
    terms: Iterable[str] = (),
    threshold: float = MATCH_THRESHOLD,
    evidence: Optional[Dict[str, List[str]]] = None,
) -> GapAssessment:
    if not retrieved:
        return GapAssessment(
            requirement=requirement,
            has_strong_match=False,
            best_candidate=None,
            gap_statement=f"No relevant experience found for: {requirement}",
        )

    top = retrieved[0]
    if any(is_match(requirement, r, terms, threshold, evidence) for r in retrieved):
        return GapAssessment(requirement=requirement, has_strong_match=True)

    return GapAssessment(
        requirement=requirement,
        has_strong_match=False,
        best_candidate=top.achievement.text,
        gap_statement=(
            f"No strong match for '{requirement}'. Closest adjacent experience: "
            f"\"{top.achievement.text}\" -- review whether this is honestly close "
            f"enough to mention as adjacent experience, or should be omitted."
        ),
    )


if __name__ == "__main__":
    # Minimal smoke test with fake data -- real usage wires this to retrieve.py output.
    from ingest import Achievement

    fake_achievement = Achievement(
        id="ach_test",
        company="Test Co",
        role="Test Role",
        date_range="2024",
        text="Built Airflow pipelines for reporting.",
        tags=["airflow", "pipeline"],
    )
    fake_retrieved = [RetrievedAchievement(achievement=fake_achievement, score=-2.5)]
    result = assess_gap("dbt metrics layer experience", fake_retrieved)
    print(result.model_dump_json(indent=2))
