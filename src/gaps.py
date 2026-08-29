"""
Gap detection: for a JD requirement where retrieval found no strong match
(below a similarity/rerank threshold), output an honest gap statement instead
of forcing a weak bullet to fit -- mirroring the "I haven't worked with X,
though my Y experience is adjacent" pattern used throughout manual CV tailoring.
"""
from typing import List, Optional
from pydantic import BaseModel

from retrieve import RetrievedAchievement

# Reranker (cross-encoder) scores are typically in a rough logit range;
# tune this threshold empirically once you see real scores from your data.
# Start conservative -- it's better to under-flag matches as gaps and review
# manually than to silently force a weak match.
MATCH_THRESHOLD = 0.0


class GapAssessment(BaseModel):
    requirement: str
    has_strong_match: bool
    best_candidate: Optional[str] = None  # closest adjacent achievement text, if any
    gap_statement: Optional[str] = None


def assess_gap(requirement: str, retrieved: List[RetrievedAchievement], threshold: float = MATCH_THRESHOLD) -> GapAssessment:
    if not retrieved:
        return GapAssessment(
            requirement=requirement,
            has_strong_match=False,
            best_candidate=None,
            gap_statement=f"No relevant experience found for: {requirement}",
        )

    top = retrieved[0]
    if top.score >= threshold:
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
