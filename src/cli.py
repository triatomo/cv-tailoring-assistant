"""
Entry point: takes a JD (text file, PDF file, or pasted text), runs the full
pipeline, and outputs:
  - a tailored CV as PDF (per-JD summary, page-1 roles filled with the
    most relevant rewritten bullets and fitted to one page, then the remaining
    roles, education, and skills on page 2), and
  - review notes printed to the terminal (rewrites, anything dropped to fit, honest gaps).

Usage:
    python src/cli.py --jd path/to/jd.txt
    python src/cli.py --jd path/to/jd.pdf
    python src/cli.py --jd-text "paste the JD text directly"
    python src/cli.py --jd path/to/jd.txt --lang de   # German CV
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from classify_jd import classify_jd
from retrieve import retrieve_for_requirement, score_against_jd
from rewrite import rewrite_achievement
from verify import verify_rewrite, get_final_bullet
from gaps import assess_gap, is_match, judge_matches, tool_terms
from embed import EMBEDDING_MODEL
from ingest import load_achievements
from cv_profile import Profile, load_profile, skill_sets_for, with_relocation, with_skills
from summary import generate_summary
from render_cv import CVBullet, CVRole, fit_page_one, write_cv
from translate_cv import LABELS, LANGUAGES, translate_cv

load_dotenv()


def _read_pdf_text(path: Path) -> str:
    """Extract plain text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print(
            "Error: reading PDF files requires the 'pypdf' package. "
            "Install it with: pip install pypdf",
            file=sys.stderr,
        )
        sys.exit(1)

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    if not text.strip():
        print(
            f"Warning: no extractable text found in {path}. "
            "The PDF may be scanned/image-based rather than text-based -- "
            "in that case, paste the JD text manually via --jd-text instead.",
            file=sys.stderr,
        )
    return text


def read_jd_text(args) -> str:
    if args.jd:
        path = Path(args.jd)
        if not path.exists():
            print(f"Error: JD file not found: {args.jd}", file=sys.stderr)
            sys.exit(1)
        if path.suffix.lower() == ".pdf":
            return _read_pdf_text(path)
        return path.read_text(encoding="utf-8")
    return args.jd_text


CANDIDATES = 15  # hybrid-search results reranked per requirement before matching


class PipelineResult(NamedTuple):
    review_md: str
    profile: Profile
    headline: str
    summary: str
    roles: List[CVRole]
    labels: Dict[str, str]


def run_pipeline(jd_text: str, top_k: int = 3, include: Optional[Set[str]] = None, lang: str = "en") -> PipelineResult:
    """include: achievement ids that must be on the CV regardless of relevance.
    lang: CV language; tailoring runs in English, then the CV is translated."""
    include = include or set()
    print("Loading embedding model...", file=sys.stderr)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Classifying JD...", file=sys.stderr)
    classification = classify_jd(jd_text)

    profile = with_skills(with_relocation(load_profile(), jd_text), classification.role_category)
    # Only achievements of roles the profile lists go on the CV; the rest stay in
    # the bank (e.g. removed working-student roles) but are never retrieved or used.
    cv_roles = {(e.company, e.role) for e in profile.experience}
    achievements = [a for a in load_achievements() if (a.company, a.role) in cv_roles]
    cv_ids = {a.id for a in achievements}
    terms = tool_terms(
        (skill for s in profile.skill_sets.values() for skill in s.skills),
        (tag for a in achievements for tag in a.tags),
    )
    unknown = include - cv_ids
    if unknown:
        print(f"Error: --include ids not on the CV: {', '.join(sorted(unknown))}", file=sys.stderr)
        sys.exit(1)

    # For each achievement, remember the JD requirement it matched best, so it
    # can be ranked for page 1 and rewritten toward that requirement.
    best_match: Dict[str, Tuple[float, str]] = {}
    gaps_section = ["## Gaps", ""]
    match_notes = ["## Matches", "", "Bullets matched to each JD requirement; (LLM) = judged by the LLM, check these.", ""]

    for requirement in classification.key_requirements:
        print(f"Job requirement: {requirement}", file=sys.stderr)
        # Rerank every hybrid-search candidate: a bullet naming the required tool can
        # score low and still be the real match (see gaps.is_match).
        retrieved = retrieve_for_requirement(requirement, classification, embedding_model, top_k=CANDIDATES)
        retrieved = [r for r in retrieved if r.achievement.id in cv_ids]

        # Rule matches (rerank score, shared tool, term_evidence) first, then the
        # bullets the LLM judges to show the requirement -- the reranker can't judge
        # responsibility requirements on its own (see gaps.JUDGE_PROMPT).
        rule_matched = [r for r in retrieved if is_match(requirement, r, terms, evidence=profile.term_evidence)]
        judged = set(judge_matches(requirement, retrieved))
        matches = rule_matched + [r for r in retrieved if r.achievement.id in judged and r not in rule_matched]
        if not matches:
            gap = assess_gap(requirement, retrieved, terms, evidence=profile.term_evidence)
            gaps_section.append(f"- **{requirement}**: {gap.gap_statement}")
            continue
        match_notes.append(
            f"- **{requirement}**: "
            + ", ".join(f"`{r.achievement.id}`" + ("" if r in rule_matched else " (LLM)") for r in matches[:top_k])
        )

        for candidate in matches[:top_k]:
            current = best_match.get(candidate.achievement.id)
            if current is None or candidate.score > current[0]:
                best_match[candidate.achievement.id] = (candidate.score, requirement)

    # Score every page-1 achievement against the JD, not just the ones that made
    # a requirement's top-k, so each page-1 role keeps its most relevant bullets.
    page_one_roles = {(e.company, e.role) for e in profile.experience if e.page == 1}
    relevance = score_against_jd(
        classification.key_requirements,
        [a for a in achievements if (a.company, a.role) in page_one_roles],
    )
    # Position in the JD of the requirement each matched bullet answers: page-1
    # bullets are listed in that order, so the CV follows the JD's priorities.
    jd_position = {
        a_id: classification.key_requirements.index(requirement)
        for a_id, (_, requirement) in best_match.items()
    }

    rewrite_notes = ["## Rewrites", ""]
    roles: List[CVRole] = []
    for entry in profile.experience:
        role_achievements = [a for a in achievements if a.company == entry.company and a.role == entry.role]
        limit = entry.max_bullets
        on_target = not entry.categories or classification.role_category in entry.categories
        min_bullets = entry.min_bullets
        fill_to = min_bullets
        if entry.page == 1:
            role_achievements.sort(key=lambda a: relevance.get(a.id, float("-inf")), reverse=True)
            if not on_target:
                limit = entry.min_bullets  # role is off-target for this JD: keep it short
            elif limit is not None:
                # On-target: fill to max_bullets, and page-1 fitting trims the least
                # relevant back toward min_bullets. A role that is the core of this JD
                # keeps all of them; fitting trims the other roles first.
                fill_to = limit
                if classification.role_category in entry.full_categories:
                    min_bullets = limit
        if entry.page == 1:
            # Pinned bullets first, then bullets matching a JD requirement, then the
            # most relevant of the rest up to fill_to.
            pinned = [a for a in role_achievements if a.id in include]
            matched = [a for a in role_achievements if a.id not in include and a.id in best_match]
            others = [a for a in role_achievements if a.id not in include and a.id not in best_match]
            chosen = pinned + matched
            if limit is not None:
                chosen = chosen[:max(limit, len(pinned))]
            chosen += others[:max(0, fill_to - len(chosen))]
            role_achievements = sorted(
                chosen, key=lambda a: (jd_position.get(a.id, float("inf")), -relevance.get(a.id, 0.0))
            )
        elif limit is not None:
            role_achievements = role_achievements[:limit]

        bullets = []
        for achievement in role_achievements:
            text = achievement.text
            match = best_match.get(achievement.id)
            if match is not None:
                print(f"Rewriting {achievement.id} toward: {match[1]}", file=sys.stderr)
                rewrite_result = rewrite_achievement(
                    achievement,
                    requirement=match[1],
                    terminology=classification.key_terminology,
                )
                verification = verify_rewrite(rewrite_result)
                text = get_final_bullet(rewrite_result, verification)
                rewrite_notes.append(f"- `{achievement.id}` ({entry.role}) toward *{match[1]}*: {text}")
                if not verification.is_supported:
                    rewrite_notes.append(
                        f"  - *(Note: rewrite was reverted to original -- flagged issues: "
                        f"{'; '.join(verification.issues)})*"
                    )
            bullets.append(
                CVBullet(
                    text=text,
                    highlights=achievement.highlights,
                    relevance=relevance.get(achievement.id),
                    achievement_id=achievement.id,
                    pinned=achievement.id in include,
                )
            )
        roles.append(
            CVRole(
                company=entry.company,
                role=entry.role,
                location=entry.location,
                date_range=entry.date_range,
                page=entry.page,
                min_bullets=min_bullets,
                bullets=bullets,
            )
        )

    print("Writing summary (cv-summary skill via Claude Pro)...", file=sys.stderr)
    cv_summary = generate_summary(jd_text, classification, profile, achievements)

    english_summary = cv_summary.summary
    summary = english_summary
    translation_failures: List[str] = []
    if lang != "en":
        print(f"Translating CV ({LANGUAGES[lang]})...", file=sys.stderr)
        profile, summary, roles, translation_failures = translate_cv(lang, profile, cv_summary.headline, summary, roles)

    # After translation: German runs longer, so fit what will actually be printed.
    print("Fitting page 1...", file=sys.stderr)
    dropped, fits = fit_page_one(profile, cv_summary.headline, summary, roles, LABELS[lang])

    output_lines = [
        f"# Tailored CV Draft -- Review Notes",
        f"",
        f"**Detected role category:** {classification.role_category}",
        f"",
        f"**Key terminology to mirror:** {', '.join(classification.key_terminology)}",
        f"",
        f"**Contact line:** {' • '.join(profile.contact)}",
        f"",
        f"**Skill set:** {' + '.join(skill_sets_for(profile, classification.role_category))}",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"**{cv_summary.headline}**",
        f"",
        english_summary,
        f"",
        *cv_summary.review_notes,
        f"",
    ]
    if lang != "en":
        output_lines.extend([f"**{LANGUAGES[lang]}:** {summary}", ""])
    if translation_failures:
        output_lines.extend([f"**Kept in English (translation failed checks):**", ""])
        output_lines.extend(f"- {f}" for f in translation_failures)
        output_lines.append("")
    if not cv_summary.is_generated:
        output_lines.append(
            f"*(Note: using the default summary from data/profile.json -- "
            f"{'; '.join(cv_summary.issues)})*"
        )
        output_lines.append("")
    if not fits:
        output_lines.append(
            "**Warning:** page 1 still overflows with every page-1 role at its min_bullets -- "
            "shorten the summary or lower min_bullets in data/profile.json."
        )
        output_lines.append("")
    output_lines.extend(rewrite_notes)
    output_lines.append("")
    if dropped:
        output_lines.extend(["## Dropped to fit page 1", ""])
        output_lines.extend(f"- `{b.achievement_id}`: {b.text}" for b in dropped)
        output_lines.append("")
    output_lines.extend(match_notes)
    output_lines.append("")
    output_lines.extend(gaps_section)

    return PipelineResult(
        review_md="\n".join(output_lines),
        profile=profile,
        headline=cv_summary.headline,
        summary=summary,
        roles=roles,
        labels=LABELS[lang],
    )


def main():
    parser = argparse.ArgumentParser(description="Tailor CV bullets to a job description.")
    parser.add_argument("--jd", type=str, help="Path to a JD text file.")
    parser.add_argument("--jd-text", type=str, help="JD text pasted directly.")
    parser.add_argument("--top-k", type=int, default=3, help="Candidates to retrieve per requirement.")
    parser.add_argument("--out", type=str, default="tailored_cv.pdf", help="Output CV PDF path (default: tailored_cv.pdf).")
    parser.add_argument(
        "--include",
        type=str,
        default="",
        help="Comma-separated achievement ids that must be on the CV, e.g. ach_11,ach_13.",
    )
    parser.add_argument(
        "--lang",
        choices=["en", *LANGUAGES],
        default="en",
        help="CV language (default: en). Tailoring runs in English; the finished CV is then translated.",
    )
    args = parser.parse_args()

    if not args.jd and not args.jd_text:
        print("Error: provide either --jd <file> or --jd-text \"<text>\"", file=sys.stderr)
        sys.exit(1)

    jd_text = read_jd_text(args)
    include = {i.strip() for i in args.include.split(",") if i.strip()}
    result = run_pipeline(jd_text, top_k=args.top_k, include=include, lang=args.lang)

    print(result.review_md)

    cv_path = Path(args.out).with_suffix(".pdf")
    write_cv(result.profile, result.headline, result.summary, result.roles, cv_path, result.labels)
    print(f"CV written to {cv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
